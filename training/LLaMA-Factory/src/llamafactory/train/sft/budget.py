# Copyright 2026 the LlamaFactory team.
#
# Budget-aware audio description: a duration head that predicts how many
# seconds remain before the next line of dialogue, read at a marker token.
#
#   variant A   <|dur|>Jack <|dur|>and ...      value exists only in the head
#   variant B   <|dur|>2.5<|dur_end|>Jack ...    value is also written as text
#
# Both use the SAME marker token <|dur|>; B merely adds a closing <|dur_end|>.
# That keeps one gather path, one head, and one embedding init for both, so the
# comparison isolates "can the model see its own estimate?" and nothing else.
#
# In both variants the head reads the hidden state AT THE MARKER POSITION -
# the state whose logits produce the following token. One rule, one code path.
import os
import torch
import torch.nn as nn

from ...extras import logging

logger = logging.get_logger(__name__)

# We reuse two tokens that already exist in Qwen's vocabulary rather than adding
# new ones. Qwen3.5-4B ties lm_head to embed_tokens, so a *new* token's output row
# can only be trained by unfreezing the whole 248k x 2560 matrix (636M params,
# ~7.6GB of optimizer state). A token the model was pretrained on already has a
# usable output row, so it can be emitted at inference with no extra parameters.
# <|box_start|>/<|box_end|> never occur in this corpus.
DUR_TOK, DUR_END = "<|box_start|>", "<|box_end|>"
SPECIALS = {"A": [DUR_TOK], "B": [DUR_TOK, DUR_END]}
MARKER = DUR_TOK          # identical in both variants
HEAD_FILE = "duration_head.pt"


class DurationHead(nn.Module):
    """Final hidden state -> one scalar (seconds)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, h: "torch.Tensor") -> "torch.Tensor":
        # Follow the dtype the backbone hands us. LLaMA-Factory upcasts trainable
        # params to fp32 unless pure_bf16 is set, so the last hidden state can be
        # either fp32 or bf16 depending on the config - match it rather than assume.
        if self.proj.weight.dtype != h.dtype:
            self.proj.to(h.dtype)
        return self.proj(h).squeeze(-1)

    def save(self, output_dir: str) -> None:
        torch.save(self.state_dict(), os.path.join(output_dir, HEAD_FILE))

    @classmethod
    def load(cls, output_dir: str, hidden_size: int, device=None):
        path = os.path.join(output_dir, HEAD_FILE)
        if not os.path.exists(path):
            return None
        head = cls(hidden_size)
        head.load_state_dict(torch.load(path, map_location="cpu"))
        return head.to(device).float().eval() if device else head.float().eval()


def register_tokens(tokenizer, variant: str) -> int:
    """Add the variant's tokens and stash the marker id on the tokenizer."""
    # already in the vocabulary - nothing to add, just record the id
    tokenizer.budget_marker_id = tokenizer.convert_tokens_to_ids(MARKER)
    return 0


def init_marker_embeddings(model, tokenizer, variant: str) -> bool:   # noqa: ARG001
    """No-op: the markers are tokens the model was already pretrained on."""
    # The markers are pretrained tokens, so their rows are already meaningful -
    # overwriting them would throw away exactly what makes them emittable.
    return True


class LastHiddenCapture:
    """Capture only the final hidden state via a hook.

    `output_hidden_states=True` returns every layer - 36 of them for Qwen3.5-4B,
    about 2.2 GB at a 12k-token sequence - when the duration head needs just the
    last. A hook on the final norm keeps one tensor instead of 36.
    """

    def __init__(self, model):
        self.h = None
        mod = self._find_norm(model)
        if mod is None:
            raise RuntimeError(
                "LastHiddenCapture could not locate the final norm module; "
                "cannot capture hidden states."
            )
        self.handle = mod.register_forward_hook(self._hook)
        self.ok = True

    @staticmethod
    def _find_norm(model):
        m = model
        for attr in ("base_model", "model"):          # unwrap PEFT / wrappers
            m = getattr(m, attr, m)
        for path in ("norm", "model.norm", "language_model.norm", "model.language_model.norm"):
            obj = m
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        # Fall back to a search: model variants move the final norm around
        # (Qwen3.5 VL nests it under model.language_model), and a silent miss here
        # surfaces much later as "NoneType is not subscriptable".
        cands = [(n, mod) for n, mod in model.named_modules()
                 if n.split(".")[-1] == "norm" and "layers" not in n]
        if cands:
            logger.info_rank0(f"LastHiddenCapture: hooking {cands[-1][0]}")
            return cands[-1][1]
        return None

    def _hook(self, _mod, _inp, out):
        self.h = out[0] if isinstance(out, tuple) else out

    def close(self):
        if self.handle is not None:
            self.handle.remove()


def gather_marker_states(hidden, input_ids, marker_id):
    """Hidden states at every marker, row-major (= the order targets were written)."""
    mask = input_ids.eq(marker_id)
    if not mask.any():
        return hidden.new_zeros((0, hidden.size(-1)))
    b, t = mask.nonzero(as_tuple=True)
    return hidden[b, t]
