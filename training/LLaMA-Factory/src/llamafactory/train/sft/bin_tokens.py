r"""Restrict embedding gradients to the newly added budget-bin tokens.

Qwen3.5 ties ``lm_head`` to ``embed_tokens``, so a token can only be *emitted*
if its row in the embedding matrix receives gradient. Making the matrix
trainable (``additional_target: embed_tokens``) achieves that, but it also lets
all 248k pretrained rows drift at the LoRA learning rate. We therefore zero the
gradient of every row except the new ones, so the pretrained vocabulary is
preserved bit-for-bit while the new tokens still learn.
"""

import torch
from ...extras import logging

logger = logging.get_logger(__name__)


def resolve_new_token_ids(tokenizer, tokens):
    """Map token strings to ids, dropping any the tokenizer does not know."""
    ids = []
    for t in tokens:
        i = tokenizer.convert_tokens_to_ids(t)
        if i is None or i == tokenizer.unk_token_id:
            logger.warning_rank0(f"Token {t} not found in tokenizer, skipping.")
            continue
        enc = tokenizer(t, add_special_tokens=False)["input_ids"]
        if len(enc) != 1:
            raise ValueError(f"{t} encodes to {len(enc)} pieces, expected 1.")
        ids.append(i)
    return ids


def restrict_embedding_grad(model, token_ids):
    """Zero embedding gradient outside ``token_ids``. Returns #hooks installed."""
    if not token_ids:
        return 0

    n = 0
    for name, param in model.named_parameters():
        # lm_head too: a model with untied output embeddings (e.g. Qwen3-VL-8B) needs
        # `additional_target: embed_tokens,lm_head`, and only the new rows may move there as well.
        # Tied models (Qwen3.5-4B) have no trainable lm_head parameter, so nothing changes for them.
        if ("embed_tokens" not in name and "lm_head" not in name) or not param.requires_grad:
            continue

        mask = torch.zeros(param.size(0), 1, dtype=torch.bool)
        mask[torch.as_tensor(token_ids)] = True

        def hook(grad, mask=mask):
            return grad * mask.to(device=grad.device)

        param.register_hook(hook)
        n += 1
        logger.info_rank0(
            f"Masked embedding grad on {name}: {len(token_ids)} of {param.size(0)} rows trainable."
        )

    if n == 0:
        logger.warning_rank0(
            "No trainable embed_tokens parameter found - new tokens cannot be emitted. "
            "Set `additional_target: embed_tokens` in the training config."
        )
    return n


class WeightedCE:
    r"""Cross-entropy that counts the new special tokens for more than one token.

    Measured at checkpoint-30 of the first overfit run, the budget-bin tokens carried
    a mean CE of 0.977 against 0.031 for every other target token - 32x higher - yet
    they are only ~5% of the targets, so they barely moved the average and the model
    settled on always emitting the majority bin. Upweighting those classes puts the
    gradient where the loss actually is.

    Liger's fused linear cross-entropy computes the per-token loss without ever
    materialising the [T, 248320] logit matrix (12 GB at 8k tokens). Taking it with
    reduction="none" means one pass yields both the weighted training loss and the
    separate bin / non-bin numbers worth logging.
    """

    def __init__(self, token_ids, weight):
        self.token_ids = list(token_ids)
        self.weight = float(weight)
        self.loss_fn = None
        self.last = {}

    def _fn(self):
        if self.loss_fn is None:
            from liger_kernel.transformers.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyLoss,
            )

            self.loss_fn = LigerFusedLinearCrossEntropyLoss(reduction="none", ignore_index=-100)
        return self.loss_fn

    def __call__(self, hidden, lm_weight, labels):
        """hidden: [B, T, H] (pre-shift). labels: [B, T]. Returns a scalar loss."""
        import torch

        h = hidden[:, :-1].reshape(-1, hidden.size(-1))
        y = labels[:, 1:].reshape(-1)
        keep = y != -100
        h, y = h[keep].to(lm_weight.dtype), y[keep]

        per_tok = self._fn()(lm_weight, h, y).float()

        ids = torch.as_tensor(self.token_ids, device=y.device)
        is_bin = torch.isin(y, ids)
        n_bin = int(is_bin.sum())
        n_oth = int((~is_bin).sum())

        w = torch.where(is_bin, self.weight, 1.0)
        loss = (per_tok * w).sum() / w.sum().clamp(min=1.0)

        self.last = {
            "ce_bin": float(per_tok[is_bin].mean()) if n_bin else 0.0,
            "ce_other": float(per_tok[~is_bin].mean()) if n_oth else 0.0,
            "n_bin": n_bin,
            "pct_bin": round(100.0 * n_bin / max(n_bin + n_oth, 1), 2),
        }
        return loss.to(hidden.dtype)


# --------------------------------------------------------------------------------------------------------------------
# new_token_rows_only: train ONLY the new rows, without copying the embedding matrices.
#
# `additional_target: embed_tokens[,lm_head]` makes PEFT keep a full trainable copy of each matrix (248k x hidden) with
# its own gradient and Adam state, although restrict_embedding_grad lets only the 9 new rows move. For untied-output
# models (Qwen3.5-9B, Qwen3-VL-8B) that is ~2B extra trainable params (~16 GB/GPU) and the run no longer fits a 48 GB
# GPU. Here the frozen base matrices get a parametrization W' = W.index_add(ids, delta) with a trainable
# delta of shape (n_new, hidden), zero-initialised, so the forward/gradients for those rows are the same as with the
# masked full copy (weight decay is 0), and every reader of `.weight` (incl. liger's fused linear CE) sees W'.
# The rows are saved next to each checkpoint as new_token_rows.safetensors (ids, delta and final rows per matrix) and
# re-applied in place at inference (load_new_token_rows) and on resume (delta restored).
# --------------------------------------------------------------------------------------------------------------------
import os
from torch import nn
from torch.nn.utils import parametrize

NEW_ROWS_FILE = "new_token_rows.safetensors"


class _RowDelta(nn.Module):
    def __init__(self, ids, hidden, dtype, device):
        super().__init__()
        self.register_buffer("ids", torch.as_tensor(ids, dtype=torch.long, device=device), persistent=False)
        self.delta = nn.Parameter(torch.zeros(len(ids), hidden, dtype=dtype, device=device))

    def forward(self, w):
        return w.index_add(0, self.ids, self.delta.to(w.dtype))


def _base_param(mod):
    return mod.parametrizations.weight.original if parametrize.is_parametrized(mod, "weight") else mod.weight


def _tied(model):
    lm = model.get_output_embeddings()
    return lm is not None and _base_param(lm) is _base_param(model.get_input_embeddings())


def _row_targets(model):
    """(name, module) for the input embeddings and, if untied, the output embeddings."""
    out = [("embed_tokens", model.get_input_embeddings())]
    lm = model.get_output_embeddings()
    if lm is not None and not _tied(model):
        out.append(("lm_head", lm))
    return out


def attach_new_token_rows(model, token_ids):
    """Freeze the base matrices and attach a trainable (n_new x hidden) delta to each. Returns #params."""
    n = 0
    tied = _tied(model)
    for name, mod in _row_targets(model):
        w = mod.weight
        w.requires_grad_(False)
        rd = _RowDelta(token_ids, w.size(1), w.dtype, w.device)
        parametrize.register_parametrization(mod, "weight", rd, unsafe=True)
        if tied:  # the output layer shares the matrix -> it must see the same delta
            parametrize.register_parametrization(model.get_output_embeddings(), "weight", rd, unsafe=True)
        rd.delta.requires_grad_(True)
        n += rd.delta.numel()
        logger.info_rank0(f"new_token_rows_only: {name} frozen{' (tied lm_head shares it)' if tied else ''}, "
                          f"trainable delta {tuple(rd.delta.shape)} on rows {token_ids}")
    return n


def _deltas(model):
    return {name: mod.parametrizations.weight[0] for name, mod in _row_targets(model)
            if parametrize.is_parametrized(mod, "weight")}


def save_new_token_rows(model, output_dir):
    from safetensors.torch import save_file
    t = {}
    for name, rd in _deltas(model).items():
        t[f"{name}.ids"] = rd.ids.cpu()
        t[f"{name}.delta"] = rd.delta.detach().float().cpu().contiguous()
        base = _base_param(model.get_input_embeddings() if name == "embed_tokens" else model.get_output_embeddings())
        t[f"{name}.rows"] = (base[rd.ids].float() + rd.delta.detach().float()).cpu().contiguous()
    if t:
        os.makedirs(output_dir, exist_ok=True)
        save_file(t, os.path.join(output_dir, NEW_ROWS_FILE))


def restore_new_token_rows(model, ckpt_dir):
    """Resume: load the saved deltas into the attached parametrizations."""
    from safetensors.torch import load_file
    f = os.path.join(ckpt_dir, NEW_ROWS_FILE)
    if not os.path.isfile(f):
        raise FileNotFoundError(f"resume with new_token_rows_only but {f} is missing")
    t = load_file(f)
    for name, rd in _deltas(model).items():
        assert torch.equal(t[f"{name}.ids"], rd.ids.cpu()), f"{name}: token ids differ from checkpoint"
        with torch.no_grad():
            rd.delta.copy_(t[f"{name}.delta"].to(rd.delta.dtype))
        logger.info_rank0(f"new_token_rows_only: restored {name} delta from {f}")


def load_new_token_rows(model, adapter_dir):
    """Inference: write the trained rows into the (plain) base matrices in place. Returns True if a file was found."""
    from safetensors.torch import load_file
    f = os.path.join(adapter_dir, NEW_ROWS_FILE)
    if not os.path.isfile(f):
        return False
    t = load_file(f)
    tgt = dict(_row_targets(model))
    for name in ("embed_tokens", "lm_head"):
        if f"{name}.rows" not in t:
            continue
        mod = tgt.get(name)
        assert mod is not None, f"checkpoint has {name} rows but the model's {name} is tied/missing"
        with torch.no_grad():
            mod.weight[t[f"{name}.ids"].to(mod.weight.device)] = t[f"{name}.rows"].to(mod.weight.dtype).to(mod.weight.device)
        logger.info_rank0(f"new_token_rows_only: loaded {t[f'{name}.ids'].numel()} {name} rows from {f}")
    return True


def new_token_rows_callback():
    from transformers import TrainerCallback

    class _SaveRows(TrainerCallback):
        def on_save(self, args, state, control, model=None, **kw):
            if state.is_world_process_zero and model is not None:
                save_new_token_rows(model, os.path.join(args.output_dir, f"checkpoint-{state.global_step}"))

        def on_train_end(self, args, state, control, model=None, **kw):
            if state.is_world_process_zero and model is not None:
                save_new_token_rows(model, args.output_dir)

    return _SaveRows()
