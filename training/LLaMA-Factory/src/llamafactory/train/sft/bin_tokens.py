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
        if "embed_tokens" not in name or not param.requires_grad:
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
