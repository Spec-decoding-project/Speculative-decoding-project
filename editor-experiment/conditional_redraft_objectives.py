"""Offline objectives for a retained original chain and a conditional branch.

No model loading or remote execution. Rows include the known anchor at zero.
Original acceptance is an OFFLINE label, never an inference-time branch gate.
The legacy conditional_suffix_loss remains unchanged for frozen experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class ObjectiveResult:
    # Sum and denominator allow true effective-batch normalization: accumulate
    # unnormalized gradients across microbatches, then divide ONCE at the step.
    numerator: Tensor
    denominator: Tensor
    weights: Tensor
    base_accepted: Tensor
    eligible: Tensor
    useful_surrogate: Tensor

    @property
    def loss(self) -> Tensor:
        """Weighted mean for this WHOLE batch, not a per-example mean."""
        return self.numerator / self.denominator.clamp_min(
            torch.finfo(self.numerator.dtype).tiny)


def _check_tokens(original: Tensor, labels: Tensor, valid: Tensor) -> None:
    if labels.ndim != 2 or labels.shape[0] < 1 or labels.shape[1] < 3:
        raise ValueError("Require a nonempty [batch, anchor + future] token panel")
    if original.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError("Original, recorded labels and validity must align")
    if original.dtype != torch.long or labels.dtype != torch.long or valid.dtype != torch.bool:
        raise ValueError("Token IDs must be int64 and validity boolean")
    if original.device != labels.device or valid.device != labels.device:
        raise ValueError("Token tensors must share a device")
    if not valid[:, 0].all() or ((~valid[:, :-1]) & valid[:, 1:]).any():
        raise ValueError("Validity must be a contiguous prefix including the anchor")
    if (labels[valid] < 0).any() or (original[valid] < 0).any():
        raise ValueError("Valid token IDs cannot be negative")
    if not torch.equal(original[:, 0], labels[:, 0]):
        raise ValueError("The known anchor must be identical")


def accepted_future(original: Tensor, labels: Tensor, valid: Tensor) -> Tensor:
    """Count leading matching FUTURE tokens, excluding the known anchor."""
    _check_tokens(original, labels, valid)
    matching = ((original[:, 1:] == labels[:, 1:]) & valid[:, 1:]).long()
    return matching.cumprod(dim=1).sum(dim=1)


def branch_progress(original: Tensor, alternative: Tensor, labels: Tensor,
                    valid: Tensor, *, cut: int = 1) -> dict[str, Tensor]:
    """Fixed-context hard coverage only; not measured generation or sampling."""
    base = accepted_future(original, labels, valid)
    revised = accepted_future(alternative, labels, valid)
    if not 1 <= cut < labels.shape[1] - 1:
        raise ValueError("Cut must leave an editable suffix")
    if not torch.equal(original[:, :cut + 1], alternative[:, :cut + 1]):
        raise ValueError("Alternative changed the frozen original prefix")
    union = torch.maximum(base, revised)
    return dict(original=base, alternative=revised, union=union,
                additional=union - base, eligible=base >= cut)


def conditional_objective(logits: Tensor, original: Tensor, labels: Tensor,
                          valid: Tensor, *, objective: str, cut: int = 1,
                          smoothing: float = 0.5, decay: float = 7.0) -> ObjectiveResult:
    """CE, standalone prefix utility, or marginal retained-chain utility.

    `ce`: suffix-relative exponential CE, matching the legacy weighted reduction.
    `prefix`: D-PACE-style detached suffix-prefix weights.
    `marginal`: the same weighting construction, counting only reach beyond a0.

    For smoothing=0, the marginal numerator has the gradient of negative
    expected additional coverage under independent suffix sampling. Normalizing
    by total detached weight or smoothing changes that exact identity. Neither
    this proxy nor the hard offline metric establishes deployed latency.

    For gradient accumulation, accumulate numerator gradients AND denominators,
    then divide gradients by the effective-batch denominator before the optimizer
    step. Taking `.loss.backward()` independently on size-one microbatches would
    silently change cross-example allocation and is prohibited by the protocol.
    """
    _check_tokens(original, labels, valid)
    if logits.ndim != 3 or logits.shape[:2] != labels.shape or logits.shape[-1] < 2:
        raise ValueError("Logits must match batch/rows and contain a vocabulary")
    if not logits.is_floating_point() or logits.device != labels.device:
        raise ValueError("Floating logits must share the token device")
    if objective not in {"ce", "prefix", "marginal"}:
        raise ValueError("Unknown objective")
    if not 1 <= cut < logits.shape[1] - 1:
        raise ValueError("Cut must leave an editable suffix")
    if not math.isfinite(smoothing) or not 0 <= smoothing <= 1:
        raise ValueError("Smoothing must be finite and in [0, 1]")
    if not math.isfinite(decay) or decay <= 0:
        raise ValueError("Decay must be finite and positive")
    if (labels[valid] >= logits.shape[-1]).any() or (original[valid] >= logits.shape[-1]).any():
        raise ValueError("Valid token outside the vocabulary")
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite logits")
    base = accepted_future(original, labels, valid)
    eligible = base >= cut
    rows = torch.arange(labels.shape[1], device=logits.device)[None]
    mask = valid & eligible[:, None] & (rows > cut)
    safe_labels = labels.masked_fill(~valid, 0)
    working = logits if logits.dtype == torch.float64 else logits.float()
    nll = F.cross_entropy(working.transpose(1, 2), safe_labels, reduction="none")
    q = torch.exp(-nll)
    suffix_q = q[:, cut + 1:]
    suffix_valid = valid[:, cut + 1:]
    raw_products = suffix_q.cumprod(dim=1)
    extra_mask = suffix_valid & eligible[:, None] & (rows[:, cut + 1:] > base[:, None])
    useful = (raw_products * extra_mask).sum(dim=1)
    if objective == "ce":
        weights = torch.exp(-(rows - cut - 1).clamp_min(0).to(working.dtype) / decay) * mask
    else:
        smoothed = (1 - smoothing) * suffix_q + smoothing
        products = smoothed.cumprod(dim=1)
        rewards = suffix_valid & eligible[:, None]
        if objective == "marginal":
            rewards = rewards & (rows[:, cut + 1:] > base[:, None])
        terms = products * rewards
        suffix_weights = terms.flip(1).cumsum(1).flip(1)
        weights = F.pad(suffix_weights, (cut + 1, 0)) * mask
    weights = weights.detach()
    numerator = (weights * nll).sum()
    denominator = weights.sum()
    return ObjectiveResult(numerator, denominator, weights, base, eligible, useful)
