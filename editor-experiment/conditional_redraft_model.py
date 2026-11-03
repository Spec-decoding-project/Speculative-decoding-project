"""Matched conditional-redraft paths over the released DFlash layer interface.

This module launches no jobs and loads no weights. The ordinary model is frozen.
Only the private upper stack, output norm, and (where needed) adapter train.
Cached tensors describe ONE anchored block; a data loader must establish their
provenance and must never supply future target features as committed context.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import weakref

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AnchoredExample:
    noise_ids: Tensor
    labels: Tensor
    valid: Tensor
    context_features: Tensor
    positions: Tensor


def make_anchored_example(recorded_ids: Tensor, teacher_features: Tensor, *,
                          anchor: int, block_size: int, mask_token_id: int) -> AnchoredExample:
    """Slice one recorded greedy response without exposing future features.

    The known anchor is a token, not an available target feature. Feature row i
    is the target representation AFTER consuming token i. Thus only rows strictly
    before `anchor` can enter this draft. Response provenance is the caller's
    responsibility; recomputed teacher logits must not replace recorded labels.
    """
    if (recorded_ids.ndim != 1 or recorded_ids.dtype != torch.long
            or teacher_features.ndim != 2
            or teacher_features.shape[0] != recorded_ids.shape[0]
            or teacher_features.device != recorded_ids.device):
        raise ValueError("Require aligned recorded tokens and teacher feature rows")
    if not 1 <= anchor < recorded_ids.shape[0] or block_size < 3 or mask_token_id < 0:
        raise ValueError("Invalid anchor, block size, or mask token")
    available = min(block_size, recorded_ids.shape[0] - anchor)
    noise = recorded_ids.new_full((1, block_size), mask_token_id)
    noise[0, 0] = recorded_ids[anchor]
    labels = recorded_ids.new_full((1, block_size), -100)
    labels[0, :available] = recorded_ids[anchor:anchor + available]
    valid = torch.arange(block_size, device=recorded_ids.device)[None] < available
    return AnchoredExample(noise, labels, valid, teacher_features[:anchor][None],
                           torch.arange(anchor + block_size, device=recorded_ids.device)[None])


def _claim_cache(cache, owner, parameters) -> None:
    """Reject cross-model caches and cache use after an optimizer update.

    This Python preflight is outside any future captured CUDA graph. A serving
    integration must preserve these checks when establishing graph/cache owners.
    """
    if cache is None:
        return
    stamp = (id(owner), tuple(p._version for p in parameters))
    existing = getattr(cache, "_conditional_redraft_stamp", None)
    if cache.get_seq_length() and existing != stamp:
        raise ValueError("Context cache belongs to another model or stale weights")
    cache._conditional_redraft_stamp = stamp


def _validate_inputs(noise: Tensor, context: Tensor, positions: Tensor, cache) -> None:
    if noise.ndim != 3 or context.ndim != 3:
        raise ValueError("Noise and context must be [batch, rows, hidden]")
    if noise.shape[0] != context.shape[0] or noise.shape[1] < 3:
        raise ValueError("Batch mismatch or no editable suffix")
    expected = (noise.shape[0], context.shape[1] + noise.shape[1])
    if tuple(positions.shape) != expected or positions.dtype != torch.long:
        raise ValueError("Positions must cover exactly new context plus the draft block")
    if positions.device != noise.device or context.device != noise.device:
        raise ValueError("Inputs must be on one device")
    # The initial implementation represents complete, unpadded contiguous
    # context. Variable-length batches require a separately validated mask path.
    past = 0 if cache is None else cache.get_seq_length()
    expected_positions = torch.arange(past, past + expected[1], device=positions.device)
    if not torch.equal(positions, expected_positions.expand(expected)):
        raise ValueError("Positions must follow the cache and precede the draft block")


@dataclass(frozen=True)
class DraftState:
    noise: Tensor
    projected_context: Tensor
    positions: Tensor
    rotary: tuple[Tensor, Tensor]
    midpoint: Tensor
    final: Tensor
    normalized: Tensor
    split: int
    producer: int
    context_past: int
    attention_mask: Tensor | None
    shared_lower_context: tuple[tuple[Tensor, Tensor], ...] | None


class _ReadOnlyContextCache:
    """Append temporary block K/V to frozen q0 context without mutating it."""

    def __init__(self, layers):
        self.layers = layers

    def update(self, keys, values, layer_idx, cache_kwargs=None):
        context_keys, context_values = self.layers[layer_idx]
        return (torch.cat((context_keys, keys), dim=-2),
                torch.cat((context_values, values), dim=-2))


class FrozenDraft(nn.Module):
    """Exact native first pass with detached intermediate-state capture."""

    def __init__(self, drafter: nn.Module, split: int = 3):
        super().__init__()
        if not 0 < split < len(drafter.layers):
            raise ValueError("Split must leave nonempty lower and upper stacks")
        self.drafter = drafter.requires_grad_(False).eval()
        self.split = split

    def train(self, mode: bool = True):
        # Training a parent module must not turn the frozen path's dropout on.
        super().train(False)
        return self

    @torch.no_grad()
    def capture(self, *, noise_embedding: Tensor, target_hidden: Tensor,
                position_ids: Tensor, past_key_values=None,
                attention_mask: Tensor | None = None) -> DraftState:
        _validate_inputs(noise_embedding, target_hidden, position_ids, past_key_values)
        _claim_cache(past_key_values, self, self.parameters())
        past = 0 if past_key_values is None else past_key_values.get_seq_length()
        model = self.drafter
        context = model.hidden_norm(model.fc(target_hidden))
        # Preserve native input dtype here, including under mixed precision.
        rotary = model.rotary_emb(noise_embedding, position_ids)
        hidden = noise_embedding
        midpoint = None
        for index, layer in enumerate(model.layers):
            hidden = layer(
                hidden_states=hidden, target_hidden=context,
                attention_mask=attention_mask, position_ids=position_ids,
                position_embeddings=rotary, past_key_value=past_key_values,
                use_cache=past_key_values is not None, is_causal=False,
            )
            if index + 1 == self.split:
                midpoint = hidden
        shared_context = None
        if past_key_values is not None:
            context_end = past + context.shape[1]
            shared_context = tuple(
                (layer.keys[..., :context_end, :].detach(),
                 layer.values[..., :context_end, :].detach())
                for layer in past_key_values.layers[:self.split]
            )
        return DraftState(
            noise=noise_embedding.detach(), projected_context=context.detach(),
            positions=position_ids, rotary=rotary, midpoint=midpoint.detach(),
            final=hidden.detach(), normalized=model.norm(hidden).detach(),
            split=self.split, producer=id(self), context_past=past,
            attention_mask=attention_mask,
            shared_lower_context=shared_context,
        )


class PrefixAdapter(nn.Module):
    """Identity feature entry plus ordered visible-token injection."""

    def __init__(self, width: int, *, device, dtype):
        super().__init__()
        self.feature = nn.Linear(width, width, bias=False, device=device, dtype=dtype)
        self.token = nn.Linear(width, width, bias=False, device=device, dtype=dtype)
        self.visible_bias = nn.Parameter(torch.zeros(width, device=device, dtype=dtype))
        self.stage_bias = nn.Parameter(torch.zeros(width, device=device, dtype=dtype))
        nn.init.eye_(self.feature.weight)
        nn.init.zeros_(self.token.weight)

    def forward(self, hidden: Tensor, noise: Tensor, prefix_embeddings: Tensor) -> Tensor:
        cut = prefix_embeddings.shape[1]
        delta = prefix_embeddings - noise[:, 1:1 + cut]
        # Project only visible rows. Do not add token work to every masked row.
        visible = self.token(delta) + self.visible_bias
        injected = F.pad(visible, (0, 0, 1, hidden.shape[1] - cut - 1))
        # FP32 trainable biases must not silently promote a BF16 entry state.
        # Preserve the backbone interface under autocast as well as deployment.
        return (self.feature(hidden) + injected + self.stage_bias).to(hidden.dtype)


class ConditionalRefiner(nn.Module):
    """One of midpoint, final, full-recompute, or fresh-shallow controls.

    `full` freezes the lower stack to isolate recomputation from trainable
    capacity. It is NOT an unconstrained fully trained conditional teacher.
    Each instance owns its trained weights and projected context cache.
    """

    MODES = frozenset({"midpoint", "final", "full", "shallow"})

    def __init__(self, first_pass: FrozenDraft, mode: str):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}")
        if any(p.requires_grad for p in first_pass.parameters()):
            raise ValueError("Ordinary drafter must be frozen")
        # Avoid including the entire frozen model in every private checkpoint.
        self._first_pass = weakref.ref(first_pass)
        self.mode = mode
        self.split = first_pass.split
        base = first_pass.drafter
        self.layers = deepcopy(base.layers[self.split:]).requires_grad_(True)
        self.norm = deepcopy(base.norm).requires_grad_(True)
        for index, layer in enumerate(self.layers):
            layer.self_attn.layer_idx = index
        parameter = next(self.layers.parameters())
        self.adapter = None if mode == "shallow" else PrefixAdapter(
            base.config.hidden_size, device=parameter.device, dtype=parameter.dtype,
        )

    def forward(self, state: DraftState, prefix_embeddings: Tensor, *,
                past_key_values=None) -> Tensor:
        first_pass = self._first_pass()
        if first_pass is None or state.producer != id(first_pass) or state.split != self.split:
            raise ValueError("State was not produced by this frozen drafter/split")
        if prefix_embeddings.ndim != 3:
            raise ValueError("Prefix embeddings must be [batch, cut, hidden]")
        cut = prefix_embeddings.shape[1]
        if (not 1 <= cut < state.noise.shape[1] - 1
                or prefix_embeddings.shape[0] != state.noise.shape[0]
                or prefix_embeddings.shape[2] != state.noise.shape[2]
                or prefix_embeddings.device != state.noise.device):
            raise ValueError("Prefix must leave an editable suffix and match the block")
        expected_past = 0 if past_key_values is None else past_key_values.get_seq_length()
        if expected_past != state.context_past:
            raise ValueError("Refiner and first-pass caches must cover the same committed context")
        _claim_cache(past_key_values, self, self.parameters())
        prefix_embeddings = prefix_embeddings.detach()
        context = state.projected_context.detach()
        common = dict(
            target_hidden=context, attention_mask=state.attention_mask,
            position_ids=state.positions, position_embeddings=state.rotary,
            past_key_value=past_key_values, use_cache=past_key_values is not None,
            is_causal=False,
        )
        if self.mode in {"full", "shallow"}:
            hidden = torch.cat((state.noise[:, :1], prefix_embeddings,
                                state.noise[:, 1 + cut:]), dim=1)
            if self.mode == "full":
                lower_common = common
                if past_key_values is not None:
                    if state.shared_lower_context is None:
                        raise ValueError("Cached full redrafting requires q0's frozen lower context")
                    # The lower weights are unchanged: reuse their already built
                    # committed K/V, but recompute EVERY conditioned block row.
                    # Only the trained upper stack needs its own persistent cache.
                    rows = state.noise.shape[1]
                    lower_common = {
                        **common, "target_hidden": context[:, :0],
                        "position_ids": state.positions[:, -rows:],
                        "position_embeddings": tuple(x[:, -rows:] for x in state.rotary),
                        "past_key_value": _ReadOnlyContextCache(state.shared_lower_context),
                        "use_cache": True,
                    }
                with torch.no_grad():
                    for layer in first_pass.drafter.layers[:self.split]:
                        hidden = layer(hidden_states=hidden, **lower_common)
        elif self.mode == "midpoint":
            hidden = state.midpoint.detach()
        else:
            hidden = state.final.detach()
        if self.adapter is not None:
            hidden = self.adapter(hidden, state.noise, prefix_embeddings)
        for layer in self.layers:
            hidden = layer(hidden_states=hidden, **common)
        return self.norm(hidden)


@dataclass(frozen=True)
class SuffixLoss:
    loss: Tensor
    weight_sum: Tensor
    supervised_tokens: Tensor
    eligible_examples: Tensor


def conditional_suffix_loss(logits: Tensor, labels: Tensor, valid: Tensor,
                            prefix_matches: Tensor, *, cut: int = 1,
                            decay: float = 7.0) -> SuffixLoss:
    """Suffix-relative exponential CE, reduced by total eligible token weight.

    Rows include the anchor. Labels are the recorded target continuation, not
    freshly substituted teacher-forced argmax tokens. Eligibility is training
    supervision only; it must NEVER serve as an oracle inference gate.
    """
    if logits.ndim != 3 or tuple(labels.shape) != tuple(logits.shape[:2]):
        raise ValueError("Logits/labels must have matching batch and block rows")
    if valid.shape != labels.shape or valid.dtype != torch.bool:
        raise ValueError("Valid-token mask must be boolean with the label shape")
    if prefix_matches.shape != labels.shape[:1] or prefix_matches.dtype != torch.bool:
        raise ValueError("Prefix eligibility must be one boolean per example")
    if not 1 <= cut < logits.shape[1] - 1 or not math.isfinite(decay) or decay <= 0:
        raise ValueError("Invalid cut or decay")
    if labels.dtype != torch.long:
        raise ValueError("Labels must be integer token IDs")
    if any(t.device != logits.device for t in (labels, valid, prefix_matches)):
        raise ValueError("Loss inputs must be on one device")
    rows = torch.arange(logits.shape[1], device=logits.device)
    mask = valid & prefix_matches[:, None] & (rows[None] > cut)
    # Ignore padded/ineligible labels even if they are negative sentinels.
    safe_labels = labels.masked_fill(~mask, 0)
    per_token = F.cross_entropy(logits.float().transpose(1, 2), safe_labels, reduction="none")
    weights = torch.exp(-(rows - cut - 1).float().clamp_min(0) / decay) * mask
    weight_sum = weights.sum()
    # The safe zero case remains connected to the graph with zero gradients.
    loss = (per_token * weights).sum() / weight_sum.clamp_min(torch.finfo(torch.float32).tiny)
    return SuffixLoss(loss, weight_sum, mask.sum(), prefix_matches.sum())
