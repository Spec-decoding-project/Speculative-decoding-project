"""Native five-layer conditional references, with and without token feedback.

All private drafter parameters train, including context projection. The original
drafter, target, embeddings, and output head remain frozen. No loading, network,
or cloud execution occurs on import.
"""
from copy import deepcopy
from dataclasses import dataclass
import weakref

import torch
from torch import Tensor, nn

from conditional_redraft_model import (
    DraftState, FrozenDraft, _claim_cache, _validate_inputs,
)

ARMS = ('feedback', 'no_feedback')


@dataclass(frozen=True)
class RawFeatureState(DraftState):
    raw_target_hidden: Tensor


class FullReferenceFirstPass(FrozenDraft):
    """Retain unprojected COMMITTED features for the private trained projection."""

    @torch.no_grad()
    def capture(self, *, noise_embedding, target_hidden, position_ids, **kwargs):
        state = super().capture(noise_embedding=noise_embedding,
            target_hidden=target_hidden, position_ids=position_ids, **kwargs)
        return RawFeatureState(**vars(state), raw_target_hidden=target_hidden.detach())


class FullConditionalReference(nn.Module):
    """Literal native input conditioning; no learned midpoint adapter.

Both arms clamp the recorded first future token in the output candidate. Only
the feedback arm feeds its embedding into the model. Clamping is done by the
caller after prediction; it must not be confused with conditioning.
"""

    def __init__(self, ordinary: FullReferenceFirstPass, mode: str):
        super().__init__()
        if mode not in ARMS:
            raise ValueError(mode)
        if any(p.requires_grad for p in ordinary.parameters()):
            raise ValueError('The original drafter must be frozen')
        self._first_pass = weakref.ref(ordinary)
        self.mode = mode
        self.drafter = deepcopy(ordinary.drafter).requires_grad_(True).eval()

    def forward(self, state: RawFeatureState, prefix_embeddings: Tensor, *, past_key_values=None):
        ordinary = self._first_pass()
        if (ordinary is None or not isinstance(state, RawFeatureState)
                or state.producer != id(ordinary) or state.split != ordinary.split):
            raise ValueError('Raw committed features must belong to this first pass')
        if prefix_embeddings.shape != (state.noise.shape[0], 1, state.noise.shape[2]):
            raise ValueError('This comparison fixes exactly one visible future token')
        if prefix_embeddings.device != state.noise.device:
            raise ValueError('Inputs must share a device')
        raw = state.raw_target_hidden.detach()
        if raw.shape[-1] != self.drafter.fc.in_features:
            raise ValueError('Wrong unprojected feature width')
        past = 0 if past_key_values is None else past_key_values.get_seq_length()
        if past != state.context_past:
            raise ValueError('Private cache and first pass cover different histories')
        _validate_inputs(state.noise, raw, state.positions, past_key_values)
        _claim_cache(past_key_values, self, self.parameters())
        noise = state.noise.detach()
        if self.mode == 'feedback':
            noise = torch.cat((noise[:, :1], prefix_embeddings.detach(), noise[:, 2:]), dim=1)
        # Native forward recomputes the PRIVATE feature projection and all five
        # layers. Reusing detached q0 projections here would lose their gradients.
        return self.drafter(noise_embedding=noise, target_hidden=raw,
            position_ids=state.positions, attention_mask=state.attention_mask,
            past_key_values=past_key_values, use_cache=past_key_values is not None,
            is_causal=False)


def restore_reference_example(target, ordinary, bank, metadata, index):
    from conditional_redraft_learning import restore_example
    state, prefix, original, labels, valid = restore_example(
        target, ordinary, bank, metadata, index)
    anchor = metadata['anchors'][index]['anchor']
    raw = bank['features'][:anchor].to(target.device)[None]
    assert raw.shape[1] == anchor
    return (RawFeatureState(**vars(state), raw_target_hidden=raw),
            prefix, original, labels, valid)


def make_full_references(ordinary, *, dtype=torch.float32):
    models = {arm: FullConditionalReference(ordinary, arm).to(dtype=dtype) for arm in ARMS}
    first = models[ARMS[0]].state_dict()
    if any(not torch.equal(v, models[ARMS[1]].state_dict()[k]) for k, v in first.items()):
        raise ValueError('Reference arms do not start from identical weights')
    owners = [{id(p) for p in model.parameters()} for model in models.values()]
    if owners[0] & owners[1] or any(ids & {id(p) for p in ordinary.parameters()} for ids in owners):
        raise ValueError('Independent arms share trainable tensors')
    if not all(p.requires_grad for model in models.values() for p in model.parameters()):
        raise ValueError('A reference parameter is accidentally frozen')
    return models
