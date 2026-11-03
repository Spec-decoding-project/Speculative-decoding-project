"""Reconstruct final first-pass features from the historical midpoint bank.

The old restore_example intentionally uses midpoint placeholders for final and
normalized fields. Never pass those placeholders to the compact editor.
This path is for no-cache historical training examples, not cached serving.
"""
from dataclasses import replace
import torch
from conditional_full_reference import restore_reference_example


@torch.no_grad()
def finish_saved_first_pass(ordinary, state):
    if state.producer != id(ordinary) or state.split != ordinary.split:
        raise ValueError('Saved state does not belong to this first pass')
    if state.context_past != 0 or state.shared_lower_context is not None:
        raise ValueError('Only no-cache historical examples are supported')
    hidden = state.midpoint
    for layer in ordinary.drafter.layers[ordinary.split:]:
        hidden = layer(hidden_states=hidden, target_hidden=state.projected_context,
                       attention_mask=state.attention_mask, position_ids=state.positions,
                       position_embeddings=state.rotary, past_key_value=None,
                       use_cache=False, is_causal=False)
    return replace(state, final=hidden.detach(),
                   normalized=ordinary.drafter.norm(hidden).detach())


@torch.no_grad()
def restore_compact_example(target, ordinary, bank, metadata, index):
    state, visible, original, labels, valid = restore_reference_example(
        target, ordinary, bank, metadata, index)
    state = finish_saved_first_pass(ordinary, state)
    scores = target.get_output_embeddings()(state.normalized[:, 1:])
    if not torch.equal(scores.argmax(-1), original[:, 1:]):
        raise ValueError('Reconstructed first pass differs from saved draft; preserve discrepancy before training')
    return state, visible[:, 0], scores, labels, valid
