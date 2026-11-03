"""Reference greedy generation with frozen q0 and one conditional alternative.

This correctness path uses eager execution, Python tree packing and host token
decisions. It is NOT an optimized serving benchmark. It creates no cloud jobs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import DynamicCache

from conditional_redraft_layout import TwoChainLayout
from conditional_redraft_model import FrozenDraft, ConditionalRefiner
from treeflash_static_verifier import StaticVerifier


@dataclass(frozen=True)
class ConditionalGeneration:
    output_ids: Tensor
    rounds: tuple[dict, ...]
    stopping_reason: str


def _features(hidden_states, layer_ids, nodes=None):
    # HF hidden_states[0] is the embedding output; layer i is at index i+1.
    states = [hidden_states[i + 1] for i in layer_ids]
    if nodes is not None:
        states = [state.index_select(1, nodes) for state in states]
    return torch.cat(states, dim=-1)


@torch.no_grad()
def generate_conditional(target, ordinary: FrozenDraft, refiner: ConditionalRefiner,
                         input_ids: Tensor, *, max_new_tokens: int,
                         eos_token_ids=(), cut: int = 1) -> ConditionalGeneration:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("Reference generation requires one nonempty prompt")
    if input_ids.dtype != torch.long or type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("Require long token IDs and a nonnegative generation cap")
    if input_ids.device != target.device:
        raise ValueError("Prompt must be on the target device")
    if refiner._first_pass() is not ordinary:
        raise ValueError("Refiner and ordinary drafter must share the frozen producer")
    block_size = ordinary.drafter.block_size
    layout = TwoChainLayout.build(block_size - 1, cut)
    eos = frozenset(eos_token_ids)
    if max_new_tokens == 0:
        return ConditionalGeneration(input_ids.clone(), (), "length")
    target.eval()
    ordinary.eval()
    refiner.eval()
    prefill = target(input_ids, use_cache=True, output_hidden_states=True)
    anchor = prefill.logits[0, -1].argmax().item()
    committed = []
    traces = []
    layer_ids = ordinary.drafter.target_layer_ids
    context_features = _features(prefill.hidden_states, layer_ids)
    # Allocate a fixed-address verifier once; no graph capture is claimed here.
    prefix_state = [(layer.keys, layer.values) for layer in prefill.past_key_values.layers]
    verifier = None
    draft_cache, refiner_cache = DynamicCache(), DynamicCache()
    embed = target.get_input_embeddings()
    readout = target.get_output_embeddings()

    while len(committed) < max_new_tokens:
        remaining = max_new_tokens - len(committed)
        if anchor in eos or remaining == 1:
            committed.append(anchor)
            break
        if verifier is None:
            verifier = StaticVerifier(
                target, prefix_state, nodes=layout.node_count,
                capacity=input_ids.shape[1] + max_new_tokens + layout.node_count,
            )
            # Context features were materialized above. Do not retain the
            # original full prefill activations and duplicate dynamic K/V.
            del prefill, prefix_state
        past = verifier.past
        noise_ids = input_ids.new_full((1, block_size), ordinary.drafter.mask_token_id)
        noise_ids[0, 0] = anchor
        state = ordinary.capture(
            noise_embedding=embed(noise_ids), target_hidden=context_features,
            position_ids=torch.arange(draft_cache.get_seq_length(), past + block_size,
                                      device=input_ids.device)[None],
            past_key_values=draft_cache,
        )
        original_tensor = readout(state.normalized[:, 1:]).argmax(-1)
        refined_hidden = refiner(state, embed(original_tensor[:, :cut]), past_key_values=refiner_cache)
        refined_tensor = torch.cat((original_tensor[:, :cut],
                                    readout(refined_hidden[:, cut + 1:]).argmax(-1)), dim=1)
        draft_cache.crop(past)
        refiner_cache.crop(past)
        original, refined = original_tensor[0].tolist(), refined_tensor[0].tolist()
        packed = layout.pack_tokens(anchor, original, refined)
        allowed = torch.tensor(layout.allowed_attention(past), device=input_ids.device)
        mask = torch.zeros((1, 1, layout.node_count, past + layout.node_count),
                           device=input_ids.device, dtype=target.dtype)
        mask.masked_fill_(~allowed[None, None], torch.finfo(target.dtype).min)
        verified = verifier(input_ids.new_tensor([packed]),
                            input_ids.new_tensor([layout.positions(past)]), mask)
        decision = layout.decide(packed, verified.logits[0].argmax(-1).tolist(),
                                 eos_token_ids=eos, max_new_tokens=remaining)
        nodes = input_ids.new_tensor(decision.accepted_nodes)
        context_features = _features(verified.hidden_states, layer_ids, nodes)
        verifier.compact(past, decision.accepted_nodes)
        committed.extend(decision.accepted_tokens)
        traces.append(dict(
            start=past, original_accepted=decision.original_accepted,
            refined_accepted=decision.refined_accepted, selected=decision.branch,
            accepted_nodes=list(decision.accepted_nodes),
            accepted_tokens=list(decision.accepted_tokens), bonus=decision.bonus_token,
            target_cache_length=verifier.past,
            draft_context_length=draft_cache.get_seq_length(),
            refiner_context_length=refiner_cache.get_seq_length(),
        ))
        if decision.bonus_token is None:
            break
        anchor = decision.bonus_token
        # A pending bonus has no target cache row. It is consumed as next round's
        # anchor, except when it already terminates this bounded generation.
        if anchor in eos or len(committed) + 1 == max_new_tokens:
            committed.append(anchor)
            break

    output = torch.cat((input_ids, input_ids.new_tensor([committed])), dim=1)
    reason = "eos" if committed and committed[-1] in eos else "length"
    return ConditionalGeneration(output, tuple(traces), reason)
