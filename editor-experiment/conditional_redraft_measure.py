"""Fixed-context complete-round calibration for conditional redrafting.

No loading, cloud submission, training or implicit downloads. Repetition resets
are outside timing; embedding, both proposals, verification, acceptance, target
feature extraction and cache compaction are inside. Only target verification
may use CUDA graphs. This is not an optimized generation benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch import Tensor
from transformers import DynamicCache

from conditional_redraft_decode import _features
from conditional_redraft_layout import GreedyDecision, TwoChainLayout
from conditional_redraft_model import FrozenDraft, ConditionalRefiner
from treeflash_static_verifier import StaticVerifier


def clone_cache(cache):
    """Measurement reset only; not a proposed serving operation."""
    result = DynamicCache()
    for i, layer in enumerate(cache.layers):
        result.update(layer.keys.detach().clone(), layer.values.detach().clone(), i)
    if hasattr(cache, "_conditional_redraft_stamp"):
        result._conditional_redraft_stamp = cache._conditional_redraft_stamp
    return result


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class RoundFixture:
    prefix_ids: Tensor
    anchor: int
    features: Tensor
    previous_length: int
    prefix_state: list
    q0_cache: DynamicCache
    refiner_caches: dict
    layout: TwoChainLayout
    verifiers: dict
    masks: dict
    positions: dict
    setup: dict


@torch.no_grad()
def prepare_fixture(target, ordinary, refiners, prefix_ids, *, incoming_rows, graph=False):
    if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1:
        raise ValueError("One explicit tokenized prefix required")
    past = prefix_ids.shape[1]
    if not 1 <= incoming_rows < past:
        raise ValueError("Need a nonempty previous context and newly committed rows")
    device = prefix_ids.device
    if graph and device.type != "cuda":
        raise ValueError("No CPU surrogate for CUDA graph measurements")
    begin = time.perf_counter()
    previous = past - incoming_rows
    layer_ids = ordinary.drafter.target_layer_ids
    target_cache = DynamicCache()
    # Two actual target calls establish earlier context and the last committed
    # block; no hidden state at/after the current anchor enters the draft.
    prefill = target(prefix_ids[:, :previous], past_key_values=target_cache,
                     use_cache=True, output_hidden_states=True, logits_to_keep=1)
    old_features = _features(prefill.hidden_states, layer_ids)
    del prefill
    tail = target(prefix_ids[:, previous:], past_key_values=target_cache,
                  use_cache=True, output_hidden_states=True, logits_to_keep=1)
    new_features = _features(tail.hidden_states, layer_ids)
    anchor = int(tail.logits[0, -1].argmax())
    prefix_state = [(layer.keys, layer.values) for layer in target_cache.layers]
    del tail
    block = ordinary.drafter.block_size
    embedding = target.get_input_embeddings()
    readout = target.get_output_embeddings()
    q0_cache = DynamicCache()
    previous_noise = prefix_ids.new_full((1, block), ordinary.drafter.mask_token_id)
    previous_noise[0, 0] = prefix_ids[0, previous]
    primed = ordinary.capture(
        noise_embedding=embedding(previous_noise), target_hidden=old_features,
        position_ids=torch.arange(previous + block, device=device)[None],
        past_key_values=q0_cache,
    )
    visible = embedding(readout(primed.normalized[:, 1:2]).argmax(-1))
    private = {}
    for mode, model in refiners.items():
        if model.mode != mode or model._first_pass() is not ordinary:
            raise ValueError("Mismatched refiner")
        cache = DynamicCache()
        model(primed, visible, past_key_values=cache)
        cache.crop(previous)
        private[mode] = cache
    q0_cache.crop(previous)
    del primed, old_features, previous_noise, visible, target_cache
    layout = TwoChainLayout.build(block - 1, cut=1)
    verifiers, masks, positions, setups = {}, {}, {}, {}
    for label, nodes in (("single", block), ("union", layout.node_count)):
        verifier = StaticVerifier(target, prefix_state, nodes=nodes, capacity=past + nodes)
        if label == "single":
            allowed = torch.cat((torch.ones(block, past, device=device, dtype=torch.bool),
                                 torch.ones(block, block, device=device, dtype=torch.bool).tril()), dim=1)
            pos = torch.arange(past, past + block, device=device)[None]
        else:
            allowed = torch.tensor(layout.allowed_attention(past), device=device)
            pos = prefix_ids.new_tensor([layout.positions(past)])
        mask = torch.zeros(1, 1, nodes, past + nodes, device=device, dtype=target.dtype)
        mask.masked_fill_(~allowed[None, None], torch.finfo(target.dtype).min)
        if graph:
            # Values change through copy_inputs; shape/addresses remain fixed.
            placeholder = prefix_ids.new_full((1, nodes), anchor)
            setups[label] = verifier.prepare_graph(prefix_state, placeholder, pos, mask)
        verifiers[label], masks[label], positions[label] = verifier, mask, pos
    synchronize(device)
    return RoundFixture(
        prefix_ids, anchor, new_features, previous, prefix_state, q0_cache,
        private, layout, verifiers, masks, positions,
        dict(seconds=time.perf_counter()-begin, incoming_rows=incoming_rows,
             target_graphs=graph, graph_setup=setups),
    )


def _single_decision(tokens, posterior):
    accepted = 0
    for token, expected in zip(tokens[1:], posterior[:-1], strict=True):
        if token != expected:
            break
        accepted += 1
    return GreedyDecision("original", accepted, accepted,
                          tuple(range(accepted+1)), tuple(tokens[:accepted+1]),
                          posterior[accepted])


@torch.no_grad()
def complete_round(target, ordinary, model, fixture, q0_cache, refiner_cache=None):
    """Returns real verification/compaction work; no saving or clocking here."""
    past = fixture.prefix_ids.shape[1]
    block = ordinary.drafter.block_size
    embedding, readout = target.get_input_embeddings(), target.get_output_embeddings()
    noise_ids = fixture.prefix_ids.new_full((1, block), ordinary.drafter.mask_token_id)
    noise_ids[0, 0] = fixture.anchor
    state = ordinary.capture(
        noise_embedding=embedding(noise_ids), target_hidden=fixture.features,
        position_ids=torch.arange(fixture.previous_length, past + block,
                                  device=noise_ids.device)[None],
        past_key_values=q0_cache,
    )
    original = readout(state.normalized[:, 1:]).argmax(-1)
    if model is None:
        label = "single"
        packed = torch.cat((noise_ids[:, :1], original), dim=1)
    else:
        label = "union"
        hidden = model(state, embedding(original[:, :1]), past_key_values=refiner_cache)
        refined_suffix = readout(hidden[:, 2:]).argmax(-1)
        # Same fixed packing as TwoChainLayout.pack_tokens, entirely on device.
        packed = torch.cat((noise_ids[:, :1], original, refined_suffix), dim=1)
        refiner_cache.crop(past)
    q0_cache.crop(past)
    verifier = fixture.verifiers[label]
    output = verifier(packed, fixture.positions[label], fixture.masks[label])
    # Host acceptance is deliberately charged. This implementation does not
    # claim that its eager draft/host decision path is a competitive runtime.
    tokens = packed[0].tolist()
    posterior = output.logits[0].argmax(-1).tolist()
    decision = (_single_decision(tokens, posterior) if model is None else
                fixture.layout.decide(tokens, posterior))
    nodes = fixture.prefix_ids.new_tensor(decision.accepted_nodes)
    new_features = _features(output.hidden_states, ordinary.drafter.target_layer_ids, nodes)
    verifier.compact(past, decision.accepted_nodes)
    return dict(
        original=tokens[1:block], packed=tokens, posterior=posterior,
        accepted=decision.accepted_tokens, bonus=decision.bonus_token,
        branch=decision.branch, original_accepted=decision.original_accepted,
        refined_accepted=decision.refined_accepted,
        committed_positions=len(decision.accepted_nodes),
        target_cache_length=verifier.past,
        draft_cache_length=q0_cache.get_seq_length(),
        refiner_cache_length=None if refiner_cache is None else refiner_cache.get_seq_length(),
        next_feature_shape=list(new_features.shape),
    )


@torch.no_grad()
def check_native_first_pass(target, ordinary, fixture):
    block = ordinary.drafter.block_size
    ids = fixture.prefix_ids.new_full((1, block), ordinary.drafter.mask_token_id)
    ids[0, 0] = fixture.anchor
    inputs = dict(
        noise_embedding=target.get_input_embeddings()(ids), target_hidden=fixture.features,
        position_ids=torch.arange(fixture.previous_length, fixture.prefix_ids.shape[1]+block,
                                  device=ids.device)[None],
    )
    reference = ordinary.drafter(**inputs, past_key_values=clone_cache(fixture.q0_cache),
                                use_cache=True, is_causal=False)
    repeated = ordinary.drafter(**inputs, past_key_values=clone_cache(fixture.q0_cache),
                               use_cache=True, is_causal=False)
    observed = ordinary.capture(**inputs, past_key_values=clone_cache(fixture.q0_cache)).normalized
    evidence = dict(reference_repeats_exactly=torch.equal(reference, repeated),
                    capture_matches_exactly=torch.equal(reference, observed),
                    max_absolute_difference=float((reference.float()-observed.float()).abs().max()),
                    finite=bool(torch.isfinite(reference).all() and torch.isfinite(observed).all()))
    return evidence, dict(reference=reference, repeated=repeated, observed=observed)


def measure_fixture(target, ordinary, refiners, fixture, *, repeats=12, warmups=2,
                    before_round=lambda: None, on_row=lambda row: None):
    if type(repeats) is not int or repeats < 2 or type(warmups) is not int or warmups < 0:
        raise ValueError("Need paired repetitions and nonnegative warmups")
    modes = ["q0", "midpoint", "final", "full", "shallow"]
    if set(refiners) != set(modes[1:]):
        raise ValueError("All four frozen initial layouts must be retained")
    rows = []
    device = fixture.prefix_ids.device
    for repetition in range(-warmups, repeats):
        # Deterministic rotation and reversal balance order without selecting
        # fast repeats or changing scientific settings in response to timings.
        order = modes[repetition % len(modes):] + modes[:repetition % len(modes)]
        if repetition % 2:
            order = order[::-1]
        for mode in order:
            before_round()
            model = refiners.get(mode)
            label = "single" if model is None else "union"
            # Reset only the independent repetition. A serving round does not
            # clone its entire context; exclude that measurement artifact.
            qcache = clone_cache(fixture.q0_cache)
            rcache = None if model is None else clone_cache(fixture.refiner_caches[mode])
            fixture.verifiers[label].load_prefix(fixture.prefix_state)
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            result = complete_round(target, ordinary, model, fixture, qcache, rcache)
            synchronize(device)
            seconds = time.perf_counter()-started
            if repetition >= 0:
                row = dict(mode=mode, repetition=repetition, complete_round_seconds=seconds,
                           result=result, peak_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                                               if device.type == "cuda" else None))
                rows.append(row)
                on_row(row)
            del qcache, rcache
    return rows
