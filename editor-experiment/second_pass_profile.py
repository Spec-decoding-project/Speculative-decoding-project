"""Three complete-round policies, with optional separately measured stage traces.

No loading/cloud activity. The historical teacher clamps the first draft guess.
Profile markers encompass real work; they are not independent kernel timings.
"""
from contextlib import nullcontext
import time
import torch
from conditional_redraft_decode import _features
from conditional_redraft_measure import clone_cache, synchronize, _single_decision

POLICIES = ('original', 'revised', 'union')


def stage(enabled, name):
    return torch.profiler.record_function('second_pass/' + name) if enabled else nullcontext()


@torch.no_grad()
def complete_profile_round(target, ordinary, model, fixture, qcache, rcache,
                           *, policy, instrument=False, evidence=None):
    if policy not in POLICIES:
        raise ValueError('Unknown verification policy')
    if policy != 'original' and (model is None or rcache is None):
        raise ValueError('Editing requires the teacher and its independent cache')
    past = fixture.prefix_ids.shape[1]
    block = ordinary.drafter.block_size
    embedding, head = target.get_input_embeddings(), target.get_output_embeddings()
    with stage(instrument, 'first_pass_inputs'):
        ids = fixture.prefix_ids.new_full((1, block), ordinary.drafter.mask_token_id)
        ids[0, 0] = fixture.anchor
        noise = embedding(ids)
        positions = torch.arange(fixture.previous_length, past+block, device=ids.device)[None]
    with stage(instrument, 'first_pass_network'):
        state = ordinary.capture(noise_embedding=noise, target_hidden=fixture.features,
                                 position_ids=positions, past_key_values=qcache)
    with stage(instrument, 'first_pass_vocabulary'):
        original = head(state.normalized[:, 1:]).argmax(-1)
    revised = None
    if policy != 'original':
        with stage(instrument, 'second_pass_input'):
            visible = embedding(original[:, :1])
        with stage(instrument, 'second_pass_network'):
            hidden = model(state, visible, past_key_values=rcache)
        with stage(instrument, 'second_pass_vocabulary'):
            suffix = head(hidden[:, 2:]).argmax(-1)
            revised = torch.cat((original[:, :1], suffix), dim=1)
    with stage(instrument, 'packing_and_draft_cache'):
        label = 'union' if policy == 'union' else 'single'
        if policy == 'union':
            packed = torch.cat((ids[:, :1], original, revised[:, 1:]), dim=1)
        else:
            packed = torch.cat((ids[:, :1], original if policy == 'original' else revised), dim=1)
        qcache.crop(past)
        if rcache is not None:
            rcache.crop(past)
    with stage(instrument, 'target_verification'):
        verifier = fixture.verifiers[label]
        output = verifier(packed, fixture.positions[label], fixture.masks[label])
    if evidence is not None:
        evidence['verification_logits'] = output.logits.detach().clone()
    with stage(instrument, 'acceptance_host_transfer'):
        tokens = packed[0].tolist()
        posterior = output.logits[0].argmax(-1).tolist()
        decision = (fixture.layout.decide(tokens, posterior) if policy == 'union'
                    else _single_decision(tokens, posterior))
    with stage(instrument, 'target_cache_and_features'):
        nodes = fixture.prefix_ids.new_tensor(decision.accepted_nodes)
        next_features = _features(output.hidden_states, ordinary.drafter.target_layer_ids, nodes)
        verifier.compact(past, decision.accepted_nodes)
    # Preserve evidence after timed computation; result materialization below is
    # included equally in complete-round timing, and is explicitly disclosed.
    return dict(policy=policy, packed=tokens, posterior=posterior,
        accepted=list(decision.accepted_tokens), bonus=decision.bonus_token,
        branch='revised' if policy=='revised' else decision.branch,
        committed_positions=len(decision.accepted_nodes),
        target_cache_length=verifier.past, draft_cache_length=qcache.get_seq_length(),
        refiner_cache_length=None if rcache is None else rcache.get_seq_length(),
        next_feature_shape=list(next_features.shape), query_count=len(tokens))


@torch.no_grad()
def audit_context(target, ordinary, model, fixture, policy):
    """Untimed independent causal scoring and a next-token cache-continuity probe.

    Any differing winners are evidence, not automatically an implementation bug.
    Returned score tensors must be recovered if differences occur.
    """
    from transformers import DynamicCache
    qcache=clone_cache(fixture.q0_cache)
    rcache=None if policy=='original' else clone_cache(fixture.refiner_caches[model.mode])
    label='union' if policy=='union' else 'single'
    verifier=fixture.verifiers[label]
    verifier.load_prefix(fixture.prefix_state)
    evidence={}
    result=complete_profile_round(target,ordinary,model,fixture,qcache,rcache,
                                  policy=policy,evidence=evidence)
    return audit_decoded_result(target,fixture,policy,result,evidence)


@torch.no_grad()
def audit_decoded_result(target,fixture,policy,result,evidence):
    """Audit an already executed round while its compacted cache is still live."""
    from transformers import DynamicCache
    label='union' if policy=='union' else 'single'
    verifier=fixture.verifiers[label]
    paths=([fixture.layout.original_nodes,fixture.layout.refined_nodes] if policy=='union'
           else [tuple(range(len(result['packed'])))])
    checks=[]
    for i,path in enumerate(paths):
        chain=fixture.prefix_ids.new_tensor([[result['packed'][j] for j in path]])
        native=target(torch.cat((fixture.prefix_ids,chain),dim=1),use_cache=False,
                      logits_to_keep=chain.shape[1]+1).logits
        # First score predicts the committed anchor; remaining scores predict
        # after each chain node under ordinary causal evaluation.
        actual=evidence['verification_logits'][:,list(path)]
        expected=native[:,1:]
        evidence[f'causal_logits_{i}']=expected.detach().clone()
        chain_tokens=chain[0].tolist()
        actual_decision=_single_decision(chain_tokens,actual[0].argmax(-1).tolist())
        causal_decision=_single_decision(chain_tokens,expected[0].argmax(-1).tolist())
        same_decision=(actual_decision.accepted_tokens==causal_decision.accepted_tokens
                       and actual_decision.bonus_token==causal_decision.bonus_token)
        checks.append(dict(path=i,accepted_decision_matches=same_decision,
            causal_accepted=list(causal_decision.accepted_tokens),causal_bonus=causal_decision.bonus_token,
            anchor_matches=int(native[0,0].argmax())==fixture.anchor,
            differing_next_token_winners=int((actual.argmax(-1)!=expected.argmax(-1)).sum()),
            max_absolute_score_difference=float((actual.float()-expected.float()).abs().max()),
            finite=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())))
    # The compacted cache contains accepted nodes, never the pending bonus.
    cache=DynamicCache()
    for i,layer in enumerate(verifier.cache.layers):
        cache.update(layer.keys[...,:verifier.past,:].clone(),
                     layer.values[...,:verifier.past,:].clone(),i)
    bonus=fixture.prefix_ids.new_tensor([[result['bonus']]])
    cached=target(bonus,past_key_values=cache,use_cache=True,logits_to_keep=1).logits
    fresh_ids=torch.cat((fixture.prefix_ids,
                        fixture.prefix_ids.new_tensor([result['accepted']]),bonus),dim=1)
    fresh=target(fresh_ids,use_cache=False,logits_to_keep=1).logits
    evidence['following_cached_logits']=cached.detach().clone()
    evidence['following_fresh_logits']=fresh.detach().clone()
    report=dict(result=result,paths=checks,
        following_token_matches=bool(torch.equal(cached.argmax(-1),fresh.argmax(-1))),
        following_score_max_difference=float((cached.float()-fresh.float()).abs().max()),
        finite=all(c['finite'] for c in checks) and bool(torch.isfinite(cached).all() and torch.isfinite(fresh).all()))
    return report,evidence


def measure_context(target, ordinary, model, fixture, *, repeats=20, warmups=3,
                    before_round=lambda:None, on_row=lambda row:None):
    if repeats < 2 or warmups < 1:
        raise ValueError('Require warmup and repeated paired measurements')
    rows, reference = [], {}
    for repeat in range(-warmups, repeats):
        order = list(POLICIES[repeat % 3:] + POLICIES[:repeat % 3])
        if repeat % 2:
            order.reverse()
        for policy in order:
            before_round()
            qcache = clone_cache(fixture.q0_cache)
            rcache = None if policy=='original' else clone_cache(fixture.refiner_caches[model.mode])
            label = 'union' if policy=='union' else 'single'
            fixture.verifiers[label].load_prefix(fixture.prefix_state)
            synchronize(fixture.prefix_ids.device)
            start = time.perf_counter()
            result = complete_profile_round(target,ordinary,model,fixture,qcache,rcache,policy=policy)
            synchronize(fixture.prefix_ids.device)
            elapsed = 1000*(time.perf_counter()-start)
            if policy in reference and result != reference[policy]:
                raise RuntimeError('Repeated round changed: preserve failure evidence before proceeding')
            reference[policy] = result
            if repeat >= 0:
                row = dict(policy=policy, repeat=repeat, wall_ms=elapsed, result=result)
                rows.append(row); on_row(row)
    return rows


def profile_context(target, ordinary, model, fixture, policy, trace_path):
    """Separate profiled round; overhead must not replace uninstrumented timing."""
    qcache = clone_cache(fixture.q0_cache)
    rcache = None if policy=='original' else clone_cache(fixture.refiner_caches[model.mode])
    label = 'union' if policy=='union' else 'single'
    fixture.verifiers[label].load_prefix(fixture.prefix_state)
    activities=[torch.profiler.ProfilerActivity.CPU]
    if fixture.prefix_ids.is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    synchronize(fixture.prefix_ids.device)
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        start=time.perf_counter()
        result=complete_profile_round(target,ordinary,model,fixture,qcache,rcache,
                                      policy=policy,instrument=True)
        synchronize(fixture.prefix_ids.device)
        wall=1000*(time.perf_counter()-start)
    prof.export_chrome_trace(str(trace_path))
    stages=[dict(name=e.key,cpu_total_us=e.cpu_time_total,
                 device_total_us=e.device_time_total,calls=e.count)
            for e in prof.key_averages() if e.key.startswith('second_pass/')]
    return dict(result=result,profiled_wall_ms=wall,stages=stages,
        limitation='Inclusive profiler times may overlap or omit CUDA graph attribution; do not sum as independent costs. No split of fused attention into context vs block cost.')
