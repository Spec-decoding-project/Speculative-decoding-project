"""Complete fixed-context rounds for the compact editor; no loading or cloud work.

First-pass logits are retained, not recomputed. The editor sees only first-pass
states and x1. Verification happens afterward, with the original branch retained.
The folded path defines its own finite-precision predictions.
"""
import torch
import time
from dataclasses import dataclass
from conditional_redraft_decode import _features
from conditional_redraft_measure import _single_decision, clone_cache, synchronize
from second_pass_profile import stage, complete_profile_round, audit_decoded_result


def _proposal_context_identity(fixture):
    def identity(tensor):
        return (id(tensor), tensor.data_ptr(), tensor._version, tuple(tensor.shape))
    return (id(fixture), identity(fixture.prefix_ids), identity(fixture.features),
            fixture.anchor, fixture.previous_length)


@dataclass(frozen=True)
class FixedCompactProposal:
    """Nondeployable cost control bound to one immutable fixed-context fixture."""
    suffix: torch.Tensor
    context_identity: tuple
    suffix_version: int
    block_size: int

    @classmethod
    def from_result(cls, result, fixture, block_size):
        if result['policy'] != 'union' or len(result['packed']) != 2*block_size-2:
            raise ValueError('Require a complete compact union result')
        if result['packed'][0] != fixture.anchor:
            raise ValueError('Proposal anchor differs from fixture')
        suffix=fixture.prefix_ids.new_tensor(result['packed'][block_size:])[None].clone()
        return cls(suffix, _proposal_context_identity(fixture), suffix._version, block_size)

    def validate(self, fixture, block_size):
        if self.context_identity != _proposal_context_identity(fixture):
            raise ValueError('Fixed proposal belongs to another or modified context')
        if self.block_size != block_size or self.suffix.shape != (1,block_size-2):
            raise ValueError('Fixed proposal shape changed')
        if self.suffix._version != self.suffix_version:
            raise ValueError('Fixed proposal tokens changed')
        if self.suffix.device != fixture.prefix_ids.device or self.suffix.dtype != fixture.prefix_ids.dtype:
            raise ValueError('Fixed proposal device or dtype mismatch')


@torch.no_grad()
def prepare_frozen_readout(editor, head):
    if editor.training or any(p.requires_grad for p in editor.parameters()):
        raise ValueError('Freeze editor before preparing serving weights')
    if getattr(head, 'bias', None) is not None:
        raise ValueError('This implementation requires a bias-free output head')
    return editor.folded_weight(head.weight).detach()


@torch.no_grad()
def compact_round(target, ordinary, editor, fixture, qcache, *,
                  policy='union', folded_weight=None, instrument=False, evidence=None,
                  fixed_proposal=None):
    if policy not in ('original', 'union'):
        raise ValueError('Preserve original branch in this experiment')
    if fixed_proposal is not None and policy != 'union':
        raise ValueError('Fixed proposal requires union verification')
    if policy == 'union' and fixed_proposal is None and (editor is None or folded_weight is None):
        raise ValueError('Frozen editor and prepared readout required')
    past = fixture.prefix_ids.shape[1]
    block = ordinary.drafter.block_size
    if fixed_proposal is not None:
        fixed_proposal.validate(fixture,block)
    embedding, head = target.get_input_embeddings(), target.get_output_embeddings()
    with stage(instrument, 'first_pass_inputs'):
        ids = fixture.prefix_ids.new_full((1, block), ordinary.drafter.mask_token_id)
        ids[0, 0] = fixture.anchor
        positions = torch.arange(fixture.previous_length, past+block, device=ids.device)[None]
        noise = embedding(ids)
    with stage(instrument, 'first_pass_network'):
        state = ordinary.capture(noise_embedding=noise, target_hidden=fixture.features,
                                 position_ids=positions, past_key_values=qcache)
    with stage(instrument, 'first_pass_vocabulary'):
        scores = head(state.normalized[:, 1:])
        original = scores.argmax(-1)
    revised = None
    if policy == 'union':
        if fixed_proposal is not None:
            with stage(instrument, 'fixed_proposal_replay'):
                revised = torch.cat((original[:, :1], fixed_proposal.suffix), dim=1)
        else:
            with stage(instrument, 'compact_editor_complete'):
                visible = embedding(original[:, 0])
                revised_scores = editor.folded_scores(state.normalized, visible,
                                                      scores[:, 1:], folded_weight)
                revised = torch.cat((original[:, :1], revised_scores.argmax(-1)), dim=1)
    with stage(instrument, 'packing_and_draft_cache'):
        label = 'union' if policy == 'union' else 'single'
        packed = torch.cat((ids[:, :1], original), dim=1)
        if revised is not None:
            packed = torch.cat((packed, revised[:, 1:]), dim=1)
        qcache.crop(past)
    with stage(instrument, 'target_verification'):
        verifier = fixture.verifiers[label]
        output = verifier(packed, fixture.positions[label], fixture.masks[label])
    with stage(instrument, 'acceptance_host_transfer'):
        tokens = packed[0].tolist()
        posterior = output.logits[0].argmax(-1).tolist()
        decision = (fixture.layout.decide(tokens, posterior) if policy == 'union'
                    else _single_decision(tokens, posterior))
    with stage(instrument, 'target_cache_and_features'):
        nodes = fixture.prefix_ids.new_tensor(decision.accepted_nodes)
        next_features = _features(output.hidden_states, ordinary.drafter.target_layer_ids, nodes)
        verifier.compact(past, decision.accepted_nodes)
    if evidence is not None:
        evidence.update(first_states=state.normalized.detach().clone(),
                        first_scores=scores.detach().clone(),
                        verification_logits=output.logits.detach().clone())
        if revised is not None and fixed_proposal is None:
            evidence.update(visible=visible.detach().clone(),
                            revised_scores=revised_scores.detach().clone())
        elif fixed_proposal is not None:
            evidence['fixed_revised_suffix']=fixed_proposal.suffix.detach().clone()
    return dict(policy=policy, packed=tokens, posterior=posterior,
                accepted=list(decision.accepted_tokens), bonus=decision.bonus_token,
                branch=decision.branch, committed_positions=len(decision.accepted_nodes),
                target_cache_length=verifier.past, draft_cache_length=qcache.get_seq_length(),
                next_feature_shape=list(next_features.shape), query_count=len(tokens))


@torch.no_grad()
def measure_compact_cost_isolation(target, ordinary, editor, fixture, *,
                                   repeats=20, warmups=3, before_round=lambda: None,
                                   on_row=lambda row: None):
    """Nondeployable fixed-proposal control, paired with identical live proposals.

    Fixture construction/cache cloning and oracle preparation are outside timing.
    First pass, packing, verification, acceptance and compaction remain timed.
    This is fixed-context profiling, never an end-to-end speedup measurement.
    """
    if repeats < 2 or warmups < 1:
        raise ValueError('Require paired repetitions and warmup')
    versions=tuple(p._version for m in (target,ordinary,editor) for p in m.parameters())
    device=fixture.prefix_ids.device
    synchronize(device);started=time.perf_counter()
    folded=prepare_frozen_readout(editor,target.get_output_embeddings())
    synchronize(device);fold_ms=1000*(time.perf_counter()-started)
    before_round()
    fixture.verifiers['union'].load_prefix(fixture.prefix_state)
    expected=compact_round(target,ordinary,editor,fixture,clone_cache(fixture.q0_cache),folded_weight=folded)
    fixed=FixedCompactProposal.from_result(expected,fixture,ordinary.drafter.block_size)
    names=('original','compact_union','fixed_proposal_union')
    reference={};rows=[]
    for repeat in range(-warmups,repeats):
        offset=repeat % len(names);order=list(names[offset:]+names[:offset])
        if repeat % 2:order.reverse()
        for name in order:
            before_round()
            policy='original' if name=='original' else 'union'
            fixture.verifiers['single' if policy=='original' else 'union'].load_prefix(fixture.prefix_state)
            cache=clone_cache(fixture.q0_cache)
            synchronize(device);started=time.perf_counter()
            result=compact_round(target,ordinary,editor,fixture,cache,policy=policy,folded_weight=folded,
                fixed_proposal=fixed if name=='fixed_proposal_union' else None)
            synchronize(device);elapsed=1000*(time.perf_counter()-started)
            if name!='original' and result!=expected:
                on_row(dict(arm=name,repeat=repeat,failure='Fixed/live result mismatch',result=result))
                raise ValueError('Preserve changed proposals, scores or cache outcome before stopping')
            if name in reference and result!=reference[name]:
                on_row(dict(arm=name,repeat=repeat,failure='Unstable result',result=result))
                raise ValueError('Unstable repeated result')
            reference[name]=result
            if repeat>=0:
                row=dict(arm=name,repeat=repeat,wall_ms=elapsed,result=result)
                rows.append(row);on_row(row)
    if versions!=tuple(p._version for m in (target,ordinary,editor) for p in m.parameters()):
        raise ValueError('Model weights changed during measurement')
    block=ordinary.drafter.block_size
    assert reference['compact_union']['packed'][:block]==reference['original']['packed']
    return dict(rows=rows,fold_setup_ms=fold_ms,proposal_results=reference,
                scope='Fixed-context profiling; supplied proposals are nondeployable. Not a learned improvement or generation speedup.',
                execution='All shared draft work eager, same target verifier implementation; fixed-proposal validation included in its time.')


@torch.no_grad()
def measure_compact_comparison(target, ordinary, teacher, editor, fixture, *,
                               repeats=20, warmups=3, before_round=lambda: None,
                               on_row=lambda row: None):
    """Matched complete rounds. Random editor measures cost, never useful gains.

    All draft networks are eager; the fixture fixes the common verifier mode.
    Prepared caches are cloned before timing for all arms. Score retention,
    embedding lookup, selection, packing, verification and compaction are timed.
    Setup is reported separately and must be amortized in any later serving claim.
    """
    if repeats < 2 or warmups < 1:
        raise ValueError('Require warmup and paired repetitions')
    head = target.get_output_embeddings()
    synchronize(fixture.prefix_ids.device)
    start = time.perf_counter()
    folded = prepare_frozen_readout(editor, head)
    synchronize(fixture.prefix_ids.device)
    setup_ms = 1000*(time.perf_counter()-start)
    modules = (target, ordinary, teacher, editor)
    versions = tuple(p._version for m in modules for p in m.parameters())
    names = ('original', 'teacher_revised', 'teacher_union', 'compact_union')
    rows, reference = [], {}
    for repeat in range(-warmups, repeats):
        order = list(names[repeat % len(names):]+names[:repeat % len(names)])
        if repeat % 2: order.reverse()
        for name in order:
            before_round()
            qcache = clone_cache(fixture.q0_cache)
            teacher_arm = name.startswith('teacher_')
            rcache = clone_cache(fixture.refiner_caches[teacher.mode]) if teacher_arm else None
            label = 'union' if name.endswith('_union') else 'single'
            fixture.verifiers[label].load_prefix(fixture.prefix_state)
            synchronize(fixture.prefix_ids.device)
            start = time.perf_counter()
            if name == 'compact_union':
                result = compact_round(target, ordinary, editor, fixture, qcache,
                                       folded_weight=folded)
            else:
                policy = name.removeprefix('teacher_')
                result = complete_profile_round(target, ordinary, teacher if teacher_arm else None,
                                                fixture, qcache, rcache, policy=policy)
            synchronize(fixture.prefix_ids.device)
            elapsed_ms = 1000*(time.perf_counter()-start)
            if name in reference and reference[name] != result:
                on_row(dict(arm=name, repeat=repeat, failure='Changed repeated result', result=result))
                raise ValueError('Preserve changed result before continuing')
            reference[name] = result
            if repeat >= 0:
                row = dict(arm=name, repeat=repeat, wall_ms=elapsed_ms, result=result)
                rows.append(row)
                on_row(row)
    if versions != tuple(p._version for m in modules for p in m.parameters()):
        raise ValueError('Frozen parameters changed; folded weights may be stale')
    original = reference['original']['packed']
    for name in ('teacher_union', 'compact_union'):
        if reference[name]['packed'][:len(original)] != original:
            raise ValueError('Original proposal changed between comparison arms')
    return dict(rows=rows, fold_setup_ms=setup_ms,
                execution='All draft networks eager; common fixture verifier mode',
                limitations=['Fixed-context cost diagnostic, not generation latency.',
                  'Different revised candidates are different downstream workloads; this measures complete policies, not isolated readout costs.',
                  'A randomly initialized editor supplies no evidence of learned gains.',
                  'Graph-optimized editor requires equally optimized draft controls in a separate comparison.',
                  'External fixture preparation and initial cache clones are outside per-round timing.'])


@torch.no_grad()
def audit_compact_round(target, ordinary, editor, fixture, folded_weight):
    fixture.verifiers['union'].load_prefix(fixture.prefix_state)
    evidence={}
    result=compact_round(target,ordinary,editor,fixture,clone_cache(fixture.q0_cache),
                         folded_weight=folded_weight,evidence=evidence)
    # Retain actual model scores on failure; do not approximate a numerical tie.
    return audit_decoded_result(target,fixture,'union',result,evidence)
