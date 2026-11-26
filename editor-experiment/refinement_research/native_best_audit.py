"""Independent reconstruction of earlier-gate native results, without GPU work."""
import argparse
import copy
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from copy_action_fit.data import SOURCES, digest, save, sha
from full_corpus_fit.data import acceptance, audit_chunk


def round_economics(baseline_ms, edited_ms, baseline_commits, edited_commits):
    """Use complete measured cycles, avoiding attribution of pending GPU work."""
    if min(baseline_ms,edited_ms,baseline_commits,edited_commits)<=0:
        raise ValueError('Positive round costs/commits required')
    return dict(
        affordable_added_ms=baseline_ms*(edited_commits/baseline_commits-1),
        measured_added_round_ms=edited_ms-baseline_ms,
        observed_extra_committed_tokens=edited_commits-baseline_commits,
        required_extra_committed_tokens=baseline_commits*(edited_ms/baseline_ms-1),
        required_total_committed_tokens=baseline_commits*edited_ms/baseline_ms,
        decode_speedup=(edited_commits/edited_ms)/(baseline_commits/baseline_ms),
        scope='At the measured complete-round cost, including all copies and scheduling. No standalone-editor or callback-wait attribution.')


def audit(path):
    prompts=json.loads((path/'prompts.json').read_text())
    manifest=json.loads((path/'run.json').read_text())
    completed=json.loads((path/'complete.json').read_text())
    rows=[json.loads(p.read_text()) for p in sorted((path/'records').glob('candidate-*.json'))]
    references=json.loads((path/'reference-results.json').read_text())
    assert len(prompts)==128 and len(references)==8
    assert manifest['prompt_digest']==digest(prompts)
    assert Counter(p['source'] for p in prompts)=={s:64 for s in SOURCES}
    assert Counter((p['source'],p['length_group']) for p in prompts)=={
        (s,q):16 for s in SOURCES for q in ('Q1','Q2','Q3','Q4')}
    assert all(p['split']=='development' for p in prompts)
    assert len({p['family_id'] for p in prompts})==128
    mapping={p['id']:p for p in prompts}
    baselines={r['prompt_id']:r for r in rows if r['phase']=='correctness' and r['mode']=='off'}
    assert len(baselines)==128
    native_positions=hashes=hits=0
    changes=[]
    for ref in references:
        b=baselines[ref['prompt_id']]
        assert all(ref['generated'][k]==b['generated'][k] for k in ('prompt','tokens','finish_reason'))
        p=mapping[ref['prompt_id']]
        native_positions+=audit_chunk([p],[ref['generated']],ref['observed']['native'])['positions_checked']
    for r in rows:
        p=mapping[r['prompt_id']]; b=baselines[r['prompt_id']]
        assert all(r['generated'][k]==b['generated'][k] for k in ('prompt','tokens','finish_reason'))
        obs=r['observed'];m=r['metrics']
        left=len(r['generated']['tokens']);prefill=0;commits=[]
        for s in obs['all_samples']:
            visible=min(left,s['sampled']);left-=visible
            if s['speculative']:commits.append(visible)
            else:prefill+=visible
        assert left==0 and prefill==1
        assert commits==m['committed_per_round'] and len(commits)==m['verification_rounds']
        assert sum(commits)==m['verification_tokens']
        assert len(commits)==sum(x['speculative'] for x in obs['rounds'])
        assert math.isclose(r['seconds']*1000,m['decode_ms']+m['prefill_ms'])
        assert r['seconds']>=r['llm_generate_seconds'] and r['drain_and_rpc_ms']>=0
        if r['mode']=='off':assert not obs['edit_calls'] and not obs['editor_ms']
        else:assert obs['edit_calls']==len(obs['editor_ms'])
        if r['phase']=='correctness':
            checked=audit_chunk([p],[r['generated']],obs['native'],hashes=True)
            native_positions+=checked['positions_checked']
            assert checked['aligned_hashes']==b['observed']['audit']['aligned_hashes']
            if r['mode']=='editor':hashes+=len(checked['aligned_hashes'])
            full=p['prompt']+r['generated']['tokens']
            for edit in obs['edits']:
                assert edit['original'][:2]==edit['edited'][:2]
                assert edit['prefix'][:len(full)]==full[:len(edit['prefix'])]
                label=full[edit['anchor']:edit['anchor']+16]
                if len(label)==16:
                    changes.append(acceptance(edit['edited'],label)-acceptance(edit['original'],label))
        if r['phase']=='timing' and r['mode']=='editor':
            assert obs['graph_misses']==obs['graph_evictions']==0
            assert obs['graph_hits']==obs['edit_calls'];hits+=obs['graph_hits']
    timing=[r for r in rows if r['phase']=='timing']
    assert len(timing)==1280 and len(rows)==1792
    for group,result in completed['groups'].items():
        cases=[r for r in timing if group=='all' or group==r['source'] or
               group==r['source']+'/'+r['length_group']]
        for mode in ('off','editor'):
            arm=[r for r in cases if r['mode']==mode]
            metrics=result['arms'][mode]
            elapsed=sum(r['seconds'] for r in arm)
            generated=sum(len(r['generated']['tokens']) for r in arm)
            rounds=sum(sum(s['speculative'] for s in r['observed']['all_samples']) for r in arm)
            committed=generated-len(arm)
            assert metrics['verification_rounds']==rounds
            assert math.isclose(metrics['tokens_per_second'],generated/elapsed)
            assert math.isclose(metrics['average_committed_per_round'],committed/rounds)
            assert math.isclose(metrics['average_decode_round_ms'],sum(r['metrics']['decode_ms'] for r in arm)/rounds)
            assert Counter(r['repeat'] for r in arm)=={i:result['prompts'] for i in range(5)}
        base,edit=result['arms']['off'],result['arms']['editor']
        expected=base['average_decode_round_ms']*(edit['average_committed_per_round']/base['average_committed_per_round']-1)
        assert math.isclose(expected,result['editing_budget_ms_per_round'],abs_tol=1e-9)
    # The raw runner also saved a callback-cost estimate under fixed noneditor
    # work. That callback can wait for pending native drafter kernels. Do not
    # interpret all of its wall time as incremental editor computation. Use the
    # actually measured complete-round difference for the requested economics.
    economics={}
    for group,result in completed['groups'].items():
        b,e=result['arms']['off'],result['arms']['editor']
        economics[group]=round_economics(b['average_decode_round_ms'],e['average_decode_round_ms'],
            b['average_committed_per_round'],e['average_committed_per_round'])
    public_groups=copy.deepcopy(completed['groups'])
    for result in public_groups.values():
        for name in ('measured_editor_ms_per_round','required_extra_committed_tokens_per_round',
                     'required_committed_tokens_per_round','noneditor_round_cost_change_ms'):
            result.pop(name,None)
        for arm in result['arms'].values():
            for old,new in (('editor_total_ms','callback_total_ms_including_native_wait'),
                            ('editor_ms_per_verification_round','callback_ms_per_round_including_native_wait'),
                            ('average_editor_call_ms','average_callback_ms_including_native_wait')):
                arm[new]=arm.pop(old)
    per_prompt=[]
    for i,p in enumerate(prompts):
        cases=[r for r in timing if r['prompt_id']==p['id']]
        means={mode:statistics.mean(r['seconds'] for r in cases if r['mode']==mode) for mode in ('off','editor')}
        per_prompt.append(dict(case=i,source=p['source'],quartile=p['length_group'],input_tokens=len(p['prompt']),
            seconds=means,speedup=means['off']/means['editor'],
            verification_rounds={mode:next(r['metrics']['verification_rounds'] for r in cases if r['mode']==mode)
                                 for mode in ('off','editor')}))
    return dict(passed=True, prompts=128, candidate_generations=len(rows),target_only_controls=len(references),
        native_argmax_positions_checked=native_positions,full_logit_hash_comparisons=hashes,
        natural_stops={source:sum(r['source']==source and r['generated']['finish_reason']=='stop'
                                 for r in baselines.values()) for source in SOURCES},
        one_pass_output_tokens={source:sum(len(r['generated']['tokens']) for r in baselines.values()
                                 if r['source']==source) for source in SOURCES},
        warm_graph_hits=hits,warm_graph_misses=0,
        native_edited_trajectory=dict(complete_windows=len(changes),repaired=sum(x>0 for x in changes),
            damaged=sum(x<0 for x in changes),prefix_delta_sum=sum(changes),
            scope='Local comparisons on edited trajectory, not matched-baseline throughput'),
        drain_and_rpc_ms=dict(mean=statistics.mean(r['drain_and_rpc_ms'] for r in timing),
                             maximum=max(r['drain_and_rpc_ms'] for r in timing)),
        per_prompt=per_prompt,groups=public_groups,round_economics=economics,graph_setup=completed['graph_setup'],
        source_artifact_hashes={p.name:sha(p) for p in (path/'run.json',path/'prompts.json',path/'complete.json')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();save(args.output,audit(args.directory))
