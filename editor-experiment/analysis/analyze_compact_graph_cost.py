"""Analyze authenticated paired fixed-context cost measurements."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,statistics
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notes/second-pass-profiling-2026-10-02/compact-graph-cost'

def analyze(rows,expected_contexts=4,repeats=20):
    names=('original','compact_union','graphed_editor_union','fixed_proposal_union')
    groups=defaultdict(dict)
    for row in rows:
        assert 'failure' not in row
        key=(row['context']['id'],row['context']['anchor_index'])
        sub=(row['arm'],row['repeat']);assert sub not in groups[key]
        assert row['wall_ms']>0
        groups[key][sub]=row
    assert len(groups)==expected_contexts
    reports=[]
    for key,g in sorted(groups.items()):
        assert set(g)=={(a,i) for a in names for i in range(repeats)}
        reference={a:g[a,0]['result'] for a in names}
        for (a,i),row in g.items():assert row['result']==reference[a]
        assert reference['compact_union']==reference['fixed_proposal_union']==reference['graphed_editor_union']
        times={a:statistics.median(g[a,i]['wall_ms'] for i in range(repeats)) for a in names}
        progress={a:reference[a]['committed_positions'] for a in names}
        assert all(v>0 for v in progress.values())
        paired=[g['compact_union',i]['wall_ms']-g['fixed_proposal_union',i]['wall_ms'] for i in range(repeats)]
        reports.append(dict(prompt=key[0],anchor_index=key[1],median_round_ms=times,committed_positions=progress,
            live_minus_replay_median_time_ms=times['compact_union']-times['fixed_proposal_union'],
            median_paired_live_minus_replay_ms=statistics.median(paired),
            paired_live_more_expensive_repetitions=sum(x>0 for x in paired),
            median_paired_eager_minus_graph_ms=statistics.median(g['compact_union',i]['wall_ms']-g['graphed_editor_union',i]['wall_ms'] for i in range(repeats)),
            median_paired_graph_minus_supplied_ms=statistics.median(g['graphed_editor_union',i]['wall_ms']-g['fixed_proposal_union',i]['wall_ms'] for i in range(repeats)),
            graph_round_overhead_percent=100*(times['graphed_editor_union']/times['original']-1),
            replay_round_overhead_percent=100*(times['fixed_proposal_union']/times['original']-1),
            live_round_overhead_percent=100*(times['compact_union']/times['original']-1),
            cost_per_position_change_percent={a:100*((times[a]/progress[a])/(times['original']/progress['original'])-1) for a in names[1:]}))
    prompts={}
    for p in sorted({r['prompt'] for r in reports}):
        subset=[r for r in reports if r['prompt']==p]
        costs={a:sum(r['median_round_ms'][a] for r in subset)/sum(r['committed_positions'][a] for r in subset) for a in names}
        prompts[p]=dict(cost_per_position_ms=costs,relative_change_percent={a:100*(costs[a]/costs['original']-1) for a in names[1:]})
    return dict(contexts=reports,prompts=prompts,independent_prompts=len(prompts),
        scope='Paired fixed-context round costs. No population confidence interval or full-generation speedup; replay is nondeployable.')

def main():
    attempt=BASE/'attempt1';manifest=json.loads((attempt/'recovery.json').read_text())
    for name,row in manifest.items():
        p=attempt/'recovered'/name
        assert p.stat().st_size==row['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']==row['local_sha256']
    summary=json.loads((attempt/'recovered/summary.json').read_text())
    assert summary['status']=='complete' and summary['contexts']==4 and summary['timing_rows']==320
    rows=json.loads((attempt/'recovered/timings.json').read_text())
    result=analyze(rows)
    for p in (attempt/'recovered').glob('*-replay-parity.json'):
        data=json.loads(p.read_text());assert data['result_exact'] and all(l['keys_exact'] and l['values_exact'] for l in data['layers'])
    setup=json.loads((attempt/'recovered/graph-setup.json').read_text())
    for p in (attempt/'recovered').glob('*-graph-parity.json'):
        data=json.loads(p.read_text());assert data['result_exact'] and all(data['scores_exact'].values()) and all(l['keys_exact'] and l['values_exact'] for l in data['layers'])
    import math
    for r in result['contexts']:
        saving=r['median_paired_eager_minus_graph_ms']
        r['graph_setup_break_even_rounds_at_this_saving']=math.ceil(setup['capture_warmup_ms']/saving) if saving>0 else None
    result['setup']=setup
    result['verified_artifacts']=len(manifest)
    (attempt/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
