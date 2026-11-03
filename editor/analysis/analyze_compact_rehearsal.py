"""Frozen final comparison; re-count saved tokens, not reported aggregate metrics."""
from pathlib import Path
from collections import defaultdict
import json,statistics,random,argparse
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'notes/second-pass-profiling-2026-10-02/compact-rehearsal'
read=lambda p:json.loads(p.read_text())

def summarize(keys,records,metadata,teacher):
    groups=defaultdict(lambda:[0,0]);totals=defaultdict(int)
    for key in keys:
        pid,i=key.split(':');m=metadata[pid]['anchors'][int(i)];r=records[key]['result']
        def count(seq):
            return next((j for j,(t,y,v) in enumerate(zip(seq,m['labels'],m['valid'])) if not v or t!=y),16)
        a=count(r['original_tokens']);b=count(r['edited_tokens']);t=teacher[key]['teacher_progress']
        assert a==r['original_progress'] and b==r['edited_progress'] and max(a,b)==r['union_progress']
        assert r['edited_tokens'][:2]==m['original'][:2]
        groups[pid][0]+=a;groups[pid][1]+=max(a,b)
        totals['repairs']+=b>a;totals['damaged_prefixes']+=b<a
        totals['damaged_complete_originals']+=a==sum(m['valid']) and b<a
        totals['captured_teacher_positions']+=max(0,min(b,t)-a)
        totals['teacher_improved_prefix_reproduced']+=t>a and b>=t
        if 2<=a<sum(m['valid']):totals['first_error_corrected']+=r['edited_tokens'][a]==m['labels'][a]
    per={pid:100*(u/a-1) for pid,(a,u) in groups.items()}
    return dict(contexts=len(keys),prompts=len(groups),**totals,equal_prompt_gain_percent=statistics.mean(per.values()),per_prompt_gain_percent=per)

def analyze(attempt='attempt2'):
    plan=read(OUT/'runner-plan.json');recovered=OUT/attempt/'recovered';receipt=read(OUT/attempt/'recovery.json')
    import hashlib
    for name,row in receipt.items():
        assert hashlib.sha256((recovered/name).read_bytes()).hexdigest()==row['local_sha256']
    assert read(recovered/'summary.json')['status']=='complete'
    teacher={f"{r['prompt']}:{r['anchor_index']}":r for r in read(OUT.parent/'correction-exposure/rows.json')}
    pids={k.split(':')[0] for k in plan['final_keys']}
    metadata={p:read(ROOT/f'experiments/results/conditional-redraft-learning-2026-09-18/run/metadata/{p}.json') for p in pids}
    sets=dict(focused_training=plan['focus_keys'],training_retention=plan['retention_keys'],
        exposed_development=[k for k in plan['final_keys'] if teacher[k]['split']=='development'],**plan['strata'])
    arms={};curves={}
    for arm,step in [('baseline',0),('ordinary',128),('targeted',128)]:
        data=read(recovered/f'{arm}-step-{step}-evaluation.json');records={r['key']:r for r in data}
        assert len(records)==len(data)==len(plan['final_keys']) and set(records)==set(plan['final_keys'])
        arms[arm]={s:summarize(keys,records,metadata,teacher) for s,keys in sets.items()}
    for arm in ('ordinary','targeted'):
        curves[arm]={}
        for step in (32,64):
            data=read(recovered/f'{arm}-step-{step}-evaluation.json');records={r['key']:r for r in data}
            assert set(records)==set(plan['monitor_keys'])
            curves[arm][step]={s:summarize([k for k in keys if k in records],records,metadata,teacher) for s,keys in sets.items()}
    differences={};rng=random.Random(201002)
    for name in sets:
        t=arms['targeted'][name]['per_prompt_gain_percent'];o=arms['ordinary'][name]['per_prompt_gain_percent']
        assert t.keys()==o.keys()
        values={p:t[p]-o[p] for p in t}
        domains={d:[v for p,v in values.items() if p.startswith(d+'-')] for d in ('gsm8k','ultrachat')}
        boots=sorted(statistics.mean([v for group in domains.values() if group for v in rng.choices(group,k=len(group))]) for _ in range(10000))
        differences[name]=dict(targeted_minus_ordinary_percentage_points=statistics.mean(values.values()),
            descriptive_paired_prompt_95_interval=[boots[249],boots[9749]],
            domains={d:statistics.mean(v) for d,v in domains.items() if v})
    result=dict(status='complete',arms=arms,curves=curves,differences=differences,
        scope='One historical initialization; exposed development; fixed final update128; no timing/speedup claim. Sample concentration and repetition are bundled.')
    (OUT/attempt/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(differences,indent=2))

if __name__=='__main__':analyze()
