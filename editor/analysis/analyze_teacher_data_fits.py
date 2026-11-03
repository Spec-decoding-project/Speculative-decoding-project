"""Independently recount the completed fits on the exposed continuity panel.

This is not the fresh final comparison and must not select a checkpoint.
"""
from pathlib import Path
from collections import defaultdict
import hashlib,json,statistics,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
BASE=ROOT/'notes/second-pass-profiling-2026-10-02/teacher-unseen-comparison'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def recount(tokens,labels,valid):
    if len(tokens)!=16 or len(labels)!=16 or len(valid)!=16:raise ValueError('Unexpected token length')
    if not valid[0] or valid!=[True]*sum(valid)+[False]*(16-sum(valid)):
        raise ValueError('Invalid reference mask')
    if tokens[0]!=labels[0]:raise ValueError('Known root was changed')
    count=0
    for i in range(1,len(labels)):
        if not valid[i] or tokens[i]!=labels[i]:break
        count+=1
    return 1+count


def main():
    import torch
    from teacher_data_comparison import ARMS,REGIMES,per_prompt_progress,paired_contrasts
    torch.set_num_threads(2)
    attempt=BASE/'fit-attempt1';manifest=read(attempt/'recovery.json');source=attempt/'recovered'
    for name,row in manifest.items():
        p=(source/name).resolve()
        if not p.is_relative_to(source.resolve()) or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256'] or row['sha256']!=row['local_sha256']:
            raise ValueError('Recovered fit evidence identity')
    summary=read(source/'summary.json')
    if summary['status']!='complete' or summary['fits']!=6 or summary['updates']!=6912:
        raise ValueError('Incomplete comparison; do not produce complete-panel conclusions')
    plan=read(BASE/'fit-plan.json')['study']
    specs=[p for p in plan['panels'] if p['split']=='old_development']
    labels={};counts={}
    for spec in specs:
        pid=spec['id'];entry=plan['old_features'][pid];p=ROOT/entry['source']
        if sha(p)!=entry['sha256']:raise ValueError('Reference feature bank changed')
        rows=torch.load(p,map_location='cpu',weights_only=True);counts[pid]=len(rows)
        for r in rows:labels[pid,r['anchor_index']]=dict(labels=r['labels'][0].tolist(),valid=r['valid'][0].tolist(),original=r['original'][0].tolist())
    endpoints={};per_arm={};all_epochs={};token_checks=0
    final=read(source/'frozen-final-checkpoints.json')
    if set(final)!={r+'--'+a for r in REGIMES for a in ARMS}:raise ValueError('Frozen endpoint set changed')
    for regime in REGIMES:
        for arm in ARMS:
            name=regime+'--'+arm;updates=read(source/name/'updates.json')
            if len(updates)!=1152 or [r['step'] for r in updates]!=list(range(1,1153)):
                raise ValueError('Incomplete or repeated updates')
            if any(sum(r['epoch']==e for r in updates)!=384 for e in (1,2,3)):raise ValueError('Epoch exposure differs')
            frozen=final[name];p=source/frozen['path']
            if sha(p)!=frozen['sha256'] or frozen['updates']!=1152:raise ValueError('Final weight identity')
            for epoch in (1,2,3):
                rows=read(source/name/f'epoch-{epoch}-development.json')
                for r in rows:
                    ref=labels[r['prompt'],r['anchor_index']];v=r['result']
                    if v['original_tokens'][1:]!=ref['original'] or v['original_tokens'][:2]!=v['edited_tokens'][:2]:
                        raise ValueError('Reference or fixed guess changed')
                    a=recount(v['original_tokens'],ref['labels'],ref['valid'])
                    b=recount(v['edited_tokens'],ref['labels'],ref['valid'])
                    if (a,b,max(a,b))!=(v['original_progress'],v['edited_progress'],v['union_progress']):
                        raise ValueError('Independent prefix count disagrees')
                    if v['fully_correct_original']!=(a==sum(ref['valid'])) or v['damaged_fully_correct']!=(a==sum(ref['valid']) and b<a):
                        raise ValueError('Fully-correct-block accounting differs')
                    token_checks+=1
                metrics=per_prompt_progress(rows,counts)
                all_epochs[name+':'+str(epoch)]=dict(equal_prompt_gain_percent=statistics.mean(r['gain_pp'] for r in metrics.values()),
                    repairs=sum(r['repairs'] for r in metrics.values()),damage=sum(r['damage'] for r in metrics.values()),
                    damaged_fully_correct_blocks=sum(r['result']['damaged_fully_correct'] for r in rows))
                if epoch==3:endpoints[regime,arm]=metrics;per_arm[name]=all_epochs[name+':3']
    contrasts=paired_contrasts(endpoints,specs)
    result=dict(status='complete',panel='64 repeatedly exposed development responses; NOT fresh final results',
        per_arm=per_arm,all_epochs=all_epochs,contrasts=contrasts,independently_recounted_context_predictions=token_checks,
        checkpoint_selection='All six epoch3 checkpoints remain frozen; do not select using these results',
        scope='Fixed-context coverage, no latency or fresh generalization claim')
    (attempt/'development-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('all_epochs',)},indent=2))


if __name__=='__main__':main()
