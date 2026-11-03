"""Independent raw-token recount of the bounded teacher-unseen collection."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,statistics
import torch
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notes/second-pass-profiling-2026-10-02/teacher-unseen-data'
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def prefix(xs,ys,valid):
    return next((i for i,(x,y,v) in enumerate(zip(xs,ys,valid)) if not v or x!=y),len(ys))

def main():
    attempt=BASE/'attempt1';src=attempt/'recovered';manifest=read(attempt/'recovery.json')
    for name,r in manifest.items():
        p=src/name;assert p.stat().st_size==r['bytes'] and sha(p)==r['local_sha256']
        if 'sha256' in r:assert r['sha256']==r['local_sha256']
    plan=read(BASE/'runner-plan.json');specs={r['id']:r for r in plan['prompts']}
    terminal=read(src/'terminal.json');compat=read(src/'compatibility.json')
    assert len(compat)==2 and all(all(r['exact'].values()) for r in compat)
    rows=[];per=[];native_timings=read(src/'timings.json') if (src/'timings.json').exists() else []
    for p in sorted((src/'features').glob('*.pt')):
        pid=p.stem;spec=specs[pid];meta=read(src/'metadata'/f'{pid}.json')
        assert meta['split']=='new_train' and meta['id']==pid
        assert sha(src/'banks'/f'{pid}.pt')==meta['bank_sha256']
        bank=torch.load(src/'banks'/f'{pid}.pt',weights_only=True,map_location='cpu')
        assert bank['ids'].tolist()[:len(spec['input_ids'])]==spec['input_ids']
        assert bank['ids'].tolist()[len(spec['input_ids']):]==meta['output_ids']
        records=torch.load(p,weights_only=True,map_location='cpu')
        assert len(records)==len(meta['anchors']) and len({r['anchor_index'] for r in records})==len(records)
        current=[]
        for r in records:
            m=meta['anchors'][r['anchor_index']];y=r['labels'][0].tolist();valid=r['valid'][0].tolist()
            original=[y[0]]+r['original'][0].tolist();teacher=original[:2]+r['teacher_winners'][0].tolist()
            assert original==m['original'] and y==m['labels'] and valid==m['valid']
            assert r['x1'].item()==original[1] and r['prompt_id']==pid and r['split']=='new_train'
            a=prefix(original,y,valid);b=prefix(teacher,y,valid);cap=sum(valid)
            assert 1<=a<=cap and 1<=b<=cap
            current.append(dict(prompt=pid,domain=spec['domain'],anchor_index=r['anchor_index'],
                anchor=m['anchor'],original=a,teacher=b,extra=max(0,b-a),reachable=2<=a<cap,
                repaired=b>a,damaged=b<a,capacity=cap))
        rows+=current
        original=sum(r['original'] for r in current);extra=sum(r['extra'] for r in current)
        per.append(dict(prompt=pid,domain=spec['domain'],contexts=len(current),original=original,extra=extra,
            gain_percent=100*extra/original,repairs=sum(r['repaired'] for r in current),
            reachable=sum(r['reachable'] for r in current),damaged=sum(r['damaged'] for r in current)))
    complete=terminal['status']=='complete' and len(per)==16
    if terminal['status']=='complete':assert complete
    def summarize(rr):
        ids={r['prompt'] for r in rr};pp=[p for p in per if p['prompt'] in ids]
        reachable=sum(r['reachable'] for r in rr);repairs=sum(r['repaired'] for r in rr)
        return dict(prompts=len(pp),contexts=len(rr),reachable=reachable,repairs=repairs,
            reachable_repair_rate=repairs/reachable if reachable else None,
            extra_positions=sum(r['extra'] for r in rr),equal_prompt_coverage_gain_percent=statistics.mean(p['gain_percent'] for p in pp) if pp else None,
            damaged=sum(r['damaged'] for r in rr))
    result={'status':'complete' if complete else 'incomplete','summary':summarize(rows),
        'domains':{d:summarize([r for r in rows if r['domain']==d]) for d in ('gsm8k','ultrachat')},
        'per_prompt':per,'timings':native_timings,'all_artifacts_authenticated':True,'training_updates':0,
        'source_manifest_sha256':sha(attempt/'recovery.json'),
        'limitations':['Sixteen preassigned new-training responses; no population claim, student improvement or speedup.',
        'Greedy bounded generation may truncate. Fixed-context original-plus-teacher coverage; checking both proposals is not free.',
        'Teacher-unseen means excluded from our teacher fine-tuning, not pretrained data.',
        'No favorable-prompt filtering or selection of final evaluation inputs based on these outputs.']}
    if complete:
        result['collection_projection_448_seconds']=sum(224*statistics.mean(t['collection_seconds'] for t in native_timings if t['domain']==d) for d in ('gsm8k','ultrachat'))
    (attempt/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    (attempt/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_prompt','timings')},indent=2))

if __name__=='__main__':main()
