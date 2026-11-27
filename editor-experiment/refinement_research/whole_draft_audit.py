"""Independent saved-token recount; imports no training code or PyTorch."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def prefix(tokens,labels):
    count=0
    for actual,expected in zip(tokens[1:],labels[1:]):
        if actual!=expected:
            break
        count+=1
    return count


def recount(result,bank):
    counts=[];seen=set()
    for row in result['rows']:
        key=(row['prompt_id'],row['anchor']);assert key not in seen;seen.add(key)
        original=bank[key]
        assert row['original_tokens']==original['draft'] and row['label_tokens']==original['labels']
        edited=row['edited_tokens'];labels=original['labels']
        assert len(edited)==16 and edited[:2]==original['draft'][:2]
        assert all(type(t) is int and 0<=t<151936 for t in edited)
        a,b=prefix(original['draft'],labels),prefix(edited,labels)
        assert row['original_progress']==a and row['edited_progress']==b
        c=sum(x==y for x,y in zip(edited[2:],labels[2:]))
        d=sum(x==y for x,y in zip(original['draft'][2:],labels[2:]))
        counts.append(dict(prompt_id=key[0],source=original['source'],original=a,edited=b,
            original_tokens=d,edited_tokens=c,kind=original['kind'],exact=edited[2:]==labels[2:]))
    assert seen==set(bank)
    metrics=summarize(counts)
    assert abs(metrics['edited_prefix']-result['mean_edited_progress'])<1e-12
    assert abs(metrics['edited_token_accuracy']-result['free_running_token_accuracy'])<1e-12
    assert metrics['improved']==result['repaired'] and metrics['damaged']==result['damaged']
    return counts,metrics


def summarize(rows):
    n=len(rows)
    return dict(blocks=n,prompts=len({r['prompt_id'] for r in rows}),
        original_prefix=sum(r['original'] for r in rows)/n,edited_prefix=sum(r['edited'] for r in rows)/n,
        original_token_accuracy=sum(r['original_tokens'] for r in rows)/(14*n),
        edited_token_accuracy=sum(r['edited_tokens'] for r in rows)/(14*n),
        improved=sum(r['edited']>r['original'] for r in rows),damaged=sum(r['edited']<r['original'] for r in rows),
        prefix_gained=sum(max(0,r['edited']-r['original']) for r in rows),
        prefix_lost=sum(max(0,r['original']-r['edited']) for r in rows),
        perfect_damaged=sum(r['kind']=='preservation' and r['edited']<15 for r in rows),
        wrong_first=sum(r['original']==0 for r in rows),exact_suffixes=sum(r['exact'] for r in rows))


def interval(rows,reference,draws=10000):
    grouped=defaultdict(list)
    assert len(rows)==len(reference)
    for a,b in zip(rows,reference):
        assert a['prompt_id']==b['prompt_id']
        grouped[a['source'],a['prompt_id']].append((a['edited']-b['edited'],a['edited_tokens']-b['edited_tokens']))
    rng=np.random.default_rng(0)
    total=np.zeros((draws,3))
    for source in sorted({k[0] for k in grouped}):
        matrix=np.array([[sum(v[0] for v in values),sum(v[1] for v in values),len(values)]
            for key,values in sorted(grouped.items()) if key[0]==source])
        indices=rng.integers(0,len(matrix),size=(draws,len(matrix)))
        total+=matrix[indices].sum(axis=1)
    return dict(prefix_delta_ci95=np.quantile(total[:,0]/total[:,2],[.025,.975]).tolist(),
        token_accuracy_delta_ci95=np.quantile(total[:,1]/(14*total[:,2]),[.025,.975]).tolist(),
        draws=draws,scope='Exploratory paired prompt bootstrap stratified by source; no correction for multiple arms/seeds')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--bank',type=Path,required=True)
    args=p.parse_args()
    assert sha(args.bank)=='c7f92fae5609137510a198297a31b5178c70c413f263f89f04560b98aa6bbb3d'
    connection=sqlite3.connect(f'file:{args.bank}?mode=ro',uri=True)
    bank={}
    for payload, in connection.execute("SELECT payload FROM examples WHERE split='development' ORDER BY id"):
        row=json.loads(payload);bank[row['prompt_id'],row['anchor']]=row
    connection.close();assert len(bank)==2048
    run=json.loads((args.output/'run.json').read_text());complete=json.loads((args.output/'complete.json').read_text())
    assert complete['passed_checks']
    candidates={}
    for arm in run['arms']:
        selection=json.loads((args.output/arm/'generator-complete.json').read_text())
        options=[]
        for path in sorted((args.output/arm).glob('calibration-*.json')):
            result=json.loads(path.read_text());step=int(path.stem.rsplit('-',1)[1])
            options.append((result['mean_edited_progress'],result['free_running_token_accuracy'],-step))
        best=max(options);assert selection['best_step']==-best[2] and tuple(selection['best_score'])==best[:2]
        steps=[]
        for path in (args.output/arm).glob('updates-*.jsonl'):
            steps.extend(json.loads(line)['step'] for line in path.read_text().splitlines())
        assert sorted(steps)==list(range(1,769))
        for kind in ('plain','gated'):
            candidates[f'{arm}/{kind}']=json.loads((args.output/arm/f'development-{kind}.json').read_text())
    candidates['original_generator']=json.loads((args.output/'original-generator-development.json').read_text())
    incumbent=json.loads((args.output/'incumbent-development.json').read_text())
    original_rows,original_metrics=recount(incumbent,bank)
    baseline=[dict(r,edited=r['original'],edited_tokens=r['original_tokens']) for r in original_rows]
    results={}
    for name,result in candidates.items():
        rows,metrics=recount(result,bank)
        assert all((a['prompt_id'],a['source'])==(b['prompt_id'],b['source']) for a,b in zip(rows,original_rows))
        results[name]=dict(**metrics,
            by_source={s:summarize([r for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})},
            versus_incumbent=interval(rows,original_rows),versus_dflash=interval(rows,baseline),
            meets_point_criteria=metrics['edited_prefix']>original_metrics['edited_prefix'] and
                metrics['edited_token_accuracy']>=metrics['original_token_accuracy'])
    report=dict(passed=True,incumbent=original_metrics,arms=results,
        scope='Existing development prompts, full-model accuracy and prefix comparison; no native speed claim')
    (args.output/'independent-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
