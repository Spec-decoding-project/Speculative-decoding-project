"""Can the saved released-score preference distinguish useful/harmful edits?

Descriptive discrimination only. No threshold is tuned, and neutral edits are
excluded from this conditional question rather than claimed as a serving gate.
"""
from pathlib import Path
import json
import sys
import numpy as np
from analyze_teacher_candidate_transfer import candidate_at, prefix


def auc(scores, labels):
    scores=np.asarray(scores);labels=np.asarray(labels,dtype=bool)
    p,n=scores[labels],scores[~labels]
    if not len(p) or not len(n):return None
    return float(((p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:])).mean())


def analyze(folder):
    folder=Path(folder);results={}
    for split in ['train','development']:
        controls={}
        for mode in ['first_guess','full_proposal']:
            rows=json.loads((folder/(mode+'-after.json' if split=='train' else mode+'-predictions.json')).read_text())
            records=[];excluded=0;neutral=0
            all_ids=sorted({r['prompt'] for r in rows})
            for r in rows:
                pid,ai=r['prompt'],r['anchor_index'];labels=r['labels'];valid=r['valid']
                if pid not in controls:
                    sub='training-controls' if split=='train' else 'controls'
                    controls[pid]={x['anchor_index']:x for x in json.loads((folder/sub/(pid+'.json')).read_text())}
                c=controls[pid][ai];original=c['original'];o=prefix(original,labels,valid);e=prefix(r['tokens'],labels,valid)
                if e==o:neutral+=1;continue
                j=min(o,e)
                # Before the first helpful/harmful edit, both histories match.
                if original[:j]!=r['tokens'][:j] or j<2:raise ValueError('Invalid comparable edit')
                old=candidate_at(c,j,original[j]);new=candidate_at(c,j,r['tokens'][j])
                if old['rank'] is None or new['rank'] is None:excluded+=1;continue
                records.append(dict(prompt=pid,anchor_index=ai,helpful=e>o,score=new['logp']-old['logp'],
                    prefix_change=e-o,original_rank=old['rank'],new_rank=new['rank']))
            scores=np.array([r['score'] for r in records]);labels=np.array([r['helpful'] for r in records])
            base=auc(scores,labels)
            # Resample all original prompts within domain. All edits belonging
            # to one selected prompt receive its same multiplicity.
            ids={p:i for i,p in enumerate(all_ids)};ix=np.array([ids[r['prompt']] for r in records])
            groups=[[ids[p] for p in all_ids if p.startswith(d+'-')] for d in ['gsm8k','ultrachat']]
            rng=np.random.default_rng(2026100494);draws=np.concatenate([rng.choice(g,(10000,len(g)),replace=True) for g in groups],axis=1)
            pos,neg=scores[labels],scores[~labels];pair=(pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])
            sampled=[]
            for draw in draws:
                w=np.bincount(draw,minlength=len(all_ids))[ix];wp,wn=w[labels],w[~labels]
                denom=wp.sum()*wn.sum()
                if denom:sampled.append(float(wp@pair@wn/denom))
            results[mode+'_'+split]=dict(helpful=int(labels.sum()),harmful=int((~labels).sum()),
                censored_outside16=excluded,neutral_prefix_edits_excluded=neutral,auc=base,
                prompt_bootstrap_interval95=np.quantile(sampled,[.025,.975]).tolist(),
                helpful_score_quartiles=np.quantile(scores[labels],[0,.25,.5,.75,1]).tolist(),
                harmful_score_quartiles=np.quantile(scores[~labels],[0,.25,.5,.75,1]).tolist(),records=records)
    return dict(results=results,interpretation='Higher score means the released shared readout favors the proposed edit more.',
        limits=['Discrimination conditional on a helpful or harmful prefix edit, not all proposals.',
                'Neutral edits and top16-censored edit pairs are separately counted; no selection policy is evaluated.',
                'Scores are not calibrated target likelihoods and reused development data cannot validate a new policy.',
                'No evidence that conditioning more strongly on this score would preserve teacher gains.',
                'No threshold tuning, neural fitting, GPU work or latency claim.'])


if __name__=='__main__':
    assert auc([2,1],[True,False])==1
    assert auc([1,2],[True,False])==0
    assert auc([1,1],[True,False])==.5
    r=analyze(sys.argv[1]);p=Path(sys.argv[2])
    if p.exists():raise ValueError('Preserve previous result')
    p.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='records'} for k,v in r['results'].items()},indent=2))
