"""Descriptive transfer audit using authenticated, already-saved predictions.

Candidate rows are the released readout rescored on the completed original
draft. They are not target probabilities or the editor's own confidences.
"""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import math
import sys


def prefix(tokens, labels, valid):
    return next((i for i,(t,y,v) in enumerate(zip(tokens,labels,valid,strict=True)) if not v or t!=y),len(tokens))


def candidate_at(control, position, token):
    if not 1<=position<=15:raise ValueError('Candidate rows correspond to positions1..15')
    ids=control['candidate_ids'][position-1];logs=control['candidate_logs'][position-1]
    if len(ids)!=16 or len(set(ids))!=16 or len(logs)!=16:raise ValueError('Not a top16 row')
    if any(not math.isfinite(v) for v in logs) or any(a<b for a,b in zip(logs,logs[1:])):
        raise ValueError('Invalid sorted log probabilities')
    if token not in ids:return dict(rank=None,logp=None)
    k=ids.index(token);return dict(rank=k+1,logp=logs[k])


def bucket(rank):
    return 'rank1' if rank==1 else 'rank2_4' if rank is not None and rank<=4 else 'rank5_16' if rank is not None else 'outside16'


def analyze(folder):
    folder=Path(folder);sources={}
    def read(name):
        p=folder/name;sources[name]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
    if read('terminal.json')['status']!='complete':raise ValueError('Incomplete source')
    results={};records=[]
    for split in ['train','development']:
        controls={}
        for mode in ['first_guess','full_proposal']:
            before=read(mode+'-before.json') if split=='train' else read('initial_'+mode+'-predictions.json')
            after=read(mode+'-after.json') if split=='train' else read(mode+'-predictions.json')
            pre={(r['prompt'],r['anchor_index']):r for r in before}
            post={(r['prompt'],r['anchor_index']):r for r in after}
            if len(pre)!=256 or len(post)!=256 or set(pre)!=set(post):raise ValueError('Incomplete paired panel')
            opportunities=defaultdict(Counter);damage=defaultdict(Counter);counters=Counter()
            for key in sorted(pre):
                pid,ai=key;a,b=pre[key],post[key];labels=a['labels'];valid=a['valid']
                if b['labels']!=labels or b['valid']!=valid:raise ValueError('Label mismatch')
                if pid not in controls:
                    sub='training-controls' if split=='train' else 'controls'
                    controls[pid]={r['anchor_index']:r for r in read(sub+'/'+pid+'.json')}
                c=controls[pid][ai];original=c['original'];op=prefix(original,labels,valid)
                ap=prefix(a['tokens'],labels,valid);bp=prefix(b['tokens'],labels,valid)
                if a['tokens'][:2]!=original[:2] or b['tokens'][:2]!=original[:2]:raise ValueError('Clamp changed')
                if a['metrics']['edited']!=ap or b['metrics']['edited']!=bp:raise ValueError('Prefix mismatch')
                counters['contexts']+=1
                if op>=2 and op<sum(valid):
                    cand=candidate_at(c,op,labels[op]);group=bucket(cand['rank']);row=opportunities[group]
                    row['opportunities']+=1
                    row['before_repairs']+=ap>op;row['after_repairs']+=bp>op
                    row['after_fixes_original_blocker']+=b['tokens'][op]==labels[op]
                    row['after_fixes_blocker_but_damages_earlier']+=b['tokens'][op]==labels[op] and bp<op
                    row['newly_learned_repairs']+=bp>op and ap<=op
                    row['lost_existing_repairs']+=ap>op and bp<=op
                    row['after_extra_positions']+=max(bp-op,0)
                    row['strong_already_passes_blocker']+=c['coverage']['conditional64']>op
                    row['after_repair_already_covered_by_strong']+=bp>op and c['coverage']['conditional64']>=bp
                    records.append(dict(kind='first_error',split=split,mode=mode,prompt=pid,anchor_index=ai,
                        original_prefix=op,before_prefix=ap,after_prefix=bp,group=group,correct_token=labels[op],
                        original_token=original[op],before_token=a['tokens'][op],after_token=b['tokens'][op],
                        strong_coverage=c['coverage']['conditional64'],**cand))
                elif op==sum(valid):counters['original_complete']+=1
                else:counters['locked_first_error']+=1
                if bp<op:
                    correct=candidate_at(c,bp,labels[bp]);wrong=candidate_at(c,bp,b['tokens'][bp])
                    row=damage[bucket(correct['rank'])];row['damaged_prefixes']+=1
                    row['damaged_complete']+=op==sum(valid)
                    row['wrong_inside16']+=wrong['rank'] is not None
                    row['wrong_outside16']+=wrong['rank'] is None
                    gap=None
                    if wrong['logp'] is not None and correct['logp'] is not None:
                        gap=correct['logp']-wrong['logp'];row['both_scored']+=1
                        row['correct_preferred_by_released']+=gap>0
                        row['wrong_preferred_by_released']+=gap<0
                    records.append(dict(kind='damage',split=split,mode=mode,prompt=pid,anchor_index=ai,
                        original_prefix=op,after_prefix=bp,correct=correct,wrong=wrong,logp_gap=gap,
                        correct_token=labels[bp],after_token=b['tokens'][bp]))
            results[mode+'_'+split]=dict(counts=dict(counters),first_error_by_candidate_rank={k:dict(v) for k,v in opportunities.items()},
                damage_by_correct_candidate_rank={k:dict(v) for k,v in damage.items()})
    # Fixed categorical standardization: apply training's within-rank repair
    # rates to the development rank distribution. Descriptive only, no fitting
    # a selector and no claim that rank removes all dataset differences.
    for mode in ['first_guess','full_proposal']:
        train=results[mode+'_train']['first_error_by_candidate_rank'];dev=results[mode+'_development']['first_error_by_candidate_rank']
        missing=[k for k,v in dev.items() if v['opportunities'] and not train.get(k,{}).get('opportunities')]
        results[mode+'_rank_standardization']=dict(uncovered_strata=missing,
            expected_dev_repairs_at_training_rates=None if missing else sum(v['opportunities']*train[k]['after_repairs']/train[k]['opportunities'] for k,v in dev.items()),
            actual_dev_repairs=sum(v['after_repairs'] for v in dev.values()),
            note='Fixed rank groups only; not a causal estimate, generalization guarantee or deployable predictor.')
    return dict(results=results,records=records,sources=sources,paid_execution=False,
        limits=['All ranking uses released shared scores recomputed on the completed first draft, not calibrated target likelihoods.',
                'Top16 omission is a censored rank, not evidence that contextual information is absent.',
                'Reference tokens classify outcomes after inference; no deployable selection policy is tested.',
                'Panels use one checkpoint seed and reused development responses.',
                'Marginal counts are descriptive; contexts within a prompt are dependent.',
                'No new training, second-pass intervention or generation latency measurement.'])


if __name__=='__main__':
    result=analyze(Path(sys.argv[1]));dest=Path(sys.argv[2])
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise ValueError('Preserve existing analysis')
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['results'],indent=2))
