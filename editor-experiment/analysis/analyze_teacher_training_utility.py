"""Training utility, with training exposure distinct from transfer evidence."""
from pathlib import Path
from collections import defaultdict
import json
import sys
import numpy as np


def prefix(tokens, labels, valid):
    n=0
    for t,y,v in zip(tokens,labels,valid,strict=True):
        if not v or t!=y:break
        n+=1
    return n


def summarize(rows):
    per=defaultdict(lambda:defaultdict(int));counts=defaultdict(int)
    for r in rows:
        pid=r['prompt'];original=r['original_prefix'];a=r['before_prefix'];b=r['after_prefix'];s=r['strong']
        counts['contexts']+=1;counts['eligible_first']+=r['eligible']
        counts['fully_correct_original']+=original==r['valid_length']
        counts['training_improves']+=b>a;counts['training_worsens']+=b<a
        counts['positions_added_by_training']+=max(b-a,0);counts['positions_lost_by_training']+=max(a-b,0)
        for name,v in [('before',a),('after',b)]:
            counts[name+'_repairs']+=v>original;counts[name+'_damages']+=v<original
            counts[name+'_damages_complete']+=original==r['valid_length'] and v<original
            counts[name+'_beyond_strong_positions']+=max(v-s,0)
        for name,v in [('original',original),('before',a),('after',b),('strong',s),
                       ('original_plus_before',max(original,a)),('original_plus_after',max(original,b)),
                       ('free_strong_plus_after',max(s,b))]:
            per[pid][name]+=v
        per[pid]['contexts']+=1
    ids=sorted(per);contrasts={}
    groups=[[i for i,p in enumerate(ids) if p.startswith(d+'-')] for d in ['gsm8k','ultrachat']]
    rng=np.random.default_rng(2026100493)
    draws=np.concatenate([rng.choice(g,(10000,len(g)),replace=True) for g in groups if g],axis=1)
    for name,ref in [('after','before'),('after','original'),('original_plus_after','original_plus_before'),
                     ('original_plus_after','strong'),('free_strong_plus_after','strong')]:
        values=np.array([100*(per[p][name]/per[p][ref]-1) for p in ids])
        contrasts[name+'_vs_'+ref]=dict(equal_prompt_percent=float(values.mean()),
            descriptive_interval95=np.quantile(values[draws].mean(axis=1),[.025,.975]).tolist(),
            positive_prompts=int((values>0).sum()),negative_prompts=int((values<0).sum()))
    return dict(counts=dict(counts),prompts=len(ids),totals={k:sum(v[k] for v in per.values()) for k in per[ids[0]]},
                contrasts=contrasts,per_prompt=dict(per))


def analyze(folder, plan):
    folder=Path(folder)
    read=lambda name:json.loads((folder/name).read_text())
    terminal=read('terminal.json')
    if terminal['status']!='complete':return dict(status='INCOMPLETE',terminal=terminal)
    audit=read('final-audit.json')
    if audit['training_updates']!=0 or not audit['frozen_weights_unchanged'] or audit['parity_vectors']!=32:
        raise ValueError('Incomplete frozen validation')
    from analyze_xpress_teacher_pilot import independently_count
    results={}
    for mode in ['first_guess','full_proposal']:
        panels={}
        for split in ['train','development']:
            before=read(mode+'-before.json') if split=='train' else read('initial_'+mode+'-predictions.json')
            after=read(mode+'-after.json') if split=='train' else read(mode+'-predictions.json')
            a={(r['prompt'],r['anchor_index']):r for r in before};b={(r['prompt'],r['anchor_index']):r for r in after}
            if len(a)!=256 or len(b)!=256 or set(a)!=set(b):raise ValueError('Missing paired contexts')
            if split=='train':
                expected={(p,i) for p in plan['training_prompts'] for i in plan['anchors']}
                if set(a)!=expected:raise ValueError('Wrong frozen training sample')
                for phase in ['before','after']:
                    check=read(mode+'-'+phase+'-audit.json')
                    if check!=dict(contexts=256,parity=8,frozen=True):raise ValueError('Missing parity audit')
            rows=[];controls={}
            for key in sorted(a):
                pid,ai=key
                if pid not in controls:
                    control_dir='training-controls' if split=='train' else 'controls'
                    controls[pid]={r['anchor_index']:r for r in read(control_dir+'/'+pid+'.json')}
                c=controls[pid][ai];r,q=a[key],b[key];labels=r['labels'];valid=r['valid']
                if q['labels']!=labels or q['valid']!=valid or r['tokens'][:2]!=c['original'][:2] or q['tokens'][:2]!=c['original'][:2]:
                    raise ValueError('Changed labels or clamped first guess')
                strong=independently_count(c['trees']['conditional64'],labels,valid)
                if strong!=c['coverage']['conditional64']:raise ValueError('Strong control count mismatch')
                original=prefix(c['original'],labels,valid);pre=prefix(r['tokens'],labels,valid);post=prefix(q['tokens'],labels,valid)
                for record,observed in [(r,pre),(q,post)]:
                    if record['metrics']['original']!=original or record['metrics']['edited']!=observed:
                        raise ValueError('Independent prefix recount failed')
                exposure=plan['exposures'][pid][str(ai)] if split=='train' else []
                eligible=bool(valid[1] and c['original'][1]==labels[1])
                if split=='train' and bool(exposure)!=eligible:raise ValueError('Unexpected training exposure')
                rows.append(dict(prompt=pid,anchor_index=ai,original_prefix=original,before_prefix=pre,after_prefix=post,
                    strong=strong,valid_length=sum(valid),eligible=eligible,training_exposure=len(exposure),
                    fully_supervised_exposures=sum(n==sum(valid) for n in exposure)))
            panels[split]=dict(all=summarize(rows),eligible_first=summarize([r for r in rows if r['eligible']]),rows=rows)
        results[mode]=panels
    return dict(status='COMPLETE_TRAINING_UTILITY_DIAGNOSTIC',results=results,
        limits=['Training and development differ in exposure and difficulty; their gap is not a randomized causal effect.',
                'Before/after compares one completed training recipe, not optimizer, loss or capacity interventions.',
                'One training seed, post hoc diagnostic on a prespecified sample; development data reused.',
                'Raw chain and free union measurements do not establish generation speedup or cost-effective selection.',
                'No new training, student, target generation or convergence test.'])


if __name__=='__main__':
    folder=Path(sys.argv[1]);plan=json.loads((folder.parent.parent/'plan.json').read_text())
    result=analyze(folder,plan);path=folder.parent/'analysis.json'
    if path.exists() and json.loads(path.read_text())!=result:raise ValueError('Preserve prior analysis')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'}))
    for mode,panels in result.get('results',{}).items():
        for split,panel in panels.items():
            print(json.dumps(dict(mode=mode,split=split,all={k:v for k,v in panel['all'].items() if k!='per_prompt'},
                                 eligible_first={k:v for k,v in panel['eligible_first'].items() if k!='per_prompt'})))
