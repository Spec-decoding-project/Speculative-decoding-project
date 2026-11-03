"""Fixed direct-versus-final readout comparison and scoped proposal opportunity."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import sys
import numpy as np


def prefix(tokens, labels, valid):
    n = 0
    for t,y,ok in zip(tokens,labels,valid,strict=True):
        if not ok or t != y:break
        n += 1
    return n


def analyze(folder):
    folder = Path(folder);sources = {}
    def read(path):
        sources[path.relative_to(folder).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    terminal = read(folder/'terminal.json')
    if terminal['status'] != 'complete':
        return dict(status='INCOMPLETE',worker=terminal,interpretation='No completed readout conclusion.')
    audit = read(folder/'final-audit.json')
    if not audit['frozen_weights_unchanged'] or audit['training_updates'] != 0:
        raise ValueError('Model changes invalidate inference intervention')
    from analyze_xpress_teacher_pilot import independently_count
    results = {}
    for mode in ('first_guess','full_proposal'):
        rows = read(folder/(mode+'-trace.json'))
        saved = {(r['prompt'],r['anchor_index']):r for r in read(folder/(mode+'-predictions.json'))}
        if len(rows)!=256 or len({(r['prompt'],r['anchor_index']) for r in rows})!=256:
            raise ValueError('Missing or duplicated trace contexts')
        if audit['completed'][mode]['helper_score_parity_checks']!=2 or not audit['completed'][mode]['saved_tokens_match']:
            raise ValueError('Neutral parity checks missing')
        per=defaultdict(lambda:defaultdict(int));counts=defaultdict(int);step_sums=[0]*7
        controls={}
        for row in rows:
            pid,ai=row['prompt'],row['anchor_index'];key=pid,ai
            if pid not in controls:
                controls[pid]={r['anchor_index']:r for r in read(folder/'controls'/(pid+'.json'))}
            c=controls[pid][ai];labels,valid=row['labels'],row['valid'];trace=row['trace']
            if [t['step'] for t in trace]!=list(range(7)) or row['original']!=c['original']:
                raise ValueError('Changed trajectory or original draft')
            strong=independently_count(c['trees']['conditional64'],labels,valid)
            if strong!=row['strong_control_coverage']:raise ValueError('Control count differs')
            if saved[key]['labels']!=labels or saved[key]['valid']!=valid or trace[-1]['tokens']!=saved[key]['tokens']:
                raise ValueError('Saved outcome did not reproduce')
            for t in trace:
                if t['tokens'][:2]!=row['original'][:2] or t['prefix']!=prefix(t['tokens'],labels,valid):
                    raise ValueError('Clamping or independent prefix mismatch')
                step_sums[t['step']]+=t['prefix']
            original=prefix(row['original'],labels,valid);direct=trace[0]['prefix'];final=trace[6]['prefix']
            for name,n in [('direct',direct),('final',final)]:
                counts[name+'_repairs_original']+=n>original;counts[name+'_damages_original']+=n<original
                counts[name+'_damages_complete_original']+=original==sum(valid) and n<original
                counts[name+'_beyond_strong_contexts']+=n>strong
                counts[name+'_beyond_strong_positions']+=max(n-strong,0)
            counts['refinement_improves_direct']+=final>direct;counts['refinement_worsens_direct']+=final<direct
            counts['refinement_ties_direct']+=final==direct
            counts['refinement_positions_gained']+=max(final-direct,0);counts['refinement_positions_lost']+=max(direct-final,0)
            for name,n in [('original',original),('direct',direct),('final',final),('strong',strong),
                           ('original_plus_direct',max(original,direct)),('original_plus_final',max(original,final)),
                           ('free_strong_plus_direct',max(strong,direct)),('free_strong_plus_final',max(strong,final))]:
                per[pid][name]+=n
        ids=sorted(per)
        if len(ids)!=64:raise ValueError('Missing prompts')
        groups=[[i for i,p in enumerate(ids) if p.startswith(d+'-')] for d in ['gsm8k','ultrachat']]
        if list(map(len,groups))!=[32,32]:raise ValueError('Domain balance changed')
        rng=np.random.default_rng(2026100492);draws=np.concatenate([rng.choice(g,(10000,len(g)),replace=True) for g in groups],axis=1)
        contrasts={}
        for name,ref in [('direct','final'),('original_plus_direct','strong'),('original_plus_final','strong'),
                         ('free_strong_plus_direct','strong'),('free_strong_plus_final','strong')]:
            values=np.array([100*(per[p][name]/per[p][ref]-1) for p in ids])
            contrasts[name+'_vs_'+ref]=dict(mean_prompt_coverage_difference_percent=float(values.mean()),
                descriptive_interval95=np.quantile(values[draws].mean(axis=1),[.025,.975]).tolist(),
                positive_prompts=int((values>0).sum()),negative_prompts=int((values<0).sum()))
        results[mode]=dict(contexts=256,prompts=64,counts=dict(counts),contrasts=contrasts,
             total_positions={k:sum(v[k] for v in per.values()) for k in per[ids[0]]},
             descriptive_step_prefix_totals=step_sums,per_prompt=dict(per))
    return dict(status='COMPLETE_READOUT_INTERVENTION',results=results,sources=sources,
        generation_speedup=False,student_compressibility=False,limits=[
        'One seed and reused development data; intervals are descriptive.',
        'Direct readout removes a component the backbone was trained through; it does not test retraining that component.',
        'Free combination concerns these particular greedy chains, not all vocabulary scores or future learned models.',
        'Known anchor and fixed first guess included consistently; runtime and target fidelity unmeasured.',
        'Intermediate steps explain the trajectory and are not post-hoc selected methods.'])


if __name__=='__main__':
    folder=Path(sys.argv[1]);result=analyze(folder);out=folder.parent/'analysis.json'
    if out.exists() and json.loads(out.read_text())!=result:raise ValueError('Preserve existing analysis')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result if result['status']=='INCOMPLETE' else dict(status=result['status'],results={k:{a:b for a,b in v.items() if a!='per_prompt'} for k,v in result['results'].items()}),indent=2))
