"""Frozen fixed30-node policy comparison for matched support training."""
import json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from analyze_alternative_first import conditioned_tree,transform,matching_prefix,fill,prefixes
from original_draft_tree_reference import build as ordinary_tree
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'notes/second-pass-profiling-2026-10-04/teacher-support-training'
NAMES=('initial_feedback','initial_no_feedback','old_support_feedback','all_support_feedback','all_support_no_feedback')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    attempt=OUT/'attempt1';manifest=read(attempt/'recovery.json')
    for name,spec in manifest.items():assert sha(attempt/'recovered'/name)==spec['sha256']
    assert read(attempt/'recovered/terminal.json')['status']=='complete'
    audit=read(attempt/'recovered/final-audit.json')
    assert audit['updates_per_arm']==[384]*3 and audit['frozen_versions_unchanged']
    bundle=read(OUT/'frozen/local-bundle.json');spec=bundle['inputs']['candidate-records.json'];src=ROOT/spec['local_path'];assert sha(src)==spec['sha256']
    refs=read(src);pred={}
    for name in NAMES:
        rows=read(attempt/'recovered'/(name+'-predictions.json'));pred[name]={(r['prompt'],r['anchor_index']):r for r in rows};assert len(pred[name])==2048
    per=defaultdict(lambda:defaultdict(int));counts=defaultdict(lambda:defaultdict(int));details=[]
    for r in refs:
        key=r['prompt'],r['anchor_index'];ids=r['candidate_ids'];q=r['q_logs'];original=r['original_tokens']
        values={'ordinary':matching_prefix(conditioned_tree(original,ids,q,{}),r['labels'],r['valid'])}
        for name in NAMES:
            p=pred[name][key]
            if name=='initial_feedback':assert p['tokens']['1']==r['teacher_tokens'] and p['candidate_logs']['1']==r['teacher_logs'][1:]
            t={ids[0][k-1]:transform(q,[q[0]]+p['candidate_logs'][str(k)]) for k in (1,2)}
            values[name]=matching_prefix(conditioned_tree(original,ids,q,t),r['labels'],r['valid'])
        # Secondary predeclared output policy permits new full-vocabulary tokens.
        # Match every condition and ordinary expansion at the same context budget.
        base=set(map(tuple,ordinary_tree(original,ids,q)['nodes']))
        mandatory={name:base|prefixes(pred[name][key]['tokens']['1'])|prefixes(pred[name][key]['tokens']['2']) for name in NAMES}
        budget=max(map(len,mandatory.values()))
        values['chain_ordinary']=matching_prefix(ordinary_tree(original,ids,q,node_budget=budget)['nodes'],r['labels'],r['valid'])
        for name in NAMES:
            values['chain_'+name]=matching_prefix(fill(mandatory[name],original,ids,q,budget),r['labels'],r['valid'])
        for name,value in values.items():per[r['prompt']][name]+=value
        for policy in ('','chain_'):
            for control in ('ordinary','initial_feedback','old_support_feedback','all_support_no_feedback'):
                delta=values[policy+'all_support_feedback']-values[policy+control];c=counts[policy+control]
                c['better_contexts']+=delta>0;c['worse_contexts']+=delta<0;c['gained_positions']+=max(delta,0);c['lost_positions']+=max(-delta,0)
        details.append(dict(prompt=key[0],anchor_index=key[1],values=values,chain_budget=budget))
    prompts=sorted(per);groups=[[i for i,p in enumerate(prompts) if p.startswith(d+'-')] for d in ('gsm8k','ultrachat')]
    rng=np.random.default_rng(202610047);draws=np.concatenate([rng.choice(g,(10000,len(g)),replace=True) for g in groups],axis=1)
    metrics={}
    passed=[]
    for policy in ('','chain_'):
        for name in NAMES:
            for control in ('ordinary',)+ (('initial_feedback','old_support_feedback','all_support_no_feedback') if name=='all_support_feedback' else ()):
                a=np.array([per[p][policy+name] for p in prompts],float);b=np.array([per[p][policy+control] for p in prompts],float);s=100*(1-b/a)
                metrics[policy+name+'/'+control]=dict(modeled_equal_round_cost_savings_percent=float(s.mean()),
                    descriptive_interval95=np.quantile(s[draws].mean(-1),[.025,.975]).tolist(),
                    domain_savings={d:float(s[g].mean()) for d,g in zip(('gsm8k','ultrachat'),groups)},
                    maximum_uniform_extra_round_cost_for_5pct=100*(.95/np.mean(b/a)-1))
        main=metrics[policy+'all_support_feedback/ordinary'];support=metrics[policy+'all_support_feedback/old_support_feedback'];feedback=metrics[policy+'all_support_feedback/all_support_no_feedback']
        if (main['modeled_equal_round_cost_savings_percent']>=10 and main['descriptive_interval95'][0]>5
          and all(v>0 for v in main['domain_savings'].values())
          and support['descriptive_interval95'][0]>0 and feedback['descriptive_interval95'][0]>0):passed.append(policy or 'score_tree')
    result=dict(status='COMPLETE',advancement_screen_passed=bool(passed),passing_policies=passed,metrics=metrics,counts=dict(counts),details=details,per_prompt=dict(per),
        limitations=['One fixed continued-training epoch, one seed, exposed development prompts',
        '10% free-cost screen leaves approximately5.6% added-cost margin for5% latency target; actual cost must be measured',
        'Coverage on recorded references is not actual generation; broader data and input effects do not establish novelty',
        'No-feedback uses one transformer proposal distribution; feedback uses two. Their complete runtime costs differ'])
    (attempt/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(screen_passed=bool(passed),metrics=metrics),indent=2))
if __name__=='__main__':main()
