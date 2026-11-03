"""Frozen comparisons for first-token-conditioned redrafting; no label-based selection."""
from pathlib import Path
import sys,json,hashlib,heapq,math
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from original_draft_tree_reference import build as ordinary, matching_prefix
from audit_branch_consistent_teacher import build as old_teacher
from student_candidate_mass_control import transform
OUT=ROOT/'notes/second-pass-profiling-2026-10-04/alternative-first-conditioning'
SOURCE=ROOT/'notes/second-pass-profiling-2026-10-03/wider-teacher-candidates-completion/consolidated'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prefixes(chain):return {tuple(chain[:i]) for i in range(1,len(chain)+1)}

def conditioned_tree(original,ids,q,by_first,budget=30):
    nodes=prefixes(original);scores={};pending=[]
    def edge(parent,token):
        row=by_first.get(parent[1],q) if len(parent)>1 else q
        return row[len(parent)-1][ids[len(parent)-1].index(token)]
    for n in sorted(nodes,key=len):scores[n]=0. if len(n)==1 else scores[n[:-1]]+edge(n[:-1],n[-1])
    def offer(parent):
        if len(parent)==len(original):return
        for token in ids[len(parent)-1]:
            child=parent+(token,)
            if child not in nodes:heapq.heappush(pending,(-(scores[parent]+edge(parent,token)),child))
    for n in sorted(nodes):offer(n)
    while len(nodes)<budget and pending:
        neg,n=heapq.heappop(pending)
        if n in nodes:continue
        nodes.add(n);scores[n]=-neg;offer(n)
    return nodes

def fill(mandatory,original,ids,q,budget):
    nodes=set(mandatory)
    pool=ordinary(original,ids,q,node_budget=budget+len(nodes))
    for n in sorted(map(tuple,pool['nodes']),key=lambda n:(-pool['log_mass'][n],n)):
        if len(nodes)==budget:break
        if len(n)==1 or n[:-1] in nodes:nodes.add(n)
    assert len(nodes)==budget
    assert all(len(n)==1 or n[:-1] in nodes for n in nodes)
    return nodes

def evaluate(r,p):
    ids=r['candidate_ids'];q=r['q_logs'];original=r['original_tokens']
    assert p['tokens']['1']==r['teacher_tokens'] and p['candidate_logs']['1']==r['teacher_logs'][1:]
    t1=transform(q,[q[0]]+p['candidate_logs']['1']);t2=transform(q,[q[0]]+p['candidate_logs']['2'])
    first,second=ids[0][:2]
    modes={'ordinary':{},'existing':{first:t1},'reused_tail':{first:t1,second:t1},'conditioned':{first:t1,second:t2}}
    trees={name:conditioned_tree(original,ids,q,mode) for name,mode in modes.items()}
    assert trees['ordinary']==set(map(tuple,ordinary(original,ids,q)['nodes']))
    assert trees['existing']==old_teacher(original,ids,q,t1)
    primary={name:matching_prefix(tree,r['labels'],r['valid']) for name,tree in trees.items()}
    # This secondary test allows the new full-vocabulary token proposals.
    # Allocate the SAME node count to both unions and the ordinary reference.
    base=trees['ordinary']|prefixes(p['tokens']['1'])
    clamp=[original[0],second]+p['tokens']['1'][2:]
    mandatory={'reused_tail':base|prefixes(clamp),'conditioned':base|prefixes(p['tokens']['2'])}
    budget=max(map(len,mandatory.values()))
    unions={name:fill(nodes,original,ids,q,budget) for name,nodes in mandatory.items()}
    unions['ordinary']=set(map(tuple,ordinary(original,ids,q,node_budget=budget)['nodes']))
    secondary={name:matching_prefix(nodes,r['labels'],r['valid']) for name,nodes in unions.items()}
    useful_new=sum(r['labels'][j] not in ids[j-1] for j in range(1,secondary['conditioned']))
    return dict(primary=primary,secondary=secondary,node_budget=budget,useful_outside_support_positions=useful_new)

def summarize(details):
    per=defaultdict(lambda:defaultdict(int));counts=defaultdict(lambda:defaultdict(int))
    for d in details:
        for panel in ('primary','secondary'):
            for name,value in d[panel].items():per[d['prompt']][panel+'/'+name]+=value
            for control in ('ordinary','reused_tail'):
                delta=d[panel]['conditioned']-d[panel][control];c=counts[panel+'/'+control]
                c['repaired_contexts']+=delta>0;c['damaged_contexts']+=delta<0
                c['gained_positions']+=max(0,delta);c['lost_positions']+=max(0,-delta)
    prompts=sorted(per);groups=[[i for i,p in enumerate(prompts) if p.startswith(d+'-')] for d in ('gsm8k','ultrachat')]
    rng=np.random.default_rng(202610045)
    draws=np.concatenate([rng.choice(g,(10000,len(g)),replace=True) for g in groups],axis=1)
    metrics={}
    for panel in ('primary','secondary'):
        actual=np.array([per[p][panel+'/conditioned'] for p in prompts],float)
        for control in ('ordinary','reused_tail')+ (('existing',) if panel=='primary' else ()):
            reference=np.array([per[p][panel+'/'+control] for p in prompts],float)
            savings=100*(1-reference/actual)
            metrics[panel+'/'+control]=dict(modeled_equal_round_cost_savings_percent=float(savings.mean()),
                descriptive_interval95=np.quantile(savings[draws].mean(-1),[.025,.975]).tolist(),
                domain_savings={d:float(savings[g].mean()) for d,g in zip(('gsm8k','ultrachat'),groups)},
                maximum_uniform_added_round_cost_percent_for_5pct=100*(.95/np.mean(reference/actual)-1),
                counts=dict(counts[panel+'/'+control]))
    return dict(metrics=metrics,per_prompt=dict(per))

def main():
    attempt=OUT/'attempt1';receipt=read(attempt/'recovery.json')
    p=attempt/'recovered/predictions.json';assert sha(p)==receipt['predictions.json']['sha256']
    audit=read(attempt/'recovered/final-audit.json');assert audit==dict(contexts=2048,teacher_unchanged=True,versions_unchanged=True)
    assert read(attempt/'recovered/terminal.json')['status']=='complete'
    predictions=read(p);by={(p['prompt'],p['anchor_index']):p for p in predictions};assert len(by)==2048
    src=SOURCE/'recovered/rows.json';assert sha(src)==read(SOURCE/'recovery.json')['rows.json']['sha256']
    records=read(src);details=[]
    for r in records:
        p=by[r['prompt'],r['anchor_index']]
        assert p['historical_primary_tokens_exact'] and p['historical_primary_scores_exact']
        details.append(dict(prompt=r['prompt'],anchor_index=r['anchor_index'],**evaluate(r,p)))
    assert len(details)==2048
    result=dict(status='COMPLETE',**summarize(details),details=details,
        input_sha256=sha(src),prediction_sha256=sha(attempt/'recovered/predictions.json'),script_sha256=sha(__file__),
        limitations=['Reused development contexts; intervals are descriptive, not fresh confirmation',
         'Reference-prefix coverage, not native tree verification or actual generation speed',
         'Primary keeps candidate support fixed; secondary matches proposal-dependent node counts',
         'No learned selector, training, runtime or novelty claim; two full teacher evaluations are not free'])
    (attempt/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['metrics'],indent=2))
if __name__=='__main__':main()
