"""Use conditional teacher scores only on its matching first-token branch."""
from pathlib import Path
from collections import defaultdict
import json,math,hashlib,heapq,sys,random
import numpy as np
from student_candidate_mass_control import transform
from original_draft_tree_reference import matching_prefix
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'notes/second-pass-profiling-2026-10-03';OUT=BASE/'teacher-chain-cost-margin'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(original,ids,q,t,budget=30,heap=True):
    mandatory={tuple(original[:i]) for i in range(1,len(original)+1)};nodes=set(mandatory)
    def edge(parent,token):
        j=len(parent)-1;logs=t if len(parent)>1 and parent[1]==original[1] else q
        return logs[j][ids[j].index(token)]
    score={}
    for n in sorted(mandatory,key=len):score[n]=0. if len(n)==1 else score[n[:-1]]+edge(n[:-1],n[-1])
    pending=[] if heap else {}
    def offer(n):
        if len(n)==len(original):return
        for token in ids[len(n)-1]:
            child=n+(token,)
            if child not in nodes:
                s=score[n]+edge(n,token)
                if heap:heapq.heappush(pending,(-s,child))
                else:pending[child]=s
    for n in sorted(nodes):offer(n)
    while len(nodes)<budget and pending:
        if heap:
            negative,n=heapq.heappop(pending);s=-negative
        else:n=min(pending,key=lambda n:(-pending[n],n));s=pending.pop(n)
        if n in nodes:continue
        nodes.add(n);score[n]=s;offer(n)
    return nodes
def synthetic():
    rng=random.Random(303)
    for _ in range(300):
        length=rng.randrange(3,6);k=rng.randrange(2,5);ids=[list(range(k)) for j in range(length-1)]
        def logs():
            result=[]
            for j in range(length-1):
                w=[rng.random()+.01 for _ in range(k)];z=sum(w)*1.1;result.append([math.log(x/z) for x in w])
            return result
        q=logs();t=logs();original=[99]+[0]*(length-1);budget=length+3
        actual=build(original,ids,q,t,budget)
        # Exhaustively score every short path using a product of conditional edges.
        frontier=[((99,),1.)];paths={}
        while frontier:
            parent,mass=frontier.pop()
            if len(parent)==length:continue
            use=t if len(parent)>1 and parent[1]==original[1] else q
            for token,lp in zip(ids[len(parent)-1],use[len(parent)-1]):
                child=parent+(token,);cm=mass*math.exp(lp);paths[child]=cm;frontier.append((child,cm))
        mandatory={tuple(original[:i]) for i in range(1,length+1)}
        expected=mandatory|set(sorted(set(paths)-mandatory,key=lambda n:(-paths[n],n))[:budget-length])
        assert actual==expected
    return 300
def main():
    assert sys.version_info[:2]==(3,11);nchecks=synthetic()
    source=BASE/'wider-teacher-candidates-completion/consolidated';p=source/'recovered/rows.json'
    assert sha(p)==read(source/'recovery.json')['rows.json']['sha256'];rows=read(p)
    prior=read(OUT/'free-first-reference.json');old={(r['prompt'],r['anchor']):r['values'] for r in prior['rows']}
    per=defaultdict(lambda:defaultdict(int));counts=defaultdict(lambda:defaultdict(int));details=[];checks=0
    for r in rows:
        key=r['prompt'],r['anchor_index'];v={}
        for k in (4,8,16):
            ids=[a[:k] for a in r['candidate_ids']];q=[a[:k] for a in r['q_logs']];raw=[a[:k] for a in r['teacher_logs']]
            for label,t in [('original',q),('teacher_mass',transform(q,raw)),('teacher_full',raw)]:
                nodes=build(r['original_tokens'],ids,q,t);assert nodes==build(r['original_tokens'],ids,q,t,heap=False);checks+=1
                name=f'{label}_k{k}';value=matching_prefix(nodes,r['labels'],r['valid']);v[name]=value
                if label=='original':assert value==old[key][f'free_k{k}']
                baseline=old[key][f'free_k{k}'];delta=value-baseline;c=counts[name]
                c['gains']+=max(0,delta);c['losses']+=max(0,-delta);c['better_contexts']+=delta>0;c['worse_contexts']+=delta<0
        for name,value in v.items():per[r['prompt']][name]+=value
        per[r['prompt']]['original_chain']+=r['original_progress'];details.append(dict(prompt=r['prompt'],anchor=key[1],values=v))
    prompts=sorted(per);groups=[np.array([i for i,p in enumerate(prompts) if p.startswith(d+'-')]) for d in ('gsm8k','ultrachat')]
    rng=np.random.default_rng(202610041);draws=np.concatenate([rng.choice(g,size=(10000,len(g)),replace=True) for g in groups],axis=1);metrics={}
    for name in v:
        k=name.rsplit('k',1)[1];values=np.array([per[p][name] for p in prompts],dtype=float);baseline=np.array([per[p]['original_k'+k] for p in prompts],dtype=float)
        saving=100*(1-baseline/values)
        metrics[name]=dict(coverage_gain_over_original_chain=float(np.mean([100*(per[p][name]/per[p]['original_chain']-1) for p in prompts])),
            modeled_zero_added_cost_savings_percent=float(saving.mean()),interval95=np.quantile(saving[draws].mean(-1),[.025,.975]).tolist(),
            domain_savings={d:float(saving[g].mean()) for d,g in zip(('gsm8k','ultrachat'),groups)},counts=dict(counts[name]))
    result=dict(status='PASS',metrics=metrics,per_prompt=dict(per),rows=details,checks=dict(exhaustive_synthetic=nchecks,independent_tree_rebuilds=checks,original_endpoints=6144),
        script_sha256=sha(Path(__file__)),input_sha256=sha(p),
        limitations=['Exploratory fixed-context policy, no training or runtime result','Teacher scores used only when branch first token matches its conditioning input',
          'Positions remain factorized given first token; no claim full sequential conditionals are known','No novelty claim; preserving conditioning is a correctness constraint, not a new learned method'])
    (OUT/'branch-consistent-teacher.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='PASS',checks=result['checks'],metrics=metrics),indent=2))
if __name__=='__main__':main()
