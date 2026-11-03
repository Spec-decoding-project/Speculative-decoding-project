"""Independent policy reconstruction and paired response analysis."""
from pathlib import Path
from collections import defaultdict
import json,math,hashlib,statistics,random,argparse
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'notes/second-pass-profiling-2026-10-03';OUT=BASE/'teacher-edit-score/attempt1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def length(tokens,labels,valid):
    return next((i for i,(a,b,v) in enumerate(zip(tokens,labels,valid)) if not v or a!=b),len(tokens))

def independent_choice(original,ids,logs):
    # Explicit sum of path probabilities at every added endpoint; independent
    # from the runtime log-domain factored-tail calculation.
    values=[]
    for j in range(2,len(original)):
        for rank in (1,2,3):
            value=sum(math.exp(sum(logs[i-1][rank if i==j else 0] for i in range(1,end+1))) for end in range(j,len(original)))
            values.append((value,-j,-rank))
    _,neg_j,neg_rank=max(values);j=-neg_j;rank=-neg_rank
    proposal=original.copy();proposal[j]=ids[j-1][rank]
    return j,rank+1,proposal

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,default=1);args=parser.parse_args()
    OUT=BASE/'teacher-edit-score'/f'attempt{args.attempt}'
    receipt=read(OUT/'recovery.json')
    for n in ('rows.json','summary.json','terminal.json'):assert sha(OUT/'recovered'/n)==receipt[n]['sha256']
    assert read(OUT/'recovered/terminal.json')['status']=='complete'
    rows=read(OUT/'recovered/rows.json');assert len(rows)==2048
    bundle=read(OUT.parent/('frozen/local-bundle.json' if args.attempt==1 else f'frozen-attempt{args.attempt}/local-bundle-attempt{args.attempt}.json'));s=bundle['inputs']['original-score-reference.json'];p=ROOT/s['local_path'];assert sha(p)==s['sha256']
    references={(r['prompt'],r['anchor_index']):r for r in read(p)}
    keys=[(r['prompt'],r['anchor_index']) for r in rows]
    assert len(set(keys))==2048 and set(keys)==set(references)
    assert read(OUT/'recovered/summary.json')['contexts']==2048
    assert read(OUT/'recovered/summary.json')['repeat_exact']==1
    per=defaultdict(lambda:defaultdict(int));counts=defaultdict(int)
    for r in rows:
        q=references[r['prompt'],r['anchor_index']];p=per[r['prompt']]
        assert q['original_progress']==r['original_progress']
        p['original']+=r['original_progress'];p['teacher_whole_union']+=r['teacher_union_progress'];p['tree']+=r['fixed_tree_progress']
        for name,logs in [('q0_policy',q['log_probabilities']),('teacher_policy',r['teacher_candidate_logs'])]:
            j,rank,proposal=independent_choice(q['original_tokens'],q['candidate_ids'],logs)
            v=r[name];assert j==v['position'] and rank==v['rank'] and proposal==v['proposal']
            n=max(r['original_progress'],length(proposal,q['labels'],q['valid']));assert n==v['union_progress'];p[name]+=n
            counts[name+'_repaired_contexts']+=n>r['original_progress']
        a,b=r['q0_policy']['union_progress'],r['teacher_policy']['union_progress']
        counts['teacher_better_contexts']+=b>a;counts['teacher_worse_contexts']+=b<a;counts['teacher_tied_contexts']+=b==a
        counts['teacher_positions_gained']+=max(0,b-a);counts['teacher_positions_lost']+=max(0,a-b)
        counts['same_action']+=(r['q0_policy']['position'],r['q0_policy']['token'])==(r['teacher_policy']['position'],r['teacher_policy']['token'])
    prompts=sorted(per);assert len(prompts)==64
    rng=random.Random(3412026);domains=[[p for p in prompts if p.startswith(d+'-')] for d in ('gsm8k','ultrachat')]
    draws=[rng.choices(domains[0],k=32)+rng.choices(domains[1],k=32) for _ in range(10000)]
    def ci(values):
        ordered=sorted(sum(values[p] for p in draw)/64 for draw in draws);return [ordered[249],ordered[9749]]
    summary={}
    for name in ('q0_policy','teacher_policy','teacher_whole_union','tree'):
        gains={p:100*(per[p][name]/per[p]['original']-1) for p in prompts}
        differences={p:100*(per[p][name]-per[p]['q0_policy'])/per[p]['original'] for p in prompts}
        summary[name]=dict(coverage_gain_percent=statistics.mean(gains.values()),descriptive_interval95=ci(gains),
            difference_from_q0_policy_pp=statistics.mean(differences.values()),difference_interval95=ci(differences),
            zero_added_cost_modeled_saving_percent=100*statistics.mean(1-per[p]['original']/per[p][name] for p in prompts),
            domains={d:statistics.mean(gains[p] for p in prompts if p.startswith(d+'-')) for d in ('gsm8k','ultrachat')})
    old=read(BASE/'sparse-edit-audit/results.json')['conditions']['expected_progress']['gain_percent']
    assert abs(summary['q0_policy']['coverage_gain_percent']-old)<1e-10
    result=dict(status='PASS',contexts=2048,responses=64,counts=dict(counts),conditions=summary,per_prompt=dict(per),
        source_hashes=dict(rows=sha(OUT/'recovered/rows.json'),analyzer=sha(Path(__file__))),
        limitations=['Reused development panel, no learned student or generation',
            'Product-marginal utility proxy; teacher score quality is tested only through resulting choices',
            'Same candidate family and one-edit constraint; no general conclusion about teacher usefulness'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='per_prompt'},indent=2))
if __name__=='__main__':main()
