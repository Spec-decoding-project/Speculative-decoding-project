"""Secondary frozen-score rule: keep q0 candidate-set mass, rerank within set."""
from pathlib import Path
from collections import defaultdict
import math,json,hashlib,random,statistics,sys
from analyze_student_score_policy import tree,prefix
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'notes/second-pass-profiling-2026-10-03';OUT=BASE/'student-score-policy/attempt1'
sys.path.insert(0,str(ROOT/'experiments'))
from sparse_edit_policy import choose

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lse(row):
    m=max(row);return m+math.log(sum(math.exp(x-m) for x in row))
def transform(original_logs,student_logs):
    assert len(original_logs)==len(student_logs)==15
    result=[original_logs[0]]
    for q,s in zip(original_logs[1:],student_logs[1:]):
        mass=min(0.,lse(q));normalizer=lse(s)
        result.append([mass+x-normalizer for x in s])
    return result

def check():
    rng=random.Random(3487)
    for _ in range(100):
        q=[rng.random()+.01 for _ in range(11)];q=[x/sum(q) for x in q]
        s=[rng.random()+.01 for _ in range(11)];s=[x/sum(s) for x in s]
        qlogs=[[math.log(x) for x in q[:4]]]*15;slogs=[[math.log(x) for x in s[:4]]]*15
        t=transform(qlogs,slogs)[1]
        expected=[sum(q[:4])*x/sum(s[:4]) for x in s[:4]]
        assert max(abs(math.exp(a)-b) for a,b in zip(t,expected))<1e-12
        assert abs(sum(math.exp(x) for x in t)+sum(q[4:])-1)<1e-12
        assert max(abs(a-b) for a,b in zip(transform(qlogs,qlogs)[1],qlogs[1]))<1e-12

def main():
    check()
    receipt=read(OUT/'recovery.json');p=OUT/'recovered/rows.json';assert sha(p)==receipt['rows.json']['sha256'];rows=read(p)
    assert len(rows)==2048
    b=read(OUT.parent/'frozen/local-bundle.json');spec=b['inputs']['original-score-reference.json'];p=ROOT/spec['local_path'];assert sha(p)==spec['sha256']
    refs={(r['prompt'],r['anchor_index']):r for r in read(p)}
    td=BASE/'teacher-edit-score/attempt2';p=td/'recovered/rows.json';assert sha(p)==read(td/'recovery.json')['rows.json']['sha256']
    teachers={(r['prompt'],r['anchor_index']):r for r in read(p)}
    primary=read(OUT/'analysis.json');assert primary['status']=='PASS'
    per=defaultdict(lambda:defaultdict(int));counts=defaultdict(int)
    for r in rows:
        key=r['prompt'],r['anchor_index'];q=refs[key];p=per[r['prompt']];p['original']+=q['original_progress']
        sources={'q0':q['log_probabilities'],'teacher':teachers[key]['teacher_candidate_logs']}
        sources.update({k:v['candidate_logs'] for k,v in r['students'].items()})
        for name,scores in sources.items():
            logs=transform(q['log_probabilities'],scores)
            assert logs[0]==q['log_probabilities'][0]
            assert all(abs(lse(a)-min(0.,lse(b)))<1e-12 for a,b in zip(logs[1:],q['log_probabilities'][1:]))
            action=choose(q['original_tokens'],q['candidate_ids'],logs)
            one=max(q['original_progress'],prefix(action['proposal'],q['labels'],q['valid']))
            nodes=tree(q['original_tokens'],q['candidate_ids'],logs)
            n=0
            for i in range(1,sum(q['valid'])+1):
                if tuple(q['labels'][:i]) not in nodes:break
                n=i
            p[name+'_one_edit']+=one;p[name+'_tree']+=n
            if name=='q0':
                assert one==teachers[key]['q0_policy']['union_progress']
                assert n==q['conditions']['k4_fixed_x1_cap30']['progress']
    prompts=sorted(per);rng=random.Random(202610032);groups=[[p for p in prompts if p.startswith(d+'-')] for d in ('gsm8k','ultrachat')]
    draws=[rng.choices(groups[0],k=32)+rng.choices(groups[1],k=32) for _ in range(10000)]
    def interval(values):
        ordered=sorted(sum(values[p] for p in draw)/64 for draw in draws);return [ordered[249],ordered[9749]]
    conditions={}
    for name in per[prompts[0]]:
        if name=='original':continue
        rule='tree' if name.endswith('_tree') else 'one_edit'
        gain={p:100*(per[p][name]/per[p]['original']-1) for p in prompts}
        delta={p:100*(per[p][name]-per[p]['q0_'+rule])/per[p]['original'] for p in prompts}
        full_delta={p:100*(per[p][name]-primary['per_prompt'][p][name])/per[p]['original'] for p in prompts}
        ratio=statistics.mean(per[p]['q0_'+rule]/per[p][name] for p in prompts)
        conditions[name]=dict(coverage_gain_percent=statistics.mean(gain.values()),difference_from_q0_pp=statistics.mean(delta.values()),difference_interval95=interval(delta),
            difference_from_full_normalization_pp=statistics.mean(full_delta.values()),full_difference_interval95=interval(full_delta),
            modeled_break_even_extra_round_cost_percent=100*(1/ratio-1),domains={d:statistics.mean(gain[p] for p in prompts if p.startswith(d+'-')) for d in ('gsm8k','ultrachat')})
    result=dict(status='PASS',contexts=2048,responses=64,conditions=conditions,per_prompt=dict(per),algebra_checks=100,
        limitations=['Secondary score rule recorded before inspecting student outcomes; no fitting or selected sweep',
        'Changes score distribution: original candidate-set mass is retained, capped at one for saved floating-point mass excess (maximum 1.877e-6)',
        'Candidate-only projection is an exact-arithmetic identity; native BF16 shape parity and total runtime are untested',
        'Reused development evidence; no generation speedup or novel method established'],source_hashes=dict(rows=sha(OUT/'recovered/rows.json'),script=sha(Path(__file__))))
    (OUT/'candidate-mass-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_prompt'},indent=2))
if __name__=='__main__':main()
