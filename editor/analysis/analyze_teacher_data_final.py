"""Report the frozen six-arm fresh comparison from independently audited tokens."""
from pathlib import Path
import hashlib,json,random,statistics,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
BASE=ROOT/'notes/second-pass-profiling-2026-10-02/teacher-unseen-comparison'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize(audit,specs,*,draws=10000):
    from teacher_data_comparison import ARMS,REGIMES,paired_contrasts
    ids=[p['id'] for p in specs];rows={r['id']:r for r in audit['per_prompt']}
    if audit['status']!='complete' or audit['prompts']!=64 or len(ids)!=len(set(ids)) or len(rows)!=64 or set(ids)!=set(rows):
        raise ValueError('Incomplete or different final panel')
    evaluations={(regime,arm):{pid:rows[pid]['evaluations'][regime+'--'+arm] for pid in ids}
                 for regime in REGIMES for arm in ARMS}
    comparisons=paired_contrasts(evaluations,specs,draws=draws)
    domains={d:[p['id'] for p in specs if p['domain']==d] for d in ('gsm8k','ultrachat')}
    if [len(v) for v in domains.values()]!=[32,32]:raise ValueError('Domain panel changed')
    rng=random.Random(20261003);boot={key:[] for key in evaluations}
    for _ in range(draws):
        sample=[rng.choice(group) for group in domains.values() for _ in group]
        for key,metrics in evaluations.items():boot[key].append(statistics.mean(metrics[p]['gain_pp'] for p in sample))
    arms={}
    for key,metrics in evaluations.items():
        samples=sorted(boot[key]);name='--'.join(key)
        arms[name]=dict(equal_prompt_coverage_gain_percent=statistics.mean(metrics[p]['gain_pp'] for p in ids),
            interval95_percent=[samples[int(.025*draws)],samples[min(draws-1,int(.975*draws))]],
            by_domain_percent={d:statistics.mean(metrics[p]['gain_pp'] for p in pp) for d,pp in domains.items()},
            first_error_repairs=sum(metrics[p]['repairs'] for p in ids),damaged_edited_prefixes=sum(metrics[p]['damage'] for p in ids),
            damaged_fully_correct_blocks=sum(rows[p]['damaged_fully_correct'][name] for p in ids))
    contrasts=comparisons['contrasts']
    # These describe observed directions, not a new post-hoc significance gate.
    directions={name:row['mean_pp']>0 for name,row in contrasts.items()}
    return dict(status='complete',per_arm=arms,paired_comparisons=comparisons,positive_contrast_point_estimates=directions,
        prompts=64,contexts=audit['contexts'],capped_native_responses=sum(bool(rows[p]['capped']) for p in ids),
        training_seeds=1,checkpoint_selection='All six epoch3 checkpoints fixed before fresh outcomes',
        inference_cost='Not measured on this panel; do not turn coverage percentages into claimed latency gains',
        scope='Fresh fixed-context union coverage. Original and edited proposals both retained. Not actual generation, novelty, or seed-robustness evidence.')


def main():
    attempt=BASE/'final-attempt1';state=read(attempt/'execution.json')
    if state.get('status')!='compute_closed_archive_retained' or state.get('scientific_status')!='fresh_evaluation_complete':
        raise ValueError('Final run incomplete; preserve partial results without full-panel conclusions')
    if not state['shutdown']['stopped'] or state['shutdown']['active_tasks'] or not state['scratch_absent']:
        raise ValueError('Resolve final compute/recovery first')
    metadata=attempt/'archive-metadata';manifest=read(attempt/'archive-metadata-recovery.json')
    name='final-audit.json';path=metadata/name;entry=manifest[name]
    if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256'] or entry['sha256']!=entry['local_sha256']:
        raise ValueError('Independent final audit changed')
    plan=read(BASE/'final-plan.json');bundle=read(BASE/'final-bundle.json')
    if sha(BASE/'final-plan.json')!=bundle['plan_sha256'] or sha(BASE/'comparison-plan.json')!=plan['study_sha256']:
        raise ValueError('Final study contract changed')
    study=read(BASE/'comparison-plan.json');specs=[p for p in study['panels'] if p['split']=='fresh_final']
    result=summarize(read(path),specs)
    result.update(audit_sha256=sha(path),final_plan_sha256=sha(BASE/'final-plan.json'))
    (attempt/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
