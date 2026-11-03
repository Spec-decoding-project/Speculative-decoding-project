"""Authenticate the two fixed fits and independently recount prefix/fidelity tokens."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
from analyze_teacher_data_fits import recount
PARENT = ROOT / 'notes/second-pass-profiling-2026-10-02/teacher-unseen-comparison'
BASE = PARENT / 'function-matching-assessment'
ARMS = ('gated_mixture', 'teacher_only')


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize(rows, references, domains):
    expected = set(references)
    actual = [(r['prompt'],r['anchor_index']) for r in rows]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Incomplete, extra or repeated predictions')
    groups = defaultdict(lambda:dict(original=0,union=0,teacher_union=0,repairs=0,damage=0,
        damaged_fully_correct=0,teacher_positions_captured=0,teacher_extra_positions=0,
        valid_rows=0,agreements=0,kl_sum=0.,teacher_native_disagreements=0))
    for r in rows:
        ref = references[r['prompt'],r['anchor_index']]
        y, valid, original = ref['labels'],ref['valid'],ref['original']
        v = r['result']
        if r['labels'] != y or r['valid'] != valid or v['original_tokens'] != original:
            raise ValueError('Labels or initial draft differ from authenticated reference')
        if v['edited_tokens'][:2] != original[:2] or r['teacher_tokens'][:2] != original[:2] or r['recorded_teacher_tokens'][:2] != original[:2]:
            raise ValueError('Fixed prefix changed')
        if ref.get('teacher') is not None and ref['teacher'] != r['recorded_teacher_tokens']:
            raise ValueError('Recorded teacher changed')
        a = recount(original,y,valid)
        b = recount(v['edited_tokens'],y,valid)
        t = recount(r['recorded_teacher_tokens'],y,valid)
        if (a,b,max(a,b)) != (v['original_progress'],v['edited_progress'],v['union_progress']):
            raise ValueError('Prefix accounting mismatch')
        if (b>a,b<a,a==sum(valid) and b<a) != (v['repaired'],v['damaged'],v['damaged_fully_correct']):
            raise ValueError('Repair/damage accounting mismatch')
        f = r['fidelity']
        agreement = sum(v['edited_tokens'][i]==r['teacher_tokens'][i] for i in range(2,16) if valid[i])
        n = sum(valid[2:])
        if n != f['valid_rows'] or agreement != f['agreements']:
            raise ValueError('Teacher agreement recount mismatch')
        g = groups[r['prompt']]
        g['original']+=a;g['union']+=max(a,b);g['teacher_union']+=max(a,t)
        g['repairs']+=b>a;g['damage']+=b<a;g['damaged_fully_correct']+=a==sum(valid) and b<a
        g['teacher_extra_positions']+=max(0,t-a)
        g['teacher_positions_captured']+=max(0,min(b,t)-a)
        g['valid_rows']+=n;g['agreements']+=agreement;g['kl_sum']+=f['kl_sum']
        g['teacher_native_disagreements']+=sum(r['teacher_tokens'][i]!=r['recorded_teacher_tokens'][i] for i in range(2,16) if valid[i])
    for pid,g in groups.items():
        g['gain_percent']=100*(g['union']/g['original']-1)
        g['agreement_percent']=100*g['agreements']/g['valid_rows']
        g['mean_kl']=g['kl_sum']/g['valid_rows']
    return dict(prompts=len(groups),contexts=len(rows),
        coverage_gain_percent=statistics.mean(g['gain_percent'] for g in groups.values()),
        teacher_agreement_percent=statistics.mean(g['agreement_percent'] for g in groups.values()),
        worker_reported_mean_kl=statistics.mean(g['mean_kl'] for g in groups.values()),
        by_domain_percent={d:statistics.mean(g['gain_percent'] for pid,g in groups.items() if domains[pid]==d) for d in ('gsm8k','ultrachat')},
        **{k:sum(g[k] for g in groups.values()) for k in ('repairs','damage','damaged_fully_correct','teacher_positions_captured','teacher_extra_positions','teacher_native_disagreements')},
        per_prompt=dict(groups))


def main():
    import torch
    torch.set_num_threads(2)
    attempt=BASE/'fit-attempt1';state=read(attempt/'execution.json')
    if state['status']!='closed' or state.get('result',{}).get('status')!='complete' or not state['shutdown']['stopped'] or state['shutdown']['active_tasks'] or not state['scratch_absent']:
        raise ValueError('Finish and recover the existing attempt first')
    source=attempt/'recovered';manifest=read(attempt/'recovery.json')
    for name,spec in manifest.items():
        p=(source/name).resolve()
        if not p.is_relative_to(source.resolve()) or p.stat().st_size!=spec['bytes'] or sha(p)!=spec['sha256'] or spec['local_sha256']!=spec['sha256']:
            raise ValueError('Recovery evidence mismatch')
    plan=read(BASE/'fit-plan.json');bundle=read(BASE/'fit-bundle.json')
    if sha(BASE/'fit-plan.json')!=bundle['plan_sha256']:raise ValueError('Frozen design changed')
    summary=read(source/'summary.json')
    if summary['status']!='complete' or summary['fits']!=2 or summary['updates']!=2304 or not summary['target_unchanged']:
        raise ValueError('Incomplete or changed study')
    frozen=read(source/'frozen-final-checkpoints.json')
    if set(frozen)!={'new_train--'+a for a in ARMS}:raise ValueError('Wrong final models')
    for name,spec in frozen.items():
        if sha(source/spec['path'])!=spec['sha256'] or spec['updates']!=1152:raise ValueError('Endpoint differs')
        updates=read(source/name/'updates.json')
        if [r['step'] for r in updates]!=list(range(1,1153)):raise ValueError('Update exposure differs')
    domains={p['id']:p['domain'] for p in plan['study']['panels']}
    refs={'development':{},'training-sample':{}}
    for p in plan['study']['panels']:
        if p['split']!='old_development':continue
        feature=plan['features'][p['id']]
        inp=bundle['inputs'][feature['relative_path']]
        path=ROOT/inp['source']
        if sha(path)!=inp['sha256']:raise ValueError('Development reference changed')
        records=torch.load(path,map_location='cpu',weights_only=True)
        for r in records:
            y=r['labels'][0].tolist()
            refs['development'][p['id'],r['anchor_index']]=dict(labels=y,valid=r['valid'][0].tolist(),original=[y[0]]+r['original'][0].tolist(),teacher=[y[0],int(r['x1'][0])]+r['teacher_winners'][0].tolist())
    prior=read(PARENT/'training-fit-diagnostic/plan.json')
    if set(plan['study']['training_probe_ids'])!={p['id'] for p in prior['prompts']}:raise ValueError('Training sample changed')
    previous=read(PARENT/'training-fit-diagnostic/attempt1/recovery.json')
    for p in prior['prompts']:
        ref=prior['reference_metadata'][p['id']];path=ROOT/ref['source']
        if sha(path)!=ref['sha256']:raise ValueError('Training native reference changed')
        pred_name='predictions/'+p['id']+'.json'
        pred_path=PARENT/'training-fit-diagnostic/attempt1/recovered'/pred_name
        if sha(pred_path)!=previous[pred_name]['sha256']:raise ValueError('Prior teacher record changed')
        oldrows=read(pred_path)['records']
        for r,oldrow in zip(read(path)['anchors'],oldrows):
            refs['training-sample'][p['id'],oldrow['anchor_index']]=dict(labels=r['labels'],valid=r['valid'],original=r['original'],teacher=oldrow['teacher_tokens'])
    results={}
    for arm in ARMS:
        name='new_train--'+arm
        for epoch in (1,2,3):
            results[arm+':'+str(epoch)+':development']=summarize(read(source/name/f'epoch-{epoch}-development.json'),refs['development'],domains)
        results[arm+':3:training-sample']=summarize(read(source/name/'epoch-3-training-sample.json'),refs['training-sample'],domains)
    mixture=results['gated_mixture:3:development'];pure=results['teacher_only:3:development']
    diffs={pid:pure['per_prompt'][pid]['gain_percent']-g['gain_percent'] for pid,g in mixture['per_prompt'].items()}
    ds=[[pid for pid in diffs if domains[pid]==d] for d in ('gsm8k','ultrachat')]
    assert list(map(len,ds))==[32,32]
    rng=random.Random(20261003)
    boot=sorted(statistics.mean(diffs[rng.choice(ids)] for ids in ds for _ in ids) for _ in range(10000))
    contrast=dict(mean_percentage_points=statistics.mean(diffs.values()),interval95=[boot[250],boot[9750]])
    decision=plan['study']['decision']['advancement']
    checks=dict(coverage=pure['coverage_gain_percent']>=decision['minimum_development_coverage_gain_percent'],
        paired_difference=boot[250]>0,domains=all(pure['by_domain_percent'][d]>mixture['by_domain_percent'][d] for d in ('gsm8k','ultrachat')),
        teacher_capture=pure['teacher_positions_captured']>mixture['teacher_positions_captured'])
    result=dict(status='complete',results=results,primary_contrast=contrast,advancement_checks=checks,screen_passed=all(checks.values()),
        scope='One seed, reused development; fixed-context coverage, not fresh confirmation or generation speed. KL is worker-computed; token agreements/progress independently recounted.')
    (attempt/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='complete',endpoints={k:{a:b for a,b in v.items() if a!='per_prompt'} for k,v in results.items() if ':3:' in k},primary_contrast=contrast,advancement_checks=checks),indent=2))


if __name__=='__main__':main()
