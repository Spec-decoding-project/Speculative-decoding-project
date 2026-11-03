"""Independent token recount using frozen audit functions, not training losses."""
from pathlib import Path
import argparse,json,statistics,hashlib
import torch
import analyze_context_smaller_update as audit_module
ROOT=Path(__file__).resolve().parents[1]
NOTES=ROOT/'notes/second-pass-profiling-2026-10-02/compact-retained-prefix'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))

def teacher_overlap(rows):
    predictions={(r['prompt'],r['anchor_index']):r['result'] for r in rows}
    bundle=read(NOTES/'local-bundle.json');plan=read(NOTES/'runner-plan.json')
    totals=dict(teacher_repaired_blocks=0,teacher_extra_positions=0,
                teacher_blocks_repaired_by_editor=0,teacher_extra_positions_captured=0)
    for spec in plan['prompts']:
        if spec['split']!='development':continue
        item=bundle['inputs'][f"features/{spec['id']}.pt"];p=ROOT/item['source']
        raw=p.read_bytes()
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
        for record in torch.load(p,map_location='cpu',weights_only=True):
            result=predictions[record['prompt_id'],record['anchor_index']]
            tokens=result['original_tokens'][:2]+record['teacher_winners'][0].tolist()
            labels=record['labels'][0].tolist();valid=record['valid'][0].tolist();t=1
            for i in range(1,16):
                if not valid[i] or tokens[i]!=labels[i]:break
                t+=1
            a=result['original_progress'];b=result['edited_progress']
            if t>a:
                totals['teacher_repaired_blocks']+=1
                totals['teacher_extra_positions']+=t-a
                totals['teacher_blocks_repaired_by_editor']+=b>a
                totals['teacher_extra_positions_captured']+=max(0,min(t,b)-a)
    assert totals['teacher_repaired_blocks']==353 and totals['teacher_extra_positions']==1136
    return totals


def main(check=False):
    audit_module.NOTES=NOTES
    refs=audit_module.references()
    if check:
        old=NOTES.parent/'context-access-training/attempt1/recovered/original-epoch-3-development.json'
        baseline=audit_module.audit(read(old),refs)
        contrast=audit_module.contrast(baseline,baseline)
        assert contrast['prompt_bootstrap_95pct']==[0,0]
        overlap=teacher_overlap(read(old))
        assert overlap['teacher_blocks_repaired_by_editor']==82 and overlap['teacher_extra_positions_captured']==184
        out=dict(status='pass',historical_contexts_recounted=2048,identity_contrast=True,teacher_overlap=overlap)
        (NOTES/'analysis-checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));return
    base=NOTES/'attempt1';source=base/'recovered'
    assert read(source/'terminal.json')['status']=='complete'
    summary=read(source/'summary.json');assert summary['updates']==2304 and summary['control_exact']
    updates=read(source/'updates.json');assert len(updates)==2304
    expected=[(e,s) for e in (1,2,3) for s in range(1,385)]
    for arm in ('uniform_ce','retained_prefix'):
        rows=[r for r in updates if r['arm']==arm]
        assert [(r['epoch'],r['step']) for r in rows]==expected
        assert all(r.get('optimizer_step',True) for r in rows)
    assert read(source/'initial-update-compatibility.json')==dict(status='exact',updates=4)
    results={}
    for arm in ('uniform_ce','retained_prefix'):
        results[arm]={}
        for epoch in (1,2,3):
            rows=read(source/f'{arm}-epoch-{epoch}-evaluation.json')
            results[arm][epoch]=audit_module.audit(rows,refs)
    control=results['uniform_ce'][3];candidate=results['retained_prefix'][3]
    historical=read(NOTES.parent/'context-access-training/attempt1/analysis.json')['development']['original']
    assert control==historical
    contrast=audit_module.contrast(candidate,control)
    positive=contrast['prompt_bootstrap_95pct'][0]>0 and all(v>0 for v in contrast['domain_mean_percentage_points'].values())
    gains=candidate['per_prompt_gain_fraction'].values()
    cost={str(h):100*(statistics.mean((1+h/100)/(1+g) for g in gains)-1) for h in (0,2,5,6.3)}
    overlaps={arm:teacher_overlap(read(source/f'{arm}-epoch-3-evaluation.json')) for arm in results}
    out=dict(status='complete_recounted',epochs=results,final_contrast=contrast,teacher_repair_overlap=overlaps,
        exploratory_screen_passed=positive,independently_recounted_contexts=12288,
        candidate_modeled_time_change_percent_by_assumed_round_overhead=cost,
        limitations=['64reusedprompts;oneinitialization','Completeobjectivepackage,notisolatedcomponentcausation',
        '12288 rows are repeated evaluations of 2048 contexts, not independent samples',
        'Costscenariosarenotmeasuredgeneration','No novelty or speedup established'])
    (base/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(uniform_gain=control['equal_prompt_union_gain_percent'],candidate_gain=candidate['equal_prompt_union_gain_percent'],contrast=contrast,screen=positive),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');main(p.parse_args().check)
