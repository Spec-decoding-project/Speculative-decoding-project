"""Authenticate saved feature banks and count actual training-loss eligibility."""
from pathlib import Path
import hashlib
import json
import statistics
import time


def main():
    import torch
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1]
    out=root/'notes/second-pass-profiling-2026-10-02/teacher-unseen-comparison'
    plan=json.loads((out/'comparison-plan.json').read_text(encoding='utf-8'))
    specs={p['id']:p for p in plan['panels']}; records=[]; start=time.time()
    inputs=dict(plan['old_features'])
    inputs.update({pid:r['features'] for pid,r in plan['reuse_pilot'].items()})
    for pid,item in inputs.items():
        p=root/item['source'];raw=p.read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('Feature identity changed')
        rows=torch.load(p,map_location='cpu',weights_only=True)
        valid=eligible=changed=useful=wrong=0
        for r in rows:
            if r['prompt_id']!=pid:raise ValueError('Mixed prompt records')
            v=r['valid'][:,2:];ok=r['teacher_winners'].eq(r['labels'][:,2:]);original=r['original'][:,1:]
            valid+=int(v.sum());eligible+=int((ok&v).sum())
            changed+=int((r['teacher_winners'].ne(original)&v).sum())
            useful+=int((ok&original.ne(r['labels'][:,2:])&v).sum());wrong+=int((~ok&v).sum())
        if valid!=eligible+wrong or useful>min(eligible,changed):raise ValueError('Count identity failed')
        records.append(dict(prompt=pid,split=specs[pid]['split'],domain=specs[pid]['domain'],valid=valid,
            eligible=eligible,changed=changed,teacher_correct_original_wrong=useful,teacher_wrong=wrong))
    summary={}
    for split in ('old_train','old_development','new_train'):
        rr=[r for r in records if r['split']==split]
        summary[split]=dict(prompts=len(rr),valid=sum(r['valid'] for r in rr),eligible=sum(r['eligible'] for r in rr),
            equal_prompt_eligible_pct=100*statistics.mean(r['eligible']/r['valid'] for r in rr),
            equal_prompt_correct_teacher_wrong_original_pct=100*statistics.mean(r['teacher_correct_original_wrong']/r['valid'] for r in rr),
            equal_prompt_teacher_changes_pct=100*statistics.mean(r['changed']/r['valid'] for r in rr))
    result=dict(summary=summary,per_prompt=records,seconds=time.time()-start,
        scope='All valid editable positions, not accepted-prefix repairs. New-training subset is the 16 collected pilot prompts. Descriptive unequal panels, not a causal explanation or generalization test.')
    (out/'teacher-guidance-support.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='per_prompt'},indent=2))


if __name__=='__main__':main()
