"""Descriptive in-sample versus development coverage; no selection or fitting."""
import json,statistics
from pathlib import Path
import torch
base=Path('notes/second-pass-profiling-2026-10-01')
source=base/'compact-training-completion/attempt2/recovered'
dest=base/'compact-training-fit-audit/attempt1'
folder=dest/'recovered'
features={}
for p in (source/'features').glob('*.pt'):
    for r in torch.load(p,map_location='cpu',weights_only=True):
        k=(r['prompt_id'],r['anchor_index'])
        assert k not in features
        features[k]=r
assert len(features)==14336
def progress(tokens,labels,valid):
    n=1
    for i in range(1,len(tokens)):
        if not valid[i] or tokens[i]!=labels[i]:break
        n+=1
    return n
output={}
for split,count in (('development',2048),('train',12288)):
    output[split]={}
    for arm in ('target_only','eligible_ce','teacher_kl'):
        rows=json.loads((folder/f'{arm}-{split}-evaluation.json').read_text())
        keys={(r['prompt'],r['anchor_index']) for r in rows}
        assert len(rows)==len(keys)==count
        groups={};original_total=student_extra=teacher_extra=overlap=teacher_repair=student_teacher_repair=damage=repair=0
        for item in rows:
            k=(item['prompt'],item['anchor_index']);r=features[k];assert r['split']==split
            labels=r['labels'][0].tolist();valid=r['valid'][0].tolist()
            original=[labels[0]]+r['original'][0].tolist()
            teacher=original[:2]+r['teacher_winners'][0].tolist()
            a=progress(original,labels,valid);t=max(a,progress(teacher,labels,valid))
            result=item['result'];s=result['union_progress']
            assert result['original_tokens']==original and result['original_progress']==a
            assert result['edited_progress']==progress(result['edited_tokens'],labels,valid)
            assert s==max(a,result['edited_progress'])
            group=groups.setdefault(k[0],[0,0,0,0]);group[0]+=a;group[1]+=s;group[2]+=t;group[3]+=1
            original_total+=a;student_extra+=s-a;teacher_extra+=t-a;overlap+=max(0,min(s,t)-a)
            teacher_repair+=t>a;student_teacher_repair+=(t>a and s>a)
            damage+=result['damaged'];repair+=s>a
        assert all(g[3]==32 for g in groups.values())
        gains={p:100*(s/a-1) for p,(a,s,t,n) in groups.items()}
        tgains={p:100*(t/a-1) for p,(a,s,t,n) in groups.items()}
        output[split][arm]=dict(contexts=count,prompts=len(groups),original_positions=original_total,
             equal_prompt_gain_percent=statistics.mean(gains.values()),extra_positions=student_extra,
             teacher_equal_prompt_gain_percent=statistics.mean(tgains.values()),teacher_extra_positions=teacher_extra,
             teacher_positions_also_covered=overlap,teacher_position_recovery_fraction=overlap/teacher_extra,
             repaired_blocks=repair,damaged_edited_blocks=damage,
             teacher_repair_blocks=teacher_repair,teacher_repair_blocks_helped=student_teacher_repair,
             teacher_block_recovery_fraction=student_teacher_repair/teacher_repair,
             domains={d:statistics.mean(v for p,v in gains.items() if p.startswith(d)) for d in ('gsm8k','ultrachat')},
             per_prompt_gain_percent=gains)
output['limitations']=['Training data are in-sample; development prompts were previously exposed.',
    'Different prompts and teacher opportunity confound causal attribution of train/development gaps.',
    'Epoch three only, one seed, no convergence or capacity-limit conclusion.',
    'Fixed-context original-plus-edited union coverage, not generation speed or standalone safety.',
    'No objective, checkpoint or model selection performed by this analysis.']
(dest/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({split:{arm:{k:v for k,v in result.items() if k!='per_prompt_gain_percent'} for arm,result in output[split].items()} for split in ('train','development')},indent=2))
