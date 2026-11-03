from pathlib import Path
import json,statistics,random,argparse

def analyze(folder):
 folder=Path(folder);results={};prompt_rows={};reference=None;all_rows={}
 rng=random.Random(701)
 def interval(values):
  n=len(values);draws=sorted(statistics.mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(10000))
  return [draws[249],draws[9749]]
 for arm in ('target_only','eligible_ce','teacher_kl'):
  results[arm]={}
  for epoch in range(1,4):
   path=folder/f'{arm}-epoch-{epoch}-evaluation.json'
   if not path.exists():continue
   rows=json.loads(path.read_text());groups={};all_rows[(arm,epoch)]=rows
   keys=[(r['prompt'],r['anchor_index']) for r in rows]
   if len(rows)!=2048 or len(set(keys))!=2048:raise ValueError('Incomplete or duplicated panel')
   signatures={(r['prompt'],r['anchor_index']):(r['result']['original_progress'],tuple(r['result']['original_tokens'])) for r in rows}
   if reference is None:reference=signatures
   elif reference!=signatures:raise ValueError('Mismatched original contexts across arms or epochs')
   for r in rows:groups.setdefault(r['prompt'],[]).append(r['result'])
   if len(groups)!=64 or any(len(rs)!=32 for rs in groups.values()):raise ValueError('Panel shape')
   progress={p:100*(sum(r['union_progress'] for r in rs)/sum(r['original_progress'] for r in rs)-1) for p,rs in groups.items()}
   prompt_rows[(arm,epoch)]=progress
   values=list(progress.values())
   results[arm][str(epoch)]={'equal_prompt_progress_percent':statistics.mean(values),'bootstrap_95_percent':interval(values),
    'extra_positions':sum(r['result']['union_progress']-r['result']['original_progress'] for r in rows),
    'repaired_blocks':sum(r['result']['repaired'] for r in rows),'damaged_edited_blocks':sum(r['result']['damaged'] for r in rows),
    'damaged_fully_correct_blocks':sum(r['result']['damaged_fully_correct'] for r in rows),
    'domains':{domain:statistics.mean(v for p,v in progress.items() if p.startswith(domain)) for domain in ('gsm8k','ultrachat')},
    'prompts_with_positive_union_gain':sum(v>0 for v in values),'per_prompt_percent':progress}
 comparisons={}
 if ('teacher_kl',3) in prompt_rows:
  for control in ('target_only','eligible_ce'):
   if (control,3) not in prompt_rows:continue
   a,b=prompt_rows[('teacher_kl',3)],prompt_rows[(control,3)]
   if a.keys()!=b.keys():raise ValueError('Unpaired prompts')
   delta=[a[p]-b[p] for p in sorted(a)]
   comparisons[control]={'mean_difference_percentage_points':statistics.mean(delta),'paired_bootstrap_95':interval(delta),'wins':sum(v>0 for v in delta),'losses':sum(v<0 for v in delta)}
 teacher=None
 feature_files=sorted((folder/'features').glob('*.pt'))
 if len(feature_files)==448:
  import torch
  totals={};seen=set();teacher_blocks={}
  for file in feature_files:
   for r in torch.load(file,map_location='cpu',weights_only=True):
    if r['split']!='development':continue
    key=(r['prompt_id'],r['anchor_index'])
    if key in seen:raise ValueError('Duplicate teacher context')
    seen.add(key)
    labels=r['labels'][0].tolist();valid=r['valid'][0].tolist()
    original=[labels[0]]+r['original'][0].tolist()
    revised=original[:2]+r['teacher_winners'][0].tolist()
    def progress(tokens):
     count=1
     for i in range(1,len(tokens)):
      if not valid[i] or tokens[i]!=labels[i]:break
      count+=1
     return count
    a,b=progress(original),progress(revised)
    if reference and reference[key]!=(a,tuple(original)):raise ValueError('Teacher and editor contexts differ')
    teacher_blocks[key]=(a,max(a,b))
    row=totals.setdefault(r['prompt_id'],[0,0]);row[0]+=a;row[1]+=max(a,b)
  if len(seen)!=2048:raise ValueError('Teacher panel incomplete')
  gains={p:100*(b/a-1) for p,(a,b) in totals.items()}
  teacher={'equal_prompt_progress_percent':statistics.mean(gains.values()),'extra_positions':sum(b-a for a,b in totals.values()),'per_prompt_percent':gains}
  for arm,epochs in results.items():
   for epoch,row in epochs.items():
    row['fraction_teacher_aggregate_extra_positions']=row['extra_positions']/teacher['extra_positions'] if teacher['extra_positions'] else None
    row['teacher_retention_note']='Aggregate opportunity ratio; not equal runtime benefit'
    recovered=0;missed=0;new=0;matched_positions=0
    for item in all_rows[(arm,int(epoch))]:
     a,t=teacher_blocks[(item['prompt'],item['anchor_index'])];student=item['result']['union_progress']
     if t>a:
      recovered+=student>a;missed+=student<=a;matched_positions+=max(0,min(student,t)-a)
     elif student>a:new+=1
    row['teacher_repair_overlap']={'teacher_repair_blocks_with_student_gain':recovered,'teacher_repair_blocks_without_student_gain':missed,'student_gain_blocks_without_teacher_gain':new,'teacher_extra_positions_also_covered':matched_positions}
 updates=json.loads((folder/'updates.json').read_text()) if (folder/'updates.json').exists() else []
 losses={}
 for arm in ('target_only','eligible_ce','teacher_kl'):
  losses[arm]={str(e):sum(r['ce']*r['valid_rows'] for r in updates if r['arm']==arm and r['epoch']==e)/sum(r['valid_rows'] for r in updates if r['arm']==arm and r['epoch']==e) for e in range(1,4) if any(r['arm']==arm and r['epoch']==e for r in updates)}
 return {'status':'complete' if all(len(r)==3 for r in results.values()) else 'partial','arms':results,'teacher':teacher,'epoch3_teacher_vs_controls':comparisons,'training_ce_by_epoch':losses,'limitations':['Exposed development prompts; exploratory intervals','Single training seed','Union preserves original coverage but incurs verification cost','No generation speedup or novelty established']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--output',required=True);a=p.parse_args()
 result=analyze(a.folder);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='arms'},indent=2))
