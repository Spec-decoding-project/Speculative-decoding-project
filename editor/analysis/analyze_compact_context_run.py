"""Frozen prompt-level analysis for a completed context-access comparison."""
from pathlib import Path
import json,statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NOTES=ROOT/'notes/second-pass-profiling-2026-10-02/context-access-training'
SOURCE=NOTES/'attempt1/recovered'
ARMS=('original','aligned_context','mismatched_context')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def summarize(rows,expected_contexts):
    assert len(rows)==expected_contexts
    groups={};keys=set();repairs=damage=full_damage=extra=0
    for row in rows:
        key=(row['prompt'],row['anchor_index']);assert key not in keys;keys.add(key)
        r=row['result'];a=r['original_progress'];b=r['edited_progress'];u=r['union_progress']
        assert u==max(a,b) and 1<=a<=16 and 1<=b<=16
        assert r['repaired']==(b>a) and r['damaged']==(b<a)
        g=groups.setdefault(row['prompt'],[0,0,0,0]);g[0]+=a;g[1]+=u;g[2]+=b;g[3]+=1
        repairs+=b>a;damage+=b<a;full_damage+=r['damaged_fully_correct'];extra+=u-a
    assert all(g[3]==32 for g in groups.values())
    gains={p:g[1]/g[0]-1 for p,g in groups.items()}
    edited={p:g[2]/g[0]-1 for p,g in groups.items()}
    inverse=statistics.mean(1/(1+x) for x in gains.values())
    return dict(contexts=len(rows),prompts=len(groups),extra_positions=extra,repaired_blocks=repairs,
        damaged_edited_blocks=damage,damaged_fully_correct_blocks=full_damage,
        equal_prompt_union_gain_percent=100*statistics.mean(gains.values()),
        equal_prompt_edited_gain_percent=100*statistics.mean(edited.values()),
        maximum_common_round_overhead_for_modeled_5pct_saving_percent=100*(.95/inverse-1),
        per_prompt_gain_fraction=gains,original_per_prompt={p:g[0] for p,g in groups.items()},
        domains={d:100*statistics.mean(v for p,v in gains.items() if p.startswith(d)) for d in ('gsm8k','ultrachat')})

def main():
    terminal=read(SOURCE/'terminal.json');summary=read(SOURCE/'summary.json')
    assert terminal['status']==summary['status']=='complete' and summary['updates']==3456
    result={'development':{},'training':{}}
    curves={}
    for arm in ARMS:
        curves[arm]={}
        for epoch in (1,2,3):
            value=summarize(read(SOURCE/f'{arm}-epoch-{epoch}-development.json'),2048)
            curves[arm][epoch]=value['equal_prompt_union_gain_percent']
            if epoch==3:result['development'][arm]=value
        result['training'][arm]=summarize(read(SOURCE/f'{arm}-epoch-3-train.json'),12288)
    for split in result:
        for arm in ARMS:
            assert result[split][arm]['original_per_prompt']==result[split]['original']['original_per_prompt']
    prompts=sorted(result['development']['original']['per_prompt_gain_fraction'])
    domains=[[i for i,p in enumerate(prompts) if p.startswith(d)] for d in ('gsm8k','ultrachat')]
    assert [len(d) for d in domains]==[32,32]
    rng=np.random.default_rng(1902)
    draws=np.concatenate([rng.choice(d,size=(20000,len(d)),replace=True) for d in domains],axis=1)
    contrasts={}
    for control in ('original','mismatched_context'):
        delta=np.array([100*(result['development']['aligned_context']['per_prompt_gain_fraction'][p]-
            result['development'][control]['per_prompt_gain_fraction'][p]) for p in prompts])
        interval=np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist()
        contrasts[control]=dict(mean_percentage_points=float(delta.mean()),prompt_bootstrap_95pct=interval,
            domain_mean_percentage_points={d:float(delta[indices].mean()) for d,indices in zip(('gsm8k','ultrachat'),domains)})
    scientific_positive=all(v['prompt_bootstrap_95pct'][0]>0 for v in contrasts.values()) and all(v>0 for v in contrasts['original']['domain_mean_percentage_points'].values())
    result.update(curves=curves,contrasts=contrasts,context_package_positive_screen=scientific_positive,
        generation_speedup_established=False,seed_count=1,
        limitations=['Previously exposed development prompts','One training seed',
          'Mismatched control resamples rows; must also beat original editor',
          'Context access, normalization, architecture and capacity are a package',
          'Retained original-branch union progress is not cost-free verification or generation speed'])
    (NOTES/'attempt1/analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('development','training')},indent=2))

if __name__=='__main__':main()
