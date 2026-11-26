"""More independent training prompts, preserving the original gate calibration."""
from collections import Counter
import json
from pathlib import Path
import sqlite3

from copy_action_fit.data import BANK_SHA256, SOURCES, digest, sha


def choose_broader(candidates, calibration, loader=None, *, preservation=600, repairable=4096):
    groups={(kind,source):[] for kind in ('preservation','repairable') for source in SOURCES}
    forbidden_prompts={r['prompt_id'] for r in calibration}
    forbidden_families={r['family_id'] for r in calibration}
    for row in candidates:
        if row['split']!='train':raise ValueError('Only training-origin candidates allowed')
        key=row['kind'],row['source']
        if key in groups:groups[key].append(row)
    selected=[];prompts=set(forbidden_prompts);families=set(forbidden_families)
    for (kind,source),rows in groups.items():
        rows.sort(key=lambda r:digest(['broader-gates-v1',kind,source,r['prompt_id'],r['anchor']]))
        needed=preservation if kind=='preservation' else repairable
        count=0
        for candidate in rows:
            if candidate['prompt_id'] in prompts or candidate['family_id'] in families:continue
            row=loader(candidate) if loader else candidate
            if (row['target_replay_verified'] is not True or len(row['draft'])!=16 or len(row['labels'])!=16
                    or row['draft'][:2]!=row['labels'][:2]
                    or row['anchor']!=row['context_start']+len(row['prefix'])
                    or (row['draft']==row['labels'])!=(kind=='preservation')):
                raise ValueError('Invalid or unreachable native training block')
            selected.append(row);prompts.add(row['prompt_id']);families.add(row['family_id']);count+=1
            if count==needed:break
        if count!=needed:raise ValueError(f'Insufficient independent prompts for {kind}/{source}: {count}/{needed}')
    return selected


def select_broader(path,parent):
    if sha(path)!=BANK_SHA256:raise ValueError('Bank fingerprint changed')
    original=json.loads((parent/'selection.json').read_text())
    calibration=json.loads((parent/'calibration-examples.json').read_text())
    if digest(calibration)!=original['calibration_sha256']:raise ValueError('Calibration changed')
    if not json.loads((parent/'complete.json').read_text())['passed_checks']:raise ValueError('Parent run incomplete')
    connection=sqlite3.connect(f'file:{Path(path)}?mode=ro',uri=True)
    try:
        if connection.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Bank integrity failed')
        calibration_lookup={(r['prompt_id'],r['anchor']):r for r in calibration}
        checked=set()
        def candidates():
            for identity,payload in connection.execute("SELECT id,payload FROM examples WHERE split='train' ORDER BY id"):
                row=json.loads(payload)
                key=row['prompt_id'],row['anchor']
                if key in calibration_lookup:
                    if row!=calibration_lookup[key]:raise ValueError('Calibration differs from immutable bank')
                    checked.add(key)
                yield dict(bank_id=identity,**{k:row[k] for k in ('split','kind','source','prompt_id','family_id','anchor')})
        def loader(row):return json.loads(connection.execute('SELECT payload FROM examples WHERE id=?',(row['bank_id'],)).fetchone()[0])
        fit=choose_broader(candidates(),calibration,loader)
        if len(checked)!=512:raise ValueError('Missing calibration rows in bank')
        development=[json.loads(r[0]) for r in connection.execute("SELECT payload FROM examples WHERE split='development' ORDER BY id")]
    finally:connection.close()
    if len(fit)!=9392 or len(calibration)!=512 or len(development)!=2048:raise ValueError('Unexpected split sizes')
    for key in ('prompt_id','family_id'):
        pools=[{r[key] for r in rows} for rows in (fit,calibration,development)]
        if len(pools[0])!=len(fit) or any(pools[i]&pools[j] for i,j in ((0,1),(0,2),(1,2))):
            raise ValueError('Prompt/family leakage')
    counts=dict(Counter(r['kind']+'/'+r['source'] for r in fit))
    weights={key:len(fit)/(4*count) for key,count in counts.items()}
    metadata=dict(schema='broader-gates-v1',bank_sha256=BANK_SHA256,fit_blocks=len(fit),
        fit_prompts=len(fit),calibration_blocks=512,development_blocks=2048,development_prompts=512,
        fit_sha256=digest(fit),calibration_sha256=digest(calibration),counts=counts,
        group_weights=weights,parent_complete_sha256=sha(parent/'complete.json'),final_test_loaded=False)
    return fit,calibration,development,metadata
