"""Matched, authenticated prompt populations for whole-draft fine-tuning."""
import json
import sqlite3

from copy_action_fit.data import SOURCES, digest
from refinement_research.gate_data import select as gate_select


def choose_training(candidates, calibration, *, per_source=4096, preservation=512):
    blocked_prompts = {r['prompt_id'] for r in calibration}
    blocked_families = {r['family_id'] for r in calibration}
    chosen = []
    for kind, count in [('preservation', preservation), ('repairable', per_source-preservation)]:
        for source in SOURCES:
            group = [r for r in candidates if r['kind'] == kind and r['source'] == source]
            group.sort(key=lambda r: digest(['whole-draft-20261127', r['prompt_id'], r['anchor']]))
            kept = 0
            for row in group:
                if row['prompt_id'] in blocked_prompts or row['family_id'] in blocked_families:
                    continue
                chosen.append(row)
                blocked_prompts.add(row['prompt_id']); blocked_families.add(row['family_id'])
                kept += 1
                if kept == count:
                    break
            if kept != count:
                raise ValueError(f'Insufficient distinct {kind}/{source}: {kept}/{count}')
    return chosen


def populations(path):
    # gate_select authenticates the entire bank SHA and checks SQLite integrity.
    gate_fit, calibration, development, metadata = gate_select(path)
    connection = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        candidates = []
        for identity, payload in connection.execute("SELECT id,payload FROM examples WHERE split='train'"):
            row = json.loads(payload)
            candidates.append(dict(bank_id=identity, **{k: row[k] for k in
                ('prompt_id', 'family_id', 'source', 'kind', 'anchor')}))
        selected = choose_training(candidates, calibration)
        training = [json.loads(connection.execute('SELECT payload FROM examples WHERE id=?',
                    (r['bank_id'],)).fetchone()[0]) for r in selected]
    finally:
        connection.close()
    for row in training:
        if (row['split'] != 'train' or not row['target_replay_verified']
                or len(row['draft']) != 16 or len(row['labels']) != 16
                or row['draft'][:2] != row['labels'][:2]
                or row['anchor'] != row['context_start']+len(row['prefix'])):
            raise ValueError('Invalid whole-draft training example')
    for field in ('prompt_id', 'family_id'):
        fit = {r[field] for r in training}
        if fit & {r[field] for r in calibration+development}:
            raise ValueError('Fine-tuning/calibration/development overlap')
    metadata.update(generator_fit_blocks=len(training), generator_fit_sha256=digest(training),
                    calibration_sha256=digest(calibration), development_sha256=digest(development))
    return training, gate_fit, calibration, development, metadata
