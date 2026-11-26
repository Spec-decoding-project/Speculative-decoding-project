"""Deterministic selection from an authenticated native SQLite bank."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3


SOURCES = ('HuggingFaceH4/ultrachat_200k', 'openai/gsm8k')
BANK_SHA256 = 'c7f92fae5609137510a198297a31b5178c70c413f263f89f04560b98aa6bbb3d'
CHECKPOINT_SHA256 = 'c79b8c59c1e2cf73a7adf7aa9738b1b273779d330fe32f8d2c4a15fd56b7e8d2'
SHARED_WEIGHTS_SHA256 = '8babb2bfe3f4c11ef283312e22ccd2e20878438c96957302ff2c462748bbd382'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def choose(rows, per_group=16, loader=None):
    groups = {(kind, source): [] for kind in ('preservation', 'repairable') for source in SOURCES}
    for row in rows:
        if row['split'] != 'train':
            raise ValueError('Only training rows may enter the pilot selection')
        if row['kind'] in ('preservation', 'repairable') and row['source'] in SOURCES:
            groups[row['kind'], row['source']].append(row)
    selected, prompts, families = [], set(), set()
    # Preserve scarce natural controls first; a prompt/family is used only once.
    for (kind, source), candidates in groups.items():
        candidates.sort(key=lambda row: digest(['copy-action-pilot-v1', kind, source,
                                                row['prompt_id'], row['anchor']]))
        count = 0
        for row in candidates:
            if row['prompt_id'] in prompts or row['family_id'] in families:
                continue
            if loader is not None:
                row = loader(row)
            if (row['target_replay_verified'] is not True or len(row['draft']) != 16
                    or len(row['labels']) != 16 or row['draft'][:2] != row['labels'][:2]
                    or row['anchor'] != row['context_start'] + len(row['prefix'])):
                raise ValueError('Unverified, incomplete, misaligned or unreachable native block')
            if (row['draft'] == row['labels']) != (kind == 'preservation'):
                raise ValueError('Native block kind differs from actual tokens')
            selected.append(row)
            prompts.add(row['prompt_id']); families.add(row['family_id'])
            count += 1
            if count == per_group:
                break
        if count != per_group:
            raise ValueError(f'Insufficient distinct native prompts for {kind}/{source}')
    return selected


def read_bank(path, *, per_group=16):
    if sha(path) != BANK_SHA256:
        raise ValueError('Full-corpus bank fingerprint differs from verified input')
    connection = sqlite3.connect(f'file:{Path(path)}?mode=ro', uri=True)
    try:
        if connection.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Bank integrity check failed')
        def candidates():
            # Sort only compact identities; do not retain 199k token prefixes.
            for identity, payload in connection.execute(
                    "SELECT id,payload FROM examples WHERE split='train' ORDER BY id"):
                row = json.loads(payload)
                yield dict(bank_id=identity, **{key:row[key] for key in
                    ('split', 'kind', 'source', 'prompt_id', 'family_id', 'anchor')})

        def load(candidate):
            return json.loads(connection.execute('SELECT payload FROM examples WHERE id=?',
                                                (candidate['bank_id'],)).fetchone()[0])

        training = choose(candidates(), per_group=per_group, loader=load)
        development = [json.loads(record[0]) for record in connection.execute(
            "SELECT payload FROM examples WHERE split='development' ORDER BY id")]
    finally:
        connection.close()
    if len(development) != 2048 or len({r['prompt_id'] for r in development}) != 512:
        raise ValueError('Development population differs from the declared protocol')
    for field in ('prompt_id', 'family_id'):
        if {r[field] for r in training} & {r[field] for r in development}:
            raise ValueError('Training/development leakage')
    metadata = dict(schema=1, bank_sha256=BANK_SHA256, selection_sha256=digest(training),
        blocks=len(training), prompts=len({r['prompt_id'] for r in training}),
        families=len({r['family_id'] for r in training}),
        counts=dict(Counter(r['kind'] + '/' + r['source'] for r in training)),
        development_blocks=len(development), development_prompts=512,
        final_test_loaded=False)
    return training, development, metadata
