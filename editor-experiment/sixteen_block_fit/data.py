"""Select a fixed diagnostic set without target inference or response rewriting."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from autoregressive_editor.data import validate_example
from one_block_fit.collect import make_bank, progress, save, sha


# Immutable outputs of completed job 8453776, pinned before either new fit.
SOURCE_HASHES = {
    'inputs.json': 'e818906d596750ccc7c6107a2782faf681657d41c1f590465a292a60d759723c',
    'greedy.json': '725f7d3bb3fe0a1b9fe949e88aa3f8955e4cc3d18c3839adda65d51bad0ab7fe',
    'native.json': '00fd4474b9bd241b77fa65c498f1b3b1b5756c013723ed4f3176c0aadad6c42b',
    'comparison.json': '0db38bbdde246fb604fbed3e9f491cde6802db7e642e9f2c7126066e3d0e25e5',
    'bank.json': '5790fa095357606f2973f7ecc0c9656de5b6c58bc1289eecfe978bfd057635b8',
}


def all_rows(inputs, greedy, native):
    fixtures = inputs['fixtures']
    ownership = {}
    for f in fixtures:
        for key in (('id', f['id']), ('family', f['family_id'])):
            if ownership.setdefault(key, f['split']) != f['split']:
                raise ValueError('Cross-split prompt/family overlap')
    rows, excluded, seen = [], [], set()
    for block in native['blocks']:
        matches = [f for f in fixtures if block['prefix'][:len(f['prompt'])] == f['prompt']]
        if len(matches) != 1:
            raise ValueError('Ambiguous or unauthenticated captured prefix')
        f = matches[0]
        full = f['prompt'] + greedy['tokens'][f['id']]
        anchor = block['anchor']
        key = (f['id'], anchor)
        if key in seen:
            raise ValueError('Duplicate native block')
        seen.add(key)
        if (block['prefix'] != full[:anchor] or len(block['prefix']) != anchor
                or anchor < len(f['prompt'])
                or block['positions'] != list(range(anchor, anchor+16))):
            raise ValueError('Captured prefix/positions differ from reference')
        labels = full[anchor:anchor+16]
        if len(labels) != 16:
            excluded.append(dict(prompt_id=f['id'], anchor=anchor, reason='incomplete_reference_window'))
            continue
        draft = block['draft']
        if len(draft) != 16 or draft[0] != labels[0] or block['native_draft_count'] != 15:
            raise ValueError('Native anchor/proposal layout invalid')
        accepted = progress(draft, labels)
        if block['num_sampled'] != accepted+1 or block['num_rejected'] != 15-accepted:
            raise ValueError('Actual native acceptance disagrees with tokens')
        if anchor+32 > 1088:
            excluded.append(dict(prompt_id=f['id'], anchor=anchor, reason='editor_context_limit'))
            continue
        row = dict(prompt_id=f['id'], family_id=f['family_id'], split=f['split'],
                   source=f['source'], source_record_sha256=f['record_sha256'], anchor=anchor,
                   prefix=block['prefix'], draft=draft, labels=labels,
                   target_replay_verified=True, native_accepted=accepted,
                   kind='wrong_first' if accepted == 0 else 'preservation' if accepted == 15 else 'repairable')
        validate_example(row, 151936)
        rows.append(row)
    return rows, excluded


def select(inputs, rows, old_bank):
    training = []
    train_ids = [f['id'] for f in inputs['fixtures'] if f['split'] == 'train']
    if len(train_ids) != 4 or len(set(train_ids)) != 4:
        raise ValueError('Require exactly four existing training prompts')
    for name in train_ids:
        candidates = sorted((r for r in rows if r['prompt_id'] == name and r['kind'] == 'repairable'),
                            key=lambda r: r['anchor'])
        if len(candidates) < 4:
            raise ValueError('Fewer than four repairable blocks in a training prompt')
        # Four order statistics, spanning earliest to latest eligible native block.
        training.extend(candidates[(k*(len(candidates)-1))//3] for k in range(4))
    by_key = {(r['prompt_id'], r['anchor']): r for r in rows}
    development = [by_key[(r['prompt_id'], r['anchor'])] for r in old_bank['held_out']]
    if len(development) != 32 or len({r['prompt_id'] for r in development}) != 8:
        raise ValueError('Development set differs from the completed one-block probe')
    preservation = [r for r in rows if r['split'] == 'validation' and r['kind'] == 'preservation']
    for r in development + preservation:
        if r['split'] != 'validation' or any(r[k] == t[k] for t in training for k in ('prompt_id', 'family_id')):
            raise ValueError('Training/development overlap')
    if len({(r['prompt_id'], r['anchor']) for r in training}) != 16:
        raise ValueError('Training blocks are not distinct')
    unique_positions = {(r['prompt_id'], p) for r in training for p in range(r['anchor']+2, r['anchor']+16)}
    return dict(training=training, development=development, preservation=preservation,
                selection='Four order statistics per training prompt: floor(k*(n-1)/3), k=0..3; anchors ascending',
                training_editable_tokens=224, training_unique_editable_positions=len(unique_positions),
                training_prompt_count=4, development_prompt_count=8,
                preservation_training_blocks=sum(r['split']=='train' and r['kind']=='preservation' for r in rows),
                preservation_scope='All naturally correct validation captures; excluded from optimization and stopping')


def audit_emitted(inputs, greedy, native):
    if greedy['tokens'] != native['tokens']:
        raise ValueError('Greedy/native output mismatch')
    indices = []
    for trial in (greedy, native):
        index = {}
        for r in trial['trace']:
            if 'prefix_sha256' in r:
                index.setdefault(r['prefix_sha256'], []).append(r)
        indices.append(index)
    count = 0
    for f in inputs['fixtures']:
        prefix = list(f['prompt'])
        for token in greedy['tokens'][f['id']]:
            digest = hashlib.sha256(json.dumps(prefix).encode()).hexdigest()
            signatures = []
            for index in indices:
                found = index.get(digest, [])
                if not found or any(r['argmax'] != token or r['position'] != len(prefix)-1
                                    or r['vocabulary_size'] != 151936 for r in found):
                    raise ValueError('Raw logit alignment failed')
                signatures.append({r['logits_sha256'] for r in found})
            if signatures[0] != signatures[1] or len(signatures[0]) != 1:
                raise ValueError('Full-vocabulary raw logit hashes differ')
            prefix.append(token)
            count += 1
    if count != 768:
        raise ValueError('Unexpected reference size')
    return count


def build(collection):
    for name, expected in SOURCE_HASHES.items():
        if sha(collection/name) != expected:
            raise ValueError(f'Previously audited capture changed: {name}')
    raw = {name: json.loads((collection/name).read_text()) for name in SOURCE_HASHES}
    inputs, greedy, native, old = (raw[n] for n in ('inputs.json', 'greedy.json', 'native.json', 'bank.json'))
    fresh = make_bank(inputs, greedy, native)
    if any(old[k] != v for k, v in fresh.items()):
        raise ValueError('Original one-block bank no longer reproduces')
    rows, excluded = all_rows(inputs, greedy, native)
    dataset = select(inputs, rows, old)
    # These observed counts were fixed before training; no hidden substitution.
    if dataset['preservation_training_blocks'] != 0 or len(dataset['preservation']) != 1:
        raise ValueError('Preservation-control scope changed')
    dataset.update(schema='sixteen-block-native-token-set-v1', provenance=SOURCE_HASHES,
                   emitted_positions_verified=audit_emitted(inputs, greedy, native),
                   eligible_counts=dict(Counter(r['split']+'/'+r['kind'] for r in rows)),
                   all_complete_native_blocks=len(rows), exclusions=excluded,
                   editor_source_hashes=inputs['editor_source_hashes'])
    return dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Refuse to overwrite a dataset')
    dataset = build(args.collection)
    save(args.output, dataset)
    print(json.dumps({k: v for k, v in dataset.items() if k not in
                      ('training', 'development', 'preservation', 'exclusions')}), flush=True)


if __name__ == '__main__':
    main()
