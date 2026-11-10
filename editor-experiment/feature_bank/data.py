"""CPU-only response validation and bounded, deterministic candidate selection."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def records(directory):
    directory = Path(directory)
    run = json.loads((directory / 'run.json').read_text())
    run_hash = digest(run)
    seen = set()
    for journal in sorted(directory.glob('shard-*.jsonl')):
        with journal.open() as stream:
            for line in stream:
                if not line.endswith('\n'):
                    raise ValueError(f'Partial journal tail: {journal}')
                row = json.loads(line)
                body = {key: value for key, value in row.items() if key != 'record_sha256'}
                if digest(body) != row['record_sha256'] or row['run_sha256'] != run_hash:
                    raise ValueError(f'Record checksum/run mismatch: {row.get("id")}')
                if row['id'] in seen:
                    raise ValueError(f'Duplicate prompt ID: {row["id"]}')
                seen.add(row['id'])
                yield row


def family(row):
    # Exact normalized message families only; semantic near-duplicates remain unproven.
    return digest([(m['role'], ' '.join(m['content'].split()).casefold()) for m in row['messages']])


def audit_sources(pilot, main):
    pilots = list(records(pilot))
    if len(pilots) > 384:
        raise ValueError('Pilot exceeds the ticket budget of 384 prompt IDs')
    by_id = {r['id']: r for r in pilots}
    counts = Counter()
    train_families, validation_families = set(), set()
    overlap = set()
    for row in records(main):
        split = row['dataset_split']
        if split not in ('train', 'validation'):
            raise ValueError(f'Unknown split: {split}')
        counts[split] += 1
        (train_families if split == 'train' else validation_families).add(family(row))
        if row['id'] in by_id:
            p = by_id[row['id']]
            if split != 'train' or p['dataset_split'] != 'train':
                raise ValueError('Pilot overlaps held-out data')
            for key in ('messages', 'prompt_token_ids', 'response_token_ids', 'finish_reason'):
                if row[key] != p[key]:
                    raise ValueError(f'Pilot/main disagreement: {row["id"]}, {key}')
            overlap.add(row['id'])
    if overlap != set(by_id):
        raise ValueError('Pilot IDs missing from main collection')
    if train_families & validation_families:
        raise ValueError('Normalized prompt family overlap across train/validation')
    return pilots, dict(counts=counts, pilot_count=len(pilots), pilot_overlap=len(overlap),
                        exact_normalized_family_overlap=0, semantic_near_duplicates='not established')


def candidates(rows, max_context=1024, anchors_per_prompt=32):
    if not 1 <= anchors_per_prompt <= 32 or not 1 <= max_context <= 1024:
        raise ValueError('Initial bank budgets: <=32 anchors/prompt and <=1024 context tokens')
    eligible, excluded = [], []
    for row in rows:
        reason = None
        if row['dataset_split'] != 'train': reason = 'not_training'
        elif row['status'] != 'generated': reason = 'not_generated'
        elif row['finish_reason'] != 'eos': reason = 'output_capped'
        elif len(row['response_token_ids']) < 16: reason = 'no_full_block'
        elif len(row['prompt_token_ids']) > max_context: reason = 'context_budget'
        if reason:
            excluded.append(dict(id=row['id'], reason=reason))
        else:
            eligible.append(row)
    # Alternate source groups so the bounded bank is not filled entirely by one domain.
    groups = {}
    for row in sorted(eligible, key=lambda r: (len(r['prompt_token_ids']), r['id'])):
        groups.setdefault(str(row['source']), []).append(row)
    ordered = []
    while any(groups.values()):
        for source in sorted(groups):
            if groups[source]: ordered.append(groups[source].pop(0))
    result = []
    for row in ordered:
        start = len(row['prompt_token_ids'])
        end = min(start + len(row['response_token_ids']) - 16, max_context)
        # Nonoverlapping full windows, bounded independently per response.
        anchors = list(range(start, end + 1, 16))[:anchors_per_prompt]
        result.append((row, anchors))
    # Round-robin over prompts before visiting a second context from any prompt.
    schedule = [(row, anchors[i]) for i in range(anchors_per_prompt)
                for row, anchors in result if i < len(anchors)]
    return schedule, excluded


def block_inputs(row, anchor, mask_token_id, block_size=16):
    ids = row['prompt_token_ids'] + row['response_token_ids']
    if anchor < len(row['prompt_token_ids']) or anchor + block_size > len(ids):
        raise ValueError('Anchor outside complete response windows')
    # Do not expose any future target token to the native draft call.
    return ids[:anchor], [ids[anchor]] + [mask_token_id] * (block_size - 1), ids[anchor:anchor + block_size]
