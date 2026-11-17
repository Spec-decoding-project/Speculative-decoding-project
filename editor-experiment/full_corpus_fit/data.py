"""Canonical native block audit, split assignment and bounded token windows."""
from collections import Counter, defaultdict
import hashlib
import json


KNOWN_DEVELOPMENT = {
    'ultrachat_200k:train_sft:145402', 'gsm8k:train:155', 'gsm8k:train:5868',
    'ultrachat_200k:train_sft:204138', 'gsm8k:train:1485', 'gsm8k:train:6867',
    'ultrachat_200k:train_sft:106750', 'ultrachat_200k:train_sft:94625',
}


def key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def split_prompts(prompts):
    grouped = defaultdict(list)
    for p in prompts:
        if p['split'] == 'validation':
            grouped[p['source']].append(p)
    development = set(KNOWN_DEVELOPMENT)
    for source, rows in grouped.items():
        if len(rows) != 512:
            raise ValueError('Expected 512 validation prompts per source')
        known = [r for r in rows if r['id'] in KNOWN_DEVELOPMENT]
        remaining = sorted((r for r in rows if r['id'] not in KNOWN_DEVELOPMENT), key=lambda r:key(r['id']))
        development.update(r['id'] for r in remaining[:256-len(known)])
    if len(development) != 512:
        raise ValueError('Invalid fixed development split')
    result = []
    for p in prompts:
        p = dict(p)
        p['source_split'] = p['split']
        if p['split'] == 'validation':
            p['split'] = 'development' if p['id'] in development else 'test'
        result.append(p)
    return result


def spread(rows, count):
    if len(rows) <= count:
        return list(rows)
    return [rows[k*(len(rows)-1)//(count-1)] for k in range(count)] if count > 1 else [rows[0]]


def acceptance(draft, labels):
    for i in range(1, 16):
        if draft[i] != labels[i]:
            return i-1
    return 15


def audit_chunk(prompts, generated, observed, *, hashes=False):
    """Authenticate source identity before constructing any windowed editor row."""
    by_external = {item['request_id']: p for p, item in zip(prompts, generated)}
    if len(generated) != len(prompts) or len(by_external) != len(prompts):
        raise ValueError('Missing or duplicate generated request')
    internals = {r['request_id'] for r in observed['trace']}
    mapping, reverse = {}, {}
    for internal in internals:
        possible = [external for external in by_external
                    if internal == external or internal.startswith(external+'-')]
        if len(possible) != 1:
            raise ValueError('Cannot map internal request ID to the returned request')
        external = possible[0]
        if external in reverse:
            raise ValueError('Multiple internal IDs for one request')
        mapping[internal], reverse[external] = by_external[external], internal
    trace = defaultdict(list)
    for r in observed['trace']:
        trace[(r['request_id'], r['prefix_sha256'])].append(r)
    full, positions_checked, aligned_hashes = {}, 0, {}
    for p, item in zip(prompts, generated):
        if item['prompt'] != p['prompt']:
            raise ValueError('Returned prompt differs from authenticated input')
        internal = reverse[item['request_id']]
        prefix = list(p['prompt'])
        for token in item['tokens']:
            digest = token_hash(prefix)
            found = trace.get((internal, digest), [])
            if not found or any(r['position'] != len(prefix)-1 or r['argmax'] != token
                                or r['vocabulary_size'] != 151936 for r in found):
                raise ValueError(f'Raw target argmax does not authenticate emitted token: {p["id"]}')
            if hashes:
                values = {r['logits_sha256'] for r in found}
                if len(values) != 1:
                    raise ValueError('Repeated canonical prefix changed logits')
                aligned_hashes[p['id']+'/'+str(len(prefix))] = next(iter(values))
            prefix.append(token)
            positions_checked += 1
        full[p['id']] = prefix
    rows, excluded, seen = [], [], set()
    for block in observed['blocks']:
        p = mapping[block['request_id']]
        anchor, tokens = block['anchor'], full[p['id']]
        pair = (p['id'], anchor)
        if pair in seen:
            raise ValueError('Duplicate native block')
        seen.add(pair)
        if (block['prefix'] != tokens[:anchor] or len(block['prefix']) != anchor
                or anchor < len(p['prompt']) or block['positions'] != list(range(anchor,anchor+16))):
            raise ValueError('Actual native context/positions differ from canonical output')
        labels = tokens[anchor:anchor+16]
        if len(labels) != 16:
            excluded.append(dict(prompt_id=p['id'], anchor=anchor, reason='incomplete_reference_window'))
            continue
        draft = block['draft']
        if len(draft) != 16 or draft[0] != labels[0] or block['native_draft_count'] != 15:
            raise ValueError('Native anchor/proposal layout mismatch')
        count = acceptance(draft, labels)
        if block['num_sampled'] != count+1 or block['num_rejected'] != 15-count:
            raise ValueError('Native acceptance disagrees with canonical target continuation')
        context_start = max(0, anchor-1024)
        row = dict(prompt_id=p['id'], source=p['source'], family_id=p['family_id'], split=p['split'],
            source_record_sha256=p['source_record_sha256'], anchor=anchor, context_start=context_start,
            prefix=block['prefix'][context_start:], draft=draft, labels=labels, native_accepted=count,
            kind='wrong_first' if count==0 else 'preservation' if count==15 else 'repairable',
            target_replay_verified=True, target_contract='invariant-native-canonical-and-acceptance-v1')
        if not row['prefix'] or len(row['prefix'])+context_start != anchor:
            raise ValueError('Editor context window is misaligned')
        rows.append(row)
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r['prompt_id']].append(r)
    selected, coverage = [], []
    for p in prompts:
        candidates = sorted(by_prompt[p['id']], key=lambda r:r['anchor'])
        natural_counts = Counter(r['kind'] for r in candidates)
        if p['split'] == 'train':
            eligible = [r for r in candidates if r['kind'] != 'wrong_first']
            preserve = [r for r in eligible if r['kind'] == 'preservation']
            # Include a naturally correct proposal when available, then spread
            # remaining choices. Never create a fake identity control.
            keep = preserve[:1]
            remaining = [r for r in eligible if r not in keep]
            keep += spread(remaining, 4-len(keep))
        else:
            keep = spread(candidates, 4)
        selected.extend(sorted(keep, key=lambda r:r['anchor']))
        coverage.append(dict(prompt_id=p['id'], split=p['split'], complete_blocks=len(candidates),
            natural_counts=dict(natural_counts), selected_blocks=len(keep),
            exclusion_reason=None if keep else 'no_reachable_full_block' if candidates else 'no_complete_native_block'))
    return dict(rows=selected, coverage=coverage, excluded=excluded,
                positions_checked=positions_checked, aligned_hashes=aligned_hashes,
                all_complete_blocks=len(rows), natural_counts=dict(Counter(r['split']+'/'+r['kind'] for r in rows)))
