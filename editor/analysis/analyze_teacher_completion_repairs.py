"""Descriptive, independent repair anatomy; never replaces the frozen primary test."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prefix(tokens, labels, valid):
    length = 0
    for guess, answer, enabled in zip(tokens, labels, valid, strict=True):
        if not enabled or guess != answer:
            break
        length += 1
    return length


def coverage(nodes, labels, valid):
    paths = {tuple(p) for p in nodes}
    if len(paths) != len(nodes) or any(len(p) > 1 and p[:-1] not in paths for p in paths):
        raise ValueError('Invalid proposal tree')
    count = 0
    for end in range(1, sum(valid) + 1):
        if tuple(labels[:end]) not in paths:
            break
        count = end
    return count


def analyze(folder):
    folder = Path(folder)
    if load(folder / 'terminal.json')['status'] != 'complete':
        raise ValueError('Only analyze a complete comparison')
    conditions = {}
    sources = {}

    def read(path):
        sources[path.relative_to(folder).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return load(path)

    for mode in ('first_guess', 'full_proposal'):
        rows = read(folder / (mode + '-predictions.json'))
        conditions[mode] = {(x['prompt'], x['anchor_index']): x for x in rows}
        if len(rows) != 256 or len(conditions[mode]) != 256:
            raise ValueError('Incomplete or duplicate trained panel')
    by_context = []
    overlap = Counter()
    token_behavior = {mode: Counter() for mode in conditions}
    for path in sorted((folder / 'controls').glob('*.json')):
        proposals = {p['anchor_index']: p for p in read(folder / 'proposals' / path.name)}
        for control in read(path):
            key = path.stem, control['anchor_index']
            original = proposals[key[1]]
            labels, valid = original['labels'], original['valid']
            base_chain = prefix(original['tokens'], labels, valid)
            base_tree = coverage(control['trees']['conditional64'], labels, valid)
            row = dict(prompt=key[0], anchor_index=key[1], original_chain=base_chain,
                       existing_conditional_coverage=base_tree, valid_positions=sum(valid))
            for mode in conditions:
                prediction = conditions[mode][key]
                if prediction['labels'] != labels or prediction['valid'] != valid:
                    raise ValueError('Unmatched reference')
                if prediction['tokens'][:2] != original['tokens'][:2]:
                    raise ValueError('Root or first guess changed')
                row[mode + '_chain'] = prefix(prediction['tokens'], labels, valid)
                row[mode + '_coverage'] = coverage(prediction['nodes'], labels, valid)
                counts = token_behavior[mode]
                for j in range(2, len(labels)):
                    if not valid[j]:
                        continue
                    before, after, answer = original['tokens'][j], prediction['tokens'][j], labels[j]
                    if before == answer:
                        counts['originally_correct_editable_tokens'] += 1
                        counts['correct_tokens_preserved'] += after == answer
                        counts['correct_tokens_damaged'] += after != answer
                    else:
                        counts['originally_wrong_editable_tokens'] += 1
                        counts['wrong_guess_repeated'] += after == before
                        counts['wrong_guess_corrected'] += after == answer
                        counts['wrong_guess_changed_to_another_wrong_token'] += after not in (before, answer)
                if 2 <= base_chain < sum(valid):
                    counts['editable_original_first_errors'] += 1
                    counts['original_first_error_corrected'] += prediction['tokens'][base_chain] == labels[base_chain]
                    counts['original_first_error_repeated'] += prediction['tokens'][base_chain] == original['tokens'][base_chain]
                if row[mode + '_coverage'] != prediction['union64']:
                    raise ValueError('Coverage disagreement')
            first_repairs = row['first_guess_chain'] > base_chain
            full_repairs = row['full_proposal_chain'] > base_chain
            overlap[('both' if first_repairs and full_repairs else 'first_guess_only'
                     if first_repairs else 'full_proposal_only' if full_repairs else 'neither')] += 1
            by_context.append(row)
    if len(by_context) != 256 or len({r['prompt'] for r in by_context}) != 64:
        raise ValueError('Incomplete context panel')
    totals = {}
    for mode in conditions:
        delta = [r[mode + '_coverage'] - r['existing_conditional_coverage'] for r in by_context]
        totals[mode] = dict(contexts_better=sum(x > 0 for x in delta),
                            contexts_worse=sum(x < 0 for x in delta),
                            contexts_tied=sum(x == 0 for x in delta),
                            gained_coverage=sum(max(x, 0) for x in delta),
                            lost_coverage=sum(max(-x, 0) for x in delta),
                            net_coverage=sum(delta))
    return dict(scope='Descriptive fixed-context anatomy; frozen primary analysis governs advancement.',
                compared_against='Released conditional64 control, identical recorded prefixes',
                original_chain_repair_overlap=dict(overlap),
                editable_token_behavior={mode: dict(counts) for mode, counts in token_behavior.items()},
                additional_value_over_existing_refinement=totals,
                contexts=by_context, sources=sources,
                caveats=['No generation speedup or student compressibility is measured.',
                         'Counts aggregate dependent contexts; they are not independent statistical samples.',
                         'Positive-context totals describe hindsight opportunity, not a deployable selector.'])


if __name__ == '__main__':
    folder = Path(sys.argv[1])
    result = analyze(folder)
    output = folder.parent / 'repair-anatomy.json'
    if output.exists() and load(output) != result:
        raise ValueError('Preserve previous analysis')
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('contexts', 'sources')}, indent=2))
