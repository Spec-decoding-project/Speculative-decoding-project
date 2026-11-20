"""Explain repair loss from saved token/action traces; no model inference."""
import argparse
from collections import Counter
import json
from pathlib import Path

from copy_action_fit.data import save, sha


def analyze(baseline, copy):
    if len(baseline['rows']) != len(copy['rows']):
        raise ValueError('Different prediction populations')
    counts = Counter()
    for control, edited in zip(baseline['rows'], copy['rows']):
        for key in ('prompt_id', 'anchor', 'original_tokens', 'label_tokens'):
            if control[key] != edited[key]:
                raise ValueError('Unpaired prediction identity or tokens')
        if control['repaired']:
            counts['control_repairs'] += 1
            if edited['repaired']:
                counts['repairs_retained'] += 1
            elif edited['edited_progress'] < edited['original_progress']:
                counts['lost_by_earlier_damage'] += 1
            else:
                first_error = edited['original_progress'] + 1
                action = edited['actions'][first_error-2]
                counts['lost_by_copying_original_error' if action == 151936
                       else 'lost_by_wrong_generated_token'] += 1
        elif edited['repaired']:
            counts['new_copy_repairs'] += 1
    counts['entire_suffix_copied'] = sum(all(t == 151936 for t in row['actions']) for row in copy['rows'])
    counts['original_suffix_token_matches'] = sum(sum(x == y for x, y in zip(
        row['original_tokens'][2:], row['label_tokens'][2:])) for row in copy['rows'])
    if counts['control_repairs'] != sum(counts[key] for key in
            ('repairs_retained', 'lost_by_earlier_damage', 'lost_by_copying_original_error',
             'lost_by_wrong_generated_token')):
        raise ValueError('Repair-loss categories do not partition the control repairs')
    if copy['repaired'] != counts['repairs_retained'] + counts['new_copy_repairs']:
        raise ValueError('COPY repair count does not match the paired trace')
    return dict(counts=counts, blocks=len(copy['rows']),
        scope='Observed action choices, not proof of why weights learned those choices')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'copy', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.baseline.read_text()), json.loads(args.copy.read_text()))
    result['input_sha256'] = dict(baseline=sha(args.baseline), copy=sha(args.copy))
    save(args.output, result)
    print(json.dumps(result, indent=2))
