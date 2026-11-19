"""Independently recount saved predictions against the immutable token bank.

This uses no training/evaluation helpers and performs no model inference.
Teacher-forced cross-entropy cannot be recovered from argmax tokens; it is
checked for a finite reported value, not claimed as independently recomputed.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prefix(candidate, labels):
    for position in range(1, len(labels)):
        if candidate[position] != labels[position]:
            return position - 1
    return len(labels) - 1


def equal_metrics(actual, expected):
    for name, value in expected.items():
        require(name in actual, f'Missing metric: {name}')
        require(math.isclose(actual[name], value, rel_tol=0, abs_tol=1e-12),
                f'Incorrect metric {name}: {actual[name]} != {value}')


def summarize(rows):
    return dict(blocks=len(rows),
                mean_original_progress=statistics.mean(r['original_progress'] for r in rows),
                mean_edited_progress=statistics.mean(r['edited_progress'] for r in rows),
                mean_union_progress=statistics.mean(r['union_progress'] for r in rows),
                repaired=sum(r['repaired'] for r in rows),
                damaged=sum(r['damaged'] for r in rows))


def audit_predictions(result, examples):
    require(len(result['rows']) == len(examples) > 0, 'Prediction population mismatch')
    seen, checked, groups, prompts = set(), [], defaultdict(list), defaultdict(list)
    matches = 0
    for actual, example in zip(result['rows'], examples):
        identity = (example['prompt_id'], example['anchor'])
        require(identity not in seen, 'Duplicate bank identity')
        seen.add(identity)
        require((actual['prompt_id'], actual['anchor']) == identity, 'Prediction identity/order mismatch')
        require(actual['original_tokens'] == example['draft'], 'Original proposal differs from bank')
        require(actual['label_tokens'] == example['labels'], 'Labels differ from bank')
        require(actual['kind'] == example['kind'], 'Block kind differs from bank')
        original, edited, labels = example['draft'], actual['edited_tokens'], example['labels']
        require(len(original) == len(edited) == len(labels) == 16, 'Incomplete token block')
        require(all(type(t) is int and 0 <= t < 151936 for t in edited), 'Invalid edited token')
        require(original[0] == labels[0] and edited[:2] == original[:2], 'Preserved seed changed')
        a, b = prefix(original, labels), prefix(edited, labels)
        row = dict(original_progress=a, edited_progress=b, union_progress=max(a, b),
                   repaired=b > a, damaged=b < a, wrong_first=original[1] != labels[1],
                   exact_suffix=edited[2:] == labels[2:])
        equal_metrics(actual, row)
        matches += sum(left == right for left, right in zip(edited[2:], labels[2:]))
        checked.append(row)
        groups['source/' + example['source']].append(row)
        groups['kind/' + example['kind']].append(row)
        prompts[example['prompt_id']].append(b - a)
    metrics = dict(**summarize(checked), wrong_first=sum(r['wrong_first'] for r in checked),
                   free_running_exact_suffix=sum(r['exact_suffix'] for r in checked) / len(checked),
                   free_running_token_accuracy=matches / (14 * len(checked)))
    equal_metrics(result, metrics)
    require(math.isfinite(result['teacher_forced_ce']) and result['teacher_forced_ce'] >= 0,
            'Invalid reported cross-entropy')
    deltas = [statistics.mean(values) for values in prompts.values()]
    return dict(metrics=metrics, groups={name:summarize(rows) for name,rows in groups.items()},
                editable_matches=matches, editable_tokens=14*len(checked),
                exact_suffixes=sum(r['exact_suffix'] for r in checked),
                prompts=len(prompts), prompt_weighted_mean_delta=statistics.mean(deltas),
                prompts_improved=sum(d > 0 for d in deltas),
                prompts_worsened=sum(d < 0 for d in deltas),
                prompts_unchanged=sum(d == 0 for d in deltas))


def audit_run(root, mode):
    info = json.loads((root/'bank.json').read_text())
    require(sha(root/'bank.sqlite') == info['bank_sha256'], 'Bank checksum mismatch')
    bank = sqlite3.connect(f'file:{root}/bank.sqlite?mode=ro', uri=True)
    require(bank.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'Bank integrity failed')
    population = bank.execute('SELECT split,count(*),count(distinct prompt_id) FROM examples GROUP BY split').fetchall()
    families = defaultdict(set)
    for split, payload in bank.execute('SELECT split,payload FROM examples'):
        families[split].add(json.loads(payload)['family_id'])
    for left, right in (('train','development'), ('train','test'), ('development','test')):
        require(not families[left] & families[right], f'Family leakage: {left}/{right}')
    rows = {split:[json.loads(r[0]) for r in bank.execute(
        'SELECT payload FROM examples WHERE split=? ORDER BY id', (split,))]
        for split in ('development','test')}
    directory = root/'training'/mode
    complete = json.loads((directory/'complete.json').read_text())
    config = json.loads((directory/'run.json').read_text())
    require(config['bank_sha256'] == complete['bank_sha256'] == info['bank_sha256'], 'Run bank differs')
    require(complete['updates'] == config['total_steps'] and complete['epochs'] == 3,
            'Incomplete optimization')
    audited, selection = {}, []
    for epoch in (1,2,3):
        result = json.loads((directory/f'development-epoch-{epoch}.json').read_text())
        audited[f'development_epoch_{epoch}'] = audit_predictions(result, rows['development'])
        selection.append((result['mean_edited_progress'], -result['teacher_forced_ce'], -epoch))
    selected = -max(selection)[2]
    require(complete['selected_epoch'] == selected, 'Checkpoint selection mismatch')
    test = json.loads((directory/'test.json').read_text())
    audited['test'] = audit_predictions(test, rows['test'])
    equal_metrics(complete['test'], audited['test']['metrics'])
    require(set(complete['test']['groups']) == set(audited['test']['groups']), 'Test group set mismatch')
    for name, expected in audited['test']['groups'].items():
        equal_metrics(complete['test']['groups'][name], expected)
    equal_metrics(complete['development'], audited[f'development_epoch_{selected}']['metrics'])
    for name in ('best','latest'):
        require(sha(directory/f'{name}.pt') == complete[f'{name}_checkpoint_sha256'],
                f'{name} checkpoint checksum mismatch')
    bank.close()
    return dict(passed=True, mode=mode, bank_sha256=info['bank_sha256'], population=population,
                split_family_intersections=0, selected_epoch=selected, audits=audited,
                limits='No model inference; cross-entropy is not independently recomputed; no speedup claim')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--mode', choices=('context','draft'), required=True)
    args=parser.parse_args()
    print(json.dumps(audit_run(args.run, args.mode), indent=2))
