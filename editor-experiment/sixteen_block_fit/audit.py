"""CPU recount of saved predictions; never runs either model or loads a checkpoint."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def prefix_length(tokens, expected):
    for index in range(1, 16):
        if tokens[index] != expected[index]:
            return index-1
    return 15


def recount(result, examples):
    assert len(result['rows']) == len(examples)
    pairs, correct = [], 0
    for row, example in zip(result['rows'], examples):
        assert (row['prompt_id'], row['anchor']) == (example['prompt_id'], example['anchor'])
        assert row['original_tokens'] == example['draft']
        assert row['label_tokens'] == example['labels']
        assert len(row['edited_tokens']) == 16
        assert row['edited_tokens'][:2] == example['draft'][:2]
        original = prefix_length(row['original_tokens'], row['label_tokens'])
        edited = prefix_length(row['edited_tokens'], row['label_tokens'])
        exact = row['edited_tokens'][2:] == row['label_tokens'][2:]
        assert (row['original_progress'], row['edited_progress'], row['union_progress']) == (original, edited, max(original, edited))
        assert (row['repaired'], row['damaged'], row['exact_suffix']) == (edited > original, edited < original, exact)
        correct += sum(a == b for a, b in zip(row['edited_tokens'][2:], row['label_tokens'][2:]))
        if 'editor_milliseconds' in row:
            times = row['editor_milliseconds']['samples']
            assert len(times) == 10 and all(math.isfinite(t) and t > 0 for t in times)
            assert statistics.median(times) == row['editor_milliseconds']['median']
        pairs.append((original, edited, exact))
    expected = dict(blocks=len(pairs), free_running_token_accuracy=correct/(14*len(pairs)),
        free_running_exact_suffix=sum(p[2] for p in pairs)/len(pairs),
        mean_original_progress=statistics.mean(p[0] for p in pairs),
        mean_edited_progress=statistics.mean(p[1] for p in pairs),
        mean_union_progress=statistics.mean(max(p[:2]) for p in pairs),
        repaired=sum(p[1] > p[0] for p in pairs), damaged=sum(p[1] < p[0] for p in pairs))
    assert all(result[k] == v for k, v in expected.items())
    return dict(**expected, correct_suffix_tokens=correct)


def audit(directory):
    data = read(directory/'dataset.json')
    results = {}
    for mode in ('context', 'draft'):
        path = directory/'training'/mode
        summary, run = read(path/'summary.json'), read(path/'run.json')
        assert run['parameters'] == 40750336 and run['effective_batch'] == 16
        assert run['metadata']['dataset_sha256'] == hashlib.sha256((directory/'dataset.json').read_bytes()).hexdigest()
        updates = [json.loads(line) for line in (path/'updates.jsonl').read_text().splitlines()]
        assert len(updates) == summary['updates'] <= 2000
        assert [u['step'] for u in updates] == list(range(1, len(updates)+1))
        assert all(u['suffix_tokens'] == 224 and all(math.isfinite(u[k]) for k in ('ce', 'gradient_norm', 'seconds')) for u in updates)
        evaluations = read(path/'training-evaluations.json')
        assert [e['step'] for e in evaluations] == list(range(25, len(updates)+1, 25))
        consecutive = 0
        for e in evaluations:
            exact = all(r['exact_suffix'] for r in e['rows'])
            assert e['free_running_exact_suffix'] == sum(r['exact_suffix'] for r in e['rows'])/16
            passing = exact and e['free_running_token_accuracy'] == 1 and e['teacher_forced_ce'] < .05
            consecutive = consecutive+1 if passing else 0
            assert e['consecutive_passes'] == consecutive
            assert consecutive <= 3
        if summary['outcome'] == 'memorization_passed':
            assert consecutive == 3 and evaluations[-1]['step'] == len(updates)
        measurements = {name: recount(read(path/(name+'.json')), data[name])
                        for name in ('training', 'development', 'preservation')}
        for name in ('development', 'preservation'):
            recount(read(path/('untrained-'+name+'.json')), data[name])
        by_source = {}
        for source in sorted({r['source'] for r in data['development']}):
            ids = [i for i, r in enumerate(data['development']) if r['source'] == source]
            dev = read(path/'development.json')
            selected = [dev['rows'][i] for i in ids]
            by_source[source] = dict(blocks=len(ids),
                mean_original_progress=statistics.mean(r['original_progress'] for r in selected),
                mean_edited_progress=statistics.mean(r['edited_progress'] for r in selected),
                repaired=sum(r['repaired'] for r in selected), damaged=sum(r['damaged'] for r in selected))
        results[mode] = dict(updates=len(updates), metrics=measurements, development_by_source=by_source)
    assert read(directory/'training/context/summary.json')['initial_weights_sha256'] == read(directory/'training/draft/summary.json')['initial_weights_sha256']
    return dict(passed=True, audit_scope='CPU metric recount from private saved tokens; no GPU rerun',
                predicted_blocks_checked=2*(16+32+1+32+1), variants=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    (args.directory/'independent-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
