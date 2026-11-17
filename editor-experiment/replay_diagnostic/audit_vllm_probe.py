"""CPU-only independent alignment of raw logit rows and emitted tokens."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def read_json(path):
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def audit_trial(fixtures, trial):
    checked, mappings = 0, {}
    for name, generated in trial['tokens'].items():
        fixture = fixtures[name]
        initial = token_hash(fixture['prompt'])
        internal = {r['request_id'] for r in trial['trace']
                    if r.get('prefix_sha256') == initial}
        if len(internal) != 1:
            raise ValueError(f'Ambiguous initial prompt: {name}')
        request = next(iter(internal))
        mappings[name] = dict(external=trial['details'][name]['request_id'], internal=request)
        for position, token in enumerate(generated):
            prefix = fixture['prompt'] + generated[:position]
            digest = token_hash(prefix)
            matched = [r for r in trial['trace'] if r.get('request_id') == request
                       and r.get('prefix_sha256') == digest]
            if not matched or any(r['position'] != len(prefix)-1 or r['argmax'] != token
                                  or r['vocabulary_size'] != 151936 for r in matched):
                raise ValueError(f'Raw logits do not authenticate token {position} of {name}')
            checked += 1
    return dict(generated_positions_verified=checked, request_mapping=mappings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--previous-report', type=Path, required=True)
    args = parser.parse_args()
    inputs = read_json(args.directory / 'inputs.json')
    fixtures = {x['id']: x for x in inputs['fixtures']}
    prior = {x['id']: x for x in read_json(args.previous_report)['fixtures']}
    for name, f in fixtures.items():
        if (f['record_sha256'] != prior[name]['record_sha256']
                or token_hash(f['prompt']) != prior[name]['prefix_sha256']
                or token_hash(f['labels']) != prior[name]['labels_sha256']):
            raise ValueError(f'Original fixture changed: {name}')
    reports = []
    for path in sorted(args.directory.glob('bi*.json*')):
        cell = read_json(path)
        if not cell['complete']:
            raise ValueError(f'Incomplete cell: {path}')
        for trial in cell['trials']:
            reports.append(dict(cell=cell['cell'], trial=trial['name'],
                                **audit_trial(fixtures, trial)))
    print(json.dumps(dict(trials=reports, total_generated_positions_audited=sum(
        r['generated_positions_verified'] for r in reports)), indent=2))


if __name__ == '__main__':
    main()
