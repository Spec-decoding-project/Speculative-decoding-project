"""Export tokens from a fully verified 128-repair/16-preservation pilot bank.

This adapter does not collect features, regenerate labels, or approve partial banks.
"""
import argparse
import json
from pathlib import Path
import sys

from .data import Bank, checksum


def export(source, responses, destination):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'feature_bank'))
    from feature_bank.inspect_bank import inspect
    from feature_bank.data import records, family
    inspection = inspect(source)  # Keep the complete-bank gate and explicit replay contract.
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source / 'manifest.json').read_text())
    audit = json.loads((source / 'audit.json').read_text())
    needed = {entry['id'] for entry in manifest['blocks']}
    originals = {r['id']: r for r in records(responses) if r['id'] in needed}
    if set(originals) != needed:
        raise ValueError('Missing authenticated response records')
    examples = []
    for entry in manifest['blocks']:
        row = originals[entry['id']]
        anchor = entry['anchor']
        tokens = row['prompt_token_ids'] + row['response_token_ids']
        if (row['record_sha256'] != entry['record_sha256'] or row['dataset_split'] != 'train' or
                row['finish_reason'] != 'eos' or tokens[anchor:anchor + 16] != entry['labels']):
            raise ValueError('Response provenance or target labels differ from verified bank')
        examples.append(dict(prompt_id=row['id'], family_id=family(row), split='train',
                             source_record_sha256=row['record_sha256'], anchor=anchor,
                             prefix=tokens[:anchor], draft=entry['original_ids'], labels=entry['labels'],
                             target_replay_verified=True))
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / 'examples.jsonl'
    with path.open('w') as stream:
        for row in examples:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    result = dict(schema='autoregressive-token-bank-v1', outcome='validated',
                  origin='verified-feature-bank', target_replay='passed', vocab_size=151936,
                  models=manifest['models'], examples=len(examples), examples_sha256=checksum(path),
                  replay_contract=inspection['replay_contract'],
                  source_bank_manifest_sha256=audit['manifest_sha256'])
    (destination / 'manifest.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return Bank(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--responses', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bank = export(args.source, args.responses, args.output)
    print(json.dumps(dict(examples=len(bank), bank_sha256=bank.fingerprint)))


if __name__ == '__main__':
    main()
