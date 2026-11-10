"""Independently re-open a completed bank before permitting training."""
import argparse
from collections import Counter
import json
from pathlib import Path

from prepare import sha256


def inspect(directory, repairs=128, preservation=16):
    import torch
    from safetensors.torch import load_file
    directory = Path(directory)
    audit = json.loads((directory / 'audit.json').read_text())
    if audit['outcome'] != 'success':
        raise ValueError('Collection did not pass; partial banks cannot unlock training')
    if sha256(directory / 'manifest.json') != audit['manifest_sha256']:
        raise ValueError('Manifest checksum mismatch')
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['schema_version'] != 1:
        raise ValueError('Unknown bank schema')
    if manifest['models'] != json.loads(Path(__file__).with_name('model_pin.json').read_text()):
        raise ValueError('Bank model pins differ from this experiment')
    counts, seen = Counter(), set()
    for entry in manifest['blocks']:
        key = (entry['id'], entry['anchor'])
        if key in seen or entry['dataset_split'] != 'train':
            raise ValueError('Duplicate block or non-training prompt')
        seen.add(key)
        name = entry['file']
        if Path(name).name != name:
            raise ValueError('Bank file must be local to the manifest directory')
        file = directory / name
        if sha256(file) != entry['sha256'] or file.stat().st_size != entry['bytes']:
            raise ValueError(f'Tensor checksum/size mismatch: {name}')
        tensors = load_file(file)
        expected = {'normalized_states': ((1, 16, 4096), torch.bfloat16),
                    'first_guess_embedding': ((1, 4096), torch.bfloat16),
                    'original_ids': ((1, 16), torch.int64), 'labels': ((1, 16), torch.int64),
                    'valid_mask': ((1, 16), torch.bool), 'anchor': ((1,), torch.int64),
                    'prefix_ids': ((1, entry['anchor']), torch.int64)}
        if set(tensors) != set(expected):
            raise ValueError('Unexpected tensor keys')
        for name, (shape, dtype) in expected.items():
            tensor = tensors[name]
            if tuple(tensor.shape) != shape or tensor.dtype != dtype or not torch.isfinite(tensor).all():
                raise ValueError(f'Tensor schema/finiteness mismatch: {name}')
        if tensors['anchor'].item() != entry['anchor'] or not tensors['valid_mask'].all():
            raise ValueError('Invalid anchor or incomplete block')
        original, labels = tensors['original_ids'][0].tolist(), tensors['labels'][0].tolist()
        if original != entry['original_ids'] or labels != entry['labels']:
            raise ValueError('Tensor/manifest token disagreement')
        if original[:2] != labels[:2]:
            raise ValueError('Selected bank has an unreachable repair')
        kind = 'preservation' if original == labels else 'repairable'
        if kind != entry['kind']:
            raise ValueError('Incorrect repair/preservation classification')
        counts[kind] += 1
    if counts.get('repairable', 0) != repairs or counts.get('preservation', 0) != preservation:
        raise ValueError(f'Insufficient bank: {dict(counts)}, expected {repairs} repairs + {preservation} preservation')
    return dict(outcome='passed', blocks=len(seen), counts=dict(counts), manifest_sha256=audit['manifest_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--repairs', type=int, default=128)
    parser.add_argument('--preservation', type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(inspect(args.directory, args.repairs, args.preservation), indent=2))
