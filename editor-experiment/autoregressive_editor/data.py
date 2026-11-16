"""Token-only examples, teacher-forcing alignment and an authenticated bank reader."""
import hashlib
import json
from pathlib import Path

MODES = ('context', 'draft')


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def condition(prefix, draft, mode):
    if mode not in MODES or not prefix or len(draft) < 3:
        raise ValueError('Need a committed prefix, anchor/first proposal/suffix, and known mode')
    if any(type(t) is not int or t < 0 for t in prefix + draft):
        raise ValueError('Token IDs must be nonnegative integers')
    ids, segments = list(prefix), [0] * len(prefix)
    if mode == 'draft':
        ids += list(draft)
        segments += [1] * len(draft)
    # Repeated seed starts the correction segment; no extra vocabulary IDs.
    ids += list(draft[:2])
    segments += [2, 2]
    return ids, segments


def teacher_forcing(example, mode):
    ids, segments = condition(example['prefix'], example['draft'], mode)
    labels = example['labels'][2:]
    start = len(ids) - 1  # The last seed token predicts the first editable token.
    return ids + labels[:-1], segments + [2] * (len(labels) - 1), start, labels


def validate_example(row, vocab_size):
    for name in ('prompt_id', 'family_id', 'source_record_sha256'):
        if not isinstance(row.get(name), str) or not row[name]:
            raise ValueError(f'Missing provenance: {name}')
    if row.get('split') not in ('train', 'validation', 'test'):
        raise ValueError('Unknown dataset split')
    if row.get('target_replay_verified') is not True:
        raise ValueError('Target replay has not passed')
    prefix, draft, labels = row['prefix'], row['draft'], row['labels']
    if not prefix or len(draft) < 3 or len(labels) != len(draft):
        raise ValueError('Incomplete token block')
    if any(type(t) is not int or not 0 <= t < vocab_size for t in prefix + draft + labels):
        raise ValueError('Token outside vocabulary')
    if row.get('anchor') != len(prefix) or draft[0] != labels[0]:
        raise ValueError('Anchor alignment mismatch')


class Bank:
    """Index offsets, not tensor features or full-vocabulary scores, in memory."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest_path = self.directory / 'manifest.json'
        self.manifest = json.loads(self.manifest_path.read_text())
        m = self.manifest
        if m.get('schema') != 'autoregressive-token-bank-v1' or m.get('outcome') != 'validated':
            raise ValueError('Require a completed, validated token bank')
        if m.get('target_replay') != 'passed' or m.get('origin') not in ('verified-feature-bank', 'synthetic-test'):
            raise ValueError('Unsupported or unverified bank provenance')
        self.path = self.directory / 'examples.jsonl'
        if checksum(self.path) != m['examples_sha256']:
            raise ValueError('Token bank checksum mismatch')
        self.fingerprint = checksum(self.manifest_path)
        self.offsets, self.prompts, self.families, self.splits = [], set(), set(), set()
        seen, ownership = set(), {}
        with self.path.open('rb') as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    raise ValueError('Incomplete journal tail')
                row = json.loads(line)
                validate_example(row, m['vocab_size'])
                key = (row['prompt_id'], row['anchor'])
                if key in seen:
                    raise ValueError('Duplicate prompt/anchor')
                seen.add(key)
                for kind, identity in [('prompt', row['prompt_id']), ('family', row['family_id'])]:
                    previous = ownership.setdefault((kind, identity), row['split'])
                    if previous != row['split']:
                        raise ValueError('Cross-split prompt/family leakage')
                self.offsets.append(offset)
                self.prompts.add(row['prompt_id'])
                self.families.add(row['family_id'])
                self.splits.add(row['split'])
        if not self.offsets or len(self.offsets) != m['examples']:
            raise ValueError('Token bank count mismatch')

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open('rb') as stream:
            stream.seek(self.offsets[index])
            return json.loads(stream.readline())

    def require_disjoint(self, other):
        if self.prompts & other.prompts or self.families & other.families:
            raise ValueError('Training and held-out banks overlap by prompt or family')
        if self.manifest['models'] != other.manifest['models']:
            raise ValueError('Banks use different model identities')
