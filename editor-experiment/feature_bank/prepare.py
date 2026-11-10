"""Download exactly one copy of the pinned drafter; verify every byte before use."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(directory, pin):
    for name, expected in pin['files'].items():
        path = Path(directory) / name
        if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError(f'Checkpoint/source fingerprint mismatch: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    pin = json.loads(Path(__file__).with_name('model_pin.json').read_text())
    args.destination.mkdir(parents=True, exist_ok=True)
    for name, expected in pin['files'].items():
        path = args.destination / name
        if path.exists():
            if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
                raise ValueError(f'Refusing to replace existing mismatched file: {path}')
            continue
        url = f"https://huggingface.co/{pin['draft_id']}/resolve/{pin['draft_revision']}/{name}"
        temporary = path.with_suffix(path.suffix + '.partial')
        print(f'Downloading {name}: {expected["bytes"]} bytes', flush=True)
        urllib.request.urlretrieve(url, temporary)
        if temporary.stat().st_size != expected['bytes'] or sha256(temporary) != expected['sha256']:
            raise ValueError(f'Download fingerprint mismatch: {name}')
        temporary.rename(path)
    verify(args.destination, pin)
    print('All pinned source and weight hashes verified.', flush=True)


if __name__ == '__main__':
    main()
