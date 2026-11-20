"""Verify preserved originals and exercise the three-checkpoint storage peak."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket

from copy_action_fit.data import CHECKPOINT_SHA256, save, sha


DRAFT_HASHES = {
    'best': CHECKPOINT_SHA256,
    'latest': '36240c9b8f5edbb9f073a5297816f4c8dfa6774e3c3eb427baed101e3b9ee0b9',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backup-attestation', type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] != 'tron61' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Storage preflight must run in a tron61 Slurm allocation')
    attestation = json.loads(args.backup_attestation.read_text())
    if attestation['draft_checkpoint_sha256'] != DRAFT_HASHES:
        raise ValueError('Independent Mac backup was not verified')
    archive = Path('/scratch0/fpajak-copy-action-20261120-originals')
    archive.mkdir(exist_ok=True)
    records = []
    # Copy and verify both completed originals before changing either home path.
    for name, expected in DRAFT_HASHES.items():
        original, preserved = args.original/f'{name}.pt', archive/f'{name}.pt'
        if sha(original) != expected:
            raise ValueError(f'Original {name} checkpoint changed')
        if not preserved.exists():
            temporary = preserved.with_suffix('.pt.tmp')
            shutil.copyfile(original, temporary)
            if sha(temporary) != expected:
                raise ValueError('Scratch preservation copy failed verification')
            temporary.replace(preserved)
        if sha(preserved) != expected:
            raise ValueError('Scratch preservation checkpoint changed')
        records.append(dict(original=str(original), preserved=str(preserved), sha256=expected))
    for record in records:
        original = Path(record['original'])
        if not original.is_symlink():
            link = original.with_suffix('.pt.copy-pilot-link')
            link.symlink_to(record['preserved'])
            link.replace(original)
        if sha(original) != record['sha256']:
            raise ValueError('Relocated original no longer resolves correctly')
    save(args.output/'original-preservation.json', dict(mac_backup=attestation, node_copies=records))
    # Physically write, fsync, and hash three whole checkpoints simultaneously:
    # two final arm checkpoints plus the next atomic replacement temporary file.
    probes = [args.output/f'storage-probe-{index}.pt' for index in range(3)]
    source = archive/'best.pt'
    try:
        for path in probes:
            with source.open('rb') as reader, path.open('wb') as writer:
                shutil.copyfileobj(reader, writer, length=8*1024**2)
                writer.flush(); os.fsync(writer.fileno())
            if sha(path) != CHECKPOINT_SHA256:
                raise ValueError('Physical checkpoint-sized storage probe failed')
        save(args.output/'storage-preflight.json', dict(passed=True,
            simultaneously_written_bytes=sum(p.stat().st_size for p in probes),
            checkpoint_copies=3, headroom_policy='Two arms plus one atomic replacement'))
    finally:
        for path in probes:
            path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
