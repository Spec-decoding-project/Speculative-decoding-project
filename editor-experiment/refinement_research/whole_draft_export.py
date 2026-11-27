"""Inventory scratch checkpoints for streaming backup; optional home staging."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--copy-checkpoints',action='store_true',
                        help='Explicitly stage additional checkpoint copies in home storage')
    args=parser.parse_args();run=args.run
    audit=json.loads((run/'output/independent-audit.json').read_text())
    assert audit['passed']
    files={}
    for arm in ('uniform','early','protect_repair'):
        destination=run/'selected-checkpoints'/arm
        if args.copy_checkpoints:
            destination.mkdir(parents=True,exist_ok=True)
        for name,metadata in [('generator-best.pt','generator-complete.json'),('gate-best.pt','gate-complete.json')]:
            expected=json.loads((run/'output'/arm/metadata).read_text())['checkpoint_sha256']
            source=run/'checkpoints'/arm/name
            assert sha(source)==expected
            target=destination/name
            if args.copy_checkpoints:
                if not target.exists():
                    temporary=target.with_suffix('.tmp.pt');shutil.copyfile(source,temporary)
                    assert sha(temporary)==expected;temporary.replace(target)
                assert sha(target)==expected
            files[str(target.relative_to(run))]=dict(bytes=source.stat().st_size,sha256=expected)
    for directory in ('output',):
        for path in sorted((run/directory).rglob('*')):
            if path.is_file():
                files[str(path.relative_to(run))]=dict(bytes=path.stat().st_size,sha256=sha(path))
    report=dict(passed=True,files=files,bytes=sum(r['bytes'] for r in files.values()),
        checkpoint_scope=('Selected pairs staged in home' if args.copy_checkpoints else
            'Selected pairs inventoried on node scratch for direct streaming; no home weight copies'),
        resumable_state='Latest optimizer checkpoints retained on node scratch')
    (run/'export-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,files=len(files),bytes=report['bytes'])))


if __name__=='__main__':main()
