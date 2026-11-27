import errno
import json
from pathlib import Path
import sys

from refinement_research import whole_draft_export as export


def test_default_export_inventories_scratch_without_staging_weight_copies(tmp_path,monkeypatch):
    (tmp_path/'output').mkdir()
    (tmp_path/'output/independent-audit.json').write_text('{"passed":true}')
    for arm in ('uniform','early','protect_repair'):
        (tmp_path/'output'/arm).mkdir();(tmp_path/'checkpoints'/arm).mkdir(parents=True)
        for name,metadata in [('generator-best.pt','generator-complete.json'),('gate-best.pt','gate-complete.json')]:
            source=tmp_path/'checkpoints'/arm/name;source.write_bytes((arm+name).encode())
            (tmp_path/'output'/arm/metadata).write_text(json.dumps({'checkpoint_sha256':export.sha(source)}))
    def disk_full(*args,**kwargs):
        raise OSError(errno.ENOSPC,'No space left on device')
    monkeypatch.setattr(export.shutil,'copyfile',disk_full)
    monkeypatch.setattr(sys,'argv',['export',str(tmp_path)])
    export.main()
    report=json.loads((tmp_path/'export-manifest.json').read_text())
    assert report['passed'] and len(report['files'])==13
    assert not (tmp_path/'selected-checkpoints').exists()
    for arm in ('uniform','early','protect_repair'):
        assert report['files'][f'selected-checkpoints/{arm}/generator-best.pt']['sha256']==export.sha(tmp_path/'checkpoints'/arm/'generator-best.pt')
