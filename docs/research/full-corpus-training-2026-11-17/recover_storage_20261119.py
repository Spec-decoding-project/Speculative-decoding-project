"""Bounded storage recovery; never changes the frozen experiment source."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import torch

from autoregressive_editor.learning import Trainer
from full_corpus_fit.train import validate_cursor
from one_block_fit.train import weights_hash

root = Path('/fs/classhomes/fpajak/speculative-decoding-project/research/full-corpus-run-20261117')
scratch = Path('/scratch0/fpajak-full-corpus-recovery-20261119')
scratch.mkdir(exist_ok=True, parents=True)
torch.set_num_threads(4)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def record(value):
    print(json.dumps(value), flush=True)

backup = json.loads(Path(sys.argv[1]).read_text())
context = root/'training/context'
complete = json.loads((context/'complete.json').read_text())
assert shutil.disk_usage(scratch).free > 3 * 1024**3
# Two independent copies exist before removing any home-file contents:
# the verified Mac backup manifest and a freshly verified node-scratch copy.
relocations = []
for filename in ('best.pt', 'latest.pt'):
    path = context/filename
    expected = complete[filename.split('.')[0]+'_checkpoint_sha256']
    assert backup['training/context/'+filename] == expected
    assert sha(path) == expected
    destination = scratch/('context-'+filename)
    if not destination.exists():
        shutil.copy2(path, destination)
    assert sha(destination) == expected
    relocations.append(dict(original=str(path), destination=str(destination), sha256=expected))
(scratch/'relocations-before-write.json').write_text(json.dumps(relocations, indent=2)+'\n')
for item in relocations:
    path, destination = Path(item['original']), Path(item['destination'])
    if not path.is_symlink():
        # Home cannot allocate even a directory entry before space is freed.
        # Both verified copies above preserve recovery across this short gap.
        path.unlink()
        path.symlink_to(destination)
    assert path.resolve() == destination
    assert sha(path) == item['sha256']
record(dict(relocations=relocations))

# Re-run the exact failing filesystem operation after relocation.
probe = root/'.write-probe-20261119'
probe.write_bytes(b'storage probe\n')
assert probe.read_bytes() == b'storage probe\n'
probe.unlink()

draft = root/'training/draft'
assert sha(draft/'latest.pt') == backup['training/draft/latest.pt']
assert sha(draft/'best.pt') == backup['training/draft/best.pt']
trainer, state = Trainer.load(draft/'latest.pt', device='cpu')
validate_cursor(state, trainer.step, 199205)
assert trainer.step == 6226 and state['epoch'] == 0 and state['cursor'] == 199205
before = weights_hash(trainer.model)
# Exercise an actual full-size atomic save on the same filesystem, and compare
# all optimizer tensors after reload. A free-space number alone is insufficient.
save_probe = root/'recovery-save-probe.pt'
trainer.save(save_probe, state)
reloaded, reloaded_state = Trainer.load(save_probe, device='cpu')
assert reloaded_state == state and reloaded.step == trainer.step
assert weights_hash(reloaded.model) == before
left, right = trainer.optimizer.state_dict(), reloaded.optimizer.state_dict()
assert left['param_groups'] == right['param_groups']
for identity, values in left['state'].items():
    for name, value in values.items():
        other = right['state'][identity][name]
        assert torch.equal(value, other) if torch.is_tensor(value) else value == other
save_probe.unlink()
del reloaded
best, best_state = Trainer.load(draft/'best.pt', device='cpu')
assert weights_hash(best.model) == before
assert best_state['selected_epoch'] == 1 and best.step == trainer.step
result = dict(storage_probe='passed', full_checkpoint_save_reload='passed',
              optimizer_roundtrip='passed', draft_step=trainer.step, draft_state=state,
              draft_weights_sha256=before, first_epoch_best_load='passed',
              relocations=relocations, source_changed=False)
(root/'recovery-storage-20261119.json').write_text(json.dumps(result, indent=2)+'\n')
record(result)
