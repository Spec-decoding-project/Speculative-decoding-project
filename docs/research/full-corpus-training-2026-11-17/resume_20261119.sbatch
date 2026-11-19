#!/bin/bash
#SBATCH --job-name=ar-full-resume
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --nodelist=tron61
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=03:00:00
#SBATCH --output=/tmp/ar-full-resume-%j.log
set -euo pipefail
project_dir=/fs/classhomes/fpajak/speculative-decoding-project
source_dir="$project_dir/research/full-corpus-fit-20261117-run"
output_dir="$project_dir/research/full-corpus-run-20261117"
install_dir=/scratch0/fpajak-vllm034-Ry21jsII
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VLLM_BATCH_INVARIANT=0
export PYTHONPATH="$source_dir:$project_dir/research/editor-feature-bank:$project_dir/research/editor-feature-bank/replay_diagnostic"
export TMPDIR="$install_dir/tmp" XDG_CACHE_HOME="$install_dir/cache"
export TRITON_CACHE_DIR="$install_dir/triton" TORCHINDUCTOR_CACHE_DIR="$install_dir/inductor"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
python_bin="$install_dir/venv/bin/python"
mkdir -p /scratch0/fpajak-full-corpus-recovery-20261119
recovery_dir=/scratch0/fpajak-full-corpus-recovery-20261119
cat > "$recovery_dir/recover-full-corpus-20261119.py" <<'RECOVERY_PY'
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
RECOVERY_PY
cat > "$recovery_dir/mac-backup-checksums.json" <<'BACKUP_JSON'
{
  "training/draft/latest.pt": "681fb92413db1b4111d6c3935d949ad9c0d1b365f90792de1c79da147aa9586f",
  "training/draft/best.pt": "6d57e0f9bfb5a9e5679fb356f3bf2e13151deb5c92189220edb5a8f037782967",
  "training/context/latest.pt": "ac112e9d404b49344835f853ff0717d9bf328c227163ce6040144f80494928bc",
  "training/context/best.pt": "94732595d3b701f66d297400d3abedc7f8b516923be295c62b01dd42a67211a9",
  "bank.sqlite": "c7f92fae5609137510a198297a31b5178c70c413f263f89f04560b98aa6bbb3d"
}
BACKUP_JSON
trap 'status=$?; printf "%s\n" "$status" > "$recovery_dir/exit-status-$SLURM_JOB_ID.txt"; cp "/tmp/ar-full-resume-$SLURM_JOB_ID.log" "$recovery_dir/" || true' EXIT
"$python_bin" "$recovery_dir/recover-full-corpus-20261119.py" "$recovery_dir/mac-backup-checksums.json"
cp "$output_dir/status.json" "$output_dir/status-before-recovery-8454498.json"
cp "$output_dir/training/draft/development-epoch-1.json" "$output_dir/development-draft-before-recovery-8454498.json"
"$python_bin" -m pip freeze > "$output_dir/packages-$SLURM_JOB_ID.txt"
cmp "$output_dir/packages-$SLURM_JOB_ID.txt" "$project_dir/research/one-block-fit-8453776/packages.txt"
"$python_bin" -u -m full_corpus_fit.pipeline --project "$project_dir" \
  --preflight "$project_dir/research/full-corpus-preflight-8454304" \
  --calibration "$project_dir/research/full-corpus-calibration-8454371/calibration.json" \
  --output "$output_dir"
