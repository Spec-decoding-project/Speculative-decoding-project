#!/bin/bash
# Recovery of setup only; protocol revision is recorded on the diagnostic ticket.
# First allocation elapsed 15m07s; this cap keeps the total below 30 minutes.
#SBATCH --job-name=vllm-recover
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --nodelist=tron61
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=00:14:00
#SBATCH --output=vllm-recover-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from Nexus project root}"
project_dir="$PWD"
previous_dir="$project_dir/research/editor-feature-bank/vllm-invariance-8452972"
output_dir="$project_dir/research/editor-feature-bank/vllm-invariance-$SLURM_JOB_ID"
install_dir=/scratch0/fpajak-vllm034-Ry21jsII
mkdir "$output_dir"
trap 'status=$?; printf "%s\n" "$status" > "$output_dir/exit-status.txt"' EXIT
test ! -e "$previous_dir/budget.json"
test ! -e "$previous_dir/runner.log"
test -d "$install_dir"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TMPDIR="$install_dir/tmp" XDG_CACHE_HOME="$install_dir/cache"
export TRITON_CACHE_DIR="$install_dir/triton" TORCHINDUCTOR_CACHE_DIR="$install_dir/inductor"
export VLLM_RECOVER_ROOT="$install_dir" VLLM_RECOVER_REPORT="$previous_dir/install-report.json"
nvidia-smi > "$output_dir/nvidia-smi.txt"
printf '%s\n' "$install_dir" > "$output_dir/environment-path.txt"
# Verify every wheel against the resolver report before reusing it. Reinstall all
# dependencies so an interrupted unpack cannot masquerade as a complete package.
.venv-qwen/bin/python - <<'PY' > "$output_dir/wheel-verification.json"
import hashlib,json,os
from pathlib import Path
from urllib.parse import unquote,urlparse
root=Path(os.environ['VLLM_RECOVER_ROOT'])
report=json.loads(Path(os.environ['VLLM_RECOVER_REPORT']).read_text())
expected={Path(unquote(urlparse(x['download_info']['url']).path)).name:
          x['download_info']['archive_info']['hashes']['sha256'] for x in report['install']}
found={}
wheelhouse=root/'verified-wheels'
wheelhouse.mkdir(exist_ok=False)
for path in (root/'tmp').rglob('*.whl'):
    if path.name not in expected:
        raise RuntimeError(f'Unexpected wheel {path.name}')
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest != expected[path.name]:
        raise RuntimeError(f'Wrong wheel hash {path.name}')
    if path.name not in found:
        (wheelhouse/path.name).symlink_to(path)
        found[path.name]=digest
if set(found) != set(expected):
    raise RuntimeError(f'Missing wheels: {sorted(set(expected)-set(found))}')
print(json.dumps(dict(verified=found,count=len(found)),indent=2))
PY
timeout 300 "$install_dir/venv/bin/python" -m pip install --no-index --no-cache-dir \
    --force-reinstall --only-binary=:all: --find-links "$install_dir/verified-wheels" \
    --report "$output_dir/install-report.json" vllm==0.34.0 > "$output_dir/install.log" 2>&1
"$install_dir/venv/bin/python" -m pip check > "$output_dir/pip-check.txt"
"$install_dir/venv/bin/python" -m pip freeze > "$output_dir/packages.txt"
export PYTHONPATH="$project_dir/research/editor-feature-bank/replay_diagnostic"
"$install_dir/venv/bin/python" -u research/editor-feature-bank/replay_diagnostic/vllm_probe.py \
    --output "$output_dir" > "$output_dir/runner.log" 2>&1
