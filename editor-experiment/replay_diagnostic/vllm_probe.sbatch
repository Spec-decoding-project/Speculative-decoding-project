#!/bin/bash
#SBATCH --job-name=vllm-invariance
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=00:30:00
#SBATCH --output=vllm-invariance-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from Nexus project root}"
project_dir="$PWD"
output_dir="$project_dir/research/editor-feature-bank/vllm-invariance-$SLURM_JOB_ID"
mkdir "$output_dir"
trap 'status=$?; printf "%s\n" "$status" > "$output_dir/exit-status.txt"' EXIT
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
nvidia-smi > "$output_dir/nvidia-smi.txt"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv > "$output_dir/device.csv"
df -h /scratch0 "$project_dir" > "$output_dir/disk.txt"
driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
if (( ${driver_version%%.*} < 580 )); then
    printf 'CUDA 13 wheel requires driver >=580; found %s. No installation or target calls.\n' "$driver_version" | tee "$output_dir/preflight-failure.txt"
    exit 20
fi
scratch_free=$(df -Pk /scratch0 | awk 'NR==2 {print $4}')
if (( scratch_free < 20 * 1024 * 1024 )); then
    printf 'Need 20 GiB node-local installation headroom; found %s KiB.\n' "$scratch_free" | tee "$output_dir/preflight-failure.txt"
    exit 21
fi
install_dir=$(mktemp -d /scratch0/fpajak-vllm034-XXXXXXXX)
printf '%s\n' "$install_dir" > "$output_dir/environment-path.txt"
export TMPDIR="$install_dir/tmp" XDG_CACHE_HOME="$install_dir/cache"
export TRITON_CACHE_DIR="$install_dir/triton" TORCHINDUCTOR_CACHE_DIR="$install_dir/inductor"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
.venv-qwen/bin/python -m venv "$install_dir/venv"
wheel_url='https://files.pythonhosted.org/packages/0b/2c/55f3b76ceadbfc6d515f16b1c17e2e69136c8e54d47be2a94b9f7326f9bb/vllm-0.34.0-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=79ca4b3cd74aec08c4bf476bdf0cb15735b605ae5368526a2faae1e30f044ca2'
timeout 900 "$install_dir/venv/bin/python" -m pip install --no-cache-dir --only-binary=:all: \
    --report "$output_dir/install-report.json" "$wheel_url" > "$output_dir/install.log" 2>&1
"$install_dir/venv/bin/python" -m pip check > "$output_dir/pip-check.txt"
"$install_dir/venv/bin/python" -m pip freeze > "$output_dir/packages.txt"
export PYTHONPATH="$project_dir/research/editor-feature-bank/replay_diagnostic"
timeout 850 "$install_dir/venv/bin/python" -u \
    research/editor-feature-bank/replay_diagnostic/vllm_probe.py --output "$output_dir" \
    > "$output_dir/runner.log" 2>&1
