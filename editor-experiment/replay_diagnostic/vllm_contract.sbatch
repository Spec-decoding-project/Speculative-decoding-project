#!/bin/bash
#SBATCH --job-name=vllm-canonical
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --nodelist=tron61
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=00:30:00
#SBATCH --output=vllm-canonical-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from Nexus project root}"
project_dir="$PWD"
install_dir=/scratch0/fpajak-vllm034-Ry21jsII
output_dir="$project_dir/research/editor-feature-bank/vllm-canonical-$SLURM_JOB_ID"
data_dir="$project_dir/data/qwen_responses_vllm_invariant_pilot_20261117"
test ! -e "$data_dir"
mkdir "$output_dir"
trap 'status=$?; printf "%s\n" "$status" > "$output_dir/exit-status.txt"' EXIT
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TMPDIR="$install_dir/tmp" XDG_CACHE_HOME="$install_dir/cache"
export TRITON_CACHE_DIR="$install_dir/triton" TORCHINDUCTOR_CACHE_DIR="$install_dir/inductor"
export PYTHONPATH="$project_dir/research/editor-feature-bank/replay_diagnostic"
test -x "$install_dir/venv/bin/python"
nvidia-smi > "$output_dir/nvidia-smi.txt"
"$install_dir/venv/bin/python" -m pip check > "$output_dir/pip-check.txt"
"$install_dir/venv/bin/python" -m pip freeze > "$output_dir/packages.txt"
cmp "$output_dir/packages.txt" "$project_dir/research/editor-feature-bank/vllm-invariance-8453091/packages.txt"
"$install_dir/venv/bin/python" -u research/editor-feature-bank/replay_diagnostic/vllm_contract.py \
    --output "$output_dir" --data-output "$data_dir" > "$output_dir/runner.log" 2>&1
