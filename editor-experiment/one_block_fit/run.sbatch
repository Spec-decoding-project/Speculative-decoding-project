#!/bin/bash
#SBATCH --job-name=ar-one-block
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --nodelist=tron61
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=00:30:00
#SBATCH --output=ar-one-block-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from Nexus project root}"
project_dir="$PWD"
install_dir=/scratch0/fpajak-vllm034-Ry21jsII
source_dir="$project_dir/research/one-block-fit-20261117"
output_dir="$project_dir/research/one-block-fit-$SLURM_JOB_ID"
mkdir "$output_dir"
trap 'status=$?; printf "%s\n" "$status" > "$output_dir/exit-status.txt"' EXIT
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TMPDIR="$install_dir/tmp" XDG_CACHE_HOME="$install_dir/cache"
export TRITON_CACHE_DIR="$install_dir/triton" TORCHINDUCTOR_CACHE_DIR="$install_dir/inductor"
export PYTHONPATH="$source_dir:$project_dir/research/editor-feature-bank:$project_dir/research/editor-feature-bank/replay_diagnostic"
nvidia-smi > "$output_dir/nvidia-smi.txt"
"$install_dir/venv/bin/python" -m pip check > "$output_dir/pip-check.txt"
"$install_dir/venv/bin/python" -m pip freeze > "$output_dir/packages.txt"
cmp "$output_dir/packages.txt" "$project_dir/research/editor-feature-bank/vllm-canonical-8453328/packages.txt"
"$install_dir/venv/bin/python" -u -m one_block_fit.collect \
  --project "$project_dir" --output "$output_dir/collection" > "$output_dir/collect.log" 2>&1
VLLM_BATCH_INVARIANT=0 "$install_dir/venv/bin/python" -u -m one_block_fit.train \
  --collection "$output_dir/collection" --output "$output_dir/training" > "$output_dir/train.log" 2>&1
