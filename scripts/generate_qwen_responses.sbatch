#!/bin/bash
#SBATCH --job-name=qwen-responses
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=03:00:00
#SBATCH --output=qwen-responses-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export HF_HOME="$PWD/data/hf-cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
QWEN_SHARD_ARGS=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    QWEN_SHARD_ARGS=(--num-shards "${QWEN_NUM_SHARDS:?Set the fixed total shard count for an array}" --shard-index "$SLURM_ARRAY_TASK_ID")
fi
.venv-qwen/bin/python -u scripts/generate_qwen_responses.py \
    --model-path data/models/qwen3-8b "${QWEN_SHARD_ARGS[@]}" "$@"
