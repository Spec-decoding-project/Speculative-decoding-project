#!/bin/bash
#SBATCH --job-name=target-replay
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=00:30:00
#SBATCH --output=target-replay-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
.venv-qwen/bin/python -u research/editor-feature-bank/replay_diagnostic/session.py \
  --target data/models/qwen3-8b --pilot data/qwen_responses_pilot \
  --bank data/editor_feature_banks/bank-20261110 \
  --generator scripts/generate_qwen_responses.py "$@"
