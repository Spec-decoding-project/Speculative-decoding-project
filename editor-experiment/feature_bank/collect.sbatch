#!/bin/bash
#SBATCH --job-name=editor-features
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=01:00:00
#SBATCH --output=editor-features-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the Nexus project root}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
.venv-qwen/bin/python -u research/editor-feature-bank/feature_bank/collect.py \
  --target data/models/qwen3-8b --draft data/models/qwen3-8b-dflash-b16 \
  --pilot data/qwen_responses_pilot --main data/qwen_responses_50k "$@"
