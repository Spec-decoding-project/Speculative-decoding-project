#!/bin/bash
#SBATCH --job-name=native-bank
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=01:00:00
#SBATCH --output=native-bank-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
bank_path="${1:?Supply a new output directory}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
.venv-qwen/bin/python -u research/editor-feature-bank/feature_bank/collect.py \
  --target data/models/qwen3-8b --draft data/models/qwen3-8b-dflash-b16 \
  --pilot data/qwen_responses_pilot --main data/qwen_responses_50k \
  --replay-contract native-cached-first-round-v1 --output "$bank_path"
.venv-qwen/bin/python research/editor-feature-bank/feature_bank/inspect_bank.py \
  "$bank_path" > "$bank_path/independent-inspection.json"
