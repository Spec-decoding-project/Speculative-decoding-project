#!/bin/bash
#SBATCH --job-name=exact-replay
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=00:30:00
#SBATCH --output=exact-replay-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from Nexus project root}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
.venv-qwen/bin/python -u research/editor-feature-bank/replay_diagnostic/exact_session.py "$@"
