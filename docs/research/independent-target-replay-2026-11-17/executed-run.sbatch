#!/bin/bash
#SBATCH --job-name=target-shape-review
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=00:10:00
#SBATCH --output=target-shape-review-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
.venv-qwen/bin/python -u research/target-shape-review-20261117-DVjVRt/diagnose.py --output research/target-shape-review-20261117-DVjVRt/output
