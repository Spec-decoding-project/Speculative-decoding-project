#!/bin/bash
#SBATCH --job-name=prompt-data
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=prompt-data-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
"${DATA_PYTHON:-python3}" scripts/collect_prompt_data.py "$@"
