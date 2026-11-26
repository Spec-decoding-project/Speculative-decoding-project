#!/bin/bash
#SBATCH --job-name=ar-copy-diagnosis
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --nodelist=tron61
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=00:15:00
#SBATCH --output=ar-copy-diagnosis-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
project_dir="$PWD"
source_dir="$project_dir/research/refinement-diagnostic-20261125-source"
output_dir="$project_dir/research/refinement-diagnostic-20261125"
old_source="$project_dir/research/copy-action-fit-20261120-r2"
install_dir=/scratch0/fpajak-vllm034-Ry21jsII
python_bin="$install_dir/venv/bin/python"
preflight_dir="$project_dir/research/refinement-diagnostic-20261125-preflight"
mkdir -p "$preflight_dir"
trap 'status=$?; printf "%s\n" "$status" > "$preflight_dir/exit-status-$SLURM_JOB_ID.txt"' EXIT
export OMP_NUM_THREADS=4 PYTHONPATH="$source_dir" VLLM_BATCH_INVARIANT=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 DO_NOT_TRACK=1
export TMPDIR="$install_dir/tmp" XDG_CACHE_HOME="$install_dir/cache"
nvidia-smi > "$preflight_dir/nvidia-smi.txt"
"$python_bin" -m pip freeze > "$preflight_dir/packages.txt"
cmp "$preflight_dir/packages.txt" "$project_dir/research/one-block-fit-8453776/packages.txt"
PYTHONPATH="$source_dir:$old_source/test-support" "$python_bin" -m pytest -q \
    "$source_dir/refinement_research/test_diagnostic.py" > "$preflight_dir/tests.txt"
"$python_bin" -u -m refinement_research.diagnose_copy \
    --bank "$project_dir/research/full-corpus-run-20261117/bank.sqlite" \
    --initial "$project_dir/research/full-corpus-run-20261117/training/draft/best.pt" \
    --pilot "$project_dir/research/copy-action-run-20261120" --output "$output_dir"
