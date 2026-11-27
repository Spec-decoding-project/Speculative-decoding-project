#!/bin/bash
#SBATCH --job-name=ar-whole-draft
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --nodelist=tron61
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=02:00:00
#SBATCH --output=ar-whole-draft-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
project_dir="$PWD"
source_dir="$project_dir/research/whole-draft-20261127-source"
run_dir="$project_dir/research/whole-draft-$SLURM_JOB_ID"
checkpoint_dir="/scratch0/fpajak-whole-draft-$SLURM_JOB_ID"
install_dir=/scratch0/fpajak-vllm034-Ry21jsII
python_bin="$install_dir/venv/bin/python"
mkdir -p "$run_dir" "$checkpoint_dir"
ln -s "$checkpoint_dir" "$run_dir/checkpoints"
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit-status.txt"' EXIT
export OMP_NUM_THREADS=4 PYTHONPATH="$source_dir" VLLM_BATCH_INVARIANT=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 DO_NOT_TRACK=1
export TMPDIR="$install_dir/tmp" XDG_CACHE_HOME="$install_dir/cache"
nvidia-smi > "$run_dir/nvidia-smi.txt"
"$python_bin" -m pip freeze > "$run_dir/packages.txt"
cmp "$run_dir/packages.txt" "$project_dir/research/one-block-fit-8453776/packages.txt"
"$python_bin" -m pip check > "$run_dir/pip-check.txt"
PYTHONPATH="$source_dir:$project_dir/research/copy-action-fit-20261120-r2/test-support" \
 "$python_bin" -m pytest -q "$source_dir/refinement_research/test_whole_draft.py" \
 "$source_dir/refinement_research/test_gate.py" "$source_dir/refinement_research/test_gate_training.py" > "$run_dir/tests.txt"
"$python_bin" -u -m refinement_research.whole_draft_fit \
 --bank "$project_dir/research/full-corpus-run-20261117/bank.sqlite" \
 --initial "$project_dir/research/full-corpus-run-20261117/training/draft/best.pt" \
 --incumbent-gate "$project_dir/research/frozen-gates-20261125/linear/best.pt" \
 --output "$run_dir/output" --checkpoints "$checkpoint_dir"
