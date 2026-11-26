#!/bin/bash
#SBATCH --job-name=ar-frozen-gates
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --nodelist=tron61
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=02:00:00
#SBATCH --output=ar-frozen-gates-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
project_dir="$PWD"
source_dir="$project_dir/research/frozen-gates-20261125-source"
output_dir="$project_dir/research/frozen-gates-20261125"
old_source="$project_dir/research/copy-action-fit-20261120-r2"
install_dir=/scratch0/fpajak-vllm034-Ry21jsII
python_bin="$install_dir/venv/bin/python"
mkdir -p "$output_dir"
trap 'status=$?; printf "%s\n" "$status" > "$output_dir/exit-status-$SLURM_JOB_ID.txt"' EXIT
export OMP_NUM_THREADS=4 PYTHONPATH="$source_dir" VLLM_BATCH_INVARIANT=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 DO_NOT_TRACK=1
export TMPDIR="$install_dir/tmp" XDG_CACHE_HOME="$install_dir/cache"
"$python_bin" -c 'import json,pathlib,time,sys; p=pathlib.Path(sys.argv[1]); p.exists() or p.write_text(json.dumps({"started_epoch":time.time(),"maximum_seconds":7200}))' "$output_dir/budget.json"
nvidia-smi > "$output_dir/nvidia-smi-$SLURM_JOB_ID.txt"
"$python_bin" -m pip check > "$output_dir/pip-check-$SLURM_JOB_ID.txt"
"$python_bin" -m pip freeze > "$output_dir/packages-$SLURM_JOB_ID.txt"
cmp "$output_dir/packages-$SLURM_JOB_ID.txt" "$project_dir/research/one-block-fit-8453776/packages.txt"
PYTHONPATH="$source_dir:$old_source/test-support" "$python_bin" -m pytest -q \
    "$source_dir/refinement_research/test_gate.py" "$source_dir/refinement_research/test_gate_training.py" \
    > "$output_dir/tests-$SLURM_JOB_ID.txt"
"$python_bin" -u -m refinement_research.train_gates \
    --bank "$project_dir/research/full-corpus-run-20261117/bank.sqlite" \
    --initial "$project_dir/research/full-corpus-run-20261117/training/draft/best.pt" --output "$output_dir"
