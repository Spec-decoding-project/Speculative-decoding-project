#!/bin/bash
#SBATCH --job-name=ar-whole-draft-audit
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --nodelist=tron61
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=ar-whole-draft-audit-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
run_dir="$PWD/research/whole-draft-8600245"
source_dir="$PWD/research/whole-draft-post-20261127"
python_bin=/scratch0/fpajak-vllm034-Ry21jsII/venv/bin/python
export OMP_NUM_THREADS=2
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/audit-exit-status.txt"' EXIT
"$python_bin" "$source_dir/whole_draft_audit.py" --output "$run_dir/output" \
 --bank "$PWD/research/full-corpus-run-20261117/bank.sqlite" > "$run_dir/audit-stdout.json"
"$python_bin" "$source_dir/whole_draft_export.py" "$run_dir"
