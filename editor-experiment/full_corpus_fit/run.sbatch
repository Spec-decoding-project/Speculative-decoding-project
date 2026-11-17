#!/bin/bash
#SBATCH --job-name=ar-full-corpus
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --nodelist=tron61
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=24:00:00
#SBATCH --output=ar-full-corpus-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
project_dir="$PWD"
install_dir=/scratch0/fpajak-vllm034-Ry21jsII
source_dir="$project_dir/research/full-corpus-fit-20261117-run"
output_dir="$project_dir/research/full-corpus-run-20261117"
preflight_dir="$project_dir/research/full-corpus-preflight-8454304"
calibration_file="${AR_FULL_CALIBRATION:?Set the successful calibration.json path}"
mkdir -p "$output_dir"
trap 'status=$?; printf "%s\n" "$status" > "$output_dir/exit-status-$SLURM_JOB_ID.txt"' EXIT
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TMPDIR="$install_dir/tmp" XDG_CACHE_HOME="$install_dir/cache"
export TRITON_CACHE_DIR="$install_dir/triton" TORCHINDUCTOR_CACHE_DIR="$install_dir/inductor"
export PYTHONPATH="$source_dir:$project_dir/research/editor-feature-bank:$project_dir/research/editor-feature-bank/replay_diagnostic"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
python_bin="$install_dir/venv/bin/python"
nvidia-smi > "$output_dir/nvidia-smi-$SLURM_JOB_ID.txt"
"$python_bin" -m pip check > "$output_dir/pip-check-$SLURM_JOB_ID.txt"
"$python_bin" -m pip freeze > "$output_dir/packages-$SLURM_JOB_ID.txt"
cmp "$output_dir/packages-$SLURM_JOB_ID.txt" "$project_dir/research/one-block-fit-8453776/packages.txt"
"$python_bin" -u -m full_corpus_fit.pipeline --project "$project_dir" --preflight "$preflight_dir" \
  --calibration "$calibration_file" --output "$output_dir"
