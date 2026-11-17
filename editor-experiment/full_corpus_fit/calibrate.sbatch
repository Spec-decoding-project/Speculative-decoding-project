#!/bin/bash
#SBATCH --job-name=ar-full-calibrate
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --nodelist=tron61
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=00:40:00
#SBATCH --output=ar-full-calibrate-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
project_dir="$PWD"
install_dir=/scratch0/fpajak-vllm034-Ry21jsII
source_dir="$project_dir/research/full-corpus-fit-20261117"
output_dir="$project_dir/research/full-corpus-calibration-$SLURM_JOB_ID"
preflight_dir="$project_dir/research/full-corpus-preflight-8454304"
mkdir "$output_dir"
trap 'status=$?; printf "%s\n" "$status" > "$output_dir/exit-status.txt"' EXIT
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TMPDIR="$install_dir/tmp" XDG_CACHE_HOME="$install_dir/cache"
export TRITON_CACHE_DIR="$install_dir/triton" TORCHINDUCTOR_CACHE_DIR="$install_dir/inductor"
export PYTHONPATH="$source_dir:$project_dir/research/editor-feature-bank:$project_dir/research/editor-feature-bank/replay_diagnostic"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
python_bin="$install_dir/venv/bin/python"
nvidia-smi > "$output_dir/nvidia-smi.txt"
"$python_bin" -m pip check > "$output_dir/pip-check.txt"
"$python_bin" -m pip freeze > "$output_dir/packages.txt"
cmp "$output_dir/packages.txt" "$project_dir/research/one-block-fit-8453776/packages.txt"
for mode in native greedy; do
  extra=()
  if [ "$mode" = greedy ]; then extra=(--greedy); fi
  VLLM_BATCH_INVARIANT=1 "$python_bin" -u -m full_corpus_fit.collect --profile "${extra[@]}" \
    --project "$project_dir" --preflight "$preflight_dir" --output "$output_dir/$mode" > "$output_dir/$mode.log" 2>&1
done
VLLM_BATCH_INVARIANT=0 "$python_bin" -m full_corpus_fit.bank \
  --preflight "$preflight_dir" --collection "$output_dir/native" --output "$output_dir/bank.sqlite" > "$output_dir/bank.log" 2>&1
VLLM_BATCH_INVARIANT=0 "$python_bin" -u -m full_corpus_fit.calibrate \
  --native "$output_dir/native" --greedy "$output_dir/greedy" \
  --previous "$project_dir/research/one-block-fit-8453776/collection" \
  --preflight "$preflight_dir" --bank "$output_dir/bank.sqlite" \
  --output "$output_dir/calibration.json" > "$output_dir/training-profile.log" 2>&1
