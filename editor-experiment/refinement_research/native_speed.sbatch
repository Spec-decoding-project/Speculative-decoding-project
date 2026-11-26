#!/bin/bash
#SBATCH --job-name=ar-native-speed
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --nodelist=tron59
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=00:45:00
#SBATCH --output=ar-native-speed-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
project_dir="$PWD"
source_dir="$project_dir/research/native-speed-20261125-source"
output_dir="$project_dir/research/native-speed-$SLURM_JOB_ID"
install_dir=/scratch0/fpajak-vllm034-Ry21jsII
python_bin="$install_dir/venv/bin/python"
mkdir -p "$output_dir" "$install_dir/tmp" "$install_dir/cache" "$install_dir/inductor"
trap 'status=$?; printf "%s\n" "$status" > "$output_dir/exit-status-$SLURM_JOB_ID.txt"' EXIT
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$source_dir:$source_dir/replay_diagnostic"
export VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export TMPDIR="$install_dir/tmp" XDG_CACHE_HOME="$install_dir/cache"
export TRITON_CACHE_DIR="$install_dir/triton" TORCHINDUCTOR_CACHE_DIR="$install_dir/inductor"
nvidia-smi > "$output_dir/nvidia-smi-$SLURM_JOB_ID.txt"
"$python_bin" -m pip check > "$output_dir/pip-check-$SLURM_JOB_ID.txt"
"$python_bin" -m pip freeze > "$output_dir/packages-$SLURM_JOB_ID.txt"
cmp "$output_dir/packages-$SLURM_JOB_ID.txt" "$project_dir/research/one-block-fit-8453776/packages.txt"
PYTHONPATH="$PYTHONPATH:$project_dir/research/copy-action-fit-20261120-r2/test-support" \
 "$python_bin" -m pytest -q "$source_dir/refinement_research/test_native_speed.py" > "$output_dir/tests.txt"
"$python_bin" -u -m refinement_research.native_speed --project "$project_dir" --output "$output_dir"
