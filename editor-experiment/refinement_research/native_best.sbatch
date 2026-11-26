#!/bin/bash
#SBATCH --job-name=ar-native-best
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --nodelist=tron50
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=01:20:00
#SBATCH --output=ar-native-best-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
project_dir="$PWD"
source_dir="$project_dir/research/native-best-20261125-source"
run_dir="$project_dir/research/native-best-$SLURM_JOB_ID"
output_dir="$run_dir/benchmark"
install_dir=/scratch0/fpajak-vllm034-Ry21jsII
python_bin="$install_dir/venv/bin/python"
mkdir -p "$run_dir" "$install_dir/tmp" "$install_dir/cache" "$install_dir/inductor"
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit-status-$SLURM_JOB_ID.txt"; if [[ -n "${monitor_pid:-}" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi' EXIT
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$source_dir:$source_dir/replay_diagnostic"
export VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export TMPDIR="$install_dir/tmp" XDG_CACHE_HOME="$install_dir/cache"
export TRITON_CACHE_DIR="$install_dir/triton" TORCHINDUCTOR_CACHE_DIR="$install_dir/inductor"
nvidia-smi > "$run_dir/nvidia-smi.txt"
printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$run_dir/cuda-visible-devices.txt"
nvidia-smi --query-gpu=timestamp,index,uuid,name,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used --format=csv -l 10 > "$run_dir/gpu-telemetry.csv" &
monitor_pid=$!
"$python_bin" -m pip freeze > "$run_dir/packages.txt"
cmp "$run_dir/packages.txt" "$project_dir/research/one-block-fit-8453776/packages.txt"
PYTHONPATH="$PYTHONPATH:$project_dir/research/copy-action-fit-20261120-r2/test-support" \
 "$python_bin" -m pytest -q "$source_dir/refinement_research/test_native_best.py" "$source_dir/refinement_research/test_native_speed.py" > "$run_dir/tests.txt"
sha256sum "$project_dir/research/full-corpus-run-20261117/training/draft/best.pt" \
 "$project_dir/research/frozen-gates-20261125/linear/best.pt" > "$run_dir/checkpoint-hashes.txt"
timeout 4700 "$python_bin" -u -m refinement_research.native_best --project "$project_dir" --output "$output_dir"
