#!/bin/bash
#SBATCH --job-name=qwen-setup
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=qwen-setup-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export HF_HOME="$PWD/data/hf-cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p .context
if [ ! -x .context/qwen-bootstrap/bin/python ]; then
    python3 -m venv .context/qwen-bootstrap
fi
.context/qwen-bootstrap/bin/python -m pip install --no-cache-dir 'uv==0.9.7'
QWEN_UV="$PWD/.context/qwen-bootstrap/bin/uv"
if [ ! -x .venv-qwen/bin/python ]; then
    "$QWEN_UV" venv --python 3.12 .venv-qwen
fi
"$QWEN_UV" pip install --no-cache --python .venv-qwen/bin/python \
    --index-url https://download.pytorch.org/whl/cu126 'torch==2.13.0'
"$QWEN_UV" pip install --no-cache --python .venv-qwen/bin/python \
    'transformers==5.15.0' 'accelerate==1.12.0' 'huggingface-hub==1.23.0' 'pyarrow==21.0.0'
.venv-qwen/bin/python - <<'PY'
import json
from pathlib import Path
import torch
import transformers
from huggingface_hub import snapshot_download

config = json.loads(Path('configs/qwen_generation.json').read_text())
print('torch', torch.__version__, 'transformers', transformers.__version__, flush=True)
snapshot_download(
    config['model_id'], revision=config['revision'],
    local_dir='data/models/qwen3-8b',
    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'], max_workers=2,
)
Path('data/models/qwen3-8b/pinned_revision.json').write_text(json.dumps({
    'model_id': config['model_id'], 'revision': config['revision'],
}) + '\n')
print('Qwen model download complete', flush=True)
PY
