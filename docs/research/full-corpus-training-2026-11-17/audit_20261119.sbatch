#!/bin/bash
# Submit with --dependency=afterok:<resume-job>. No GPU inference or training.
#SBATCH --job-name=ar-results-audit
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --nodelist=tron61
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/fs/classhomes/fpajak/speculative-decoding-project/research/full-corpus-results-audit-20261119/job-%j.log
set -euo pipefail
project=/fs/classhomes/fpajak/speculative-decoding-project
audit_dir="$project/research/full-corpus-results-audit-20261119"
output_dir="$project/research/full-corpus-run-20261117"
python_bin=/scratch0/fpajak-vllm034-Ry21jsII/venv/bin/python
trap 'status=$?; printf "%s\n" "$status" > "$audit_dir/exit-status-$SLURM_JOB_ID.txt"' EXIT
for mode in context draft; do
  "$python_bin" "$audit_dir/audit_results.py" --run "$output_dir" --mode "$mode" > "$audit_dir/$mode.json.tmp"
  mv "$audit_dir/$mode.json.tmp" "$audit_dir/$mode.json"
done
cmp "$output_dir/development-draft-before-recovery-8454498.json" "$output_dir/training/draft/development-epoch-1.json"
"$python_bin" - "$project" "$audit_dir" <<'PY'
from pathlib import Path
import hashlib,json,os,sys
from datetime import datetime,timezone
project,audit=map(Path,sys.argv[1:])
root=project/'research/full-corpus-run-20261117'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()
completion=json.loads((root/'complete.json').read_text())
assert completion['result_sha256']==sha(root/'result.json')
original=json.loads((project/'research/full-corpus-preflight-8454304/summary.json').read_text())['source_hashes']
for name,expected in original.items():
    assert sha(project/'data/qwen_responses_50k'/name)==expected,name
for mode in ('context','draft'):
    assert json.loads((audit/f'{mode}.json').read_text())['passed']
result=dict(passed=True,completed_at=datetime.now(timezone.utc).isoformat(),
    job_id=os.environ['SLURM_JOB_ID'],source_files_unchanged=len(original),
    resumed_first_epoch_development_identical=True,
    audit_source_sha256=sha(audit/'audit_results.py'),
    audits={mode:sha(audit/f'{mode}.json') for mode in ('context','draft')},
    result_sha256=sha(root/'result.json'))
(audit/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
PY
