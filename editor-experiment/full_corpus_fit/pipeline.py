"""Run the authorized full-data pipeline, with durable stage status and results."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from full_corpus_fit.preflight import save, sha


def now():
    return datetime.now(timezone.utc).isoformat()


def report(output):
    variants={m:json.loads((output/'training'/m/'complete.json').read_text()) for m in ('context','draft')}
    bank=json.loads((output/'bank.json').read_text())
    result=dict(completed_at=now(),variants=variants,bank=bank,
                scope='Full prompt inventory; up to four complete reachable training blocks per prompt; saved-context evaluation')
    save(output/'result.json',result)
    lines=['# Full-corpus autoregressive editor training results','',
        'Both variants completed three epochs. The table reports reserved final-test results after development-only checkpoint selection.','',
        '| Measurement | Context plus seed | Whole-draft conditioning |','|---|---:|---:|']
    fields=[('Selected epoch',lambda r:r['selected_epoch']),
        ('Training updates',lambda r:r['updates']),
        ('Original correct prefix',lambda r:r['test']['mean_original_progress']),
        ('Edited correct prefix',lambda r:r['test']['mean_edited_progress']),
        ('Union correct prefix',lambda r:r['test']['mean_union_progress']),
        ('Repaired test blocks',lambda r:r['test']['repaired']),
        ('Damaged test blocks',lambda r:r['test']['damaged']),
        ('Exact suffix fraction',lambda r:r['test']['free_running_exact_suffix']),
        ('Editable token accuracy',lambda r:r['test']['free_running_token_accuracy']),
        ('Editor latency, ms',lambda r:r['isolated_latency_ms'])]
    for name,getter in fields:
        lines.append(f'| {name} | {getter(variants["context"])} | {getter(variants["draft"])} |')
    lines+=['','All original source prompts were processed; actual training coverage and exclusion counts are recorded in bank.json.',
        'Original response files were not changed. Native target/draft parameters remained frozen.',
        'Correct-prefix coverage and isolated FP32 editor latency do not establish full-generation speedup.',
        'Per-source/preservation results, checkpoint hashes and raw predictions are in result.json and training/{context,draft}/.','']
    (output/'REPORT.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--preflight',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    config=dict(preflight_sha256=sha(args.preflight/'summary.json'),calibration_sha256=sha(args.calibration),
                source_manifest_sha256=sha(Path(__file__).resolve().parents[1]/'source-manifest.json'))
    fingerprint=args.output/'pipeline.json'
    if fingerprint.exists() and json.loads(fingerprint.read_text()) != config:
        raise ValueError('Refuse to resume the pipeline with changed inputs/source')
    save(fingerprint,config)
    if (args.output/'complete.json').exists():
        print('Full pipeline already complete',flush=True)
        return
    stages=[('collection',['full_corpus_fit.collect','--project',str(args.project),
        '--preflight',str(args.preflight),'--calibration',str(args.calibration),
        '--output',str(args.output/'collection')]),
        ('bank',['full_corpus_fit.bank','--preflight',str(args.preflight),
        '--collection',str(args.output/'collection'),'--output',str(args.output/'bank.sqlite')])]
    for mode in ('context','draft'):
        stages.append((mode,['full_corpus_fit.train','--bank',str(args.output/'bank.sqlite'),
                            '--output',str(args.output/'training'/mode),'--mode',mode]))
    start=time.monotonic()
    for name,arguments in stages:
        state=dict(phase=name,started_at=now(),job_id=os.environ.get('SLURM_JOB_ID'),
                   pipeline_seconds=time.monotonic()-start,command=[sys.executable,'-m',*arguments])
        save(args.output/'status.json',state)
        print(json.dumps(state),flush=True)
        env=dict(os.environ,VLLM_BATCH_INVARIANT='1' if name=='collection' else '0')
        with (args.output/(name+'.log')).open('a') as log:
            log.write('\nSTART '+now()+'\n');log.flush()
            completed=subprocess.run(state['command'],env=env,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            save(args.output/'status.json',dict(**state,failed_at=now(),returncode=completed.returncode))
            raise RuntimeError(f'Pipeline stopped at {name}; preserved outputs can be inspected/resumed')
    original=json.loads((args.preflight/'summary.json').read_text())['source_hashes']
    for name,expected in original.items():
        if sha(args.project/'data/qwen_responses_50k'/name) != expected:
            raise ValueError(f'Original response source changed during the run: {name}')
    save(args.output/'original-source-integrity.json',dict(passed=True,files_checked=len(original),checked_at=now()))
    report(args.output)
    completed=dict(phase='complete',completed_at=now(),job_id=os.environ.get('SLURM_JOB_ID'),
                   pipeline_seconds=time.monotonic()-start,result_sha256=sha(args.output/'result.json'))
    save(args.output/'complete.json',completed)
    save(args.output/'status.json',completed)
    print(json.dumps(completed),flush=True)


if __name__=='__main__':
    main()
