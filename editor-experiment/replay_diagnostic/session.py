"""Three-fixture target-only replay diagnosis; never trains or changes labels.

Run with SLURM. Baseline exits 1 on the original equality failure unless
--wait-for-probes is selected. That mode keeps the frozen model in the same
allocation for a subsequently written probes.py + probes.ready pair, then exits
with the baseline verdict. The original source/data stay untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'feature_bank'))
from collect import loading_ok, target_fingerprints, write_json
from data import block_inputs, records
from prepare import sha256


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--generator', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-for-probes', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('GPU allocation required')
    run = json.loads((args.pilot / 'run.json').read_text())
    if sha256(args.generator) != run['generator_sha256']:
        raise ValueError('Original response generator fingerprint mismatch')
    manifest = json.loads((args.bank / 'manifest.partial.json').read_text())
    if len(manifest) != 2:
        raise ValueError('Expected two original bank controls')
    audit = json.loads((args.bank / 'audit.json').read_text())
    failed = audit['active_fixture']
    selected = {x['id']: x for x in manifest + [failed]}
    fixtures = []
    for row in records(args.pilot):
        if row['id'] not in selected:
            continue
        metadata = selected[row['id']]
        if 'record_sha256' in metadata and metadata['record_sha256'] != row['record_sha256']:
            raise ValueError('Control source record changed')
        prefix, _, labels = block_inputs(row, metadata['anchor'], 151669)
        if metadata.get('labels', labels) != labels:
            raise ValueError('Control labels changed')
        fixtures.append(dict(row=row, prefix=prefix, labels=labels, anchor=metadata['anchor']))
    if len(fixtures) != 3:
        raise ValueError('Missing fixed diagnostic fixture')
    fixtures.sort(key=lambda f: f['row']['id'])
    report = dict(
        schema=1, purpose='target replay diagnostic; no training',
        source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
        source_run=run, source_generator_sha256=sha256(args.generator),
        source_bank_audit_sha256=sha256(args.bank / 'audit.json'),
        target_files=target_fingerprints(args.target),
        runtime=dict(python=platform.python_version(), torch=torch.__version__,
                     transformers=transformers.__version__, cuda=torch.version.cuda,
                     device=torch.cuda.get_device_name(), hostname=platform.node(),
                     job_id=os.environ.get('SLURM_JOB_ID'), seed=0,
                     attention='sdpa', dtype='bfloat16', allow_tf32=False),
        baseline=[])
    write_json(args.output / 'setup.json', report)
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    target, info = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, device_map='cuda:0',
        attn_implementation='sdpa', local_files_only=True, output_loading_info=True)
    loading_ok(info)
    target.eval().requires_grad_(False)
    report['target_config'] = target.config.to_dict()
    report['generation_config'] = target.generation_config.to_dict()
    report['runtime']['device_total_bytes'] = torch.cuda.get_device_properties(0).total_memory
    started = time.monotonic()
    with torch.inference_mode():
        for fixture in fixtures:
            row, prefix, labels = fixture['row'], fixture['prefix'], fixture['labels']
            rendered = tokenizer.apply_chat_template(row['messages'], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            if tokenizer.encode(rendered, add_special_tokens=False) != row['prompt_token_ids']:
                raise ValueError('Prompt tokenization mismatch')
            if hashlib.sha256(rendered.encode()).hexdigest() != row['rendered_prompt_sha256']:
                raise ValueError('Rendered prompt hash mismatch')
            item = dict(id=row['id'], record_sha256=row['record_sha256'],
                        anchor=fixture['anchor'], prompt_length=len(row['prompt_token_ids']),
                        prefix_sha256=digest_ids(prefix), label_sha256=digest_ids(labels),
                        input_length=len(prefix)+15, readout_rows=16,
                        cache=False, implicit_positions=list(range(len(prefix)+15)), repeats=[])
            for repeat in range(3):
                anchor_output = target(torch.tensor([prefix], device='cuda'), use_cache=False,
                                       output_hidden_states=True, logits_to_keep=1)
                anchor_actual = anchor_output.logits.argmax(-1).item()
                del anchor_output
                output = target(torch.tensor([prefix + labels[:-1]], device='cuda'),
                                use_cache=False, logits_to_keep=16)
                logits = output.logits[0]
                actual = logits.argmax(-1).tolist()
                mismatch = [dict(position=i, expected=a, actual=b)
                            for i, (a, b) in enumerate(zip(labels, actual)) if a != b]
                item['repeats'].append(dict(repeat=repeat, anchor_equal=anchor_actual==labels[0],
                    anchor_actual=anchor_actual, matches=16-len(mismatch), mismatches=mismatch,
                    logits_dtype=str(logits.dtype)))
                print(json.dumps(dict(id=row['id'], **item['repeats'][-1])), flush=True)
                del output, logits
            report['baseline'].append(item)
    torch.cuda.synchronize()
    report['baseline_seconds'] = time.monotonic() - started
    report['baseline_pass'] = all(r['anchor_equal'] and r['matches']==16
                                 for f in report['baseline'] for r in f['repeats'])
    write_json(args.output / 'baseline.json', report)
    print('BASELINE_PASS=' + str(report['baseline_pass']), flush=True)
    if args.wait_for_probes:
        print('Waiting at most 20 minutes for the bounded probe script', flush=True)
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline and not (args.output / 'probes.ready').exists():
            time.sleep(2)
        if (args.output / 'probes.ready').exists():
            probe = args.output / 'probes.py'
            report['probe_source_sha256'] = sha256(probe)
            with torch.inference_mode():
                runpy.run_path(str(probe), init_globals=dict(
                    target=target, tokenizer=tokenizer, fixtures=fixtures, report=report,
                    args=args, torch=torch, write_json=write_json,
                    forward_budget_seconds=max(0, 600-report['baseline_seconds'])))
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    write_json(args.output / 'result.json', report)
    raise SystemExit(0 if report['baseline_pass'] else 1)


if __name__ == '__main__':
    main()
