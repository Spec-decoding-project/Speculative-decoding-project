"""Supported invariant validation, followed by the authorized four-prompt pilot.

The new data directory is created only after full greedy/DFlash agreement and
EOS completion. All engine processes share 4800 target calls / 600 target seconds.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from audit_vllm_probe import audit_trial, token_hash
from vllm_probe import prepare, sha256, first_difference, compare
from vllm_probe_worker import save


def cell_name(invariant, dflash, token_budget, pilot):
    return f'bi{invariant}-dflash{dflash}-budget{token_budget}-pilot{pilot}'


def child(args):
    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams
    if vllm.__version__ != '0.34.0':
        raise RuntimeError('Unreviewed vLLM version')
    torch.set_num_threads(4)
    fixtures = json.loads((args.output / 'inputs.json').read_text())['fixtures']
    by_id = {f['id']: f for f in fixtures}
    # Keep the same attention/KV contract in validation and full-response runs.
    maximum = 2304
    config = dict(model=str(args.project / 'data/models/qwen3-8b'), dtype='bfloat16',
        tensor_parallel_size=1, max_model_len=maximum, max_num_seqs=4,
        max_num_batched_tokens=args.token_budget,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        enforce_eager=True, async_scheduling=False, seed=0,
        kv_cache_memory_bytes=2048 * 1024**2,
        attention_config={'backend': 'FLASH_ATTN'},
        worker_extension_cls='vllm_contract_worker.ContractWorker',
        generation_config='vllm', disable_log_stats=False)
    if args.dflash:
        config['speculative_config'] = dict(method='dflash',
            model=str(args.project / 'data/models/qwen3-8b-dflash-b16'),
            num_speculative_tokens=15, max_model_len=maximum)
    cell = cell_name(args.invariant, args.dflash, args.token_budget, args.pilot)
    result = dict(cell=cell, config=config, complete=False, trials=[],
        runtime=dict(vllm=vllm.__version__, torch=torch.__version__, cuda=torch.version.cuda,
            transformers=transformers.__version__, python=platform.python_version(),
            gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
            host=platform.node(), job_id=os.environ.get('SLURM_JOB_ID')))
    path = args.output / f'{cell}.json'
    try:
        llm = LLM(**config)
        startup = llm.collective_rpc('probe_finish')[0]
        result['startup'] = startup
        if (startup['model_class'] != 'Qwen3ForCausalLM'
                or startup['invariant_active'] != bool(args.invariant)
                or not startup['attention']
                or any(x['batch_invariant'] != bool(args.invariant)
                       or x['cls'] != 'FlashAttentionImpl' for x in startup['attention'])):
            raise RuntimeError('Canonical target/attention activation not confirmed')
        sampling = SamplingParams(temperature=0, max_tokens=2048 if args.pilot else 16,
                                  ignore_eos=not bool(args.pilot), logprobs=5, seed=0)
        result['sampling'] = dict(temperature=0, max_tokens=sampling.max_tokens,
                                 ignore_eos=sampling.ignore_eos, seed=0, logprobs=5)
        plan = [('full', fixtures)] if args.pilot else [
            ('repeat0', fixtures), ('repeat1', list(reversed(fixtures))), ('repeat2', fixtures)]
        if args.invariant and args.token_budget == 256 and not args.pilot:
            plan.append(('alone', fixtures[:1]))
        for name, batch in plan:
            before = llm.collective_rpc('probe_begin', args=(name, True))[0]
            started = time.monotonic()
            outputs = llm.generate([{'prompt_token_ids': f['prompt']} for f in batch],
                                   sampling, use_tqdm=False)
            wall = time.monotonic() - started
            if len(outputs) != len(batch):
                raise RuntimeError('Missing generated fixture')
            observed = llm.collective_rpc('probe_finish')[0]
            tokens, details = {}, {}
            for fixture, output in zip(batch, outputs):
                completion = output.outputs[0]
                ids = list(completion.token_ids)
                if output.prompt_token_ids != fixture['prompt'] or not ids:
                    raise RuntimeError('Misaligned/empty generated output')
                if not args.pilot and len(ids) != 16:
                    raise RuntimeError('Incomplete diagnostic window')
                tokens[fixture['id']] = ids
                details[fixture['id']] = dict(request_id=output.request_id,
                    finish_reason=completion.finish_reason, stop_reason=completion.stop_reason,
                    text=completion.text,
                    historical_first_block_difference=first_difference(fixture['labels'], ids[:16]))
            trial = dict(name=name, tokens=tokens, details=details,
                trace=observed['trace'], timings=observed['timings'], wall_seconds=wall,
                target_calls=observed['budget']['calls']-before['calls'],
                target_seconds=observed['budget']['seconds']-before['seconds'],
                peak_allocated_bytes=observed['peak_allocated_bytes'])
            trial['alignment_audit'] = audit_trial(by_id, trial)
            result['trials'].append(trial)
            save(path, result)
            print(json.dumps(dict(cell=cell, trial=name, target_calls=trial['target_calls'],
                token_counts={k: len(v) for k, v in tokens.items()})), flush=True)
        result['metrics'] = [dataclasses.asdict(m) if dataclasses.is_dataclass(m) else repr(m)
                             for m in llm.get_metrics()]
        result['complete'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        save(path, result)


def execute(args, invariant, dflash, token_budget, pilot=0):
    name = cell_name(invariant, dflash, token_budget, pilot)
    path = args.output / f'{name}.json'
    if path.exists():
        raise RuntimeError(f'Refusing to overwrite a prior cell: {name}')
    env = dict(os.environ, VLLM_BATCH_INVARIANT=str(invariant),
        VLLM_PROBE_BUDGET=str(args.output / 'budget.json'),
        VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_USE_FLASHINFER_SAMPLER='0',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1')
    command = [sys.executable, str(Path(__file__).resolve()), '--child',
        '--project', str(args.project), '--output', str(args.output),
        '--invariant', str(invariant), '--dflash', str(dflash),
        '--token-budget', str(token_budget), '--pilot', str(pilot)]
    with (args.output / f'{name}.log').open('w') as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'{name} failed; see partial result/log; no automatic retry')
    return json.loads(path.read_text())


def require_exact(comparison, expected_rows):
    if (any(v is not None for v in comparison['token_differences'].values())
            or comparison['matching_prefix_rows'] < expected_rows
            or comparison['unequal_full_logit_rows']):
        raise ValueError('Exact canonical comparison failed')


def publish(args, inputs, greedy, verified, comparison):
    # Imports are CPU-only; prepare() already added the original bank helper path.
    from data import digest, records
    originals = {r['id']: r for r in records(args.project / 'data/qwen_responses_pilot')
                 if r['id'] in greedy['tokens']}
    if len(originals) != 4 or args.data_output.exists():
        raise ValueError('Missing originals or existing publication destination')
    for name in originals:
        if (greedy['details'][name]['finish_reason'] != 'stop'
                or verified['details'][name]['finish_reason'] != 'stop'):
            raise ValueError('Capped/non-EOS response: pilot cannot pass')
    run = dict(schema_version=2, stage='canonical_target_pilot',
        canonical_contract='qwen3-8b-vllm034-invariant-20261117',
        status='verified_pilot_not_a_training_bank',
        target_revision=inputs['draft_pin']['target_revision'],
        draft_revision=inputs['draft_pin']['draft_revision'],
        source_hashes=inputs['source_hashes'],
        greedy_evidence_sha256=sha256(args.output / f'{cell_name(1,0,256,1)}.json'),
        verification_evidence_sha256=sha256(args.output / f'{cell_name(1,1,256,1)}.json'),
        source_run_file_sha256=sha256(args.project / 'data/qwen_responses_pilot/run.json'),
        checkpoint_hashes=inputs['target_hashes'], max_new_tokens=2048,
        batch_invariant=True, chunked_prefill=True, max_model_len=2304,
        runtime=json.loads((args.output / f'{cell_name(1,0,256,1)}.json').read_text())['runtime'],
        packages_sha256=sha256(args.output / 'packages.txt'),
        job_id=os.environ.get('SLURM_JOB_ID'))
    result_rows, changes = [], []
    for name, original in originals.items():
        tokens = greedy['tokens'][name]
        row = dict(id=name, source=original['source'], dataset_split=original['dataset_split'],
            messages=original['messages'], prompt_token_ids=original['prompt_token_ids'],
            response_token_ids=tokens, response_text=greedy['details'][name]['text'],
            status='verified_pilot', finish_reason='eos',
            original_record_sha256=original['record_sha256'], run_sha256=digest(run))
        row['record_sha256'] = digest(row)
        result_rows.append(row)
        changes.append(dict(id=name, old_length=len(original['response_token_ids']),
            new_length=len(tokens), first_difference=first_difference(original['response_token_ids'], tokens),
            old_response_sha256=token_hash(original['response_token_ids']),
            new_response_sha256=token_hash(tokens), record_sha256=row['record_sha256']))
    staging = args.data_output.with_name(args.data_output.name + '.staging')
    staging.mkdir(parents=True, exist_ok=False)
    save(staging / 'run.json', run)
    journal = staging / 'shard-00000.jsonl'
    journal.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in result_rows))
    validation = dict(passed=True, count=4, responses=changes,
        shared_full_logit_rows=comparison['matching_prefix_rows'],
        journal_sha256=sha256(journal), run_file_sha256=sha256(staging/'run.json'),
        run_content_sha256=digest(run),
        training_authorized=False, feature_bank_complete=False)
    save(staging/'verification.json', validation)
    # Independent journal checksum/run audit using the existing reader.
    if len(list(records(staging))) != 4:
        raise ValueError('Published journal integrity failure')
    staging.rename(args.data_output)
    return dict(path=str(args.data_output), **validation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-output', type=Path)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--invariant', type=int, choices=(0,1), default=1)
    parser.add_argument('--dflash', type=int, choices=(0,1), default=0)
    parser.add_argument('--token-budget', type=int, choices=(64,256), default=256)
    parser.add_argument('--pilot', type=int, choices=(0,1), default=0)
    args = parser.parse_args()
    args.project, args.output = args.project.resolve(), args.output.resolve()
    if args.child:
        child(args)
        return
    if args.data_output is None:
        raise ValueError('An explicit new versioned data destination is required')
    args.data_output = args.data_output.resolve()
    if args.data_output.exists() or (args.output/'budget.json').exists():
        raise ValueError('Refusing to overwrite data or reset a prior budget')
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = prepare(args.project)
    names = ('vllm_contract.py','vllm_contract_worker.py','audit_vllm_probe.py',
             'vllm_probe.py','vllm_probe_worker.py','vllm_contract.sbatch')
    inputs['source_hashes'] = {name: sha256(Path(__file__).with_name(name)) for name in names}
    save(args.output/'inputs.json', inputs)
    save(args.output/'budget.json', dict(calls=0, seconds=0.0, max_calls=4800, max_seconds=600))
    cells, comparisons = {}, {}
    for invariant, dflash, tokens in [(1,0,256),(1,1,256),(1,0,64),(1,1,64),(0,0,256),(0,1,256)]:
        cells[(invariant,dflash,tokens)] = execute(args,invariant,dflash,tokens)
        if invariant and dflash:
            for a,b in zip(cells[(1,0,tokens)]['trials'][:3],cells[(1,1,tokens)]['trials'][:3]):
                require_exact(compare(a,b),64)
    for invariant, tokens in [(1,256),(1,64),(0,256)]:
        a,b = cells[(invariant,0,tokens)]['trials'],cells[(invariant,1,tokens)]['trials']
        for i in range(3):
            c = compare(a[i],b[i]);comparisons[f'bi{invariant}-budget{tokens}-repeat{i}'] = c
            if invariant:
                require_exact(c,64)
        if invariant:
            for method, trials in [('AR',a),('DFlash',b)]:
                for i in (1,2):
                    c=compare(trials[0],trials[i]);require_exact(c,64)
                    comparisons[f'{method}-budget{tokens}-repeat0-vs-{i}']=c
    for dflash in (0,1):
        trials=cells[(1,dflash,256)]['trials']
        c=compare(trials[0],trials[3]);require_exact(c,16)
        comparisons[f'dflash{dflash}-batch-companion']=c
        for i in range(3):
            c=compare(trials[i],cells[(1,dflash,64)]['trials'][i]);require_exact(c,64)
            comparisons[f'dflash{dflash}-chunks-repeat{i}']=c
    for key,cell in cells.items():
        if key[0] and key[1]:
            for trial in cell['trials']:
                steps=[r for r in trial['trace'] if r.get('sampling')]
                if not any(any(v>0 for v in s['num_rejected']) for s in steps[:-1]):
                    raise ValueError('Missing rejection followed by continuation')
    save(args.output/'validation.json', dict(passed=True, comparisons=comparisons))
    print('Supported invariant validation passed; beginning authorized full-response pilot.',flush=True)
    greedy=execute(args,1,0,256,pilot=1)['trials'][0]
    if any(d['finish_reason'] != 'stop' for d in greedy['details'].values()):
        save(args.output/'pilot-failure.json',dict(reason='capped/non-EOS greedy response',
            finishes={k:d['finish_reason'] for k,d in greedy['details'].items()}))
        raise ValueError('Full greedy pilot did not finish within its fixed cap')
    verified=execute(args,1,1,256,pilot=1)['trials'][0]
    c=compare(greedy,verified)
    save(args.output/'pilot-comparison.json',c)
    require_exact(c,sum(len(x) for x in greedy['tokens'].values()))
    publication=publish(args,inputs,greedy,verified,c)
    save(args.output/'result.json',dict(complete=True, validation_passed=True,
        publication=publication, budget=json.loads((args.output/'budget.json').read_text())))
    print(json.dumps(publication),flush=True)


if __name__ == '__main__':
    main()
