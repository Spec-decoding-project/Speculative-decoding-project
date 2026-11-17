"""Bounded genuine-greedy versus DFlash diagnostic; never trains or relabels.

Six isolated engine processes: off/on x AR/DFlash, then on x AR/DFlash with
chunked prefill. Core repeats use fresh requests and reversed order on repeat 1.
256 generation steps at worst, plus <=8 extra chunk steps; 36 calls reserved
for initialization. Worker hooks enforce the shared 300-call/600-second cap.
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

from vllm_probe_worker import save

FIXTURES = {'ultrachat_200k:train_sft:37680': (24, 6),
            'ultrachat_200k:train_sft:61759': (22, 15),
            'ultrachat_200k:train_sft:164610': (19, 5), 'gsm8k:train:3821': (38, 5)}
SOURCE_SHA = '421e593f7f2ab56bfac09adc84f7d887455ebeb0955685ed8c22abf9b43f6ac0'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(project):
    bank = Path(__file__).resolve().parents[1] / 'feature_bank'
    sys.path.insert(0, str(bank))
    from data import records
    if sha256(project / 'scripts/generate_qwen_responses.py') != SOURCE_SHA:
        raise ValueError('Original generator changed')
    target = project / 'data/models/qwen3-8b'
    expected = json.loads((bank / 'target_files.json').read_text())
    if json.loads((target / 'pinned_revision.json').read_text()) != {
            'model_id': 'Qwen/Qwen3-8B', 'revision': expected['revision']}:
        raise ValueError('Target revision changed')
    hashes = {}
    for item in expected['files']:
        path = target / item['rfilename']
        if path.stat().st_size != item['size']:
            raise ValueError(f'Wrong target size: {path.name}')
        if 'lfs' in item:
            value = sha256(path)
            wanted = item['lfs']['sha256']
        else:
            value = hashlib.sha1(f'blob {item["size"]}\0'.encode() + path.read_bytes()).hexdigest()
            wanted = item['blobId']
        if value != wanted:
            raise ValueError(f'Wrong target hash: {path.name}')
        hashes[path.name] = value
    draft_pin = json.loads((bank / 'model_pin.json').read_text())
    for name, item in draft_pin['files'].items():
        path = project / 'data/models/qwen3-8b-dflash-b16' / name
        if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError(f'Wrong draft file: {name}')
    fixtures = []
    for row in records(project / 'data/qwen_responses_pilot'):
        if row['id'] not in FIXTURES:
            continue
        anchor, focus = FIXTURES[row['id']]
        if len(row['prompt_token_ids']) != anchor or len(row['response_token_ids']) < 16:
            raise ValueError('Invalid first-block fixture')
        fixtures.append(dict(id=row['id'], prompt=row['prompt_token_ids'],
            labels=row['response_token_ids'][:16], record_sha256=row['record_sha256'], focus=focus))
    if len(fixtures) != 4:
        raise ValueError('Missing fixtures')
    fixtures.sort(key=lambda f: list(FIXTURES).index(f['id']))
    return dict(fixtures=fixtures, target_hashes=hashes, draft_pin=draft_pin,
                source_generator_sha256=SOURCE_SHA)


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return dict(position=i, left=x, right=y)
    return None if len(a) == len(b) else dict(position=min(len(a), len(b)), length_mismatch=True)


def compare(a, b):
    """Only compare raw logits on exactly matching reconstructed token prefixes."""
    def group(trial):
        result = {}
        for row in trial['trace']:
            if 'logits_sha256' in row:
                result.setdefault(row['prefix_sha256'], []).append(row)
        return result
    left, right = group(a), group(b)
    shared = sorted(left.keys() & right.keys())
    return dict(token_differences={key: first_difference(a['tokens'][key], b['tokens'][key])
                for key in a['tokens'].keys() & b['tokens'].keys()},
                matching_prefix_rows=len(shared),
                unequal_full_logit_rows=[dict(prefix_sha256=key, left=left[key], right=right[key])
                    for key in shared if len({r['logits_sha256']
                                              for r in left[key] + right[key]}) != 1])


def child(args):
    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams
    if vllm.__version__ != '0.34.0':
        raise RuntimeError('Unreviewed vLLM version')
    fixtures = json.loads((args.output / 'inputs.json').read_text())['fixtures']
    torch.set_num_threads(4)
    config = dict(model=str(args.project / 'data/models/qwen3-8b'), dtype='bfloat16',
        tensor_parallel_size=1, max_model_len=256, max_num_seqs=4,
        max_num_batched_tokens=64 if args.chunked else 256,
        enable_chunked_prefill=bool(args.chunked), enable_prefix_caching=False,
        enforce_eager=True, async_scheduling=False, seed=0,
        kv_cache_memory_bytes=512 * 1024**2,
        attention_config={'backend': 'FLASH_ATTN'},
        worker_extension_cls='vllm_probe_worker.ProbeWorker',
        generation_config='vllm', disable_log_stats=False)
    if args.dflash:
        config['speculative_config'] = dict(method='dflash',
            model=str(args.project / 'data/models/qwen3-8b-dflash-b16'),
            num_speculative_tokens=15, max_model_len=256)
    cell = f'bi{args.invariant}-dflash{args.dflash}-chunk{args.chunked}'
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
        if startup['model_class'] != 'Qwen3ForCausalLM':
            raise RuntimeError('Wrong target implementation')
        if startup['invariant_active'] != bool(args.invariant):
            raise RuntimeError('Batch invariant dispatch state disagrees with request')
        if not startup['attention'] or any(x['batch_invariant'] != bool(args.invariant)
                                         for x in startup['attention']):
            raise RuntimeError('Attention invariance not confirmed')
        # Native greedy sampling: no forced labels, trace replay, or draft injection.
        sampling = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True,
                                  logprobs=5, seed=0)
        plan = [('chunked', fixtures)] if args.chunked else [
            ('repeat0', fixtures), ('repeat1', list(reversed(fixtures))), ('repeat2', fixtures)]
        if args.invariant and not args.chunked:
            plan.append(('alone', fixtures[:1]))
        for name, batch in plan:
            # Capture hashes on every repeat to check logits as well as selected IDs.
            before = llm.collective_rpc('probe_begin', args=(name, True))[0]
            began = time.monotonic()
            outputs = llm.generate([{'prompt_token_ids': f['prompt']} for f in batch],
                                   sampling, use_tqdm=False)
            elapsed = time.monotonic() - began
            if len(outputs) != len(batch):
                raise RuntimeError('Missing generated fixture output')
            observed = llm.collective_rpc('probe_finish')[0]
            tokens, details = {}, {}
            for fixture, output in zip(batch, outputs):
                completion = output.outputs[0]
                ids = list(completion.token_ids)
                if output.prompt_token_ids != fixture['prompt'] or len(ids) != 16:
                    raise RuntimeError('Output alignment/length mismatch')
                tokens[fixture['id']] = ids
                details[fixture['id']] = dict(request_id=output.request_id,
                    historical_difference=first_difference(fixture['labels'], ids),
                    selected_logprobs=[{str(k): v.logprob for k, v in row.items()}
                                       for row in completion.logprobs])
            result['trials'].append(dict(name=name, tokens=tokens, details=details,
                trace=observed['trace'], timings=observed['timings'],
                wall_seconds=elapsed, target_calls=observed['budget']['calls']-before['calls'],
                target_seconds=observed['budget']['seconds']-before['seconds'],
                peak_allocated_bytes=observed['peak_allocated_bytes']))
            save(path, result)
        result['metrics'] = [dataclasses.asdict(m) if dataclasses.is_dataclass(m) else repr(m)
                             for m in llm.get_metrics()]
        result['complete'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        save(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--invariant', type=int, choices=(0, 1), default=0)
    parser.add_argument('--dflash', type=int, choices=(0, 1), default=0)
    parser.add_argument('--chunked', type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.project = args.project.resolve()
    if args.child:
        child(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = prepare(args.project)
    inputs['source_hashes'] = {name: sha256(Path(__file__).with_name(name)) for name in
                              ('vllm_probe.py', 'vllm_probe_worker.py', 'vllm_probe.sbatch')}
    save(args.output / 'inputs.json', inputs)
    budget = args.output / 'budget.json'
    if budget.exists():
        raise RuntimeError('Refusing to reset an existing diagnostic budget')
    save(budget, dict(calls=0, seconds=0.0))
    matrix = [(1, 0, 0), (1, 1, 0), (0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 1)]
    cells = {}
    for invariant, dflash, chunked in matrix:
        env = dict(os.environ, VLLM_BATCH_INVARIANT=str(invariant),
                   VLLM_PROBE_BUDGET=str(budget), VLLM_WORKER_MULTIPROC_METHOD='spawn',
                   HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1')
        command = [sys.executable, str(Path(__file__).resolve()), '--child',
                   '--project', str(args.project), '--output', str(args.output),
                   '--invariant', str(invariant), '--dflash', str(dflash), '--chunked', str(chunked)]
        cell = f'bi{invariant}-dflash{dflash}-chunk{chunked}'
        with (args.output / f'{cell}.log').open('w') as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f'{cell} failed: see its log and partial result; no automatic retry')
        cells[cell] = json.loads((args.output / f'{cell}.json').read_text())
    comparisons = {}
    for invariant in (0, 1):
        a = cells[f'bi{invariant}-dflash0-chunk0']['trials']
        b = cells[f'bi{invariant}-dflash1-chunk0']['trials']
        for i in range(3):
            comparisons[f'bi{invariant}-AR-vs-DFlash-repeat{i}'] = compare(a[i], b[i])
        for name, trials in [('AR', a), ('DFlash', b)]:
            for i in (1, 2):
                comparisons[f'bi{invariant}-{name}-repeat0-vs-{i}'] = compare(trials[0], trials[i])
            if invariant:
                comparisons[f'{name}-batch-vs-alone'] = compare(trials[0], trials[3])
                other = cells[f'bi1-dflash{int(name == "DFlash")}-chunk1']['trials'][0]
                comparisons[f'{name}-prefill-chunks'] = compare(trials[0], other)
    save(args.output / 'comparisons.json', dict(comparisons=comparisons,
        budget=json.loads(budget.read_text()), complete=True,
        scope='four first-block fixtures; instrumented eager execution; no universal guarantee'))


if __name__ == '__main__':
    main()
