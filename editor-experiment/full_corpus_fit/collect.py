"""Resumable native collection on every authenticated prompt."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import time

from full_corpus_fit.data import audit_chunk, key, split_prompts
from full_corpus_fit.preflight import save, sha


def bucket(prompt):
    n = len(prompt['prompt'])
    return next((i for i, limit in enumerate((128, 512, 1024, 2240)) if n <= limit), 4)


def load_prompts(preflight):
    summary = json.loads((preflight/'summary.json').read_text())
    if sha(preflight/'prompts.jsonl') != summary['index_sha256']:
        raise ValueError('Authenticated prompt index changed')
    with (preflight/'prompts.jsonl').open() as stream:
        prompts = split_prompts([json.loads(line) for line in stream])
    if Counter(p['split'] for p in prompts) != {'train':50000, 'development':512, 'test':512}:
        raise ValueError('Full-corpus split sizes changed')
    return sorted(prompts, key=lambda p:(bucket(p), key(p['id'])))


def profile_prompts(prompts):
    prior = ('ultrachat_200k:train_sft:37680', 'ultrachat_200k:train_sft:61759',
             'ultrachat_200k:train_sft:164610', 'gsm8k:train:3821')
    training = [p for p in prompts if p['split']=='train']
    chosen = {p['id']:p for p in training if p['id'] in prior}
    for b, count in enumerate((32,32,24,16,14)):
        rows = [p for p in training if bucket(p)==b and p['id'] not in chosen]
        chosen.update((p['id'],p) for p in rows[:count])
    for p in training:
        if len(chosen) >= 128:
            break
        chosen.setdefault(p['id'],p)
    if len(chosen) != 128:
        raise ValueError('Unexpected calibration count')
    return sorted(chosen.values(), key=lambda p:(bucket(p),key(p['id'])))


def read_chunk(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def write_chunk(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with gzip.open(temporary, 'wt', compresslevel=3) as stream:
        json.dump(data, stream, separators=(',', ':'))
    temporary.replace(path)


def run(args):
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm_probe import prepare
    if vllm.__version__ != '0.34.0' or os.environ.get('VLLM_BATCH_INVARIANT') != '1':
        raise ValueError('Require the pinned invariant target environment')
    torch.set_num_threads(4)
    all_prompts = load_prompts(args.preflight)
    prompts = profile_prompts(all_prompts) if args.profile else all_prompts
    args.output.mkdir(exist_ok=True, parents=True)
    config = dict(model=str(args.project/'data/models/qwen3-8b'), dtype='bfloat16',
        tensor_parallel_size=1, max_model_len=4096, max_num_seqs=4, max_num_batched_tokens=256,
        enable_chunked_prefill=True, enable_prefix_caching=False, enforce_eager=True,
        async_scheduling=False, seed=0, kv_cache_memory_bytes=2048*1024**2,
        attention_config={'backend':'FLASH_ATTN'}, generation_config='vllm',
        worker_extension_cls='full_corpus_fit.capture.CorpusCaptureWorker')
    if not args.greedy:
        config['speculative_config'] = dict(method='dflash',
            model=str(args.project/'data/models/qwen3-8b-dflash-b16'), num_speculative_tokens=15,
            max_model_len=4096)
    run_config = dict(schema=1, preflight_sha256=sha(args.preflight/'summary.json'),
        profile=args.profile, greedy=args.greedy, config=config,
        prompt_ids=[p['id'] for p in prompts], max_new_tokens=64,
        chunk_size=8 if args.profile else 64, editor_context_tokens=1024,
        source_hashes={name:sha(Path(__file__).parent/name) for name in ('collect.py','capture.py','data.py')},
        labels='New versioned invariant-native continuation; original response journals read-only')
    if not args.profile:
        if args.calibration is None:
            raise ValueError('Full collection requires the successful calibration artifact')
        calibration=json.loads(args.calibration.read_text())
        if (not calibration['agreement']['passed'] or calibration['agreement']['prompts'] != 128
                or calibration['source_hashes'] != run_config['source_hashes']
                or calibration['native_config'] != config):
            raise ValueError('Full collection differs from the validated calibration contract')
        run_config['calibration_sha256']=sha(args.calibration)
    run_path = args.output/'run.json'
    if run_path.exists() and json.loads(run_path.read_text()) != run_config:
        raise ValueError('Cannot resume collection with changed configuration')
    save(run_path, run_config)
    if (args.output/'complete.json').exists():
        print('Collection already complete; refusing duplicate inference', flush=True)
        return
    # The existing pin checker streams checksums; it does not load target weights.
    pins = prepare(args.project)
    save(args.output/'model-pins.json', pins)
    start = time.monotonic()
    llm = LLM(**config)
    startup = llm.collective_rpc('corpus_finish')[0]
    if (not startup['invariant_active'] or startup['model_class'] != 'Qwen3ForCausalLM'
            or not startup['attention'] or any(not a['batch_invariant'] for a in startup['attention'])):
        raise ValueError('Canonical target dispatch/attention unverified')
    save(args.output/'startup.json', dict(**startup, torch=str(torch.__version__), vllm=vllm.__version__,
        gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda, initialization_seconds=time.monotonic()-start))
    chunk_size = run_config['chunk_size']
    chunk_reports, selected_total = [], 0
    loop_start = time.monotonic()
    for index, lo in enumerate(range(0, len(prompts), chunk_size)):
        batch = prompts[lo:lo+chunk_size]
        path = args.output/f'chunk-{index:05d}.json.gz'
        if path.exists():
            old = read_chunk(path)
            if old['prompt_ids'] != [p['id'] for p in batch] or old['run_sha256'] != sha(run_path):
                raise ValueError('Previously completed chunk is incompatible')
            audit = audit_chunk(batch, old['generated'], old['observed'], hashes=args.profile)
            if audit != old['audit']:
                raise ValueError('Previously completed chunk failed re-audit')
            report = old['measurement']
        else:
            llm.collective_rpc('corpus_begin', args=(args.profile,))
            before = time.monotonic()
            outputs = llm.generate([{'prompt_token_ids':p['prompt']} for p in batch],
                SamplingParams(temperature=0, max_tokens=64, seed=0), use_tqdm=False)
            seconds = time.monotonic()-before
            observed = llm.collective_rpc('corpus_finish')[0]
            generated = [dict(request_id=o.request_id, prompt=list(o.prompt_token_ids),
                         tokens=list(o.outputs[0].token_ids), finish_reason=o.outputs[0].finish_reason)
                         for o in outputs]
            audit = audit_chunk(batch, generated, observed, hashes=args.profile)
            report = dict(index=index, prompts=len(batch), prompt_tokens=sum(len(p['prompt']) for p in batch),
                generated_tokens=sum(len(g['tokens']) for g in generated), seconds=seconds,
                buckets=dict(Counter(bucket(p) for p in batch)), selected_blocks=len(audit['rows']),
                covered_prompts=sum(c['selected_blocks']>0 for c in audit['coverage']))
            write_chunk(path, dict(prompt_ids=[p['id'] for p in batch], run_sha256=sha(run_path),
                generated=generated, observed=observed, audit=audit, measurement=report))
        chunk_reports.append(report)
        selected_total += len(audit['rows'])
        status = dict(completed_prompts=lo+len(batch), total_prompts=len(prompts),
            selected_blocks=selected_total, measured_generation_seconds=sum(r['seconds'] for r in chunk_reports),
            session_seconds=time.monotonic()-loop_start, last_chunk=report,
            projected_total_generation_seconds=sum(r['seconds'] for r in chunk_reports)/(lo+len(batch))*len(prompts))
        save(args.output/'progress.json', status)
        print(json.dumps(status), flush=True)
    manifest = {p.name:sha(p) for p in sorted(args.output.glob('chunk-*.json.gz'))}
    save(args.output/'complete.json', dict(run_sha256=sha(run_path), chunks=manifest,
        prompts=len(prompts), selected_blocks=selected_total, measurements=chunk_reports,
        total_generation_seconds=sum(r['seconds'] for r in chunk_reports)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--calibration', type=Path)
    args = parser.parse_args()
    if args.greedy and not args.profile:
        raise ValueError('Independent greedy reference is bounded to calibration prompts')
    run(args)


if __name__ == '__main__':
    main()
