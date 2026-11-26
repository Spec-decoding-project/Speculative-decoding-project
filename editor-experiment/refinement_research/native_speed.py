"""Native correctness gate, then bounded paired full-generation timing."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from copy_action_fit.data import digest, save, sha


def select(prompts):
    development = [p for p in prompts if p['split'] == 'development']
    selected = []
    for label, lo, hi in [('short', 1, 128), ('long', 1024, 1900)]:
        candidates = sorted((p for p in development if lo <= len(p['prompt']) <= hi),
                            key=lambda p:digest(p['id']))
        if len(candidates) < 4:
            raise ValueError('Insufficient prespecified prompt-length stratum')
        selected.extend(dict(p, length_group=label) for p in candidates[:4])
    return selected


def compare(reference, generated):
    if (generated['prompt'] != reference['prompt'] or generated['tokens'] != reference['tokens']
            or generated['finish_reason'] != reference['finish_reason']):
        raise ValueError('Native output/termination differs from target-only greedy reference')


def configuration(project, greedy):
    config = dict(model=str(project/'data/models/qwen3-8b'), dtype='bfloat16',
        tensor_parallel_size=1, max_model_len=4096, max_num_seqs=1, max_num_batched_tokens=256,
        enable_chunked_prefill=True, enable_prefix_caching=False, enforce_eager=True,
        async_scheduling=False, seed=0, kv_cache_memory_bytes=2048*1024**2,
        attention_config={'backend':'FLASH_ATTN'}, generation_config='vllm',
        worker_extension_cls='refinement_research.native_speed_worker.NativeSpeedWorker')
    if not greedy:
        config['speculative_config'] = dict(method='dflash',
            model=str(project/'data/models/qwen3-8b-dflash-b16'), num_speculative_tokens=15,
            max_model_len=4096)
    return config


def child(args):
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from full_corpus_fit.data import audit_chunk
    if vllm.__version__ != '0.34.0' or os.environ.get('VLLM_BATCH_INVARIANT') != '1':
        raise ValueError('Pinned canonical target required')
    torch.set_num_threads(4)
    prompts = json.loads((args.output/'prompts.json').read_text())
    greedy = args.phase == 'reference'
    llm = LLM(**configuration(args.project, greedy))
    startup = llm.collective_rpc('corpus_finish')[0]
    if not startup['invariant_active'] or not startup['attention'] or any(
            not a['batch_invariant'] for a in startup['attention']):
        raise ValueError('Canonical invariant target not active')
    save(args.output/(args.phase+'-startup.json'), dict(**startup, torch=str(torch.__version__),
         vllm=vllm.__version__, gpu=torch.cuda.get_device_name()))
    records = []

    def generate(p, mode, *, audit=False, reference=None, clear=False, budget=64, stop=None):
        llm.collective_rpc('speed_begin', args=(mode, audit, reference, clear))
        start = time.perf_counter()
        output = llm.generate([{'prompt_token_ids':p['prompt']}],
            SamplingParams(temperature=0, seed=0, max_tokens=budget,
                           stop_token_ids=[] if stop is None else [stop]), use_tqdm=False)[0]
        seconds = time.perf_counter()-start
        observed = llm.collective_rpc('speed_finish')[0]
        result = dict(request_id=output.request_id, prompt=list(output.prompt_token_ids),
            tokens=list(output.outputs[0].token_ids), finish_reason=output.outputs[0].finish_reason)
        if audit:
            audited = audit_chunk([p], [result], observed['native'])
            observed['audit'] = dict(positions_checked=audited['positions_checked'],
                                     selected_blocks=len(audited['rows']))
        row = dict(prompt_id=p['id'], length_group=p['length_group'], prompt_tokens=len(p['prompt']),
                   mode=mode, precision=observed['precision'], seconds=seconds,
                   generated=result, observed=observed, budget=budget, stop_token=stop)
        return row

    def checkpoint(row, phase):
        row['phase'] = phase
        records.append(row)
        save(args.output/(args.phase+'-results.json'), records)
        print(json.dumps(dict(phase=phase, case=len(records), group=row['length_group'],
            mode=row['mode'], seconds=row['seconds'], tokens=len(row['generated']['tokens']),
            rounds=len(row['observed']['rounds']), graph_misses=row['observed']['graph_misses'])), flush=True)

    if greedy:
        for p in prompts:
            checkpoint(generate(p, 'off', audit=True), 'reference')
        # End conditions use the first reference's fourth token as a real stop ID.
        p = prompts[0]
        stop = records[0]['generated']['tokens'][3]
        for budget, token in [(1,None),(2,None),(17,None),(64,stop)]:
            checkpoint(generate(p, 'off', audit=True, budget=budget, stop=token), 'stop-reference')
        return

    refs = json.loads((args.output/'reference-results.json').read_text())
    initial = args.project/'research/full-corpus-run-20261117/training/draft/best.pt'
    gate = args.project/'research/broader-gates-20261125/linear/best.pt'
    load = llm.collective_rpc('speed_load', args=(str(initial),str(gate),'fp32'))[0]
    save(args.output/'editor-loaded.json', load)
    # Correctness precedes any interpretation of benchmark timing.
    for p, ref in zip(prompts, refs):
        full = p['prompt']+ref['generated']['tokens']
        for mode in ('off','identity','editor'):
            row = generate(p, mode, audit=True, reference=full, clear=mode=='editor')
            compare(ref['generated'], row['generated'])
            checkpoint(row, 'correctness')
    p, ref = prompts[0], refs[0]
    full = p['prompt']+ref['generated']['tokens']
    for mode in ('oracle','reject_first'):
        row = generate(p, mode, audit=True, reference=full)
        compare(ref['generated'], row['generated'])
        if mode == 'oracle' and 0 not in row['observed']['rejected']:
            raise ValueError('All-accepted fixture did not exercise all acceptance')
        if mode == 'reject_first' and set(row['observed']['sampled']) != {1}:
            raise ValueError('First-rejection fixture did not force rejection')
        checkpoint(row, 'verifier-fixture')
    for ref in refs[8:]:
        row = generate(p, 'editor', audit=True, reference=full,
                       budget=ref['budget'], stop=ref['stop_token'])
        compare(ref['generated'], row['generated'])
        checkpoint(row, 'stop-fixture')
    save(args.output/'correctness-passed.json', dict(passed=True, prompts=8,
        controls=['original','identity','fp32_editor','all_accepted','first_rejected','budgets_1_2_17','stop_token'],
        scope='Replacement only; native cache progression checked by successive exact greedy prefixes/output. No union claim.'))

    # Per-prompt empty-cache run includes graph creation. Warm paired runs follow.
    for p, ref in zip(prompts, refs):
        for mode in ('off','editor'):
            row = generate(p, mode, clear=mode=='editor')
            compare(ref['generated'], row['generated'])
            checkpoint(row, 'cold' if mode=='editor' else 'warmup')
        for repeat in range(5):
            for mode in (('off','editor') if repeat%2 == 0 else ('editor','off')):
                row = generate(p, mode)
                compare(ref['generated'], row['generated'])
                row['repeat'] = repeat
                if mode == 'editor' and row['observed']['graph_misses']:
                    raise ValueError('Warm benchmark unexpectedly recaptured a graph')
                checkpoint(row, 'timing')

    # Precision comparison is separate, after the fixed FP32 timing candidate.
    llm.collective_rpc('speed_load', args=(str(initial),str(gate),'bf16'))
    for p, ref in zip(prompts, refs):
        row = generate(p, 'editor', audit=True, reference=p['prompt']+ref['generated']['tokens'], clear=True)
        compare(ref['generated'], row['generated'])
        checkpoint(row, 'bf16-correctness')
    save(args.output/'complete.json', summarize(records))


def summarize(records):
    timing = [r for r in records if r['phase']=='timing']
    groups = {}
    for group in ('all','short','long'):
        cases = [r for r in timing if group=='all' or r['length_group']==group]
        by_prompt = {}
        for identity in sorted({r['prompt_id'] for r in cases}):
            arms = {mode:[r for r in cases if r['prompt_id']==identity and r['mode']==mode]
                    for mode in ('off','editor')}
            medians = {mode:statistics.median(r['seconds'] for r in rows) for mode,rows in arms.items()}
            by_prompt[identity] = dict(seconds=medians, speedup=medians['off']/medians['editor'],
                tokens=len(arms['off'][0]['generated']['tokens']),
                rounds={mode:len(rows[0]['observed']['rounds']) for mode,rows in arms.items()},
                editor_median_ms=statistics.median(x for r in arms['editor'] for x in r['observed']['editor_ms']))
        totals = {mode:sum(r['seconds'][mode] for r in by_prompt.values()) for mode in ('off','editor')}
        groups[group] = dict(prompts=len(by_prompt), total_median_seconds=totals,
            speedup=totals['off']/totals['editor'], per_prompt=by_prompt,
            tokens=sum(r['tokens'] for r in by_prompt.values()))
    return dict(passed_correctness=True, groups=groups, repeats=5, optimizer_updates=0,
        scope='Exploratory batch-one replacement, invariant target/drafter eager; editor CUDA graphs reused. Eight development prompts, 64-token cap; no union or non-invariant performance claim.',
        cold_seconds=sum(r['seconds'] for r in records if r['phase']=='cold'),
        peak_allocated_bytes=max(r['observed']['peak_allocated_bytes'] for r in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project','output'):parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--phase', choices=['reference','candidate'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase:
        child(args)
        return
    if (args.output/'run.json').exists():
        raise ValueError('Fresh output directory required; do not overwrite attempts')
    from full_corpus_fit.collect import load_prompts
    preflight = args.project/'research/full-corpus-preflight-8454304'
    prompts = select(load_prompts(preflight))
    save(args.output/'prompts.json', prompts)
    save(args.output/'run.json', dict(schema='native-speed-v1', prompts_sha256=digest(prompts),
        initial_sha256=sha(args.project/'research/full-corpus-run-20261117/training/draft/best.pt'),
        gate_sha256=sha(args.project/'research/broader-gates-20261125/linear/best.pt'),
        configurations={p:configuration(args.project,p=='reference') for p in ('reference','candidate')},
        max_tokens=64, repeats=5, graph_cache_entries=64, maximum_seconds=2700,
        source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('native_speed*.py')},
        final_test_loaded=False, training=False))
    started = time.monotonic()
    for phase in ('reference','candidate'):
        subprocess.run([sys.executable,'-u','-m','refinement_research.native_speed',
            '--project',str(args.project),'--output',str(args.output),'--phase',phase],
            check=True, timeout=max(1,2650-(time.monotonic()-started)))


if __name__ == '__main__':main()
