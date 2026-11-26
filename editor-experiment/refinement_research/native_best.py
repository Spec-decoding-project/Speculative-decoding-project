"""Paired native benchmark of the exact earlier 3.6997 linear gate; no fitting."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

from copy_action_fit.data import SOURCES, digest, save, sha
from refinement_research.native_speed import compare, configuration

GATE_SHA = '5ab048b622beffb10ab8d282027cb2510f5439e4621793d5a82e687e696d679b'
REPEATS = 5


def select(prompts, per_quartile=16):
    selected = []
    for source in SOURCES:
        population = sorted((p for p in prompts if p['split']=='development' and p['source']==source),
                            key=lambda p:(len(p['prompt']), digest(p['id'])))
        if len(population) != 256:
            raise ValueError('Changed development population')
        for q in range(4):
            group = population[q*64:(q+1)*64]
            chosen = sorted(group, key=lambda p:digest(['native-earlier-gate-v1',p['id']]))[:per_quartile]
            selected.extend(dict(p, length_group=f'Q{q+1}', population_length_range=
                                 [len(group[0]['prompt']),len(group[-1]['prompt'])]) for p in chosen)
    return sorted(selected, key=lambda p:digest(['native-pair-order-v1',p['id']]))


def counts(samples, visible):
    """Clip terminal sampled overshoot/EOS against actually returned tokens."""
    remaining = visible
    verify_tokens = prefill_tokens = rounds = 0
    per_round = []
    for sample in samples:
        n = sample['sampled']
        if n < 0:
            raise ValueError('Negative native sample count')
        emitted = min(remaining, n)
        remaining -= emitted
        if sample['speculative']:
            rounds += 1
            verify_tokens += emitted
            per_round.append(emitted)
        else:
            prefill_tokens += emitted
    if remaining or prefill_tokens != 1 or not rounds:
        raise ValueError('Expected one prefill anchor and at least one verification round')
    return dict(verification_rounds=rounds, verification_tokens=verify_tokens,
                prefill_tokens=prefill_tokens, committed_per_round=per_round)


def aggregate(rows):
    totals = {}
    for mode in ('off','editor'):
        arm = [r for r in rows if r['mode']==mode]
        n = sum(r['metrics']['verification_rounds'] for r in arm)
        tokens = sum(len(r['generated']['tokens']) for r in arm)
        committed = sum(r['metrics']['verification_tokens'] for r in arm)
        ms = sum(r['metrics']['decode_ms'] for r in arm)
        seconds = sum(r['seconds'] for r in arm)
        editor_ms = sum(sum(r['observed']['editor_ms']) for r in arm)
        totals[mode] = dict(requests=len(arm), output_tokens=tokens, seconds=seconds,
            tokens_per_second=tokens/seconds, verification_rounds=n,
            decode_ms=ms, average_decode_round_ms=ms/n,
            committed_tokens=committed, average_committed_per_round=committed/n,
            decode_tokens_per_second=1000*committed/ms,
            prefill_and_request_ms=sum(r['metrics']['prefill_ms'] for r in arm),
            editor_total_ms=editor_ms, editor_ms_per_verification_round=editor_ms/n,
            editor_calls=sum(r['observed']['edit_calls'] for r in arm),
            average_editor_call_ms=editor_ms/sum(r['observed']['edit_calls'] for r in arm) if mode=='editor' else 0,
            graph_hits=sum(r['observed']['graph_hits'] for r in arm),
            graph_misses=sum(r['observed']['graph_misses'] for r in arm))
    b,e = totals['off'],totals['editor']
    gain = e['average_committed_per_round']/b['average_committed_per_round']-1
    budget = b['average_decode_round_ms']*gain
    cost = e['editor_ms_per_verification_round']
    extra = b['average_committed_per_round']*cost/b['average_decode_round_ms']
    return dict(arms=totals, end_to_end_speedup=b['seconds']/e['seconds'],
        latency_increase_percent=100*(e['seconds']/b['seconds']-1),
        throughput_change_percent=100*(b['seconds']/e['seconds']-1),
        decode_speedup=e['decode_tokens_per_second']/b['decode_tokens_per_second'],
        committed_gain_percent=100*gain,
        editing_budget_ms_per_round=budget,
        measured_editor_ms_per_round=cost,
        required_extra_committed_tokens_per_round=extra,
        required_committed_tokens_per_round=b['average_committed_per_round']+extra,
        observed_extra_committed_tokens_per_round=e['average_committed_per_round']-b['average_committed_per_round'],
        observed_round_cost_increase_ms=e['average_decode_round_ms']-b['average_decode_round_ms'],
        noneditor_round_cost_change_ms=e['average_decode_round_ms']-cost-b['average_decode_round_ms'])


def summarize(records, prompts):
    rows=[r for r in records if r['phase']=='timing']
    groups={}
    for source in ('all',*SOURCES):
        for q in (None,'Q1','Q2','Q3','Q4') if source!='all' else (None,):
            cases=[r for r in rows if (source=='all' or r['source']==source) and
                   (q is None or r['length_group']==q)]
            ids=sorted({r['prompt_id'] for r in cases})
            if not ids:continue
            pairs={pid:{mode:[r for r in cases if r['prompt_id']==pid and r['mode']==mode]
                        for mode in ('off','editor')} for pid in ids}
            for arms in pairs.values():
                for arm in arms.values():
                    if sorted(r['repeat'] for r in arm)!=list(range(REPEATS)):
                        raise ValueError('Missing/duplicate paired repeat')
                    if len({digest(r['metrics']['committed_per_round']) for r in arm})!=1:
                        raise ValueError('Native round trajectory changes across repeats')
                for r in arms['editor']:compare(arms['off'][0]['generated'],r['generated'])
            result=aggregate(cases)
            result.update(prompts=len(ids), prompt_tokens_range=[min(r['prompt_tokens'] for r in cases),
                          max(r['prompt_tokens'] for r in cases)], repeats=REPEATS,
                          repeat_speedups=[aggregate([r for r in cases if r['repeat']==i])['end_to_end_speedup']
                                           for i in range(REPEATS)])
            means=[(statistics.mean(r['seconds'] for r in pairs[pid]['off']),
                    statistics.mean(r['seconds'] for r in pairs[pid]['editor'])) for pid in ids]
            rng=random.Random(0); boot=[]
            for _ in range(10000):
                chosen=rng.choices(means,k=len(means))
                boot.append(sum(p[0] for p in chosen)/sum(p[1] for p in chosen))
            boot.sort();result['prompt_bootstrap_speedup_ci95']=[boot[249],boot[9749]]
            result['prompts_faster']=sum(b>e for b,e in means)
            groups[source+('/'+q if q else '')]=result
    cold=[r for r in records if r['phase']=='cold']
    return dict(schema='native-earlier-gate-v1', groups=groups, repeats=REPEATS,
        prompts=len(prompts), matched_tokens=True, optimizer_updates=0, final_test_loaded=False,
        graph_setup=dict(empty_cache_requests=len(cold),
            total_capture_and_warmup_ms=sum(r['observed']['graph_preparation_ms'] for r in cold),
            total_new_graphs=sum(r['observed']['graph_misses'] for r in cold),
            total_cold_end_to_end_seconds=sum(r['seconds'] for r in cold),
            cold_output_tokens=sum(len(r['generated']['tokens']) for r in cold),
            cold_by_source={source:dict(requests=sum(r['source']==source for r in cold),
                seconds=sum(r['seconds'] for r in cold if r['source']==source),
                graph_preparation_ms=sum(r['observed']['graph_preparation_ms'] for r in cold if r['source']==source))
                for source in SOURCES}),
        correctness_generations=sum(r['phase']=='correctness' for r in records),
        timed_generations=len(rows),
        peak_allocated_bytes=max(r['observed']['peak_allocated_bytes'] for r in records),
        scope='FP32 earlier linear gate; canonical invariant target/drafter eager; standard editor kernels with reusable CUDA graphs. Development, not fresh confirmation. 64-token cap, batch one.',
        economics_assumption='Break-even editor budget holds baseline noneditor round work constant. Measured noneditor changes reported separately. Decode begins before first draft, after synchronized prefill anchor; includes last unused proposal and all engine gaps/copies.')


def child(args):
    import os
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from full_corpus_fit.data import audit_chunk
    if vllm.__version__!='0.34.0' or os.environ.get('VLLM_BATCH_INVARIANT')!='1':
        raise ValueError('Pinned invariant runtime required')
    torch.set_num_threads(4)
    prompts=json.loads((args.output/'prompts.json').read_text())
    greedy=args.phase=='reference'
    started=time.perf_counter()
    llm=LLM(**configuration(args.project,greedy))
    startup=llm.collective_rpc('corpus_finish')[0]
    if not startup['invariant_active'] or not startup['attention'] or any(
            not a['batch_invariant'] for a in startup['attention']):
        raise ValueError('Canonical target not active')
    save(args.output/(args.phase+'-startup.json'),dict(**startup,torch=str(torch.__version__),
        vllm=vllm.__version__,gpu=torch.cuda.get_device_name(),initialization_seconds=time.perf_counter()-started))
    if not greedy:
        initial=args.project/'research/full-corpus-run-20261117/training/draft/best.pt'
        gate=args.project/'research/frozen-gates-20261125/linear/best.pt'
        loaded=llm.collective_rpc('speed_load',args=(str(initial),str(gate),'fp32','standard',GATE_SHA))[0]
        save(args.output/'editor-loaded.json',loaded)
    records=[]
    def run(p, mode, phase, *, audit=False, reference=None, clear=False, repeat=None):
        llm.collective_rpc('speed_begin',args=(mode,audit,reference,clear,not greedy))
        start=time.perf_counter()
        out=llm.generate([{'prompt_token_ids':p['prompt']}],
            SamplingParams(temperature=0,seed=0,max_tokens=64),use_tqdm=False)[0]
        user_end=time.perf_counter()
        end=llm.collective_rpc('speed_stop')[0]
        observed=llm.collective_rpc('speed_finish')[0]
        generated=dict(request_id=out.request_id,prompt=list(out.prompt_token_ids),
            tokens=list(out.outputs[0].token_ids),finish_reason=out.outputs[0].finish_reason)
        metrics={}
        if not greedy:
            metrics=counts(observed['all_samples'],len(generated['tokens']))
            boundary=observed['decode_started']
            if not start<=boundary<=end:raise ValueError('Worker/driver clock boundary invalid')
            if metrics['verification_rounds']!=sum(r['speculative'] for r in observed['rounds']):
                raise ValueError('Verifier round accounting disagrees')
            metrics.update(decode_ms=1000*(end-boundary),prefill_ms=1000*(boundary-start))
        if audit:
            checked=audit_chunk([p],[generated],observed['native'],hashes=not greedy)
            observed['audit']=dict(positions_checked=checked['positions_checked'],
                                  aligned_hashes=checked['aligned_hashes'])
        row=dict(prompt_id=p['id'],source=p['source'],length_group=p['length_group'],
            prompt_tokens=len(p['prompt']),mode=mode,phase=phase,repeat=repeat,
            seconds=end-start,llm_generate_seconds=user_end-start,drain_and_rpc_ms=1000*(end-user_end),
            generated=generated,metrics=metrics,observed=observed)
        records.append(row)
        save(args.output/'records'/f'{args.phase}-{len(records):05}.json',row)
        print(json.dumps(dict(phase=phase,case=len(records),source=p['source'],group=p['length_group'],
            mode=mode,seconds=row['seconds'],rounds=metrics.get('verification_rounds'),
            graph_misses=observed['graph_misses'])),flush=True)
        return row
    if greedy:
        # Eight independent target-only controls, one from each source/quartile.
        seen=set()
        for p in prompts:
            group=(p['source'],p['length_group'])
            if group not in seen:
                run(p,'off','reference',audit=True);seen.add(group)
        save(args.output/'reference-results.json',records)
        return
    refs={r['prompt_id']:r for r in json.loads((args.output/'reference-results.json').read_text())}
    for index,p in enumerate(prompts):
        baseline=run(p,'off','correctness',audit=True)
        if p['id'] in refs:compare(refs[p['id']]['generated'],baseline['generated'])
        full=p['prompt']+baseline['generated']['tokens']
        edited=run(p,'editor','correctness',audit=True,reference=full,clear=True)
        compare(baseline['generated'],edited['generated'])
        if baseline['observed']['audit']['aligned_hashes']!=edited['observed']['audit']['aligned_hashes']:
            raise ValueError('Target full logits changed between paths')
        warm=run(p,'off','warmup')
        compare(baseline['generated'],warm['generated'])
        cold=run(p,'editor','cold',clear=True)
        compare(baseline['generated'],cold['generated'])
        for repeat in range(REPEATS):
            for mode in (('off','editor') if (repeat+index)%2==0 else ('editor','off')):
                r=run(p,mode,'timing',repeat=repeat)
                compare(baseline['generated'],r['generated'])
                if mode=='editor' and r['observed']['graph_misses']:
                    raise ValueError('Warm timing recaptured a graph')
        save(args.output/'progress.json',dict(completed_prompts=index+1,total_prompts=len(prompts)))
    save(args.output/'complete.json',summarize(records,prompts))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--phase',choices=['reference','candidate'])
    args=parser.parse_args()
    if args.phase:
        child(args);return
    if args.output.exists():raise ValueError('Fresh output directory required')
    (args.output/'records').mkdir(parents=True)
    from full_corpus_fit.collect import load_prompts
    preflight=args.project/'research/full-corpus-preflight-8454304'
    prompts=select(load_prompts(preflight))
    save(args.output/'prompts.json',prompts)
    save(args.output/'run.json',dict(schema='native-earlier-gate-v1',gate_sha256=GATE_SHA,
        prompt_ids=[p['id'] for p in prompts],prompt_digest=digest(prompts),
        prompt_index_sha256=sha(preflight/'prompts.jsonl'),prompts=len(prompts),repeats=REPEATS,
        selection='16 deterministic hash samples per source-specific development length quartile; 64/source',
        baseline_configuration=configuration(args.project,False),budget_tokens=64,precision='fp32',
        editor_kernels='standard',source_hashes={str(p.relative_to(Path(__file__).parent)):sha(p)
             for p in sorted(Path(__file__).parent.glob('*.py'))},optimizer_updates=0,final_test_loaded=False))
    for phase,timeout in [('reference',600),('candidate',4200)]:
        subprocess.run([sys.executable,'-u','-m','refinement_research.native_best','--project',str(args.project),
            '--output',str(args.output),'--phase',phase],check=True,timeout=timeout)


if __name__=='__main__':main()
