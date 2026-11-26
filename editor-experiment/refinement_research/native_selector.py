"""Bounded pre-editor selector fit and paired native evaluation; frozen editor."""
import argparse
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

from copy_action_fit.data import SOURCES, digest, save, sha
from refinement_research.native_best import GATE_SHA, counts
from refinement_research.native_best_audit import round_economics
from refinement_research.native_speed import compare, configuration
from refinement_research.selector import fit, label_shadow, select_prompts

MODES=('off','selector_probe','editor','selected')
REPEATS=3


def config(project, greedy):
    c=configuration(project,greedy)
    c['worker_extension_cls']='refinement_research.selector_worker.SelectorWorker'
    return c


def summarize(records):
    rows=[r for r in records if r['phase']=='timing']
    groups={}
    for source in ('all',*SOURCES):
        for q in ((None,) if source=='all' else (None,'Q1','Q2','Q3','Q4')):
            cases=[r for r in rows if (source=='all' or r['source']==source) and (q is None or r['length_group']==q)]
            ids=sorted({r['prompt_id'] for r in cases})
            arms={}
            for mode in MODES:
                arm=[r for r in cases if r['mode']==mode]
                for pid in ids:
                    paired=[r for r in arm if r['prompt_id']==pid]
                    baseline=next(r for r in cases if r['prompt_id']==pid and r['mode']=='off')
                    if sorted(r['repeat'] for r in paired)!=list(range(REPEATS)):
                        raise ValueError('Missing paired repeat')
                    if len({digest(r['metrics']['committed_per_round']) for r in paired})!=1:
                        raise ValueError('Native trajectory changed between repeats')
                    for r in paired:compare(baseline['generated'],r['generated'])
                rounds=sum(r['metrics']['verification_rounds'] for r in arm)
                seconds=sum(r['seconds'] for r in arm)
                committed=sum(r['metrics']['verification_tokens'] for r in arm)
                ms=sum(r['metrics']['decode_ms'] for r in arm)
                calls=sum(r['observed']['edit_calls'] for r in arm)
                decisions=[d for r in arm for d in r['observed']['selector_decisions']]
                arms[mode]=dict(requests=len(arm),seconds=seconds,
                    output_tokens=sum(len(r['generated']['tokens']) for r in arm),
                    tokens_per_second=sum(len(r['generated']['tokens']) for r in arm)/seconds,
                    verification_rounds=rounds,decode_ms=ms,average_round_ms=ms/rounds,
                    committed_tokens=committed,committed_per_round=committed/rounds,
                    editor_calls=calls,selector_calls=len(decisions),
                    selected_fraction=sum(d['selected'] for d in decisions)/len(decisions) if decisions else None,
                    graph_hits=sum(r['observed']['graph_hits'] for r in arm),
                    graph_misses=sum(r['observed']['graph_misses'] for r in arm))
            b=arms['off'];comparisons={}
            for mode in MODES[1:]:
                a=arms[mode]
                means=[tuple(statistics.mean(r['seconds'] for r in cases if r['prompt_id']==pid and r['mode']==m)
                             for m in ('off',mode)) for pid in ids]
                rng=random.Random(0);boot=[]
                for _ in range(10000):
                    sample=rng.choices(means,k=len(means))
                    boot.append(sum(a for a,b in sample)/sum(b for a,b in sample))
                boot.sort()
                comparisons[mode]=dict(speedup=b['seconds']/a['seconds'],
                    throughput_change_percent=100*(b['seconds']/a['seconds']-1),
                    repeat_speedups=[sum(r['seconds'] for r in cases if r['mode']=='off' and r['repeat']==i)/
                        sum(r['seconds'] for r in cases if r['mode']==mode and r['repeat']==i) for i in range(REPEATS)],
                    prompt_bootstrap_ci95=[boot[249],boot[9749]],prompts_faster=sum(a>b for a,b in means),
                    economics=round_economics(b['average_round_ms'],a['average_round_ms'],
                                             b['committed_per_round'],a['committed_per_round']))
            groups[source+('/'+q if q else '')]=dict(prompts=len(ids),
                prompt_length_range=[min(r['prompt_tokens'] for r in cases),max(r['prompt_tokens'] for r in cases)],
                arms=arms,comparisons=comparisons)
    return dict(schema='native-selector-v1',groups=groups,matched_tokens=True,repeats=REPEATS,
        timed_generations=len(rows),editor_optimizer_updates=0,final_test_evaluated=False,
        cold={mode:dict(requests=sum(r['phase']=='cold' and r['mode']==mode for r in records),
            capture_ms=sum(r['observed']['graph_preparation_ms'] for r in records if r['phase']=='cold' and r['mode']==mode),
            captures=sum(r['observed']['graph_misses'] for r in records if r['phase']=='cold' and r['mode']==mode),
            seconds=sum(r['seconds'] for r in records if r['phase']=='cold' and r['mode']==mode)) for mode in ('editor','selected')})


def child(args):
    import os
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from full_corpus_fit.data import audit_chunk
    if vllm.__version__!='0.34.0' or os.environ.get('VLLM_BATCH_INVARIANT')!='1':
        raise ValueError('Pinned invariant runtime required')
    torch.set_num_threads(4)
    groups=json.loads((args.output/'prompts.json').read_text())
    greedy=args.phase=='reference'
    started=time.perf_counter()
    llm=LLM(**config(args.project,greedy))
    startup=llm.collective_rpc('corpus_finish')[0]
    if not startup['invariant_active'] or not startup['attention'] or any(not a['batch_invariant'] for a in startup['attention']):
        raise ValueError('Canonical target inactive')
    save(args.output/(args.phase+'-startup.json'),dict(**startup,torch=str(torch.__version__),
        vllm=vllm.__version__,gpu=torch.cuda.get_device_name(),initialization_seconds=time.perf_counter()-started))
    if not greedy:
        initial=args.project/'research/full-corpus-run-20261117/training/draft/best.pt'
        gate=args.project/'research/frozen-gates-20261125/linear/best.pt'
        loaded=llm.collective_rpc('speed_load',args=(str(initial),str(gate),'fp32','standard',GATE_SHA))[0]
        save(args.output/(args.phase+'-editor-loaded.json'),loaded)
    if args.phase=='evaluation':
        model=json.loads((args.output/'selector.json').read_text())
        loaded=llm.collective_rpc('selector_load',args=(model,))[0]
        if loaded['selector_digest']!=digest(model):raise ValueError('Selector serialization changed')
        save(args.output/'selector-loaded.json',loaded)
    records=[]
    def run(p,mode,phase,*,audit=False,reference=None,clear=False,repeat=None):
        llm.collective_rpc('selector_request',args=(len(p['prompt']),))
        llm.collective_rpc('speed_begin',args=(mode,audit,reference,clear,not greedy))
        start=time.perf_counter()
        out=llm.generate([{'prompt_token_ids':p['prompt']}],SamplingParams(temperature=0,seed=0,max_tokens=64),use_tqdm=False)[0]
        returned=time.perf_counter()
        end=llm.collective_rpc('speed_stop')[0]
        observed=llm.collective_rpc('speed_finish')[0]
        if mode in ('off','selector_probe') and observed['edit_calls']:
            raise ValueError('Skip control unexpectedly invoked the editor')
        if mode=='selected' and observed['edit_calls']!=sum(d['selected'] for d in observed['selector_decisions']):
            raise ValueError('Selector decisions do not match actual editor calls')
        generated=dict(request_id=out.request_id,prompt=list(out.prompt_token_ids),
                       tokens=list(out.outputs[0].token_ids),finish_reason=out.outputs[0].finish_reason)
        metrics={}
        if not greedy:
            metrics=counts(observed['all_samples'],len(generated['tokens']))
            boundary=observed['decode_started']
            if not start <= boundary <= end:raise ValueError('Invalid worker clock boundary')
            if metrics['verification_rounds']!=sum(r['speculative'] for r in observed['rounds']):
                raise ValueError('Round accounting differs')
            metrics.update(decode_ms=1000*(end-boundary),prefill_ms=1000*(boundary-start))
        if audit:
            checked=audit_chunk([p],[generated],observed['native'],hashes=not greedy)
            observed['audit']=dict(positions_checked=checked['positions_checked'],aligned_hashes=checked['aligned_hashes'])
        row=dict(prompt_id=p['id'],source=p['source'],length_group=p['length_group'],prompt_tokens=len(p['prompt']),
            mode=mode,phase=phase,repeat=repeat,seconds=end-start,llm_generate_seconds=returned-start,
            generated=generated,metrics=metrics,observed=observed)
        records.append(row)
        save(args.output/'records'/f'{args.phase}-{len(records):05}.json',row)
        print(json.dumps(dict(phase=phase,index=len(records),mode=mode,seconds=end-start,
            rounds=metrics.get('verification_rounds'),edits=observed['edit_calls'],misses=observed['graph_misses'])),flush=True)
        return row
    seen=set()
    if args.phase=='collection':
        labels={}
        for name in ('fit','calibration'):
            labels[name]=[]
            for i,p in enumerate(groups[name]):
                key=(p['source'],p['length_group']);audit=key not in seen;seen.add(key)
                r=run(p,'shadow',name,audit=audit,clear=True)
                labels[name].extend(label_shadow(p,r))
                save(args.output/'progress.json',dict(phase=name,completed=i+1,total=len(groups[name])))
            save(args.output/(name+'-labels.json'),labels[name])
        model=fit(labels['fit'],labels['calibration'])
        save(args.output/'selector.json',model)
        save(args.output/'frozen-selector.json',dict(sha256=sha(args.output/'selector.json'),digest=digest(model)))
        print(json.dumps(dict(selector_threshold=model['threshold'],calibration=model['calibration'])),flush=True)
        return
    if greedy:
        for p in groups['evaluation']:
            key=(p['source'],p['length_group'])
            if key not in seen:run(p,'off','reference',audit=True);seen.add(key)
        save(args.output/'reference-results.json',records)
        return
    refs={r['prompt_id']:r for r in json.loads((args.output/'reference-results.json').read_text())}
    for i,p in enumerate(groups['evaluation']):
        audit=p['id'] in refs
        baseline=run(p,'off','correctness',audit=audit)
        if audit:compare(refs[p['id']]['generated'],baseline['generated'])
        full=p['prompt']+baseline['generated']['tokens']
        for mode in MODES[1:]:
            if audit:
                r=run(p,mode,'correctness',audit=True,reference=full)
                compare(baseline['generated'],r['generated'])
                if baseline['observed']['audit']['aligned_hashes']!=r['observed']['audit']['aligned_hashes']:
                    raise ValueError('Target full logits differ between arms')
        for mode in ('selected','editor'):
            r=run(p,mode,'cold',clear=True)
            compare(baseline['generated'],r['generated'])
        # Warm both trajectories into the shared cache, outside timing.
        for mode in MODES:
            compare(baseline['generated'],run(p,mode,'warmup')['generated'])
        for repeat in range(REPEATS):
            offset=(i+repeat)%len(MODES)
            for mode in MODES[offset:]+MODES[:offset]:
                r=run(p,mode,'timing',repeat=repeat)
                compare(baseline['generated'],r['generated'])
                if r['observed']['graph_misses']:raise ValueError('Warm run recaptured graph')
        save(args.output/'progress.json',dict(phase='evaluation',completed=i+1,total=len(groups['evaluation'])))
    save(args.output/'complete.json',summarize(records))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--phase',choices=['collection','reference','evaluation'])
    args=parser.parse_args()
    if args.phase:child(args);return
    if args.output.exists():raise ValueError('Fresh output directory required')
    (args.output/'records').mkdir(parents=True)
    from full_corpus_fit.collect import load_prompts
    preflight=args.project/'research/full-corpus-preflight-8454304'
    groups=select_prompts(load_prompts(preflight))
    save(args.output/'prompts.json',groups)
    save(args.output/'run.json',dict(schema='native-selector-v1',prompt_digest=digest(groups),
        prompt_index_sha256=sha(preflight/'prompts.jsonl'),gate_sha256=GATE_SHA,repeats=REPEATS,
        source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        configuration=config(args.project,False),budget_tokens=64,editor_optimizer_updates=0,final_test_evaluated=False))
    for phase,timeout in [('collection',1500),('reference',600),('evaluation',2600)]:
        subprocess.run([sys.executable,'-u','-m','refinement_research.native_selector','--project',str(args.project),
                        '--output',str(args.output),'--phase',phase],check=True,timeout=timeout)


if __name__=='__main__':main()
