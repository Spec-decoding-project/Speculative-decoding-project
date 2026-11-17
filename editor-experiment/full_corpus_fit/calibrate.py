"""Validate the enlarged context contract and measure collection/training rates."""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import statistics
import time

from full_corpus_fit.bank import Bank
from full_corpus_fit.collect import bucket, load_prompts, read_chunk
from full_corpus_fit.preflight import save, sha


def comparison(native, greedy, previous):
    outputs, hashes = [], []
    for path in (native, greedy):
        complete = json.loads((path/'complete.json').read_text())
        tokens, aligned = {}, {}
        for name, expected in complete['chunks'].items():
            if sha(path/name) != expected:
                raise ValueError('Calibration chunk changed')
            chunk = read_chunk(path/name)
            for identity, generated in zip(chunk['prompt_ids'], chunk['generated']):
                tokens[identity] = generated['tokens']
            aligned.update(chunk['audit']['aligned_hashes'])
        outputs.append(tokens)
        hashes.append(aligned)
    if outputs[0] != outputs[1] or hashes[0] != hashes[1] or len(outputs[0]) != 128:
        raise ValueError('New context contract failed exact native/greedy token or full-logit agreement')
    old = json.loads((previous/'native.json').read_text())
    matched = 0
    for name, expected in old['tokens'].items():
        if name in outputs[0]:
            if expected != outputs[0][name]:
                raise ValueError('Previously validated canonical continuation changed')
            matched += 1
    if matched != 4:
        raise ValueError('Missing four fixed canonical training fixtures')
    return dict(passed=True, prompts=128, emitted_tokens=sum(map(len,outputs[0].values())),
                matched_full_logit_rows=len(hashes[0]), previous_canonical_fixtures=matched)


def main():
    import torch
    from autoregressive_editor.learning import Trainer
    from autoregressive_editor.model import Config
    from one_block_fit.train import causal_check, weights_hash
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--greedy',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--preflight',type=Path,required=True)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import os
    if os.environ.get('VLLM_BATCH_INVARIANT') != '0':
        raise ValueError('Target invariance leaked into editor profiling')
    torch.set_num_threads(4)
    agreement=comparison(args.native,args.greedy,args.previous)
    bank=Bank(args.bank)
    ids=bank.ids('train')
    if not ids:
        raise ValueError('No calibration training examples')
    by_bucket=defaultdict(list)
    for identity in ids:
        r=bank.row(identity)
        by_bucket[next((i for i,x in enumerate((128,512,1024,2240)) if r['anchor']<=x),4)].append(identity)
    rates={}
    for mode in ('context','draft'):
        trainer=Trainer(Config(),mode,seed=0,lr=1e-3,device='cuda')
        if weights_hash(trainer.model) != '672080cdb5b25a32ae824198f3549e43979352bab0f621e75379dd9c46d03ab8':
            raise ValueError('Editor initialization changed')
        causal_check(trainer,max((bank.row(i) for i in ids),key=lambda r:len(r['prefix'])))
        samples={}
        for group, candidates in sorted(by_bucket.items()):
            rows=[bank.row(candidates[i%len(candidates)]) for i in range(32)]
            trainer.update(rows)
            times=[]
            for _ in range(5):
                torch.cuda.synchronize(); start=time.perf_counter()
                result=trainer.update(rows)
                torch.cuda.synchronize(); times.append(time.perf_counter()-start)
                if result['suffix_tokens'] != 448:
                    raise ValueError('Full-run effective batch normalization failed')
            samples[str(group)]=dict(seconds_per_update=statistics.median(times),samples=times,
                examples_per_update=32,mean_context=statistics.mean(len(r['prefix']) for r in rows))
        rates[mode]=samples
        del trainer
        torch.cuda.empty_cache()
    prompts=load_prompts(args.preflight)
    population=Counter(bucket(p) for p in prompts)
    measured=json.loads((args.native/'complete.json').read_text())['measurements']
    per_bucket=defaultdict(list)
    for r in measured:
        groups={int(k):v for k,v in r['buckets'].items()}
        if len(groups)==1:
            per_bucket[next(iter(groups))].append(r['seconds']/r['prompts'])
    global_rate=sum(r['seconds'] for r in measured)/128
    projected_generation=sum(n*statistics.mean(per_bucket[b]) if per_bucket[b] else n*global_rate
                             for b,n in population.items())
    training_population=Counter(bucket(p) for p in prompts if p['split']=='train')
    projected_training={}
    for mode in rates:
        fallback=max(v['seconds_per_update'] for v in rates[mode].values())
        projected_training[mode]=sum(n*4/32*3*rates[mode].get(str(b),
            {'seconds_per_update':fallback})['seconds_per_update'] for b,n in training_population.items())
    native_run=json.loads((args.native/'run.json').read_text())
    result=dict(agreement=agreement,training_rates=rates,population_by_prompt_length=population,
        native_seconds_per_prompt_by_length={str(b):statistics.mean(v) for b,v in per_bucket.items()},
        projected_native_generation_seconds=projected_generation,
        projected_training_seconds_three_epochs=projected_training,
        source_hashes=native_run['source_hashes'],native_config=native_run['config'],
        estimation_scope='Up to four blocks per prompt; calibration oversamples long inputs; excludes queue/startup/audit/evaluation overhead',
        peak_editor_gpu_bytes=torch.cuda.max_memory_allocated())
    save(args.output,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
