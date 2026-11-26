"""Exact-shape CUDA graph prototype, with cold preparation and warm full-call cost."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from autoregressive_editor.data import condition
from copy_action_fit.data import CHECKPOINT_SHA256, SHARED_WEIGHTS_SHA256, digest, save, sha
from copy_action_fit.train import require
from refinement_research.gate import generator_hash
from refinement_research.train_gates import initialize, progress


class CapturedEditor:
    """One graph for one prefix length. Does not implement a serving graph cache."""
    def __init__(self,model,prefix,draft):
        self.model=model
        ids,segments=condition(prefix,draft,'draft')
        device=model.token.weight.device
        self.tokens=torch.tensor([ids],dtype=torch.long,device=device)
        self.types=torch.tensor([segments],dtype=torch.long,device=device)
        self.source=torch.tensor([draft],dtype=torch.long,device=device)
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):model.decode_tensors(self.tokens,self.types,self.source)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output,_=model.decode_tensors(self.tokens,self.types,self.source)

    def propose(self,prefix,draft):
        ids,segments=condition(prefix,draft,'draft')
        if len(ids)!=self.tokens.shape[1] or len(draft)!=16:
            raise ValueError('Graph shape mismatch; recapture or a serving cache is required')
        # These CPU construction and transfer costs are included in the timed call.
        self.tokens.copy_(torch.tensor([ids],dtype=torch.long))
        self.types.copy_(torch.tensor([segments],dtype=torch.long))
        self.source.copy_(torch.tensor([draft],dtype=torch.long))
        self.graph.replay()
        return self.output.flatten().tolist()


def measure(call):
    torch.cuda.synchronize();started=time.perf_counter()
    output=call()
    torch.cuda.synchronize()
    return output,1000*(time.perf_counter()-started)


def profile_eager(model,row,output):
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            for _ in range(3):model.propose(row['prefix'],row['draft'])
            torch.cuda.synchronize()
        path=output/'eager-profile.json';profile.export_chrome_trace(str(path))
        trace=json.loads(path.read_text())['traceEvents']
        kernels=[e for e in trace if e.get('cat')=='kernel' and e.get('ph')=='X']
        if not kernels:return dict(status='unavailable',reason='Profiler returned no CUDA kernel events')
        totals={}
        for event in kernels:
            name=event['name'];totals[name]=totals.get(name,0)+event['dur']/1000
        return dict(status='collected',calls=3,kernels=len(kernels),
            summed_kernel_ms_per_call=sum(totals.values())/3,
            largest_kernels_ms_per_call={k:v/3 for k,v in sorted(totals.items(),key=lambda p:-p[1])[:10]},
            scope='Instrumented sum of kernel durations, not unprofiled wall latency or exclusive bottleneck attribution')
    except RuntimeError as error:
        return dict(status='unavailable',reason=str(error),scope='Exact-output wall timing remains separate')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('initial','parent','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4);args.output.mkdir(parents=True,exist_ok=True)
    require(not (args.output/'complete.json').exists(),'Completed prototype must not be overwritten')
    budget=args.output/'budget.json'
    if not budget.exists():save(budget,dict(started_epoch=time.time(),maximum_seconds=1200))
    deadline=json.loads(budget.read_text())['started_epoch']+1140
    parent=json.loads((args.parent/'complete.json').read_text())
    require(parent['passed_checks'] and sha(args.initial)==CHECKPOINT_SHA256,'Parent/input provenance failed')
    # Read only previously audited development inputs; no new labels or target calls.
    payloads=json.loads((args.parent/'linear/development.json').read_text())['rows']
    import sqlite3
    bank=args.parent.parent/'full-corpus-run-20261117/bank.sqlite'
    from copy_action_fit.data import BANK_SHA256
    require(sha(bank)==BANK_SHA256,'Bank changed')
    connection=sqlite3.connect(f'file:{bank}?mode=ro',uri=True)
    rows=[json.loads(r[0]) for r in connection.execute("SELECT payload FROM examples WHERE split='development' ORDER BY id")]
    connection.close();require(len(rows)==2048 and len(payloads)==2048,'Changed development population')
    indices=[i*2047//63 for i in range(64)]
    contract=dict(schema='fixed-shape-graph-prototype-v1',kind='linear',blocks=64,
        parent_complete_sha256=sha(args.parent/'complete.json'),
        checkpoint_sha256=sha(args.parent/'linear/best.pt'),inputs_sha256=digest([rows[i] for i in indices]),
        warmups=2,repeats=10,interleaved_order=True,graph_warmups=3,optimizer_updates=0,final_test_loaded=False,
        source_hashes={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    require(contract['checkpoint_sha256']==parent['arms']['linear']['checkpoint_sha256'],'Gate checkpoint changed')
    save(args.output/'run.json',contract)
    model=initialize(args.initial,'linear')
    checkpoint=torch.load(args.parent/'linear/best.pt',map_location='cpu',weights_only=True)
    model.gate_head.load_state_dict(checkpoint['gate'],strict=True)
    records=[];changed=0;started=time.monotonic()
    for number,index in enumerate(indices):
        require(time.time()<deadline-30,'Prototype budget exhausted; no automatic extension')
        row,previous=rows[index],payloads[index]
        require((row['prompt_id'],row['anchor'])==(previous['prompt_id'],previous['anchor']) and
                row['draft']==previous['original_tokens'] and row['labels']==previous['label_tokens'],
                'Timing input differs from audited parent')
        expected=model.propose(row['prefix'],row['draft'])
        require(expected==previous['edited_tokens'],'Tensor refactor changed saved full-editor outputs')
        require(model.propose(row['prefix'],row['draft'],cached=False)==expected,'Cached/full mismatch')
        torch.cuda.synchronize();before=torch.cuda.memory_allocated();start=time.perf_counter()
        graph=CapturedEditor(model,row['prefix'],row['draft'])
        torch.cuda.synchronize();cold_ms=1000*(time.perf_counter()-start)
        memory=torch.cuda.memory_allocated()-before
        altered=list(row['draft']);altered[2]=(altered[2]+1)%model.config.vocab_size
        other=model.propose(row['prefix'],altered)
        require(graph.propose(row['prefix'],altered)==other,'Graph reused stale input/cache state')
        changed+=other!=expected
        require(graph.propose(row['prefix'],row['draft'])==expected,'Graph failed to restore input')
        calls={'eager':lambda:model.propose(row['prefix'],row['draft']),
               'graph':lambda:graph.propose(row['prefix'],row['draft'])}
        for call in calls.values():
            for _ in range(2):require(call()==expected,'Warmup output changed')
        samples={'eager':[],'graph':[]}
        for repeat in range(10):
            for method in (('eager','graph') if repeat%2==0 else ('graph','eager')):
                actual,ms=measure(calls[method]);require(actual==expected,'Timed output changed')
                samples[method].append(ms)
        records.append(dict(prompt_id=row['prompt_id'],anchor=row['anchor'],prefix_length=len(row['prefix']),
            eager_ms=statistics.median(samples['eager']),graph_ms=statistics.median(samples['graph']),
            capture_and_warmup_ms=cold_ms,additional_allocated_bytes=memory,samples_ms=samples))
        del graph,calls;gc.collect();torch.cuda.empty_cache()
        if (number+1)%8==0:
            save(args.output/'timing.json',records)
            progress(args.output,phase='graph_timing',completed=number+1,blocks=64)
    save(args.output/'timing.json',records)
    require(generator_hash(model)==SHARED_WEIGHTS_SHA256,'Generator changed')
    require(sha(args.parent/'linear/best.pt')==contract['checkpoint_sha256'],'Gate artifact changed')
    profile=profile_eager(model,rows[indices[32]],args.output);save(args.output/'profile-summary.json',profile)
    eager=statistics.median(r['eager_ms'] for r in records);graphed=statistics.median(r['graph_ms'] for r in records)
    quality=parent['arms']['linear']['development'];delta=quality['mean_edited_progress']-quality['mean_original_progress']
    complete=dict(passed_checks=True,blocks=64,exact_parent_outputs=True,cached_full='passed',
        changed_input_controls=64,changed_output_controls=changed,all_repeats_match=True,
        eager_median_ms=eager,graph_median_ms=graphed,editor_latency_ratio=eager/graphed,
        median_capture_and_warmup_ms=statistics.median(r['capture_and_warmup_ms'] for r in records),
        maximum_additional_allocated_bytes=max(r['additional_allocated_bytes'] for r in records),
        distinct_tested_prefix_lengths=len({r['prefix_length'] for r in records}),
        warm_break_even_baseline_round_ms=graphed*(quality['mean_original_progress']+1)/delta if delta>0 else None,
        generator_weights_sha256=generator_hash(model),gate_checkpoint_sha256=contract['checkpoint_sha256'],
        profile=profile,elapsed_seconds=time.monotonic()-started,optimizer_updates=0,final_test_loaded=False,
        scope='Warm fixed-shape editor prototype including input/output copies; cold capture separate; no serving cache or native generation speedup')
    save(args.output/'complete.json',complete);progress(args.output,phase='complete',eager_ms=eager,graph_ms=graphed)


if __name__=='__main__':main()
