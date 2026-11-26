"""Paired editor-only kernel attribution, on native-pilot inputs."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from copy_action_fit.data import save
from refinement_research.editor_kernels import editor_kernels
from refinement_research.graph_replay import CapturedEditor
from refinement_research.train_gates import initialize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','parent','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    from vllm.model_executor.determinism import batch_invariant
    batch_invariant.init_batch_invariance()
    rows=json.loads((args.parent/'candidate-results.json').read_text())
    inputs=[r['observed']['edits'][0] for r in rows if r['phase']=='correctness' and r['mode']=='editor']
    if len(inputs)!=8:raise ValueError('Require eight completed native inputs')
    model=initialize(args.project/'research/full-corpus-run-20261117/training/draft/best.pt','linear')
    state=torch.load(args.project/'research/broader-gates-20261125/linear/best.pt',weights_only=True,map_location='cpu')
    model.gate_head.load_state_dict(state['gate']);model.requires_grad_(False)
    results=[]
    for index,row in enumerate(inputs):
        prefix,draft=row['prefix'][-1024:],row['original']
        for mode in ('invariant','standard','invariant'):
            with editor_kernels(mode):
                expected=model.propose(prefix,draft)
                graph=CapturedEditor(model,prefix,draft)
            for _ in range(2):assert graph.propose(prefix,draft)==expected
            wall,gpu=[],[]
            for _ in range(5):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize();start=time.perf_counter();a.record()
                actual=graph.propose(prefix,draft)
                b.record();torch.cuda.synchronize()
                wall.append(1000*(time.perf_counter()-start));gpu.append(a.elapsed_time(b))
                assert actual==expected and batch_invariant._batch_invariant_MODE
            results.append(dict(case=index,mode=mode,prefix_length=len(prefix),output=expected,
                wall_ms=wall,gpu_ms=gpu))
            del graph;gc.collect();torch.cuda.empty_cache()
    for i in range(8):
        a,b,c=results[3*i:3*i+3]
        assert a['output']==c['output']
    save(args.output,dict(cases=8,records=results,
        changed_output_cases=sum(results[3*i]['output']!=results[3*i+1]['output'] for i in range(8)),
        medians_ms={m:statistics.median(statistics.median(r['wall_ms']) for r in results if r['mode']==m)
                    for m in ('invariant','standard')},
        restored_invariant_outputs=True,scope='Editor-only, fixed weights/inputs/GPU; ATen dispatch is the changed variable.'))


if __name__=='__main__':main()
