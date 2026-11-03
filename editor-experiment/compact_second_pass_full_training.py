"""Controlled full-exposure editor fit; no allocation on import."""
from pathlib import Path
import json
import time
from second_pass_profile_worker import write, sha


def run(plan, paths, out, persist, deadline):
    import torch
    from safetensors.torch import load_file
    from conditional_redraft_loading import load_target, load_ordinary_drafter
    from conditional_full_reference_precision import FullReferenceFirstPass, FullConditionalReference, check_numerical_buffers
    from conditional_redraft_learning import state_digest
    from conditional_redraft_measure import prepare_fixture, check_native_first_pass
    from compact_second_pass_prototype import CompactSecondPass
    from compact_second_pass_decode import prepare_frozen_readout, audit_compact_round, measure_compact_comparison
    from second_pass_profile import audit_context
    from compact_second_pass_features import extract_record,materialize
    from compact_second_pass_learning import update,evaluate_batch,ARMS
    from copy import deepcopy
    import statistics
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    def guard():
        if time.time()>deadline-120:raise TimeoutError('Recovery reserve reached')
    guard()
    if not torch.cuda.is_available():raise RuntimeError('Requires actual CUDA measurements')
    torch.set_num_threads(4)
    torch.manual_seed(plan['editor_seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    if sha(paths['teacher'])!=plan['teacher_file_sha256']:raise ValueError('Teacher identity')
    target=load_target(paths['target'],device='cuda',dtype=torch.bfloat16)
    ordinary=FullReferenceFirstPass(load_ordinary_drafter(paths['draft'],device='cuda',dtype=torch.bfloat16))
    teacher=FullConditionalReference(ordinary,'feedback').to(dtype=torch.bfloat16).eval()
    teacher.load_state_dict(load_file(paths['teacher'],device='cuda'),strict=True)
    teacher.requires_grad_(False)
    if state_digest(teacher.state_dict())!=plan['teacher_state_sha256']:raise ValueError('Teacher deployment changed')
    check_numerical_buffers(ordinary,teacher)
    versions=tuple(p._version for m in (target,ordinary,teacher) for p in m.parameters())
    write(out/'runtime.json',dict(torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
        training='FP32 master parameters, BF16 autocast',deployment='BF16 frozen copy',purpose='three-arm learning screen'))
    (out/'features').mkdir(exist_ok=True)
    records=[];timings=[]
    for spec in plan['prompts']:
        guard();data=Path(paths['data'])
        meta=json.loads((data/'metadata'/(spec['id']+'.json')).read_text())
        bank_path=data/'banks'/(spec['id']+'.pt')
        if sha(bank_path)!=meta['bank_sha256'] or meta['split']!=spec['split']:raise ValueError('Input identity')
        bank=torch.load(bank_path,map_location='cpu',weights_only=True)
        for index in range(len(meta['anchors'])):
            guard();torch.cuda.synchronize();start=time.perf_counter()
            try:record=extract_record(target,ordinary,teacher,bank,meta,index)
            except BaseException:
                write(out/'extraction-failure.json',dict(prompt=spec['id'],index=index))
                from conditional_full_reference import restore_reference_example
                from compact_second_pass_bank import finish_saved_first_pass
                state,visible,original,labels,valid=restore_reference_example(target,ordinary,bank,meta,index)
                state=finish_saved_first_pass(ordinary,state)
                with torch.no_grad():scores=target.get_output_embeddings()(state.normalized[:,1:])
                torch.save(dict(states=state.normalized.cpu(),midpoint=state.midpoint.cpu(),
                    raw_features=state.raw_target_hidden.cpu(),scores=scores.cpu(),original=original.cpu(),
                    labels=labels.cpu(),valid=valid.cpu()),out/'extraction-failure.pt')
                persist();raise
            torch.cuda.synchronize();timings.append(dict(prompt=spec['id'],anchor=index,split=spec['split'],seconds=time.perf_counter()-start))
            records.append(record)
        torch.save(records[-len(meta['anchors']):],out/'features'/(spec['id']+'.pt'))
        if len(timings)%512==0:
            write(out/'extraction-timings.json',timings);persist()
        del bank
    write(out/'extraction-timings.json',timings);persist()
    training=[r for r in records if r['split']=='train']
    development=[r for r in records if r['split']=='development']
    if len(training)!=12288 or len(development)!=2048:raise ValueError('Full exposure incomplete')
    if {r['prompt_id'] for r in training}&{r['prompt_id'] for r in development}:raise ValueError('Split overlap')
    import random
    orders=[]
    for epoch in range(plan['epochs']):
        order=list(range(len(training)));random.Random(plan['order_seed']+epoch).shuffle(order);orders.append(order)
    write(out/'training-orders.json',orders)
    torch.manual_seed(plan['editor_seed'])
    initial=CompactSecondPass(target.config.hidden_size,rank=256,block_size=ordinary.drafter.block_size).cuda().float()
    torch.save(initial.state_dict(),out/'initial.pt')
    embedding,head=target.get_input_embeddings(),target.get_output_embeddings()
    rows=[];completed=[]
    def save_checkpoint(editor,optimizer,arm,epoch,step):
        destination=out/(arm+'-resume.pt')
        temporary=out/(arm+'-resume.tmp')
        torch.save(dict(model=editor.state_dict(),optimizer=optimizer.state_dict(),
                        arm=arm,epoch=epoch,step=step,rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(),plan=plan),temporary)
        temporary.replace(destination)
        write(out/'updates.json',rows);persist()
    for arm in ARMS:
        editor=deepcopy(initial);optimizer=torch.optim.AdamW(editor.parameters(),lr=plan['learning_rate'],weight_decay=.01)
        for epoch,order in enumerate(orders,1):
            for step,start_index in enumerate(range(0,len(order),32),1):
                guard();torch.cuda.synchronize();start=time.perf_counter()
                selected=[training[i] for i in order[start_index:start_index+32]]
                batches=[materialize(selected[j:j+4],embedding,head,'cuda') for j in range(0,32,4)]
                from torch.nn.attention import sdpa_kernel,SDPBackend
                with sdpa_kernel(SDPBackend.MATH):
                    metrics=update(editor,optimizer,batches,head.weight,arm=arm,
                                   autocast=lambda:torch.autocast('cuda',dtype=torch.bfloat16))
                torch.cuda.synchronize()
                rows.append(dict(arm=arm,epoch=epoch,step=step,seconds=time.perf_counter()-start,**metrics))
                del batches
                if step%64==0:save_checkpoint(editor,optimizer,arm,epoch,step)
            save_checkpoint(editor,optimizer,arm,epoch,step)
            deployed=deepcopy(editor).to(dtype=torch.bfloat16).eval().requires_grad_(False)
            checkpoint=out/(arm+'-epoch-'+str(epoch)+'.pt');torch.save(deployed.state_dict(),checkpoint)
            replay=deepcopy(deployed);replay.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
            with torch.no_grad():
                folded=deployed.folded_weight(head.weight);replay_folded=replay.folded_weight(head.weight)
            evaluations=[]
            for index,r in enumerate(development):
                guard();batch=materialize([r],embedding,head,'cuda')
                result=evaluate_batch(deployed,batch,head.weight,folded_weight=folded)
                if index==0 and evaluate_batch(replay,batch,head.weight,folded_weight=replay_folded)!=result:
                    raise ValueError('Checkpoint roundtrip')
                evaluations.append(dict(arm=arm,epoch=epoch,prompt=r['prompt_id'],anchor_index=r['anchor_index'],result=result))
            write(out/(arm+'-epoch-'+str(epoch)+'-evaluation.json'),evaluations)
            completed.append(dict(arm=arm,epoch=epoch,evaluations=len(evaluations)))
            write(out/'progress.json',dict(completed=completed,updates=len(rows)))
            persist();del deployed,replay,folded,replay_folded
        del editor,optimizer
    if versions!=tuple(p._version for m in (target,ordinary,teacher) for p in m.parameters()):raise ValueError('Frozen weights changed')
    write(out/'summary.json',dict(status='complete',records=len(records),updates=len(rows),completed=completed,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        limitation='Single seed; 64 previously exposed development prompts; fixed-context coverage, not generation speedup. Epoch three primary, all epochs reported.'))
    persist()
