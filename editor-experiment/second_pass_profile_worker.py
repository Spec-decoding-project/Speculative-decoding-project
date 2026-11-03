"""Released-weight profiling worker. Call explicitly on already staged assets.

No training, downloader, provider allocation or execution on import. The caller
must supply an external deadline, recovery controller and approved spending cap.
"""
from pathlib import Path
import hashlib
import json
import time


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def run(plan,paths,out,persist,deadline):
    import torch
    from safetensors.torch import load_file
    from conditional_redraft_loading import load_target,load_ordinary_drafter
    from conditional_full_reference_precision import FullReferenceFirstPass,FullConditionalReference,check_numerical_buffers
    from conditional_redraft_learning import state_digest
    from conditional_redraft_measure import prepare_fixture,check_native_first_pass
    from second_pass_profile import POLICIES,measure_context,profile_context,audit_context
    out=Path(out);out.mkdir(parents=True,exist_ok=True)

    def guard():
        if time.time()>deadline-120:raise TimeoutError('Two-minute recovery reserve reached')

    guard()
    if not torch.cuda.is_available():raise RuntimeError('Released-weight timing requires CUDA')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    if sha(paths['teacher'])!=plan['teacher_file_sha256']:
        raise ValueError('Teacher file identity changed')
    target=load_target(paths['target'],device='cuda',dtype=torch.bfloat16)
    ordinary=FullReferenceFirstPass(load_ordinary_drafter(paths['draft'],device='cuda',dtype=torch.bfloat16))
    teacher=FullConditionalReference(ordinary,'feedback').to(dtype=torch.bfloat16).eval()
    teacher.load_state_dict(load_file(paths['teacher'],device='cuda'),strict=True)
    teacher.requires_grad_(False)
    digest=state_digest(teacher.state_dict())
    if digest!=plan['teacher_state_sha256']:raise ValueError('Teacher deployment weights differ')
    check_numerical_buffers(ordinary,teacher)
    versions=[p._version for module in (target,ordinary,teacher) for p in module.parameters()]
    write(out/'runtime.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        dtype='bfloat16',target_graph=True,draft_graph=False,
        note='Diagnostic execution policy; not an optimized serving speed comparison'))
    persist()
    rows=[];setups=[];profiles=[]
    for spec in plan['contexts']:
        guard()
        data=Path(paths['data'])
        metadata=json.loads((data/'metadata'/(spec['id']+'.json')).read_text())
        bank_path=data/'banks'/(spec['id']+'.pt')
        if sha(bank_path)!=metadata['bank_sha256']:raise ValueError('Input bank identity changed')
        if metadata['split']!='development':raise ValueError('Only declared development bank permitted')
        bank=torch.load(bank_path,map_location='cpu',weights_only=True)
        anchor=metadata['anchors'][spec['anchor_index']]['anchor']
        if anchor!=spec['anchor']:raise ValueError('Frozen anchor changed')
        ids=bank['ids'][:anchor].to(target.device)[None]
        fixture=prepare_fixture(target,ordinary,{'feedback':teacher},ids,
            incoming_rows=spec['incoming_rows'],graph=True)
        check,tensors=check_native_first_pass(target,ordinary,fixture)
        if not all(check[k] for k in ('reference_repeats_exactly','capture_matches_exactly','finite')):
            torch.save(tensors,out/'native-capture-failure.pt');persist()
            raise ValueError('Native capture check failed')
        del tensors
        setups.append(dict(context=spec,setup=fixture.setup,first_pass_check=check))
        write(out/'setups.json',setups);persist()

        def save_row(row):
            rows.append(dict(context=spec,**row))

        try:
            measured=measure_context(target,ordinary,teacher,fixture,repeats=plan['repeats'],
                warmups=plan['warmups'],before_round=guard,on_row=save_row)
        finally:
            write(out/'rounds.json',rows);persist()
        reference={r['policy']:r['result'] for r in measured}
        a,b,c=[reference[x] for x in POLICIES]
        block=ordinary.drafter.block_size
        if not (a['query_count']==b['query_count']==block and c['query_count']==2*block-2
                and a['packed'][:2]==b['packed'][:2]==c['packed'][:2]
                and a['packed']==c['packed'][:block] and b['packed'][2:]==c['packed'][block:]):
            raise ValueError('Cross-policy proposal assembly mismatch')
        eos=target.config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos])
        if any(any(t in eos for t in r['accepted']+[r['bonus']]) for r in reference.values()):
            write(out/'terminal-context.json',dict(context=spec,results=reference));persist()
            raise ValueError('Terminal context requires explicit EOS-aware accounting')
        if spec['profile']:
            for policy in POLICIES:
                guard()
                label=f"{spec['id']}-{spec['anchor_index']}-{policy}"
                audit,tensors=audit_context(target,ordinary,teacher,fixture,policy)
                write(out/(label+'.audit.json'),audit)
                disagreement=(not audit['following_token_matches'] or
                    any(not p['anchor_matches'] or not p['accepted_decision_matches'] for p in audit['paths']))
                if disagreement or not audit['finite'] or any(p['differing_next_token_winners'] for p in audit['paths']):
                    torch.save({k:v.cpu() for k,v in tensors.items()},out/(label+'.scores.pt'))
                    persist()
                if disagreement or not audit['finite']:
                    raise ValueError('Accepted decision or following cache differs; evidence preserved')
                del tensors
                profile=profile_context(target,ordinary,teacher,fixture,policy,out/(label+'.trace.json'))
                if profile['result']!=reference[policy]:
                    write(out/'profile-mismatch.json',dict(context=spec,profile=profile,reference=reference[policy]));persist()
                    raise ValueError('Instrumentation changed decoded result')
                profiles.append(dict(context=spec,policy=policy,**profile))
                write(out/'profiles.json',profiles);persist()
        del bank,fixture,ids
    final_versions=[p._version for module in (target,ordinary,teacher) for p in module.parameters()]
    if versions!=final_versions or state_digest(teacher.state_dict())!=digest:
        raise ValueError('Frozen weights changed')
    write(out/'summary.json',dict(status='complete',contexts=len(plan['contexts']),
        timing_rows=len(rows),profiled_rounds=len(profiles),teacher_state_sha256=digest,
        generation_speedup_claim=False,
        limitation='Historical x1-conditioned teacher; fixed-context diagnostics; cross-shape fidelity must be separately audited before serving claims.'))
    persist()
