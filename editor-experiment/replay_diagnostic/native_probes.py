"""Actual pinned spec_generate verifier and targeted numerical probes.

Run as session.py's second stage. Oracle injection replaces only the drafter's
15 sampled tokens, transparently, to hold the verified prefix fixed. It is not
an editor or a performance result. Original source and labels are never changed.
"""
from contextlib import contextmanager, nullcontext
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

from collect import native_class, loading_ok
from prepare import verify
from transformers import DynamicCache, GenerationConfig
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

started = time.monotonic()
pin = json.loads(Path('research/editor-feature-bank/feature_bank/model_pin.json').read_text())
draft_path = Path('data/models/qwen3-8b-dflash-b16')
verify(draft_path,pin)
draft_class, _ = native_class(draft_path)
draft, load_info = draft_class.from_pretrained(draft_path,dtype=torch.bfloat16,
    device_map='cuda:0',attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
loading_ok(load_info)
draft.eval().requires_grad_(False)
report['native_draft_pin']=pin
report['native_source_sha256']=hashlib.sha256((draft_path/'dflash.py').read_bytes()).hexdigest()
qwen_source=Path(inspect.getfile(type(target.model.layers[0])))
report['qwen_source_sha256']=hashlib.sha256(qwen_source.read_bytes()).hexdigest()
report['native_modes']=[]
report['original_generation']=[]
report['native_weights_trainable']=dict(target=sum(p.numel() for p in target.parameters() if p.requires_grad),
                                        draft=sum(p.numel() for p in draft.parameters() if p.requires_grad))
trace_state=dict(enabled=False,index=0,values={})


def budget():
    if time.monotonic()-started >= forward_budget_seconds:
        raise RuntimeError('Ten-minute forward/setup budget exhausted')


def checksum(ids):
    return hashlib.sha256(json.dumps(ids.detach().cpu().tolist()).encode()).hexdigest()


def compare_logits(logits, labels):
    actual=logits.argmax(-1).flatten().tolist()
    if len(actual)!=len(labels):
        raise ValueError('Prediction/label offset mismatch')
    last=logits.reshape(-1,logits.shape[-1])[-1].float()
    top=last.topk(5)
    return dict(matches=sum(a==b for a,b in zip(actual,labels)),
        mismatches=[dict(position=i+16-len(labels),expected=a,actual=b)
                    for i,(a,b) in enumerate(zip(labels,actual)) if a!=b],
        last=dict(expected=labels[-1],actual=int(last.argmax()),
                  token279=float(last[279]),token3070=float(last[3070]),
                  margin279_minus3070=float(last[279]-last[3070]),
                  top5=[dict(id=i,logit=v) for i,v in zip(top.indices.tolist(),top.values.tolist())]))


def trace_hook(name):
    def hook(module, inputs, output):
        if trace_state['enabled']:
            value=output[0] if isinstance(output,tuple) else output
            trace_state['values'][name]=value[0,trace_state['index']].detach().clone()
    return hook


hooks=[]
hooks.append(target.model.embed_tokens.register_forward_hook(trace_hook('embedding')))
first=target.model.layers[0]
for name in ('input_layernorm','self_attn.q_proj','self_attn.k_proj','self_attn.v_proj',
             'self_attn.q_norm','self_attn.k_norm','self_attn.o_proj',
             'post_attention_layernorm','mlp.gate_proj','mlp.up_proj','mlp.down_proj'):
    hooks.append(first.get_submodule(name).register_forward_hook(trace_hook('layer0.'+name)))
for i,layer in enumerate(target.model.layers):
    hooks.append(layer.register_forward_hook(trace_hook(f'layer{i}.output')))
hooks.append(target.model.norm.register_forward_hook(trace_hook('final_norm')))


def trace_diff(reference,other):
    if reference.keys()!=other.keys():
        raise ValueError('Layer tracing paths differ')
    rows=[]
    for key,a in reference.items():
        b=other[key]
        rows.append(dict(site=key,equal=torch.equal(a,b),changed=int((a!=b).sum()),
                         max_abs=float((a.float()-b.float()).abs().max())))
    return rows


def cached(fixture,ones_mask=False):
    budget()
    prefix,labels=fixture['prefix'],fixture['labels']
    cache=DynamicCache()
    logits=[]
    trace_state['values']={}
    for i in range(16):
        budget()
        ids=torch.tensor([prefix if i==0 else [labels[i-1]]],device=target.device)
        positions=torch.arange(0 if i==0 else len(prefix)+i-1,len(prefix)+i,device=target.device)[None]
        trace_state.update(enabled=i==15,index=0)
        kwargs=dict(position_ids=positions,past_key_values=cache,use_cache=True,
                    logits_to_keep=1,output_hidden_states=True)
        if ones_mask:
            kwargs['attention_mask']=torch.ones((1,len(prefix)+i),dtype=torch.long,device=target.device)
        output=target(ids,**kwargs)
        logits.append(output.logits.clone())
        del output
    trace_state['enabled']=False
    return compare_logits(torch.cat(logits,dim=1),labels),trace_state['values']


def uncached(fixture):
    budget()
    prefix,labels=fixture['prefix'],fixture['labels']
    ids=torch.tensor([prefix+labels[:-1]],device=target.device)
    trace_state.update(enabled=True,index=len(prefix)+14,values={})
    out=target(ids,use_cache=False,logits_to_keep=16,output_hidden_states=True)
    trace_state['enabled']=False
    return compare_logits(out.logits,labels),trace_state['values']


def native(fixture,oracle):
    budget()
    prefix,labels=fixture['prefix'],fixture['labels']
    target_calls=[]
    trace_state['values']={}
    sample_globals=vars(sys.modules[type(draft).__module__])
    original_sample=sample_globals['sample']
    sampled_calls=[]
    def controlled_sample(logits,temperature=0):
        sampled_calls.append(list(logits.shape))
        if oracle and logits.shape[1]==15:
            return torch.tensor([labels[1:]],device=logits.device)
        return original_sample(logits,temperature)
    def pre_hook(module,inputs,kwargs):
        ids=inputs[0]
        cache=kwargs['past_key_values']
        entry=dict(input_ids=ids.detach().clone(),positions=kwargs['position_ids'][0].tolist(),
                   cache_before=int(cache.get_seq_length()),use_cache=kwargs['use_cache'],
                   logits_to_keep=kwargs.get('logits_to_keep','all'),
                   attention_mask=kwargs.get('attention_mask'))
        if entry['attention_mask'] is not None:
            raise ValueError('Unexpected changed native attention-mask contract')
        target_calls.append(entry)
        trace_state.update(enabled=len(target_calls)==2,index=14)
    def post_hook(module,inputs,kwargs,output):
        target_calls[-1]['logits']=output.logits.detach().clone()
        target_calls[-1]['cache_after']=int(kwargs['past_key_values'].get_seq_length())
        trace_state['enabled']=False
    pre=target.register_forward_pre_hook(pre_hook,with_kwargs=True)
    post=target.register_forward_hook(post_hook,with_kwargs=True)
    sample_globals['sample']=controlled_sample
    try:
        # max_new_tokens=1 executes exactly one native draft/verify round, then
        # truncates the returned output. Hooks retain the full verification call.
        draft.spec_generate(target,torch.tensor([prefix],device=target.device),
                            max_new_tokens=1,stop_token_ids=None,temperature=0)
    finally:
        sample_globals['sample']=original_sample
        pre.remove();post.remove();trace_state['enabled']=False
    if len(target_calls)!=2 or [x[1] for x in sampled_calls]!=[1,15,16]:
        raise ValueError('Unexpected native generate call sequence')
    prefill,verify=target_calls
    candidate=verify['input_ids'][0]
    posterior=verify['logits'][0,:15].argmax(-1)
    acceptance=int((candidate[1:]==posterior).cumprod(0).sum())
    expected_candidate=torch.tensor(labels,device=target.device)
    candidate_matches=candidate==expected_candidate
    first_error=next((i for i,v in enumerate(candidate_matches.tolist()) if not v),16)
    if oracle and not bool(candidate_matches[1:].all()):
        raise ValueError('Oracle candidate injection failed')
    if verify['cache_before']!=len(prefix) or verify['positions']!=list(range(len(prefix),len(prefix)+16)):
        raise ValueError('Native cache/position alignment mismatch')
    result=dict(oracle_candidate=oracle,anchor_equal=int(prefill['logits'].argmax())==labels[0],
        candidate_sha256=checksum(candidate),candidate_equal_saved=bool(candidate_matches.all()),
        first_candidate_error=first_error,native_accepted_proposals=acceptance,
        posterior_on_candidate=compare_logits(verify['logits'][:,:15],labels[1:]),
        interpretation='Posterior after first candidate error is not on the saved context',
        calls=[dict(input_length=x['input_ids'].shape[1],input_sha256=checksum(x['input_ids']),
                    positions=x['positions'],cache_before=x['cache_before'],cache_after=x['cache_after'],
                    use_cache=x['use_cache'],logits_to_keep=x['logits_to_keep'],attention_mask=None)
               for x in target_calls])
    return result,trace_state['values']


@contextmanager
def no_reduced_bf16():
    old=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    try: yield
    finally: torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=old


@contextmanager
def fp32_transformer_linears():
    originals=[]
    for module in target.model.modules():
        if isinstance(module,torch.nn.Linear):
            originals.append((module,module.forward))
            def forward(x,m=module):
                return F.linear(x.float(),m.weight.float(),None if m.bias is None else m.bias.float()).to(x.dtype)
            module.forward=forward
    try: yield
    finally:
        for module,original in originals:
            module.forward=original


report['default_reduced_bf16']=str(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
modes=[('default',nullcontext),('math_sdpa',lambda:sdpa_kernel(SDPBackend.MATH)),
       ('bf16_reductions_disabled',no_reduced_bf16),('fp32_transformer_linears',fp32_transformer_linears)]
try:
    for name,context in modes:
        with context():
            for fixture in fixtures:
                budget()
                c,ct=cached(fixture)
                u,ut=uncached(fixture)
                n,nt=native(fixture,oracle=True)
                entry=dict(mode=name,id=fixture['row']['id'],cached=c,uncached=u,oracle_native=n,
                           cached_vs_native=trace_diff(ct,nt),cached_vs_uncached=trace_diff(ct,ut))
                if name=='default':
                    entry['oracle_native_repeats']=[n]+[native(fixture,oracle=True)[0] for _ in range(2)]
                    entry['actual_dflash_candidate']=native(fixture,oracle=False)[0]
                    entry['cached_with_ones_mask'],ones_trace=cached(fixture,ones_mask=True)
                    entry['cached_mask_trace']=trace_diff(ct,ones_trace)
                    cfg=target.generation_config
                    config=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=16,use_cache=True,
                        eos_token_id=cfg.eos_token_id,bos_token_id=cfg.bos_token_id,pad_token_id=cfg.pad_token_id)
                    generated=[]
                    for repeat in range(3):
                        budget()
                        ids=torch.tensor([fixture['prefix']],device=target.device)
                        result=target.generate(input_ids=ids,attention_mask=torch.ones_like(ids),generation_config=config)
                        actual=result[0,len(fixture['prefix']):].tolist()
                        generated.append(dict(repeat=repeat,length=len(actual),
                            mismatches=[dict(position=i,expected=a,actual=b)
                              for i,(a,b) in enumerate(zip(fixture['labels'],actual)) if a!=b]))
                    report['original_generation'].append(dict(id=fixture['row']['id'],repeats=generated))
                report['native_modes'].append(entry)
                report['native_seconds_so_far']=time.monotonic()-started
                write_json(args.output/'native.partial.json',report)
                first_difference=next((r['site'] for r in entry['cached_vs_native'] if not r['equal']),None)
                print(json.dumps(dict(mode=name,id=fixture['row']['id'],cached_matches=c['matches'],
                    uncached_matches=u['matches'],native_accepted=n['native_accepted_proposals'],
                    native_mismatches=n['posterior_on_candidate']['mismatches'],first_difference=first_difference)),flush=True)
finally:
    for hook in hooks: hook.remove()
    trace_state['enabled']=False
torch.cuda.synchronize()
report['native_seconds']=time.monotonic()-started
report['total_forward_and_draft_setup_seconds']=report['baseline_seconds']+report['native_seconds']
write_json(args.output/'native.json',report)
