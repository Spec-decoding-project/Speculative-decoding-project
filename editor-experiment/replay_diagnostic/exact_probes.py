"""Targeted probes after the four-fixture baseline is inspected.

Compare identical saved token prefixes, isolate the first query projection on
identical input rows, and separate upstream differences from vocabulary readout.
No label replacement or training occurs. The final FP32-linear intervention is
diagnostic and limited to the newly failing fixture, not a validated correction.
"""
from contextlib import contextmanager
import time
from torch.nn import functional as F


def difference(a, b):
    if a.shape != b.shape:
        raise ValueError('Cannot compare differently shaped trace rows')
    delta = a.float()-b.float()
    return dict(equal=torch.equal(a,b),changed=int((a!=b).sum()),
                max_abs=float(delta.abs().max()),rms=float(delta.square().mean().sqrt()),
                reference_rms=float(a.float().square().mean().sqrt()))


def trace_difference(a,b):
    if list(a)!=list(b):
        raise ValueError('Trace sites/order differ')
    return [dict(site=key,**difference(value,b[key])) for key,value in a.items()]


def pair_margins(hidden, fixture, weight):
    row = F.linear(hidden.float()[None],weight)[0]
    result = logits_summary(row[None], [fixture['labels'][fixture['focus']]], 0)['focus']
    result['position']=fixture['focus']
    return result


@contextmanager
def fp32_linears():
    originals=[]
    for module in harness.target.model.modules():
        if isinstance(module,torch.nn.Linear):
            originals.append((module,module.forward))
            def forward(x,m=module):
                return F.linear(x.float(),m.weight.float(),None if m.bias is None else m.bias.float()).to(x.dtype)
            module.forward=forward
    try:
        yield
    finally:
        for module,method in originals: module.forward=method


started=time.monotonic()
report['traces']=[]
weight_fp32=harness.target.lm_head.weight.float()
query=harness.target.model.layers[0].self_attn.q_proj

for fixture in fixtures:
    cached=harness.cached_saved(fixture,until=fixture['focus'],trace=True)
    cached_trace=harness.trace
    cached_q=harness.q_input
    native=harness.native(fixture,trace=True)
    native_trace=harness.trace
    native_q=harness.q_input
    index=fixture['focus']-1
    expected_native_row=native_q[:,index:index+1].contiguous()
    if cached_q.shape[1]!=1:
        raise ValueError('Expected single-token cached query')
    row_equal=torch.equal(cached_q,expected_native_row)
    if not row_equal:
        raise ValueError('First query input differs: shape-only projection isolation invalid')
    # This holds the weights and selected input vector exactly fixed. The only
    # difference is evaluating that vector alone or inside a 16-row matrix.
    isolation=[]
    for repeat in range(3):
        whole=query(native_q)[0,index]
        alone=query(expected_native_row)[0,0]
        isolation.append(dict(repeat=repeat,**difference(alone,whole),
                              whole_equals_native_trace=torch.equal(whole,native_trace['layer0.self_attn.q_proj']),
                              alone_equals_cached_trace=torch.equal(alone,cached_trace['layer0.self_attn.q_proj'])))
    previous=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        no_reduce=difference(query(expected_native_row)[0,0],query(native_q)[0,index])
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=previous
    fp32_whole=F.linear(native_q.float(),query.weight.float())[0,index].to(native_q.dtype)
    fp32_alone=F.linear(expected_native_row.float(),query.weight.float())[0,0].to(native_q.dtype)
    item=dict(id=fixture['row']['id'],anchor=fixture['anchor'],focus=fixture['focus'],
              cached=cached,native=native,layer_rows=trace_difference(cached_trace,native_trace),
              projection_isolation=dict(input_row_exact=row_equal,cached_shape=list(cached_q.shape),
                  block_shape=list(native_q.shape),input_row_sha256=tensor_digest(cached_q),
                  weight_sha256=tensor_digest(query.weight),default_reduced_bf16=str(previous),
                  repeated_default=isolation,reductions_disabled=no_reduce,fp32_then_bf16=difference(fp32_alone,fp32_whole)),
              final_hidden=difference(cached_trace['final_norm'],native_trace['final_norm']),
              cached_hidden_fp32_head=pair_margins(cached_trace['final_norm'],fixture,weight_fp32),
              native_hidden_fp32_head=pair_margins(native_trace['final_norm'],fixture,weight_fp32))
    report['traces'].append(item)
    write_json(args.output/'traces.partial.json',report)
    print({'id':item['id'],'focus':item['focus'],'same_projection_input':row_equal,
           'projection_difference':isolation[0],
           'cached_fp32_head_margin':item['cached_hidden_fp32_head']['signed_margin'],
           'native_fp32_head_margin':item['native_hidden_fp32_head']['signed_margin']},flush=True)

del weight_fp32
new=fixtures[0]
if new['row']['id']!='ultrachat_200k:train_sft:37680':
    raise ValueError('Wrong primary fixture')
with fp32_linears():
    report['fp32_linears_new_fixture']=dict(
        cached=harness.cached_saved(new,until=new['focus']),native=harness.native(new))
report['fp32_intervention_scope']='Only new fixture; no all-fixture correction or timing validation'
torch.cuda.synchronize()
report['probe_wall_seconds']=time.monotonic()-started
report['target_calls_after_probes']=harness.calls
write_json(args.output/'probes.json',report)
