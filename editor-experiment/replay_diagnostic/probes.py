"""Readout/cache probes, run after the repeated baseline in session.py.

No correction or contract change is applied. Only the three fixed fixtures run.
Globals are supplied by the bounded session; raw source records stay on Nexus.
"""
import hashlib
import json
import time
from torch.nn import functional as F
from transformers import GenerationConfig

started = time.monotonic()
report['probes'] = []


def budget():
    if time.monotonic() - started >= forward_budget_seconds:
        raise RuntimeError('Ten-minute total forward-probe budget exhausted')


def summary(logits, expected):
    values = logits.float()
    winner = int(values.argmax())
    maximum = values.max()
    ties = torch.where(values == maximum)[0].tolist()
    top = values.topk(5)
    return dict(expected=expected, actual=winner, dtype=str(logits.dtype),
                expected_logit=float(values[expected]), actual_logit=float(values[winner]),
                expected_minus_actual=float(values[expected]-values[winner]),
                maximum_tie_ids=ties,
                top5=[dict(id=i, logit=v) for i,v in zip(top.indices.tolist(),top.values.tolist())])


def comparison(logits, labels):
    actual = logits.argmax(-1).reshape(-1).tolist()
    return dict(matches=sum(a==b for a,b in zip(actual,labels)),
                mismatches=[dict(position=i,expected=a,actual=b)
                            for i,(a,b) in enumerate(zip(labels,actual)) if a!=b],
                last=summary(logits.reshape(-1,logits.shape[-1])[-1],labels[-1]))


def input_digest(ids):
    return hashlib.sha256(json.dumps(ids.detach().cpu().tolist()).encode()).hexdigest()


weight_fp32 = target.lm_head.weight.float()
for fixture in fixtures:
    budget()
    row, prefix, labels = fixture['row'], fixture['prefix'], fixture['labels']
    device = target.device
    full = torch.tensor([prefix + labels[:-1]], device=device)
    base = target(full, use_cache=False, logits_to_keep=16, output_hidden_states=True)
    hidden = base.hidden_states[-1][:,-16:]
    original_logits = base.logits.clone()
    item = dict(id=row['id'], anchor=fixture['anchor'], input_sha256=input_digest(full),
                base=comparison(original_logits,labels),
                hidden_shape=list(hidden.shape), head_weight_dtype=str(target.lm_head.weight.dtype))
    head_bf16 = target.lm_head(hidden)
    item['direct_head_equals_base'] = torch.equal(head_bf16,original_logits)
    if not item['direct_head_equals_base']:
        raise ValueError('Cannot isolate head: direct readout differs from model output')
    head_one = torch.cat([target.lm_head(hidden[:,i:i+1]) for i in range(16)],dim=1)
    head_fp32 = F.linear(hidden.float(),weight_fp32)
    head_fp32_one = torch.cat([F.linear(hidden[:,i:i+1].float(),weight_fp32)
                              for i in range(16)],dim=1)
    item['same_hidden_bf16_one_row'] = comparison(head_one,labels)
    item['same_hidden_fp32_16_rows'] = comparison(head_fp32,labels)
    item['same_hidden_fp32_one_row'] = comparison(head_fp32_one,labels)
    item['head_row_comparison'] = dict(
        bf16_argmax_changes=int((head_one.argmax(-1)!=head_bf16.argmax(-1)).sum()),
        bf16_max_abs=float((head_one-head_bf16).abs().max()),
        fp32_argmax_changes=int((head_fp32_one.argmax(-1)!=head_fp32.argmax(-1)).sum()),
        fp32_max_abs=float((head_fp32_one-head_fp32).abs().max()))
    # Keep the saved prefix identical in every path. No generated alternative
    # token is fed to the teacher-forced cached replay.
    cached_hidden, cached_logits, step_inputs = [], [], []
    cache = None
    for i in range(16):
        budget()
        ids = torch.tensor([prefix if i==0 else [labels[i-1]]],device=device)
        positions = torch.arange(0 if i==0 else len(prefix)+i-1,len(prefix)+i,device=device)[None]
        mask = torch.ones((1,len(prefix)+i),dtype=torch.long,device=device)
        out = target(ids, attention_mask=mask, position_ids=positions,
                     past_key_values=cache,use_cache=True,logits_to_keep=1,output_hidden_states=True)
        cache=out.past_key_values
        cached_hidden.append(out.hidden_states[-1][:,-1:].clone())
        cached_logits.append(out.logits.clone())
        step_inputs.append(dict(step=i,input_length=ids.shape[1],mask_length=mask.shape[1],
                                position_start=int(positions[0,0]),position_end=int(positions[0,-1]),
                                cache_after=int(cache.get_seq_length()),input_sha256=input_digest(ids)))
        del out
    cached_hidden=torch.cat(cached_hidden,dim=1)
    cached_logits=torch.cat(cached_logits,dim=1)
    item['cached_saved_prefix']=comparison(cached_logits,labels)
    item['cached_positions']=step_inputs
    item['cached_hidden_vs_batched']=dict(
        equal=torch.equal(cached_hidden,hidden),
        max_abs=float((cached_hidden-hidden).abs().max()),
        final_max_abs=float((cached_hidden[:,-1]-hidden[:,-1]).abs().max()),
        final_changed_dimensions=int((cached_hidden[:,-1]!=hidden[:,-1]).sum()))
    item['cached_hidden_fp32']=comparison(F.linear(cached_hidden.float(),weight_fp32),labels)
    item['cached_hidden_bf16_16_rows']=comparison(target.lm_head(cached_hidden),labels)
    del cache
    # Original generator configuration, bounded to the first 16 output tokens.
    # All fixtures are the first block, so no omitted earlier generation exists.
    if len(row['prompt_token_ids']) != len(prefix):
        raise ValueError('Original-generate comparison requires first-block fixtures')
    source_config=target.generation_config
    config=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=16,
        bos_token_id=source_config.bos_token_id,eos_token_id=source_config.eos_token_id,
        pad_token_id=source_config.pad_token_id,use_cache=True)
    item['original_generation_config']=config.to_dict()
    generated_runs=[]
    for repetition in range(3):
        budget()
        ids=torch.tensor([prefix],device=device)
        generated=target.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
                                 generation_config=config)
        actual=generated[0,len(prefix):].tolist()
        mismatches=[dict(position=i,expected=a,actual=b)
                    for i,(a,b) in enumerate(zip(labels,actual)) if a!=b]
        generated_runs.append(dict(repeat=repetition,length=len(actual),matches=16-len(mismatches)
                                   if len(actual)==16 else None,mismatches=mismatches))
    item['original_generate']=generated_runs
    # Control for the explicit mask: preserve sequence length/cache mode/head rows.
    budget()
    mask_out=target(full,attention_mask=torch.ones_like(full),use_cache=False,
                    logits_to_keep=16,output_hidden_states=True)
    item['explicit_ones_mask']=comparison(mask_out.logits,labels)
    item['explicit_mask_hidden_equal']=torch.equal(mask_out.hidden_states[-1][:,-16:],hidden)
    del mask_out
    # Retain only last readout row with the same complete backbone input.
    one_out=target(full,use_cache=False,logits_to_keep=1,output_hidden_states=True)
    item['one_row_model']=summary(one_out.logits[0,-1],labels[-1])
    item['one_row_model_hidden_equal']=torch.equal(one_out.hidden_states[-1][:,-16:],hidden)
    del one_out
    report['probes'].append(item)
    write_json(args.output/'probes.partial.json',report)
    print(json.dumps(dict(id=row['id'],base=item['base'],
                          cached=item['cached_saved_prefix'],generation=generated_runs)),flush=True)
    del base,hidden,original_logits,head_bf16,head_one,head_fp32,head_fp32_one,cached_logits,cached_hidden
torch.cuda.synchronize()
report['probe_seconds']=time.monotonic()-started
report['total_forward_seconds']=report['baseline_seconds']+report['probe_seconds']
write_json(args.output/'probes.json',report)
