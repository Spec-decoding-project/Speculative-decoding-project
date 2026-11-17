"""Bounded four-fixture original-generator/native-logit comparison, no training.

Stage one reproduces the saved outputs using the authenticated original method.
Stage two is a separately supplied probes.py after the baseline is reviewed.
Every top-level target forward, including generate(), counts toward 300 calls.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import runpy
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'feature_bank'))
from collect import loading_ok, native_class, target_fingerprints, write_json
from data import block_inputs, records
from prepare import sha256, verify


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def logits_summary(logits, labels, focus):
    logits = logits.reshape(-1, logits.shape[-1])
    actual = logits.argmax(-1).tolist()
    if len(actual) != len(labels):
        raise ValueError('Misaligned replay logits')
    row = logits[focus].float()
    expected = labels[focus]
    alternate = 16984 if expected == 12946 else 3070 if expected == 279 else next(
        token for token in row.topk(2).indices.tolist() if token != expected)
    top = row.topk(5)
    return dict(matches=sum(a == b for a, b in zip(actual, labels)),
                mismatches=[dict(position=i, expected=a, actual=b) for i, (a, b) in enumerate(zip(labels, actual)) if a != b],
                focus=dict(position=focus, expected=expected, actual=int(row.argmax()), alternate=alternate,
                           expected_logit=float(row[expected]), alternate_logit=float(row[alternate]),
                           signed_margin=float(row[expected]-row[alternate]),
                           maximum_tie_ids=torch.where(row == row.max())[0].tolist(),
                           top5=[dict(id=i, logit=v) for i,v in zip(top.indices.tolist(),top.values.tolist())]))


class Harness:
    def __init__(self, target, draft, backend, output):
        self.target, self.draft, self.backend, self.output = target, draft, backend, output
        self.calls = 0
        self.seconds = 0.0
        self.limit = 300
        self.max_seconds = 600
        self.started = None
        self.before = target.register_forward_pre_hook(self._before, with_kwargs=True)
        self.after = target.register_forward_hook(self._after, with_kwargs=True)
        self.capture_trace = False
        self.trace_row = 0
        self.trace = {}
        self.trace_handles = []
        modules = [('embedding', target.model.embed_tokens)]
        layer = target.model.layers[0]
        for name in ('input_layernorm','self_attn.q_proj','self_attn.q_norm',
                     'self_attn.k_proj','self_attn.k_norm','self_attn.v_proj',
                     'self_attn.o_proj','post_attention_layernorm','mlp.gate_proj',
                     'mlp.up_proj','mlp.down_proj'):
            modules.append(('layer0.'+name, layer.get_submodule(name)))
        modules += [(f'layer{i}.output', layer) for i, layer in enumerate(target.model.layers)]
        modules.append(('final_norm', target.model.norm))
        for name,module in modules:
            self.trace_handles.append(module.register_forward_hook(self._trace_hook(name)))
        self.q_input = None
        self.trace_handles.append(layer.get_submodule('self_attn.q_proj').register_forward_pre_hook(self._q_input))

    def _before(self, module, args, kwargs):
        if self.calls >= self.limit or self.seconds >= self.max_seconds:
            raise RuntimeError('Diagnostic forward budget exhausted')
        torch.cuda.synchronize()
        self.calls += 1
        self.started = time.monotonic()

    def _after(self, module, args, kwargs, output):
        torch.cuda.synchronize()
        self.seconds += time.monotonic()-self.started

    def _trace_hook(self, name):
        def hook(module, args, output):
            if self.capture_trace:
                value = output[0] if isinstance(output, tuple) else output
                self.trace[name] = value[0,self.trace_row].detach().clone()
        return hook

    def _q_input(self, module, args):
        if self.capture_trace:
            self.q_input = args[0].detach().clone()

    def reset_trace(self):
        self.trace = {}
        self.q_input = None

    def original_generate(self, fixture):
        captured, inputs = [], []
        def pre(module, args, kwargs):
            ids = kwargs.get('input_ids', args[0] if args else None)
            cache = kwargs.get('past_key_values')
            inputs.append(dict(ids=ids[0].tolist(), cache_before=0 if cache is None else cache.get_seq_length(),
                               positions=None if kwargs.get('position_ids') is None else kwargs['position_ids'][0].tolist()))
        def post(module, args, kwargs, output):
            captured.append(output.logits[:,-1:].detach().clone())
        p = self.target.register_forward_pre_hook(pre, with_kwargs=True)
        q = self.target.register_forward_hook(post, with_kwargs=True)
        try:
            generated = self.backend.generate(fixture['row'])
        finally:
            p.remove(); q.remove()
        actual = generated['response_token_ids']
        result = logits_summary(torch.cat(captured,dim=1), fixture['labels'], fixture['focus'])
        result.update(generated_length=len(actual), generated_equal_saved=actual==fixture['labels'],
                      original_method_called=True, input_prefix_hash=digest(generated['prompt_token_ids']),
                      source_prefixes_equal_saved=all(
                          x['ids'] == (fixture['prefix'] if i==0 else [fixture['labels'][i-1]])
                          for i,x in enumerate(inputs)),
                      calls=[dict(input_length=len(x['ids']), input_sha256=digest(x['ids']),
                                  cache_before=x['cache_before'], positions=x['positions']) for x in inputs])
        return result

    def cached_saved(self, fixture, until=None, trace=False):
        from transformers import DynamicCache
        labels, prefix = fixture['labels'], fixture['prefix']
        steps = len(labels) if until is None else until+1
        cache, values = DynamicCache(), []
        self.reset_trace()
        for i in range(steps):
            self.capture_trace = trace and i == fixture['focus']
            self.trace_row = 0
            ids = torch.tensor([prefix if i == 0 else [labels[i-1]]], device=self.target.device)
            out = self.target(ids, position_ids=torch.arange(0 if i==0 else len(prefix)+i-1,len(prefix)+i,
                device=self.target.device)[None],past_key_values=cache,use_cache=True,logits_to_keep=1)
            values.append(out.logits.detach().clone())
        self.capture_trace = False
        return logits_summary(torch.cat(values,dim=1),labels[:steps],fixture['focus'])

    def native(self, fixture, oracle=True, trace=False):
        labels, prefix = fixture['labels'], fixture['prefix']
        calls = []
        globs = vars(sys.modules[type(self.draft).__module__])
        sample = globs['sample']
        def replacement(logits, temperature=0):
            if oracle and logits.shape[1] == 15:
                return torch.tensor([labels[1:]], device=logits.device)
            return sample(logits,temperature)
        def pre(module,args,kwargs):
            ids = args[0]
            calls.append(dict(ids=ids.detach().clone(),positions=kwargs['position_ids'][0].tolist(),
                              cache_before=int(kwargs['past_key_values'].get_seq_length())))
            self.capture_trace = trace and len(calls)==2
            self.trace_row = fixture['focus']-1
        def post(module,args,kwargs,out):
            calls[-1]['logits'] = out.logits.detach().clone()
            calls[-1]['cache_after'] = int(kwargs['past_key_values'].get_seq_length())
            self.capture_trace = False
        self.reset_trace()
        p=self.target.register_forward_pre_hook(pre,with_kwargs=True)
        q=self.target.register_forward_hook(post,with_kwargs=True)
        globs['sample']=replacement
        try:
            self.draft.spec_generate(self.target,torch.tensor([prefix],device=self.target.device),
                                     max_new_tokens=1,stop_token_ids=None,temperature=0)
        finally:
            p.remove();q.remove();globs['sample']=sample;self.capture_trace=False
        if len(calls)!=2 or calls[1]['cache_before']!=len(prefix):
            raise ValueError('Unexpected native call/cache layout')
        if calls[1]['positions']!=list(range(len(prefix),len(prefix)+16)):
            raise ValueError('Native position mismatch')
        candidate=calls[1]['ids'][0]
        if oracle and candidate.tolist()!=labels:
            raise ValueError('Saved-candidate diagnostic did not receive exactly the saved candidate')
        values=torch.cat((calls[0]['logits'][:,-1:],calls[1]['logits'][:,:15]),dim=1)
        result=logits_summary(values,labels,fixture['focus'])
        first_error=next((i for i,(a,b) in enumerate(zip(candidate.tolist(),labels)) if a!=b),16)
        result.update(oracle=oracle,first_candidate_error=first_error,
                      accepted_proposals=int((candidate[1:]==values[0,1:].argmax(-1)).cumprod(0).sum()),
                      candidate_sha256=digest(candidate.tolist()),
                      calls=[dict(input_length=x['ids'].shape[1],input_sha256=digest(x['ids'][0].tolist()),
                                  positions=x['positions'],cache_before=x['cache_before'],cache_after=x['cache_after']) for x in calls])
        return result

    def uncached(self, fixture):
        ids=torch.tensor([fixture['prefix']+fixture['labels'][:-1]],device=self.target.device)
        output=self.target(ids,use_cache=False,logits_to_keep=16)
        return logits_summary(output.logits,fixture['labels'],fixture['focus'])

    def close(self):
        self.before.remove();self.after.remove()
        for handle in self.trace_handles: handle.remove()


def main():
    global torch
    import torch
    import transformers
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wait-for-probes',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    target_path=Path('data/models/qwen3-8b')
    draft_path=Path('data/models/qwen3-8b-dflash-b16')
    pilot=Path('data/qwen_responses_pilot')
    original=Path('scripts/generate_qwen_responses.py')
    run=json.loads((pilot/'run.json').read_text())
    if sha256(original)!=run['generator_sha256']:
        raise ValueError('Original generator fingerprint mismatch')
    configs={'ultrachat_200k:train_sft:37680':(24,6), 'ultrachat_200k:train_sft:61759':(22,15),
             'ultrachat_200k:train_sft:164610':(19,5),'gsm8k:train:3821':(38,5)}
    fixtures=[]
    for row in records(pilot):
        if row['id'] in configs:
            anchor,focus=configs[row['id']]
            prefix,_,labels=block_inputs(row,anchor,151669)
            if len(row['prompt_token_ids'])!=anchor:
                raise ValueError('Expected first-block fixture')
            fixtures.append(dict(row=row,prefix=prefix,labels=labels,anchor=anchor,focus=focus))
    if len(fixtures)!=4:
        raise ValueError('Missing fixed fixture')
    fixtures.sort(key=lambda f:(f['row']['id']!='ultrachat_200k:train_sft:37680',f['row']['id']))
    report=dict(schema=1,purpose='exact original saved-token diagnostic; no training',
                source_sha256=sha256(Path(__file__)),source_run=run,
                source_generator_sha256=sha256(original),target_files=target_fingerprints(target_path),
                runtime=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
                             cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),hostname=platform.node(),
                             job_id=os.environ.get('SLURM_JOB_ID'),seed=0,allow_tf32=False),baseline=[])
    sys.path.insert(0,str(original.parent.resolve()))
    spec=importlib.util.spec_from_file_location('original_qwen_generator',original)
    source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    backend=source.QwenBackend(target_path,run['generation'])
    backend.generation_config.max_new_tokens=16
    target=backend.model.eval().requires_grad_(False)
    report['generation_config']=backend.generation_config.to_dict()
    report['source_cap_change']='Only max_new_tokens shortened from 2048 to 16; original generate method called unchanged'
    pin=json.loads(Path('research/editor-feature-bank/feature_bank/model_pin.json').read_text())
    verify(draft_path,pin)
    cls,_=native_class(draft_path)
    draft,info=cls.from_pretrained(draft_path,dtype=torch.bfloat16,device_map='cuda:0',
                                  attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
    loading_ok(info);draft.eval().requires_grad_(False)
    report['draft_pin']=pin
    report['qwen_implementation_sha256']=sha256(Path(inspect.getfile(type(target.model.layers[0]))))
    report['target_config']=target.config.to_dict()
    report['ownership']=dict(target_trainable=sum(p.numel() for p in target.parameters() if p.requires_grad),
                             draft_trainable=sum(p.numel() for p in draft.parameters() if p.requires_grad))
    harness=Harness(target,draft,backend,args.output)
    try:
        with torch.inference_mode():
            for fixture in fixtures:
                row=fixture['row']
                item=dict(id=row['id'],anchor=fixture['anchor'],focus=fixture['focus'],
                          record_sha256=row['record_sha256'],prefix_sha256=digest(fixture['prefix']),
                          labels_sha256=digest(fixture['labels']),generation=[],native=[],uncached=[])
                for repetition in range(3):
                    item['generation'].append(harness.original_generate(fixture))
                    if item['generation'][-1]['input_prefix_hash']!=digest(fixture['prefix']):
                        raise ValueError('Original tokenizer prompt differs from saved prefix')
                    item['native'].append(harness.native(fixture))
                    item['uncached'].append(harness.uncached(fixture))
                if row['id']=='ultrachat_200k:train_sft:37680':
                    item['cached_saved_prefix']=[harness.cached_saved(fixture,until=fixture['focus']) for _ in range(3)]
                    item['ordinary_dflash']=harness.native(fixture,oracle=False)
                report['baseline'].append(item)
                print(json.dumps(dict(id=row['id'],generation=[x['matches'] for x in item['generation']],
                                      native=[x['matches'] for x in item['native']],
                                      native_focus=item['native'][0]['focus'])),flush=True)
                write_json(args.output/'baseline.partial.json',report)
            report['baseline_exact']=all(x['matches']==16 for f in report['baseline'] for x in f['native']+f['generation'])
            report['baseline_target_calls']=harness.calls
            report['baseline_target_seconds']=harness.seconds
            write_json(args.output/'baseline.json',report)
            print(f'BASELINE_EXACT={report["baseline_exact"]}; TARGET_CALLS={harness.calls}',flush=True)
            if args.wait_for_probes:
                deadline=time.monotonic()+1200
                print('Waiting for probes.ready within this allocation',flush=True)
                while time.monotonic()<deadline and not (args.output/'probes.ready').exists(): time.sleep(2)
                if (args.output/'probes.ready').exists():
                    report['probe_sha256']=sha256(args.output/'probes.py')
                    runpy.run_path(str(args.output/'probes.py'),init_globals=dict(
                        torch=torch,harness=harness,fixtures=fixtures,report=report,write_json=write_json,
                        args=args,tensor_digest=tensor_digest,logits_summary=logits_summary))
    except Exception as error:
        import traceback
        report['error']=str(error);report['traceback']=traceback.format_exc()
        raise
    finally:
        report['target_calls']=harness.calls;report['target_seconds']=harness.seconds
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        harness.close();write_json(args.output/'result.json',report)
    raise SystemExit(0 if report['baseline_exact'] else 1)


if __name__=='__main__': main()
