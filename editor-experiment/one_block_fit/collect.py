"""Capture real native proposals and authenticate a one-block/held-out token set."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def progress(draft, labels):
    n = 0
    for a, b in zip(draft[1:], labels[1:]):
        if a != b:
            break
        n += 1
    return n


def make_bank(inputs, greedy, native):
    """CPU-only audit; labels never enter the proposal-capture worker."""
    fixtures = {x['id']: x for x in inputs['fixtures']}
    rows, excluded, seen, counts = [], [], set(), Counter()
    for block in native['blocks']:
        matches = [f for f in fixtures.values()
                   if block['prefix'][:len(f['prompt'])] == f['prompt']]
        if len(matches) != 1:
            raise ValueError('Ambiguous or unauthenticated captured prefix')
        f = matches[0]
        full = f['prompt'] + greedy['tokens'][f['id']]
        anchor = block['anchor']
        key = (f['id'], anchor)
        if key in seen:
            raise ValueError('Duplicate native block')
        seen.add(key)
        if block['prefix'] != full[:anchor] or len(block['prefix']) != anchor:
            raise ValueError('Captured prefix differs from canonical generation')
        if anchor < len(f['prompt']) or block['positions'] != list(range(anchor, anchor+16)):
            raise ValueError('Native block position alignment failed')
        labels = full[anchor:anchor+16]
        if len(labels) != 16:
            excluded.append(dict(id=f['id'], anchor=anchor, reason='incomplete_reference_window'))
            continue
        draft = block['draft']
        if len(draft) != 16 or draft[0] != labels[0] or block['native_draft_count'] != 15:
            raise ValueError('Native anchor/proposal layout invalid')
        accepted = progress(draft, labels)
        if block['num_sampled'] != accepted+1 or block['num_rejected'] != 15-accepted:
            raise ValueError('Captured proposal and actual native acceptance disagree')
        if anchor+32 > 1088:
            excluded.append(dict(id=f['id'], anchor=anchor, reason='editor_context_limit'))
            continue
        if counts[f['id']] >= 4:
            excluded.append(dict(id=f['id'], anchor=anchor, reason='four_block_budget'))
            continue
        counts[f['id']] += 1
        row = dict(prompt_id=f['id'], family_id=f['family_id'], split=f['split'],
                   source_record_sha256=f['record_sha256'], anchor=anchor,
                   prefix=block['prefix'], draft=draft, labels=labels,
                   target_replay_verified=True, native_accepted=accepted,
                   kind='wrong_first' if accepted == 0 else 'preservation' if accepted == 15 else 'repairable')
        rows.append(row)
    repairs = [r for r in rows if r['split'] == 'train' and r['kind'] == 'repairable']
    order = {f['id']: i for i, f in enumerate(inputs['fixtures'])}
    repairs.sort(key=lambda r: (order[r['prompt_id']], r['anchor']))
    if not repairs:
        raise ValueError('No real repairable training block within the fixed budget')
    train = repairs[0]
    test = [r for r in rows if r['split'] == 'validation']
    required = {f['id'] for f in fixtures.values() if f['split'] == 'validation'}
    if {r['prompt_id'] for r in test} != required:
        raise ValueError('A held-out prompt lacks a complete evaluable block')
    if any(r['prompt_id'] == train['prompt_id'] or r['family_id'] == train['family_id'] for r in test):
        raise ValueError('Training/held-out overlap')
    return dict(schema='one-block-native-token-set-v1', training=[train], held_out=test,
                all_captured_rows=rows, excluded=excluded,
                class_counts=dict(Counter(r['kind'] for r in test)),
                scope='One training block; eight held-out validation prompts; not a full training bank')


def prepare(args):
    from vllm_probe import prepare as authenticate_models
    from feature_bank.data import records, family, source_name
    pins = authenticate_models(args.project)
    pilot = list(records(args.project/'data/qwen_responses_vllm_invariant_pilot_20261117'))
    wanted = [f['id'] for f in pins['fixtures']]
    pilot.sort(key=lambda r: wanted.index(r['id']))
    train_families = {family(r) for r in pilot}
    valid = []
    for r in records(args.project/'data/qwen_responses_50k'):
        if (r['dataset_split'] == 'validation' and r['status'] == 'generated'
                and r['finish_reason'] == 'eos' and len(r['prompt_token_ids']) <= 1024
                and len(r['response_token_ids']) >= 16 and family(r) not in train_families):
            valid.append(r)
    valid.sort(key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
    chosen, groups, families = [], Counter(), set(train_families)
    for r in valid:
        source = source_name(r)
        if groups[source] >= 4 or family(r) in families:
            continue
        chosen.append(r)
        groups[source] += 1
        families.add(family(r))
    if len(chosen) != 8 or sorted(groups.values()) != [4, 4]:
        raise ValueError('Expected four held-out prompts from each of two sources')
    fixtures = [dict(id=r['id'], prompt=r['prompt_token_ids'],
                     family_id=family(r), record_sha256=r['record_sha256'],
                     split=r['dataset_split'], source=source_name(r),
                     canonical_tokens=r['response_token_ids'][:64] if r in pilot else None)
                for r in pilot+chosen]
    pins['fixtures'] = fixtures
    pins['selection'] = dict(validation_order='SHA256(prompt ID)', counts=dict(groups),
                             validation_eligible=len(valid), family_overlap=0)
    pins['source_hashes'] = {str(p.relative_to(Path(__file__).parent)):sha(p)
                            for p in Path(__file__).parent.glob('*.py')}
    import autoregressive_editor
    model_dir = Path(autoregressive_editor.__file__).parent
    pins['editor_source_hashes'] = {p.name:sha(p) for p in model_dir.glob('*.py')}
    save(args.output/'inputs.json', pins)


def child(args):
    import platform
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from audit_vllm_probe import audit_trial
    assert vllm.__version__ == '0.34.0'
    torch.set_num_threads(4)
    inputs = json.loads((args.output/'inputs.json').read_text())
    fixtures = inputs['fixtures']
    config = dict(model=str(args.project/'data/models/qwen3-8b'), dtype='bfloat16',
        tensor_parallel_size=1, max_model_len=2304, max_num_seqs=4,
        max_num_batched_tokens=256, enable_chunked_prefill=True, enable_prefix_caching=False,
        enforce_eager=True, async_scheduling=False, seed=0,
        kv_cache_memory_bytes=2048*1024**2, attention_config={'backend':'FLASH_ATTN'},
        worker_extension_cls='one_block_fit.capture_worker.FitCaptureWorker',
        generation_config='vllm')
    if args.native:
        config['speculative_config'] = dict(method='dflash',
            model=str(args.project/'data/models/qwen3-8b-dflash-b16'), num_speculative_tokens=15,
            max_model_len=2304)
    llm = LLM(**config)
    startup = llm.collective_rpc('probe_finish')[0]
    if (not startup['invariant_active'] or startup['model_class'] != 'Qwen3ForCausalLM'
            or not startup['attention'] or any(not a['batch_invariant'] for a in startup['attention'])):
        raise ValueError('Invariant target activation unverified')
    llm.collective_rpc('probe_begin', args=('native' if args.native else 'greedy', True))
    start = time.monotonic()
    outputs = llm.generate([{'prompt_token_ids':f['prompt']} for f in fixtures],
        SamplingParams(temperature=0, max_tokens=64, seed=0), use_tqdm=False)
    elapsed = time.monotonic()-start
    observed = llm.collective_rpc('probe_finish')[0]
    trial = dict(tokens={}, details={}, **observed, wall_seconds=elapsed, config=config,
                 runtime=dict(torch=str(torch.__version__), vllm=vllm.__version__,
                              gpu=torch.cuda.get_device_name(), python=platform.python_version()))
    assert len(outputs) == len(fixtures)
    for f, out in zip(fixtures, outputs):
        assert out.prompt_token_ids == f['prompt']
        trial['tokens'][f['id']] = list(out.outputs[0].token_ids)
        trial['details'][f['id']] = dict(request_id=out.request_id,
            finish_reason=out.outputs[0].finish_reason)
        if f['canonical_tokens'] is not None and trial['tokens'][f['id']] != f['canonical_tokens']:
            raise ValueError('Existing canonical training fixture changed')
    trial['alignment_audit'] = audit_trial({f['id']:f for f in fixtures}, trial)
    save(args.output/('native.json' if args.native else 'greedy.json'), trial)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    if args.child:
        child(args)
        return
    args.output.mkdir(exist_ok=False, parents=True)
    prepare(args)
    save(args.output/'budget.json', dict(calls=0,seconds=0.0,max_calls=500,max_seconds=120))
    env = dict(os.environ, VLLM_BATCH_INVARIANT='1', VLLM_USE_FLASHINFER_SAMPLER='0',
        VLLM_PROBE_BUDGET=str(args.output/'budget.json'), VLLM_WORKER_MULTIPROC_METHOD='spawn',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1')
    for native in (False,True):
        command = [sys.executable,'-m','one_block_fit.collect','--child',
                   '--project',str(args.project),'--output',str(args.output)]
        if native:
            command += ['--native']
        with (args.output/('native.log' if native else 'greedy.log')).open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    from vllm_probe import compare
    from vllm_contract import require_exact
    greedy = json.loads((args.output/'greedy.json').read_text())
    native = json.loads((args.output/'native.json').read_text())
    comparison = compare(greedy,native)
    require_exact(comparison,sum(map(len,greedy['tokens'].values())))
    save(args.output/'comparison.json',comparison)
    bank = make_bank(json.loads((args.output/'inputs.json').read_text()),greedy,native)
    bank['provenance'] = {name:sha(args.output/name) for name in ('inputs.json','greedy.json','native.json','comparison.json')}
    save(args.output/'bank.json',bank)
    print(json.dumps(dict(training=bank['training'][0]['prompt_id'],held_out=len(bank['held_out']),
                          classes=bank['class_counts'],bank_sha256=sha(args.output/'bank.json'))),flush=True)


if __name__ == '__main__':
    main()
