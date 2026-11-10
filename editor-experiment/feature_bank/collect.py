"""Bounded teacher-free Qwen3-8B bank. Run inside a single-GPU SLURM job.

No optimization occurs here. A failed check writes failure.json and stops.
Raw prompt text and tensors stay on the server; audit.json contains aggregate evidence.
"""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import types
from typing import Optional

from data import audit_sources, block_inputs, candidates, digest
from prepare import sha256, verify


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def target_fingerprints(directory):
    expected = json.loads(Path(__file__).with_name('target_files.json').read_text())
    pin = json.loads((directory / 'pinned_revision.json').read_text())
    if pin != {'model_id': 'Qwen/Qwen3-8B', 'revision': expected['revision']}:
        raise ValueError('Target revision marker mismatch')
    results = {}
    for item in expected['files']:
        path = directory / item['rfilename']
        if path.stat().st_size != item['size']:
            raise ValueError(f'Target size mismatch: {path.name}')
        if 'lfs' in item:
            value = sha256(path)
            if value != item['lfs']['sha256']:
                raise ValueError(f'Target SHA256 mismatch: {path.name}')
        else:
            value = hashlib.sha1(f'blob {item["size"]}\0'.encode() + path.read_bytes()).hexdigest()
            if value != item['blobId']:
                raise ValueError(f'Target Git blob mismatch: {path.name}')
        results[path.name] = value
    return results


def native_class(directory):
    """Execute hash-verified released model source, without its dataset CLI imports.

    utils.py imports datasets for unused benchmark helpers. Extract only the three
    original pure helper function definitions; no model source is transformed.
    """
    import torch
    package = types.ModuleType('pinned_feature_dflash')
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    helpers = types.ModuleType(package.__name__ + '.utils')
    helpers.__dict__.update(torch=torch, Optional=Optional)
    tree = ast.parse((directory / 'utils.py').read_text())
    names = {'build_target_layer_ids', 'extract_context_feature', 'sample'}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in definitions} != names:
        raise ValueError('Pinned native helper functions not found')
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(directory / 'utils.py'), 'exec'), helpers.__dict__)
    sys.modules[helpers.__name__] = helpers
    spec = importlib.util.spec_from_file_location(package.__name__ + '.dflash', directory / 'dflash.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DFlashDraftModel, helpers.extract_context_feature


def loading_ok(info):
    for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        if info.get(key):
            raise ValueError(f'Incomplete checkpoint loading: {key}: {info[key]}')


def readout_control(states, first_embedding, head):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from compact_second_pass_prototype import CompactSecondPass
    with torch.inference_mode():
        torch.manual_seed(0)
        editor = CompactSecondPass(4096).to(states.device).eval()
        count = sum(p.numel() for p in editor.parameters() if p.requires_grad)
        if count != 4204032:
            raise ValueError(f'Unexpected editor parameter count: {count}')
        s, e = states.float(), first_embedding.float()
        z = editor.coefficients(s, e)
        # Preserve native [1,15,H] readout row shape in the identity serving check.
        serving_states = states[:, 1:].clone()
        serving_states[:, 1:] += (z @ editor.up.T).to(states.dtype)
        baseline = head(states[:, 1:])
        identity = head(serving_states)
        if not torch.equal(baseline, identity):
            raise ValueError('Identity editor changed native readout')
        # Full vocabulary, chunked only along vocabulary dimension; no stored logits.
        editor.up.normal_(std=1e-3)
        z = editor.coefficients(s, e)
        direct_parts, folded_parts = [], []
        max_error = 0.0
        for w in head.weight.split(8192):
            w = w.float()
            direct = (s[:, 2:] + z @ editor.up.T) @ w.T
            folded = s[:, 2:] @ w.T + z @ (w @ editor.up).T
            torch.testing.assert_close(direct, folded, rtol=2e-4, atol=2e-3)
            max_error = max(max_error, (direct - folded).abs().max().item())
            direct_parts.append(direct)
            folded_parts.append(folded)
        changed = (torch.cat(direct_parts, -1).argmax(-1) != torch.cat(folded_parts, -1).argmax(-1)).sum().item()
        if changed:
            raise ValueError('FP32 direct/folded argmax disagreement')
    return dict(editor_trainable_parameters=count, identity_native_bitwise_equal=True,
                nonzero_direct_folded_max_abs=max_error, nonzero_direct_folded_argmax_changes=changed,
                direct_folded_atol=2e-3, direct_folded_rtol=2e-4, seed=0)


def run(args, report):
    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('A GPU allocation is required')
    report['runtime'] = dict(python=platform.python_version(), torch=torch.__version__,
                             transformers=transformers.__version__, cuda=torch.version.cuda,
                             gpu=torch.cuda.get_device_name(), job_id=os.environ.get('SLURM_JOB_ID'),
                             hostname=platform.node())
    pin = json.loads(Path(__file__).with_name('model_pin.json').read_text())
    report['model_pin'] = pin
    print('Verifying target and draft files', flush=True)
    report['target_files'] = target_fingerprints(args.target)
    verify(args.draft, pin)
    rows, report['source_audit'] = audit_sources(args.pilot, args.main)
    schedule, excluded = candidates(rows, args.max_context, args.anchors_per_prompt)
    report['candidate_budget'] = len(schedule)
    report['excluded_prompts'] = dict(Counter(r['reason'] for r in excluded))
    write_json(args.output / 'excluded_prompts.json', excluded)
    report['source_files'] = {str(p): sha256(p) for root in (args.pilot, args.main)
                              for p in sorted(root.glob('*.json*'))}
    print(f'{len(schedule)} bounded candidates; loading frozen models', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    target, info = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa',
        local_files_only=True, output_loading_info=True)
    loading_ok(info)
    cls, extract = native_class(args.draft)
    draft, info = cls.from_pretrained(args.draft, dtype=torch.bfloat16, device_map='cuda:0',
                                     attn_implementation='sdpa', local_files_only=True, output_loading_info=True)
    loading_ok(info)
    for model in (target, draft):
        model.eval().requires_grad_(False)
    if (target.config.hidden_size, target.config.vocab_size, draft.block_size, draft.mask_token_id,
        draft.target_layer_ids) != (4096, 151936, 16, 151669, [1, 9, 17, 25, 33]):
        raise ValueError('Model interface mismatch')
    if not 0 <= draft.mask_token_id < target.model.embed_tokens.num_embeddings:
        raise ValueError('Mask ID outside frozen embedding table')
    report['tokenizer'] = dict(length=len(tokenizer), mask_id=draft.mask_token_id,
                               mask_token=tokenizer.convert_ids_to_tokens(draft.mask_token_id),
                               chat_template_sha256=hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest())
    report['ownership'] = {name: dict(total=sum(p.numel() for p in model.parameters()),
                                     trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
                           for name, model in [('target', target), ('draft', draft), ('head', target.lm_head)]}
    report['loading'] = 'complete; no missing, unexpected, or mismatched parameters'
    selected, tried, counts = [], [], Counter()
    checked_prompts = set()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode():
        for row, anchor in schedule:
            if time.monotonic() - started > args.seconds:
                report['budget_stop'] = 'wall_time'
                break
            fixture = dict(id=row['id'], source=row['source'], anchor=anchor)
            report['active_fixture'] = fixture
            write_json(args.output / 'audit.partial.json', report)
            prefix, noise, labels = block_inputs(row, anchor, draft.mask_token_id)
            if row['id'] not in checked_prompts:
                rendered = tokenizer.apply_chat_template(row['messages'], tokenize=False,
                                                         add_generation_prompt=True, enable_thinking=False)
                if tokenizer.encode(rendered, add_special_tokens=False) != row['prompt_token_ids']:
                    raise ValueError('Saved prompt tokenization differs from pinned tokenizer')
                if hashlib.sha256(rendered.encode()).hexdigest() != row['rendered_prompt_sha256']:
                    raise ValueError('Rendered prompt fingerprint mismatch')
                if row['response_token_ids'][-1] not in (151645, 151643):
                    raise ValueError('EOS-labelled response has no EOS')
                checked_prompts.add(row['id'])
            prefix_ids = torch.tensor([prefix], device='cuda')
            output = target(prefix_ids, use_cache=False, output_hidden_states=True, logits_to_keep=1)
            if output.logits.argmax(-1).item() != labels[0]:
                report['target_mismatch'] = dict(position=0, expected=labels[0], actual=output.logits.argmax(-1).item())
                raise ValueError('Target anchor replay mismatch; no relabeling permitted')
            features = extract(output.hidden_states, draft.target_layer_ids)
            del output
            # Independent causal replay: last 16 logits predict anchor + 15 labels.
            replay = target(torch.tensor([prefix + labels[:-1]], device='cuda'),
                            use_cache=False, logits_to_keep=16).logits.argmax(-1)[0].tolist()
            if replay != labels:
                report['target_mismatch'] = dict(expected=labels, actual=replay,
                    positions=[i for i, (a, b) in enumerate(zip(labels, replay)) if a != b])
                raise ValueError('Target continuation replay mismatch; investigate numerical/alignment causes')
            kwargs = dict(target_hidden=features, noise_embedding=target.model.embed_tokens(torch.tensor([noise], device='cuda')),
                          position_ids=torch.arange(anchor + 16, device='cuda')[None], use_cache=False, is_causal=False)
            captured = []
            handle = draft.norm.register_forward_hook(lambda module, inputs, output: captured.append(output.detach().clone()))
            try:
                states = draft(**kwargs)
            finally:
                handle.remove()
            if len(captured) != 1 or not torch.equal(states, captured[0]) or not torch.isfinite(states).all():
                raise ValueError('Captured state is not finite final native normalized output')
            original = torch.tensor([noise], device='cuda')
            scores = target.lm_head(states[:, 1:])
            original[:, 1:] = scores.argmax(-1)
            if not tried:
                repeated = draft(**kwargs)
                if not torch.equal(states, repeated):
                    raise ValueError('Native first pass is not deterministic')
                # Changing every future target label cannot change the constructed inputs.
                altered = dict(row, response_token_ids=list(row['response_token_ids']))
                offset = anchor - len(row['prompt_token_ids'])
                altered['response_token_ids'][offset + 1:offset + 16] = [0] * 15
                altered_prefix, altered_noise, _ = block_inputs(altered, anchor, draft.mask_token_id)
                if (prefix, noise) != (altered_prefix, altered_noise):
                    raise ValueError('Future label leakage into draft inputs')
                report['readout_control'] = readout_control(states, target.model.embed_tokens(original[:, 1]), target.lm_head)
                report['native_controls'] = dict(final_norm_capture_bitwise_equal=True, repeated_forward_bitwise_equal=True,
                    future_label_perturbation_inputs_equal=True, target_context_positions=[0, anchor - 1],
                    native_readout_shape=list(scores.shape), target_layer_offset=1,
                    attention='sdpa, noncausal draft, causal target', cache='disabled; native forward first round only')
            original_ids = original[0].tolist()
            kind = ('wrong_first' if original_ids[1] != labels[1] else
                    'preservation' if original_ids == labels else 'repairable')
            counts[kind] += 1
            entry = dict(fixture, kind=kind, original_ids=original_ids, labels=labels,
                         record_sha256=row['record_sha256'], dataset_split=row['dataset_split'])
            tried.append(entry)
            with (args.output / 'attempted.jsonl').open('a') as stream:
                stream.write(json.dumps(entry, sort_keys=True) + '\n')
            limit = args.repairs if kind == 'repairable' else args.preservation if kind == 'preservation' else 0
            if sum(s['kind'] == kind for s in selected) < limit:
                name = f'block-{len(selected):04d}.safetensors'
                tensors = dict(normalized_states=states.cpu().contiguous(),
                               first_guess_embedding=target.model.embed_tokens(original[:, 1]).cpu().contiguous(),
                               original_ids=original.cpu(), labels=torch.tensor([labels]),
                               valid_mask=torch.ones((1, 16), dtype=torch.bool),
                               prefix_ids=prefix_ids.cpu(), anchor=torch.tensor([anchor]))
                save_file(tensors, args.output / name)
                selected.append(dict(entry, file=name, sha256=sha256(args.output / name), bytes=(args.output / name).stat().st_size))
                write_json(args.output / 'manifest.partial.json', selected)
                print(f'{len(tried)} candidates: saved {len(selected)} ({kind})', flush=True)
            del states, captured, features, kwargs, scores, prefix_ids
            if sum(s['kind'] == 'repairable' for s in selected) == args.repairs and sum(s['kind'] == 'preservation' for s in selected) == args.preservation:
                break
        torch.cuda.synchronize()
    report.update(elapsed_capture_seconds=time.monotonic() - started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  candidates_tried=len(tried), classifications=dict(counts), selected_counts=dict(Counter(s['kind'] for s in selected)),
                  selected_sources=dict(Counter(str(s['source']) for s in selected)),
                  selected_distinct_prompts=len({s['id'] for s in selected}),
                  tensor_bytes=sum(s['bytes'] for s in selected))
    complete = (report['selected_counts'].get('repairable', 0) == args.repairs and
                report['selected_counts'].get('preservation', 0) == args.preservation)
    report['outcome'] = 'success' if complete else 'inconclusive: bounded selection exhausted'
    write_json(args.output / 'attempted.json', tried)
    write_json(args.output / 'manifest.json', dict(schema_version=1, models=pin, blocks=selected))
    report['manifest_sha256'] = sha256(args.output / 'manifest.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('target', 'draft', 'pilot', 'main', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--repairs', type=int, default=128)
    p.add_argument('--preservation', type=int, default=16)
    p.add_argument('--max-context', type=int, default=1024)
    p.add_argument('--anchors-per-prompt', type=int, default=32)
    p.add_argument('--seconds', type=int, default=2700)
    args = p.parse_args()
    if not 1 <= args.repairs <= 128 or not 0 <= args.preservation <= 16 or not 1 <= args.seconds <= 2700:
        p.error('Initial bank limits: 1..128 repairs, 0..16 preservation, <=2700 seconds')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  source_sha256={f.name: sha256(f) for f in Path(__file__).parent.glob('*.py')})
    report['source_sha256']['compact_second_pass_prototype.py'] = sha256(
        Path(__file__).resolve().parents[1] / 'compact_second_pass_prototype.py')
    try:
        run(args, report)
    except Exception as error:
        report.update(outcome='failed check; training blocked', error=str(error), traceback=traceback.format_exc())
        write_json(args.output / 'failure.json', report)
        raise
    finally:
        write_json(args.output / 'audit.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
