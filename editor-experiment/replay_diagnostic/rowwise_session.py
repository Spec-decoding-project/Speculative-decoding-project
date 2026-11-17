"""Bounded original-generator versus single-row-linear native verification test.

Exactly 279 planned target calls: 216 repeated source/native baseline calls,
35 cached trace calls, 24 modified native calls, and four ordinary-control calls.
No precision switch, label rewriting, recollection, or training.
"""
import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

import torch
import transformers

import exact_session as original_diagnostic
from exact_session import (Harness, block_inputs, digest, loading_ok, native_class,
                           records, sha256, target_fingerprints, verify, write_json)
from rowwise import single_row_linears

original_diagnostic.torch = torch


class TimedHarness(Harness):
    def __init__(self, *args):
        self.forward_times = []
        super().__init__(*args)

    def _after(self, module, args, kwargs, output):
        previous = self.seconds
        super()._after(module, args, kwargs, output)
        self.forward_times.append(self.seconds - previous)


def timed(harness, callback):
    torch.cuda.synchronize()
    started, calls = time.monotonic(), harness.calls
    index = len(harness.forward_times)
    result = callback()
    torch.cuda.synchronize()
    result['measurement'] = dict(wall_seconds=time.monotonic()-started,
                                 target_calls=harness.calls-calls,
                                 target_forward_seconds=harness.forward_times[index:])
    return result


def differences(reference, observed):
    if list(reference) != list(observed):
        raise ValueError('Trace sites/order differ')
    result = []
    for site, a in reference.items():
        b = observed[site]
        if a.shape != b.shape:
            raise ValueError('Trace shapes differ')
        delta = a.float()-b.float()
        result.append(dict(site=site, equal=torch.equal(a, b), changed=int((a != b).sum()),
                           max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt())))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    target_path = Path('data/models/qwen3-8b')
    draft_path = Path('data/models/qwen3-8b-dflash-b16')
    pilot = Path('data/qwen_responses_pilot')
    original = Path('scripts/generate_qwen_responses.py')
    run = json.loads((pilot/'run.json').read_text())
    if sha256(original) != run['generator_sha256']:
        raise ValueError('Original generator fingerprint mismatch')
    configs = {'ultrachat_200k:train_sft:37680': (24, 6), 'ultrachat_200k:train_sft:61759': (22, 15),
               'ultrachat_200k:train_sft:164610': (19, 5), 'gsm8k:train:3821': (38, 5)}
    fixtures = []
    for row in records(pilot):
        if row['id'] in configs:
            anchor, focus = configs[row['id']]
            prefix, _, labels = block_inputs(row, anchor, 151669)
            if len(row['prompt_token_ids']) != anchor:
                raise ValueError('Expected first-block fixture')
            fixtures.append(dict(row=row, prefix=prefix, labels=labels, anchor=anchor, focus=focus))
    if len(fixtures) != 4:
        raise ValueError('Missing fixture')
    fixtures.sort(key=lambda f: (f['row']['id'] != 'ultrachat_200k:train_sft:37680', f['row']['id']))
    planned = sum(3*(16+2) + f['focus']+1 + 3*2 for f in fixtures) + 4
    if planned != 279 or planned > 300:
        raise ValueError('Unexpected forward budget')
    report = dict(schema=1, purpose='single-row linear correction diagnostic; no training',
                  planned_target_calls=planned, source_run=run,
                  source_hashes={name: sha256(Path(__file__).with_name(name)) for name in
                                 ('rowwise_session.py', 'rowwise.py', 'exact_session.py')},
                  original_generator_sha256=sha256(original), target_files=target_fingerprints(target_path),
                  runtime=dict(python=platform.python_version(), torch=torch.__version__,
                               transformers=transformers.__version__, cuda=torch.version.cuda,
                               gpu=torch.cuda.get_device_name(), hostname=platform.node(),
                               job_id=os.environ.get('SLURM_JOB_ID'), seed=0, allow_tf32=False,
                               deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                               reduced_bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
                  fixtures=[])
    sys.path.insert(0, str(original.parent.resolve()))
    spec = importlib.util.spec_from_file_location('original_qwen_generator', original)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    backend = source.QwenBackend(target_path, run['generation'])
    backend.generation_config.max_new_tokens = 16
    target = backend.model.eval().requires_grad_(False)
    report['generation_config'] = backend.generation_config.to_dict()
    report['source_cap_change'] = 'Only max_new_tokens 2048 -> 16; original method unchanged'
    report['target_config'] = target.config.to_dict()
    report['qwen_implementation_sha256'] = sha256(Path(inspect.getfile(type(target.model.layers[0]))))
    pin = json.loads(Path('research/editor-feature-bank/feature_bank/model_pin.json').read_text())
    verify(draft_path, pin)
    cls, _ = native_class(draft_path)
    draft, info = cls.from_pretrained(draft_path, dtype=torch.bfloat16, device_map='cuda:0',
        attn_implementation='sdpa', local_files_only=True, output_loading_info=True)
    loading_ok(info)
    draft.eval().requires_grad_(False)
    report['draft_pin'] = pin
    report['ownership'] = dict(target_trainable=sum(p.numel() for p in target.parameters() if p.requires_grad),
                               draft_trainable=sum(p.numel() for p in draft.parameters() if p.requires_grad))
    harness = TimedHarness(target, draft, backend, args.output)
    references = {}
    try:
        with torch.inference_mode():
            for fixture in fixtures:
                item = dict(id=fixture['row']['id'], anchor=fixture['anchor'], focus=fixture['focus'],
                            record_sha256=fixture['row']['record_sha256'],
                            prefix_sha256=digest(fixture['prefix']), labels_sha256=digest(fixture['labels']),
                            generation=[], native=[], rowwise=[])
                report['fixtures'].append(item)
                for repeat in range(3):
                    generated = timed(harness, lambda: harness.original_generate(fixture))
                    item['generation'].append(generated)
                    if (not generated['generated_equal_saved'] or not generated['source_prefixes_equal_saved']
                            or generated['input_prefix_hash'] != digest(fixture['prefix'])):
                        raise ValueError('Original generator no longer reproduces saved first block')
                    item['native'].append(timed(harness, lambda: harness.native(fixture, trace=repeat == 0)))
                    if repeat == 0:
                        default_trace = harness.trace
                item['cached_trace_replay'] = timed(harness, lambda: harness.cached_saved(
                    fixture, until=fixture['focus'], trace=True))
                cached_trace = harness.trace
                references[item['id']] = cached_trace
                item['cached_vs_default_layers'] = differences(cached_trace, default_trace)
                write_json(args.output/'baseline.partial.json', report)
                print(json.dumps(dict(stage='baseline', id=item['id'],
                    generation=[x['matches'] for x in item['generation']],
                    native=[x['matches'] for x in item['native']])), flush=True)
            report['baseline_red_reproduced'] = all(
                x['mismatches'] == [dict(position=6, expected=12946, actual=16984)]
                for x in report['fixtures'][0]['native'])
            report['baseline_target_calls'] = harness.calls
            if harness.calls != 251 or not report['baseline_red_reproduced']:
                raise ValueError('Expected baseline/call count did not reproduce')
            write_json(args.output/'baseline.json', report)

            for fixture, item in zip(fixtures, report['fixtures']):
                for repeat in range(3):
                    with single_row_linears(target) as state:
                        result = timed(harness, lambda: harness.native(fixture, trace=repeat == 0))
                    result['intervention'] = dict(target_calls=state['target_calls'],
                        split_modules=dict(state['split_modules']), linear_calls=state['linear_calls'],
                        traced=repeat == 0)
                    item['rowwise'].append(result)
                    if repeat == 0:
                        item['cached_vs_rowwise_layers'] = differences(references[item['id']], harness.trace)
                        q = next(x for x in item['cached_vs_rowwise_layers'] if x['site'] == 'layer0.self_attn.q_proj')
                        item['single_row_query_matches_reference'] = q['equal']
                default = item['native'][1:]
                modified = item['rowwise'][1:]
                item['instrumented_timing'] = dict(
                    scope='Median of two non-traced measured rounds after one traced round; no speedup claim',
                    default_round_seconds=statistics.median(x['measurement']['wall_seconds'] for x in default),
                    rowwise_round_seconds=statistics.median(x['measurement']['wall_seconds'] for x in modified),
                    default_block_seconds=statistics.median(x['measurement']['target_forward_seconds'][1] for x in default),
                    rowwise_block_seconds=statistics.median(x['measurement']['target_forward_seconds'][1] for x in modified))
                write_json(args.output/'rowwise.partial.json', report)
                print(json.dumps(dict(stage='rowwise', id=item['id'], matches=[x['matches'] for x in item['rowwise']],
                    focus=item['rowwise'][0]['focus'], query_exact=item['single_row_query_matches_reference'],
                    timing=item['instrumented_timing'])), flush=True)

            new = fixtures[0]
            report['ordinary_default'] = timed(harness, lambda: harness.native(new, oracle=False))
            with single_row_linears(target):
                report['ordinary_rowwise'] = timed(harness, lambda: harness.native(new, oracle=False))
            report['all_fixtures_exact'] = all(x['matches'] == 16 and x['accepted_proposals'] == 15
                for f in report['fixtures'] for x in f['rowwise'])
            report['all_query_rows_exact'] = all(f['single_row_query_matches_reference'] for f in report['fixtures'])
            if harness.calls != planned:
                raise ValueError(f'Unexpected final target call count: {harness.calls} != {planned}')
            report['complete'] = True
    except Exception as error:
        report['error'] = str(error)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['target_calls'] = harness.calls
        report['target_seconds'] = harness.seconds
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['process_wall_seconds'] = time.monotonic()-started
        harness.close()
        write_json(args.output/'result.json', report)
    raise SystemExit(0 if report['all_fixtures_exact'] else 1)


if __name__ == '__main__':
    main()
