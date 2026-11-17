"""Independent fixed-token target shape diagnostic. No labels/model files change."""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def jd(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    import torch
    import transformers
    from transformers import DynamicCache
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    feature_dir = Path('research/editor-feature-bank/feature_bank').resolve()
    sys.path.insert(0, str(feature_dir))
    from collect import target_fingerprints
    from data import records
    model_path = Path('data/models/qwen3-8b')
    pilot = Path('data/qwen_responses_pilot')
    original = Path('scripts/generate_qwen_responses.py')
    run = json.loads((pilot / 'run.json').read_text())
    assert sha(original) == run['generator_sha256'], 'Original generator hash mismatch'
    configs = {'ultrachat_200k:train_sft:37680': (24, 6, 16984),
               'ultrachat_200k:train_sft:61759': (22, 15, 3070),
               'gsm8k:train:3821': (38, 5, None)}
    fixtures = []
    for row in records(pilot):
        if row['id'] in configs:
            anchor, focus, alternate = configs[row['id']]
            assert len(row['prompt_token_ids']) == anchor
            assert len(row['response_token_ids']) >= 16
            fixtures.append(dict(row=row, prefix=row['prompt_token_ids'],
                                 labels=row['response_token_ids'][:16], focus=focus, alternate=alternate))
    assert len(fixtures) == 3
    fixtures.sort(key=lambda f: f['row']['id'])
    report = dict(purpose='Independent target-only execution-shape diagnosis; first 16 tokens only',
                  code_sha256=sha(__file__), original_generator_sha256=sha(original),
                  source_run_sha256=sha(pilot / 'run.json'), target_files=target_fingerprints(model_path),
                  dependency_sha256={p.name: sha(p) for p in feature_dir.glob('*.py')},
                  baseline=[], probes=[], complete=False,
                  runtime=dict(python=platform.python_version(), torch=torch.__version__,
                               transformers=transformers.__version__, cuda=torch.version.cuda,
                               gpu=torch.cuda.get_device_name(), host=platform.node(),
                               job_id=os.environ.get('SLURM_JOB_ID'), tf32=False,
                               deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                               cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG')))
    save(args.output / 'setup.json', report)
    sys.path.insert(0, str(original.parent.resolve()))
    spec = importlib.util.spec_from_file_location('original_collector', original)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    backend = source.QwenBackend(model_path, run['generation'])
    backend.generation_config.max_new_tokens = 16
    target = backend.model.eval().requires_grad_(False)
    report['generation_config'] = backend.generation_config.to_dict()
    report['source_change'] = 'max_new_tokens 2048 -> 16 only'
    report['target_config'] = target.config.to_dict()
    report['qwen_implementation_sha256'] = sha(inspect.getfile(type(target)))
    report['target_trainable_parameters'] = sum(p.numel() for p in target.parameters() if p.requires_grad)
    budget = dict(calls=0, target_seconds=0.0, limit=300, seconds_limit=120)

    def before(module, a, kw):
        if budget['calls'] >= budget['limit'] or budget['target_seconds'] >= budget['seconds_limit']:
            raise RuntimeError('Bounded diagnostic forward limit reached')
        torch.cuda.synchronize()
        budget['calls'] += 1
        budget['call_start'] = time.monotonic()

    def after(module, a, kw, out):
        torch.cuda.synchronize()
        budget['target_seconds'] += time.monotonic() - budget['call_start']

    h1 = target.register_forward_pre_hook(before, with_kwargs=True)
    h2 = target.register_forward_hook(after, with_kwargs=True)

    def td(t):
        return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    def summarize(logits, fixture):
        logits = logits.reshape(-1, logits.shape[-1])
        assert logits.shape[0] == 16 and torch.isfinite(logits).all()
        labels = fixture['labels']
        actual = logits.argmax(-1).tolist()
        i = fixture['focus']
        row = logits[i].float()
        expected = labels[i]
        alt = fixture['alternate']
        if alt is None:
            alt = next(t for t in row.topk(2).indices.tolist() if t != expected)
        vals, ids = row.topk(5)
        return dict(matches=sum(a == b for a, b in zip(actual, labels)), actual=actual,
                    mismatches=[dict(position=i, absolute_position=len(fixture['prefix'])+i,
                                     expected=y, actual=p) for i,(y,p) in enumerate(zip(labels, actual)) if y != p],
                    logits_sha256=td(logits), logits_dtype=str(logits.dtype),
                    focus=dict(position=i, expected=expected, selected=int(row.argmax()), alternate=alt,
                               expected_logit=float(row[expected]), alternate_logit=float(row[alt]),
                               margin=float(row[expected]-row[alt]),
                               tie_ids=torch.where(row == row.max())[0].tolist(),
                               top5=[dict(id=t, logit=v) for t,v in zip(ids.tolist(), vals.tolist())]))

    def original_generate(fixture):
        logits, observed = [], []
        def pre(module, a, kw):
            x = kw.get('input_ids', a[0] if a else None)
            observed.append(x[0].tolist())
        def post(module, a, kw, out):
            logits.append(out.logits[:, -1:].detach().clone())
        p = target.register_forward_pre_hook(pre, with_kwargs=True)
        q = target.register_forward_hook(post, with_kwargs=True)
        try:
            generated = backend.generate(fixture['row'])
        finally:
            p.remove(); q.remove()
        assert generated['prompt_token_ids'] == fixture['prefix'], 'Prompt retokenization differs'
        assert generated['rendered_prompt_sha256'] == fixture['row']['rendered_prompt_sha256']
        result = summarize(torch.cat(logits, dim=1), fixture)
        result['generated_equal_saved'] = generated['response_token_ids'] == fixture['labels']
        result['input_prefixes_equal_saved'] = observed == [fixture['prefix']] + [[t] for t in fixture['labels'][:-1]]
        return result

    def cached_chunks(fixture, size):
        cache = DynamicCache()
        prefix = fixture['prefix']
        ids = torch.tensor([prefix], device=target.device)
        out = target(ids, position_ids=torch.arange(len(prefix), device=target.device)[None],
                     past_key_values=cache, use_cache=True, logits_to_keep=1, output_hidden_states=True)
        values = [out.logits.detach().clone()]
        cache_steps = []
        for start in range(0, 16, size):
            end = min(start + size, 16)
            assert cache.get_seq_length() == len(prefix)+start
            x = torch.tensor([fixture['labels'][start:end]], device=target.device)
            out = target(x, position_ids=torch.arange(len(prefix)+start, len(prefix)+end, device=target.device)[None],
                         past_key_values=cache, use_cache=True, output_hidden_states=True)
            assert cache.get_seq_length() == len(prefix)+end
            cache_steps.append(dict(before=len(prefix)+start, after=len(prefix)+end, input_rows=end-start))
            values.append(out.logits.detach().clone())
        # Prefill predicts anchor; block row i predicts label i+1. Ignore final bonus.
        result = summarize(torch.cat(values, dim=1)[:, :16], fixture)
        result['chunk_size'] = size
        result['cache_steps'] = cache_steps
        return result

    def uncached(fixture):
        x = torch.tensor([fixture['prefix'] + fixture['labels'][:-1]], device=target.device)
        out = target(x, use_cache=False, logits_to_keep=16)
        return summarize(out.logits, fixture)

    # 3*(3*(16+2+1)) = 171 baseline calls.
    # 3*(17+9+5+3) = 102 chunk-probe calls.
    # 3*2*2 = 12 deterministic-mode cached-block calls. Total 285, cap 300.
    report['planned_target_calls'] = 285
    try:
        with torch.inference_mode():
            for fixture in fixtures:
                item = dict(id=fixture['row']['id'], record_sha256=fixture['row']['record_sha256'],
                            prefix_sha256=jd(fixture['prefix']), labels_sha256=jd(fixture['labels']),
                            anchor=len(fixture['prefix']), focus=fixture['focus'],
                            generation=[], cached_block16=[], uncached=[])
                for repeat in range(3):
                    item['generation'].append(original_generate(fixture))
                    item['cached_block16'].append(cached_chunks(fixture, 16))
                    item['uncached'].append(uncached(fixture))
                item['repeat_logits_bitwise_identical'] = {
                    k: len({r['logits_sha256'] for r in item[k]}) == 1
                    for k in ('generation', 'cached_block16', 'uncached')}
                report['baseline'].append(item)
                save(args.output / 'baseline.partial.json', report)
                print(json.dumps(dict(id=item['id'], matches={k:[r['matches'] for r in item[k]]
                    for k in ('generation','cached_block16','uncached')},
                    repeat_logits_bitwise_identical=item['repeat_logits_bitwise_identical'])), flush=True)
            report['baseline_exact'] = all(r['matches'] == 16 for f in report['baseline']
                                          for k in ('generation', 'cached_block16', 'uncached') for r in f[k])
            report['baseline_budget'] = dict(budget)
            save(args.output / 'baseline.json', report)
            print('BASELINE_EXACT=' + str(report['baseline_exact']) + '; waiting for probes.ready', flush=True)
            deadline = time.monotonic() + 180
            while not (args.output / 'probes.ready').exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError('Probe review marker timed out; baseline preserved')
                time.sleep(1)
            for fixture in fixtures:
                item = dict(id=fixture['row']['id'], chunks=[])
                for size in (1, 2, 4, 8):
                    item['chunks'].append(cached_chunks(fixture, size))
                report['probes'].append(item)
                save(args.output / 'probes.partial.json', report)
                print(json.dumps(dict(id=item['id'], chunks={r['chunk_size']: r['mismatches'] for r in item['chunks']})), flush=True)
            # Same settings and shapes; change only PyTorch's deterministic flag.
            previous = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(True)
            report['deterministic_mode'] = []
            try:
                for fixture in fixtures:
                    item = dict(id=fixture['row']['id'], repeats=[])
                    try:
                        item['repeats'] = [cached_chunks(fixture, 16) for _ in range(2)]
                    except RuntimeError as error:
                        item['unsupported_error'] = str(error)
                        report['deterministic_mode'].append(item)
                        break
                    report['deterministic_mode'].append(item)
                    print(json.dumps(dict(id=item['id'], deterministic_block_matches=[x['matches'] for x in item['repeats']])), flush=True)
            finally:
                torch.use_deterministic_algorithms(previous)
            report['complete'] = True
    except Exception as error:
        report['error'] = str(error)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        h1.remove(); h2.remove()
        report['budget'] = {k:v for k,v in budget.items() if k != 'call_start'}
        report['wall_seconds'] = time.monotonic() - started
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        save(args.output / 'result.json', report)
    print('DIAGNOSTIC_COMPLETE; exact-token assertion=' + str(report['baseline_exact']), flush=True)
    raise SystemExit(0 if report['baseline_exact'] else 1)


if __name__ == '__main__':
    main()
