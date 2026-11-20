"""A bounded, paired COPY-action pilot; final-test rows are never loaded."""
import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import time

import numpy as np
import torch

from autoregressive_editor.model import Config
from copy_action_fit.data import (BANK_SHA256, CHECKPOINT_SHA256, SHARED_WEIGHTS_SHA256,
                                  digest, read_bank, save, sha)
from copy_action_fit.learning import PilotTrainer, aggregate, evaluate, inputs, passed, weights_hash
from full_corpus_fit.audit_results import audit_predictions


def require(condition, message):
    if not condition:
        raise ValueError(message)


def status(output, **fields):
    record = dict(utc_epoch=time.time(), **fields)
    save(output/'progress.json', record)
    print(json.dumps(record), flush=True)


@torch.no_grad()
def causal_check(model, row):
    changed = deepcopy(row)
    changed['labels'][2] = (changed['labels'][2]+1) % model.config.vocab_size
    x, s, start, _, _ = inputs(model, row)
    y, t, _, _, _ = inputs(model, changed)
    a, _ = model(x, s, logit_start=start)
    b, _ = model(y, t, logit_start=start)
    require(torch.equal(a[:, 0], b[:, 0]), 'A future label changed the first action')


def audit(result, examples, variant, vocab_size):
    report = audit_predictions(result, examples)
    copies = 0
    for row in result['rows']:
        require(len(row['actions']) == 14, 'Missing action trace')
        for offset, action in enumerate(row['actions'], start=2):
            require(type(action) is int and 0 <= action < vocab_size + (variant == 'copy'),
                    'Invalid output action')
            emitted = row['original_tokens'][offset] if action == vocab_size else action
            require(emitted == row['edited_tokens'][offset], 'Action emitted the wrong token')
            copies += action == vocab_size
    require(copies == result['copy_actions'], 'Incorrect COPY count')
    return dict(passed=True, copy_actions=copies, **report)


@torch.no_grad()
def timing(model, examples):
    records = []
    for row in examples:
        expected = model.propose(row['prefix'], row['draft'], 'draft')
        require(model.propose(row['prefix'], row['draft'], 'draft', cached=False) == expected,
                'Cached/full editor proposals differ')
        for _ in range(2):
            model.propose(row['prefix'], row['draft'], 'draft')
        samples = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            actual = model.propose(row['prefix'], row['draft'], 'draft')
            torch.cuda.synchronize()
            samples.append(1000*(time.perf_counter()-start))
            require(actual == expected, 'Timing repetitions changed emitted tokens')
        records.append(dict(prompt_id=row['prompt_id'], anchor=row['anchor'],
                            median_ms=statistics.median(samples), samples_ms=samples))
    return dict(blocks=len(records), median_editor_ms=statistics.median(r['median_ms'] for r in records),
                cached_full='passed', rows=records,
                scope='FP32 cached editor only; 14 sequential outputs; 2 warmups/10 samples per block')


def uncertainty(left, right, *, draws=10000):
    """Paired bootstrap over prompts, keeping each prompt's four blocks together."""
    values = defaultdict(list)
    require(len(left) == len(right), 'Unpaired uncertainty populations')
    for a, b in zip(left, right):
        require((a['prompt_id'], a['anchor']) == (b['prompt_id'], b['anchor']),
                'Unpaired uncertainty identities')
        values[a['prompt_id']].append(a['edited_progress']-b['edited_progress'])
    deltas = np.array([statistics.mean(values[key]) for key in sorted(values)], dtype=np.float64)
    rng = np.random.default_rng(0)
    sampled = np.empty(draws)
    for start in range(0, draws, 100):
        count = min(100, draws-start)
        sampled[start:start+count] = deltas[rng.integers(len(deltas), size=(count, len(deltas)))].mean(1)
    return dict(mean_delta=float(deltas.mean()), ci95=np.quantile(sampled, [.025, .975]).tolist(),
                prompts=len(deltas), bootstrap_draws=draws, seed=0,
                scope='Exploratory development comparison, paired resampling by prompt')


def fit(args, variant, training, development, contract_sha, deadline):
    output = args.output/variant
    output.mkdir(exist_ok=True)
    checkpoint = output/'checkpoint.pt'
    if (output/'complete.json').exists():
        result = json.loads((output/'complete.json').read_text())
        require(result['contract_sha256'] == contract_sha and sha(checkpoint) == result['checkpoint_sha256'],
                'Completed arm provenance changed')
        return result
    if checkpoint.exists():
        trainer, state = PilotTrainer.load(checkpoint, device='cuda')
        require(state['contract_sha256'] == contract_sha and trainer.variant == variant,
                'Checkpoint provenance/arm mismatch')
    else:
        trainer = PilotTrainer(Config(), variant, device='cuda')
        source_step = trainer.warm_start(args.checkpoint)
        shared = weights_hash(trainer.model, shared=True)
        require(shared == SHARED_WEIGHTS_SHA256, 'Shared initial weights changed')
        state = dict(contract_sha256=contract_sha, consecutive=0, history=[],
                     shared_initial_sha256=shared, source_step=source_step, outcome=None)
        initial = evaluate(trainer.model, training, variant)
        save(output/'initial-training.json', initial)
        if variant == 'baseline':
            # Replay every saved development proposal before making any update.
            saved = json.loads(args.reference.read_text())
            replay = evaluate(trainer.model, development, variant)
            require(len(saved['rows']) == len(replay['rows']), 'Reference population changed')
            for old, new in zip(saved['rows'], replay['rows']):
                for key in ('prompt_id', 'anchor', 'original_tokens', 'label_tokens', 'edited_tokens'):
                    require(old[key] == new[key], f'Baseline runtime replay changed {key}')
            require(abs(saved['teacher_forced_ce']-replay['teacher_forced_ce']) < 1e-5,
                    'Baseline runtime replay changed loss')
            save(output/'initial-development.json', replay)
            save(output/'baseline-replay.json', dict(passed=True, blocks=len(development),
                 reference_sha256=sha(args.reference), shared_weights_sha256=shared,
                 teacher_forced_ce_delta=replay['teacher_forced_ce']-saved['teacher_forced_ce']))
        trainer.save(checkpoint, state)
    expected_parameters = 40750336 + (257 if variant == 'copy' else 0)
    require(sum(p.numel() for p in trainer.model.parameters()) == expected_parameters, 'Parameter count changed')
    require({id(p) for p in trainer.model.parameters()} ==
            {id(p) for group in trainer.optimizer.param_groups for p in group['params']},
            'Optimizer does not exclusively own editor parameters')
    for row in (training[0], training[32], training[-1]):
        causal_check(trainer.model, row)
    started = time.monotonic()
    with (output/'updates.jsonl').open('a') as log:
        while state['outcome'] is None and trainer.step < 2000:
            if time.time() >= deadline:
                state['outcome'] = 'time_limit'
                break
            torch.cuda.synchronize()
            begin = time.perf_counter()
            update = trainer.update(training)
            torch.cuda.synchronize()
            update['seconds'] = time.perf_counter()-begin
            log.write(json.dumps(update)+'\n'); log.flush()
            if trainer.step == 1 or trainer.step % 10 == 0:
                status(args.output, phase='training', variant=variant, **update,
                       arm_seconds_left=max(0, deadline-time.time()))
            if trainer.step % 25 == 0:
                metrics = evaluate(trainer.model, training, variant)
                state['consecutive'] = state['consecutive']+1 if passed(metrics) else 0
                summary = dict(step=trainer.step, consecutive=state['consecutive'], **aggregate(metrics, training))
                state['history'].append(summary)
                save(output/'training-evaluations.json', state['history'])
                if state['consecutive'] == 3:
                    state['outcome'] = 'memorization_passed'
                trainer.save(checkpoint, state)
                status(args.output, phase='training_evaluation', variant=variant, **summary)
    if state['outcome'] is None:
        state['outcome'] = 'update_limit'
    final_train = evaluate(trainer.model, training, variant)
    final_hash = weights_hash(trainer.model)
    require(weights_hash(trainer.model, shared=True) != SHARED_WEIGHTS_SHA256,
            'Shared weights did not change during optimization')
    trainer.save(checkpoint, state)
    del trainer
    gc.collect(); torch.cuda.empty_cache()
    trainer, restored_state = PilotTrainer.load(checkpoint, device='cuda')
    require(restored_state == state and weights_hash(trainer.model) == final_hash,
            'Reload changed checkpoint provenance/weights')
    require(evaluate(trainer.model, training, variant) == final_train,
            'Checkpoint reload changed training loss or emitted tokens')
    save(output/'final-training.json', final_train)
    status(args.output, phase='development', variant=variant, updates=trainer.step)
    dev = evaluate(trainer.model, development, variant)
    save(output/'development.json', dev)
    audits = dict(training=audit(final_train, training, variant, trainer.model.config.vocab_size),
                  development=audit(dev, development, variant, trainer.model.config.vocab_size))
    save(output/'audit.json', audits)
    timed = timing(trainer.model, [development[i*(len(development)-1)//63] for i in range(64)])
    save(output/'timing.json', timed)
    result = dict(variant=variant, contract_sha256=contract_sha, outcome=state['outcome'],
        updates=trainer.step, consecutive_passes=state['consecutive'], parameters=expected_parameters,
        checkpoint_sha256=sha(checkpoint), weights_sha256=final_hash,
        shared_initial_sha256=state['shared_initial_sha256'], checkpoint_reload='passed',
        cached_full='passed on 64 fixed development blocks', independent_prediction_audit='passed',
        training=aggregate(final_train, training), development=aggregate(dev, development),
        median_editor_ms=timed['median_editor_ms'], elapsed_seconds=time.monotonic()-started,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(output/'complete.json', result)
    del trainer
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'checkpoint', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('VLLM_BATCH_INVARIANT', '0') == '0', 'Unexpected target invariance override')
    torch.set_num_threads(4)
    require(torch.cuda.is_available(), 'Pilot requires the allocated GPU')
    args.output.mkdir(parents=True, exist_ok=True)
    started = json.loads((args.output/'budget.json').read_text())['started_epoch']
    require(time.time() < started+6600, 'Paired pilot budget already exhausted; no automatic extension')
    require(sha(args.checkpoint) == CHECKPOINT_SHA256, 'Warm-start checkpoint fingerprint changed')
    training, development, selection = read_bank(args.bank)
    require(selection['selection_sha256'] == 'f286a92360b74187f78f93ff798caf3f9a5a9be2157d145126287b34c0485809',
            'Native selection differs from prelaunch selection')
    save(args.output/'selection.json', selection)
    save(args.output/'training-examples.json', training)
    source_root = Path(__file__).resolve().parent.parent
    paths = [p for package in ('copy_action_fit', 'autoregressive_editor')
             for p in (source_root/package).glob('*.py')]
    paths.append(source_root/'full_corpus_fit'/'audit_results.py')
    contract = dict(schema='copy-action-pilot-v1', config=asdict(Config()), selection=selection,
        checkpoint_sha256=CHECKPOINT_SHA256, reference_sha256=sha(args.reference),
        shared_weights_sha256=SHARED_WEIGHTS_SHA256,
        source_hashes={str(p.relative_to(source_root)):sha(p) for p in sorted(paths)},
        lr=.0001, weight_decay=.01, clip_norm=1., effective_batch=64, microbatch=1, seed=0,
        precision='FP32', max_updates_per_arm=2000, evaluation_interval=25,
        required_consecutive_passes=3, maximum_total_seconds=7200,
        maximum_optimization_seconds_per_arm=3150,
        selection_rule='Training exact suffixes=100% and induced token CE<0.05; no dev selection',
        python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), final_test_loaded=False)
    path = args.output/'run.json'
    require(not path.exists() or json.loads(path.read_text()) == contract, 'Run contract changed on resume')
    save(path, contract)
    status(args.output, phase='starting', selection=selection)
    baseline = fit(args, 'baseline', training, development, digest(contract),
                   min(time.time()+3150, started+3300))
    copy = fit(args, 'copy', training, development, digest(contract),
               min(time.time()+3150, started+6600))
    baseline_rows = json.loads((args.output/'baseline'/'development.json').read_text())['rows']
    copy_rows = json.loads((args.output/'copy'/'development.json').read_text())['rows']
    original = [dict(row, edited_progress=row['original_progress']) for row in baseline_rows]
    intervals = dict(baseline_minus_original=uncertainty(baseline_rows, original),
                     copy_minus_original=uncertainty(copy_rows, original),
                     copy_minus_baseline=uncertainty(copy_rows, baseline_rows))
    require(sha(args.bank) == BANK_SHA256 and sha(args.checkpoint) == CHECKPOINT_SHA256,
            'Completed full-corpus artifacts changed')
    result = dict(passed_checks=True, baseline=baseline, copy=copy, uncertainty=intervals,
        elapsed_total_seconds=time.time()-started, final_test_loaded=False,
        scope='64-block training fit and exploratory development coverage; no native acceptance or speedup claim')
    save(args.output/'complete.json', result)
    status(args.output, phase='complete', baseline_outcome=baseline['outcome'], copy_outcome=copy['outcome'],
           baseline_prefix=baseline['development']['mean_edited_progress'],
           copy_prefix=copy['development']['mean_edited_progress'],
           original_prefix=copy['development']['mean_original_progress'])


if __name__ == '__main__':
    main()
