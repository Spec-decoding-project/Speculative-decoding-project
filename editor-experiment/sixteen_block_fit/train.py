"""Fit sixteen fixed blocks; evaluate development only after fixing checkpoints."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import statistics
import time

import torch

from autoregressive_editor.learning import Trainer, evaluate
from autoregressive_editor.model import Config
from one_block_fit.collect import save, sha
from one_block_fit.train import causal_check, detailed, passed, weights_hash
from sixteen_block_fit.data import build


INITIAL_HASH = '672080cdb5b25a32ae824198f3549e43979352bab0f621e75379dd9c46d03ab8'


def fit(args, dataset, mode):
    path = args.output/mode
    path.mkdir(exist_ok=False)
    rows = dataset['training']
    torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(Config(), mode, seed=0, lr=1e-3, device='cuda')
    parameters = sum(p.numel() for p in trainer.model.parameters())
    if parameters != 40750336:
        raise ValueError('Full-size editor parameter count changed')
    owned = {id(p) for p in trainer.model.parameters()}
    optimized = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
    if owned != optimized:
        raise ValueError('Optimizer ownership failure')
    initial_hash = weights_hash(trainer.model)
    if initial_hash != INITIAL_HASH:
        raise ValueError('Initial weights differ from the controlled one-block experiment')
    for row in rows:
        causal_check(trainer, row)
    save(path/'initial-training.json', detailed(trainer.model, rows, mode))
    metadata = dict(dataset_sha256=sha(args.dataset), mode=mode,
                    training_keys=[dict(prompt_id=r['prompt_id'], anchor=r['anchor']) for r in rows])
    save(path/'run.json', dict(config=asdict(trainer.model.config), mode=mode, seed=0, lr=1e-3,
         parameters=parameters, initial_weights_sha256=initial_hash, optimizer='AdamW',
         weight_decay=0.0, clip_norm=1.0, precision='FP32', microbatch=1, effective_batch=16,
         suffix_tokens_per_update=224, sampling='All sixteen blocks in fixed dataset order each update',
         max_updates=2000, max_training_seconds=600, evaluation_interval=25,
         required_consecutive_passes=3, optimizer_ownership='Editor parameters only; target/drafter not loaded',
         metadata=metadata, torch=str(torch.__version__), python=platform.python_version(),
         gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda))
    consecutive, evaluations, updates = 0, [], []
    outcome = 'update_limit'
    started = time.monotonic()
    with (path/'updates.jsonl').open('w') as log:
        while trainer.step < 2000:
            if time.monotonic()-started >= 600:
                outcome = 'time_limit'
                break
            torch.cuda.synchronize()
            before = time.perf_counter()
            update = trainer.update(rows)
            torch.cuda.synchronize()
            update['seconds'] = time.perf_counter()-before
            if update['suffix_tokens'] != 224:
                raise ValueError('Effective-batch token normalization changed')
            updates.append(update)
            log.write(json.dumps(update)+'\n')
            log.flush()
            if trainer.step % 25 == 0:
                metrics = evaluate(trainer.model, rows, mode)
                consecutive = consecutive+1 if passed(metrics) else 0
                evaluations.append(dict(step=trainer.step, consecutive_passes=consecutive, **metrics))
                save(path/'training-evaluations.json', evaluations)
                print(json.dumps(dict(mode=mode, step=trainer.step, ce=metrics['teacher_forced_ce'],
                      exact=metrics['free_running_exact_suffix'], accuracy=metrics['free_running_token_accuracy'],
                      consecutive_passes=consecutive)), flush=True)
                if consecutive == 3:
                    outcome = 'memorization_passed'
                    break
    elapsed = time.monotonic()-started
    final_hash = weights_hash(trainer.model)
    if final_hash == initial_hash:
        raise ValueError('Optimizer did not change weights')
    for row in rows:
        causal_check(trainer, row)
        if trainer.model.propose(row['prefix'], row['draft'], mode) != trainer.model.propose(
                row['prefix'], row['draft'], mode, cached=False):
            raise ValueError('Cached versus full-sequence editor output mismatch')
    final_train = evaluate(trainer.model, rows, mode)
    checkpoint = path/'checkpoint.pt'
    trainer.save(checkpoint, metadata)
    del trainer
    torch.cuda.empty_cache()
    restored, saved = Trainer.load(checkpoint, device='cuda')
    if saved != metadata or weights_hash(restored.model) != final_hash:
        raise ValueError('Checkpoint state/provenance changed')
    if evaluate(restored.model, rows, mode) != final_train:
        raise ValueError('Reloaded checkpoint training outputs changed')
    # No excluded-prompt results are computed until the checkpoint is fixed.
    results = {}
    for name in ('training', 'development', 'preservation'):
        results[name] = detailed(restored.model, dataset[name], mode, timing=True)
        save(path/(name+'.json'), results[name])
    del restored
    torch.cuda.empty_cache()
    untrained = Trainer(Config(), mode, seed=0, lr=1e-3, device='cuda')
    if weights_hash(untrained.model) != initial_hash:
        raise ValueError('Initial-state reconstruction failed')
    for name in ('development', 'preservation'):
        save(path/('untrained-'+name+'.json'), detailed(untrained.model, dataset[name], mode))
    summary = dict(outcome=outcome, updates=len(updates), training_seconds=elapsed,
         initial_weights_sha256=initial_hash, final_weights_sha256=final_hash,
         checkpoint_sha256=sha(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
         checkpoint_reload='passed', cached_full_decoding='passed', future_label_check='passed',
         mean_first_100_update_seconds=statistics.mean(r['seconds'] for r in updates[:100]),
         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
         **{name: {k: v for k, v in result.items() if k != 'rows'} for name, result in results.items()})
    save(path/'summary.json', summary)
    print(json.dumps(dict(mode=mode, outcome=outcome, updates=len(updates),
          development_original=results['development']['mean_original_progress'],
          development_edited=results['development']['mean_edited_progress'],
          preservation_damaged=results['preservation']['damaged'])), flush=True)
    del untrained
    torch.cuda.empty_cache()
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    if os.environ.get('VLLM_BATCH_INVARIANT', '0') != '0':
        raise ValueError('Target-only invariant configuration leaked into editor process')
    dataset = json.loads(args.dataset.read_text())
    if dataset != build(args.collection):
        raise ValueError('Dataset differs from deterministic capture re-audit/selection')
    import autoregressive_editor
    editor_dir = Path(autoregressive_editor.__file__).parent
    for name, expected in dataset['editor_source_hashes'].items():
        if sha(editor_dir/name) != expected:
            raise ValueError('Editor source changed after native collection')
    args.output.mkdir(exist_ok=False, parents=True)
    results = {mode: fit(args, dataset, mode) for mode in ('context', 'draft')}
    save(args.output/'result.json', dict(variants=results, dataset_sha256=sha(args.dataset),
         scope='Sixteen repairable blocks/four training prompts; eight development prompts; one excluded preservation control; no full-generation speedup claim'))


if __name__ == '__main__':
    main()
