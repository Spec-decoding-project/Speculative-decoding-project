"""Fit one real block, then evaluate fixed checkpoints on excluded prompts."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import torch

from autoregressive_editor.learning import Trainer, evaluate, tensors
from autoregressive_editor.model import Config
from one_block_fit.collect import make_bank, save, sha, progress


def weights_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def passed(metrics):
    return (metrics['teacher_forced_ce'] < 0.05
            and metrics['free_running_exact_suffix'] == 1.0
            and metrics['free_running_token_accuracy'] == 1.0)


@torch.no_grad()
def causal_check(trainer, row):
    changed = deepcopy(row)
    changed['labels'][2] = (changed['labels'][2]+1) % trainer.model.config.vocab_size
    a, sa, start, _ = tensors(trainer.model,row,trainer.mode)
    b, sb, _, _ = tensors(trainer.model,changed,trainer.mode)
    x, _ = trainer.model(a,sa,logit_start=start)
    y, _ = trainer.model(b,sb,logit_start=start)
    if not torch.equal(x[:,0],y[:,0]):
        raise ValueError('A future correct token leaked into an earlier prediction')


@torch.no_grad()
def detailed(model, rows, mode, timing=False):
    result = evaluate(model,rows,mode)
    for row, metric in zip(rows,result['rows']):
        edited = model.propose(row['prefix'],row['draft'],mode)
        metric.update(original_tokens=row['draft'],edited_tokens=edited,
                      label_tokens=row['labels'],kind=row['kind'])
        if timing:
            for _ in range(2):
                model.propose(row['prefix'],row['draft'],mode)
            times = []
            for _ in range(10):
                torch.cuda.synchronize()
                started = time.perf_counter()
                measured = model.propose(row['prefix'],row['draft'],mode)
                torch.cuda.synchronize()
                times.append((time.perf_counter()-started)*1000)
                if measured != edited:
                    raise ValueError('Editor changed output during timing')
            metric['editor_milliseconds'] = dict(median=statistics.median(times), samples=times)
    result['mean_original_progress'] = statistics.mean(r['original_progress'] for r in result['rows'])
    result['mean_edited_progress'] = statistics.mean(r['edited_progress'] for r in result['rows'])
    result['mean_union_progress'] = statistics.mean(r['union_progress'] for r in result['rows'])
    result['progress_unit'] = 'Consecutive matching proposals after the target anchor; maximum 15'
    if timing:
        medians = [r['editor_milliseconds']['median'] for r in result['rows']]
        result['median_editor_milliseconds'] = statistics.median(medians)
        result['timing_scope'] = 'FP32 cached editor proposal, 14 sequential outputs, two warmups/ten samples per block; target/drafter unloaded'
    return result


def fit(args, bank, mode):
    path = args.output/mode
    path.mkdir(exist_ok=False)
    row = bank['training'][0]
    trainer = Trainer(Config(), mode, seed=0, lr=1e-3, device='cuda')
    parameters = sum(p.numel() for p in trainer.model.parameters())
    if parameters != 40750336:
        raise ValueError('Full-size editor parameter count changed')
    owned = {id(p) for p in trainer.model.parameters()}
    optimized = {id(p) for g in trainer.optimizer.param_groups for p in g['params']}
    if owned != optimized:
        raise ValueError('Optimizer ownership failure')
    initial_hash = weights_hash(trainer.model)
    causal_check(trainer,row)
    initial = detailed(trainer.model,[row],mode)
    save(path/'initial-training.json',initial)
    metadata = dict(bank_sha256=sha(args.collection/'bank.json'), training_ids=[row['prompt_id']],
                    training_anchors=[row['anchor']], mode=mode)
    save(path/'run.json',dict(config=asdict(trainer.model.config),mode=mode,seed=0,lr=1e-3,
         parameters=parameters,initial_weights_sha256=initial_hash,optimizer='AdamW',
         weight_decay=0.0,clip_norm=1.0,precision='FP32',microbatch=1,
         max_updates=2000,max_training_seconds=600,evaluation_interval=25,
         required_consecutive_passes=3,optimizer_ownership='editor parameters only',
         metadata=metadata,torch=str(torch.__version__),python=platform.python_version(),
         gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda))
    consecutive, evaluations, updates = 0, [], []
    started = time.monotonic()
    outcome = 'update_limit'
    with (path/'updates.jsonl').open('w') as log:
        while trainer.step < 2000:
            if time.monotonic()-started >= 600:
                outcome = 'time_limit'
                break
            torch.cuda.synchronize()
            t = time.perf_counter()
            update = trainer.update([row])
            torch.cuda.synchronize()
            update['seconds'] = time.perf_counter()-t
            updates.append(update)
            log.write(json.dumps(update)+'\n')
            log.flush()
            if trainer.step % 25 == 0:
                metrics = evaluate(trainer.model,[row],mode)
                consecutive = consecutive+1 if passed(metrics) else 0
                record = dict(step=trainer.step,consecutive_passes=consecutive,**metrics)
                evaluations.append(record)
                save(path/'training-evaluations.json',evaluations)
                print(json.dumps(dict(mode=mode,step=trainer.step,ce=metrics['teacher_forced_ce'],
                    exact=metrics['free_running_exact_suffix'],passes=consecutive)),flush=True)
                if consecutive == 3:
                    outcome = 'memorization_passed'
                    break
    elapsed = time.monotonic()-started
    final_hash = weights_hash(trainer.model)
    if final_hash == initial_hash:
        raise ValueError('Optimizer did not change weights')
    causal_check(trainer,row)
    proposal = trainer.model.propose(row['prefix'],row['draft'],mode)
    if proposal != trainer.model.propose(row['prefix'],row['draft'],mode,cached=False):
        raise ValueError('Cached versus full-sequence editor output mismatch')
    checkpoint = path/'checkpoint.pt'
    trainer.save(checkpoint,metadata)
    final_train = evaluate(trainer.model,[row],mode)
    del trainer
    torch.cuda.empty_cache()
    restored, saved = Trainer.load(checkpoint,device='cuda')
    if saved != metadata or weights_hash(restored.model) != final_hash:
        raise ValueError('Checkpoint state/provenance changed')
    if evaluate(restored.model,[row],mode) != final_train:
        raise ValueError('Checkpoint predictions changed')
    # Held-out metrics are computed only after stopping and fixing the checkpoint.
    train_metrics = detailed(restored.model,[row],mode,timing=True)
    test_metrics = detailed(restored.model,bank['held_out'],mode,timing=True)
    save(path/'final-training.json',train_metrics)
    save(path/'held-out.json',test_metrics)
    del restored
    torch.cuda.empty_cache()
    untrained = Trainer(Config(),mode,seed=0,lr=1e-3,device='cuda')
    if weights_hash(untrained.model) != initial_hash:
        raise ValueError('Initial-state reconstruction failed')
    save(path/'untrained-held-out.json',detailed(untrained.model,bank['held_out'],mode))
    summary = dict(outcome=outcome,updates=len(updates),training_seconds=elapsed,
         initial_weights_sha256=initial_hash,final_weights_sha256=final_hash,
         checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
         checkpoint_reload='passed',cached_full_decoding='passed',future_label_check='passed',
         first_100_update_seconds=statistics.mean(r['seconds'] for r in updates[:100]),
         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
         training={k:v for k,v in train_metrics.items() if k!='rows'},
         held_out={k:v for k,v in test_metrics.items() if k!='rows'})
    save(path/'summary.json',summary)
    print(json.dumps(dict(mode=mode,outcome=outcome,updates=len(updates),
                          held_out_original=test_metrics['mean_original_progress'],
                          held_out_edited=test_metrics['mean_edited_progress'])),flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    # Explicitly use ordinary PyTorch for the editor, not vLLM's global overrides.
    if __import__('os').environ.get('VLLM_BATCH_INVARIANT','0') != '0':
        raise ValueError('Target-only invariant configuration leaked into editor process')
    bank = json.loads((args.collection/'bank.json').read_text())
    inputs = json.loads((args.collection/'inputs.json').read_text())
    import autoregressive_editor
    editor_dir = Path(autoregressive_editor.__file__).parent
    for name,expected in inputs['editor_source_hashes'].items():
        if sha(editor_dir/name) != expected:
            raise ValueError('Editor implementation changed after data collection')
    for name,expected in bank['provenance'].items():
        if sha(args.collection/name) != expected:
            raise ValueError('Collection evidence changed')
    fresh = make_bank(inputs,json.loads((args.collection/'greedy.json').read_text()),
                      json.loads((args.collection/'native.json').read_text()))
    if any(bank[k] != v for k,v in fresh.items()):
        raise ValueError('Independent native token-set audit failed')
    if len(bank['training']) != 1 or len({r['prompt_id'] for r in bank['held_out']}) != 8:
        raise ValueError('Unreviewed training/evaluation scope')
    args.output.mkdir(exist_ok=False,parents=True)
    results = {mode:fit(args,bank,mode) for mode in ('context','draft')}
    if results['context']['initial_weights_sha256'] != results['draft']['initial_weights_sha256']:
        raise ValueError('Variants did not start from identical weights')
    save(args.output/'result.json',dict(variants=results,bank_sha256=sha(args.collection/'bank.json'),
         scope='One real training block; eight excluded validation prompts; no full-generation speedup claim'))


if __name__ == '__main__':
    main()
