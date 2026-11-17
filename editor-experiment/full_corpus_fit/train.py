"""Three-epoch full-bank training with deterministic resume and excluded tests."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import uuid

import torch

from autoregressive_editor.learning import Trainer
from autoregressive_editor.model import Config
from full_corpus_fit.bank import Bank
from full_corpus_fit.preflight import save, sha
from one_block_fit.train import causal_check, detailed, weights_hash


INITIAL_HASH='672080cdb5b25a32ae824198f3549e43979352bab0f621e75379dd9c46d03ab8'


def epoch_order(ids, epoch):
    result=list(ids)
    random.Random(1701+epoch).shuffle(result)
    return result


def learning_rate(step, total_steps):
    warmup=max(1,int(total_steps*.02))
    if step <= warmup:
        return .001*step/warmup
    fraction=(step-warmup)/max(1,total_steps-warmup)
    return .001*(.1+.9*(1+math.cos(math.pi*fraction))/2)


def key(result):
    return (result['mean_edited_progress'], -result['teacher_forced_ce'])


def validate_cursor(state, step, examples):
    epoch,cursor=state['epoch'],state['cursor']
    if (not 0 <= epoch <= 3 or not 0 <= cursor <= examples
            or (epoch==3 and cursor!=0)
            or (cursor%32 and cursor!=examples)
            or step != epoch*math.ceil(examples/32)+math.ceil(cursor/32)):
        raise ValueError('Checkpoint epoch/cursor does not match optimizer steps')


def aggregate(result, rows):
    output={k:v for k,v in result.items() if k!='rows'}
    grouped=defaultdict(list)
    for example, metric in zip(rows,result['rows']):
        grouped['source/'+example['source']].append(metric)
        grouped['kind/'+example['kind']].append(metric)
    output['groups']={name:dict(blocks=len(items),
        mean_original_progress=statistics.mean(r['original_progress'] for r in items),
        mean_edited_progress=statistics.mean(r['edited_progress'] for r in items),
        mean_union_progress=statistics.mean(r['union_progress'] for r in items),
        repaired=sum(r['repaired'] for r in items),damaged=sum(r['damaged'] for r in items))
        for name,items in grouped.items()}
    return output


def train(args):
    if os.environ.get('VLLM_BATCH_INVARIANT','0') != '0':
        raise ValueError('Target-only invariant overrides must not affect the editor')
    torch.set_num_threads(4)
    bank=Bank(args.bank)
    bank_info=json.loads(args.bank.with_suffix('.json').read_text())
    if bank.metadata['profile'] or bank.metadata['prompt_count'] != 51024:
        raise ValueError('Full training requires the complete 51,024-prompt processing inventory')
    if sha(args.bank) != bank_info['bank_sha256']:
        raise ValueError('Training bank checksum mismatch')
    ids=bank.ids('train')
    if not ids:
        raise ValueError('No eligible full-corpus training examples')
    dev_rows=[bank.row(i) for i in bank.ids('development')]
    if not dev_rows:
        raise ValueError('No development examples')
    args.output.mkdir(parents=True,exist_ok=True)
    import autoregressive_editor
    editor_root=Path(autoregressive_editor.__file__).parent
    steps_per_epoch=math.ceil(len(ids)/32)
    total_steps=3*steps_per_epoch
    run=dict(schema=1,bank_sha256=bank_info['bank_sha256'],config=asdict(Config()),mode=args.mode,
        source_hashes={p.name:sha(p) for p in sorted(editor_root.glob('*.py'))},
        runner_sha256=sha(Path(__file__)),seed=0,epochs=3,effective_batch=32,microbatch=1,
        peak_lr=.001,weight_decay=.01,precision='FP32',gradient_clip=1.,
        scheduler='2% linear warmup then cosine decay to 10% of peak',
        training_examples=len(ids),steps_per_epoch=steps_per_epoch,total_steps=total_steps,
        selection='Highest development mean edited prefix; lower CE breaks ties',
        initial_weights_sha256=INITIAL_HASH)
    path=args.output/'run.json'
    if path.exists() and json.loads(path.read_text()) != run:
        raise ValueError('Refuse resume with changed bank, source or optimization contract')
    save(path,run)
    if (args.output/'complete.json').exists():
        print('Variant already complete; no duplicate training',flush=True)
        return
    run_hash=sha(path)
    last=args.output/'latest.pt'
    best_path=args.output/'best.pt'
    if last.exists():
        trainer,state=Trainer.load(last,device='cuda')
        if state['run_sha256'] != run_hash:
            raise ValueError('Checkpoint provenance mismatch')
    else:
        trainer=Trainer(Config(),args.mode,seed=0,lr=.001,device='cuda')
        if weights_hash(trainer.model) != INITIAL_HASH:
            raise ValueError('Full training did not start at the shared initialization')
        state=dict(run_sha256=run_hash,epoch=0,cursor=0,best_key=None,best_epoch=None,
                   initial_weights_sha256=INITIAL_HASH)
    if sum(p.numel() for p in trainer.model.parameters()) != 40750336:
        raise ValueError('Editor parameter count changed')
    validate_cursor(state,trainer.step,len(ids))
    if {id(p) for p in trainer.model.parameters()} != {id(p) for g in trainer.optimizer.param_groups for p in g['params']}:
        raise ValueError('Optimizer includes parameters outside the editor')
    for group in trainer.optimizer.param_groups:
        group['weight_decay']=.01
    for identity in (ids[0],ids[-1],ids[len(ids)//2]):
        causal_check(trainer,bank.row(identity))
    attempt=uuid.uuid4().hex[:12]
    started=time.monotonic()
    measured_updates=[]
    with (args.output/f'updates-{attempt}.jsonl').open('w') as log:
        for epoch in range(state['epoch'],3):
            order=epoch_order(ids,epoch)
            cursor=state['cursor'] if epoch==state['epoch'] else 0
            while cursor < len(order):
                batch_ids=order[cursor:cursor+32]
                rows=[bank.row(i) for i in batch_ids]
                lr=learning_rate(trainer.step+1,total_steps)
                for group in trainer.optimizer.param_groups:
                    group['lr']=lr
                torch.cuda.synchronize(); begin=time.perf_counter()
                update=trainer.update(rows)
                torch.cuda.synchronize(); seconds=time.perf_counter()-begin
                cursor+=len(batch_ids)
                measured_updates.append(seconds)
                update.update(epoch=epoch+1,cursor=cursor,lr=lr,seconds=seconds,
                              batch_examples=len(batch_ids),attempt=attempt)
                log.write(json.dumps(update)+'\n');log.flush()
                state.update(epoch=epoch,cursor=cursor)
                if trainer.step%100==0 or cursor==len(order):
                    recent=statistics.mean(measured_updates[-100:])
                    progress=dict(mode=args.mode,phase='training',step=trainer.step,total_steps=total_steps,
                        epoch=epoch+1,epochs=3,examples_in_epoch=cursor,examples_per_epoch=len(ids),
                        loss=update['ce'],lr=lr,seconds_per_recent_update=recent,
                        estimated_remaining_update_seconds=(total_steps-trainer.step)*recent,
                        session_seconds=time.monotonic()-started)
                    save(args.output/'progress.json',progress)
                    print(json.dumps(progress),flush=True)
                if trainer.step%500==0 or cursor==len(order):
                    trainer.save(last,state)
            dev=detailed(trainer.model,dev_rows,args.mode)
            save(args.output/f'development-epoch-{epoch+1}.json',dev)
            save(args.output/f'development-summary-epoch-{epoch+1}.json',aggregate(dev,dev_rows))
            score=key(dev)
            if state['best_key'] is None or score > tuple(state['best_key']):
                state.update(best_key=list(score),best_epoch=epoch+1)
                trainer.save(best_path,dict(state,selected_epoch=epoch+1))
            state.update(epoch=epoch+1,cursor=0)
            trainer.save(last,state)
            print(json.dumps(dict(mode=args.mode,completed_epoch=epoch+1,
                development_edited=dev['mean_edited_progress'],development_original=dev['mean_original_progress'],
                development_ce=dev['teacher_forced_ce'],best_epoch=state['best_epoch'])),flush=True)
    final_step=trainer.step
    del trainer
    torch.cuda.empty_cache()
    selected, selected_state=Trainer.load(best_path,device='cuda')
    if selected_state['run_sha256'] != run_hash or selected_state['selected_epoch'] != state['best_epoch']:
        raise ValueError('Best checkpoint provenance/selection mismatch')
    selected_epoch=state['best_epoch']
    dev_saved=json.loads((args.output/f'development-epoch-{selected_epoch}.json').read_text())
    dev_reloaded=detailed(selected.model,dev_rows,args.mode)
    if dev_reloaded != dev_saved:
        raise ValueError('Selected checkpoint reload changed development outputs')
    # Final-test rows are loaded and scored only after all epochs and selection.
    test_rows=[bank.row(i) for i in bank.ids('test')]
    if not test_rows:
        raise ValueError('No final-test examples')
    test=detailed(selected.model,test_rows,args.mode)
    save(args.output/'test.json',test)
    timing_rows=[test_rows[i*(len(test_rows)-1)//63] for i in range(64)] if len(test_rows)>64 else test_rows
    timed=detailed(selected.model,timing_rows,args.mode,timing=True)
    save(args.output/'test-timing.json',timed)
    for row in timing_rows:
        if selected.model.propose(row['prefix'],row['draft'],args.mode) != selected.model.propose(
                row['prefix'],row['draft'],args.mode,cached=False):
            raise ValueError('Cached/full editor outputs disagree on final-test timing sample')
    result=dict(mode=args.mode,epochs=3,updates=final_step,selected_epoch=selected_epoch,
        parameters=40750336,bank_sha256=bank_info['bank_sha256'],best_checkpoint_sha256=sha(best_path),
        latest_checkpoint_sha256=sha(last),selected_weights_sha256=weights_hash(selected.model),
        checkpoint_reload='passed',cached_full_check=f'passed on {len(timing_rows)} fixed test blocks',
        development=aggregate(dev_reloaded,dev_rows),test=aggregate(test,test_rows),
        isolated_latency_ms=timed['median_editor_milliseconds'],timing_blocks=len(timing_rows),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),session_seconds=time.monotonic()-started,
        scope='Saved-context coverage and isolated FP32 latency; no full-generation speedup measurement')
    save(args.output/'complete.json',result)
    save(args.output/'progress.json',dict(phase='complete',mode=args.mode,updates=final_step,
        selected_epoch=selected_epoch,test_original=test['mean_original_progress'],test_edited=test['mean_edited_progress']))
    print(json.dumps(result,indent=2),flush=True)
    bank.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('context','draft'),required=True)
    args=p.parse_args()
    train(args)


if __name__=='__main__':
    main()
