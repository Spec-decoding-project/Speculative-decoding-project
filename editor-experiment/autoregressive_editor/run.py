"""Explicit bounded training/evaluation commands for a validated token bank."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import time

import torch

from .data import Bank
from .learning import Trainer, evaluate
from .model import Config, TokenEditor


def write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def synchronize(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('describe', 'train', 'evaluate'))
    parser.add_argument('--experiment', type=Path, default=Path(__file__).with_name('experiment.json'))
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--seconds', type=int, default=1800)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repairs', type=int, choices=(1, 16, 128), default=1)
    parser.add_argument('--preservation', type=int, default=0)
    args = parser.parse_args()
    spec = json.loads(args.experiment.read_text())
    config = Config(**spec['model'])
    if args.command == 'describe':
        # No checkpoint download or full-sized model allocation.
        with torch.device('meta'):
            model = TokenEditor(config)
        print(json.dumps(dict(experiment=spec, parameters=sum(p.numel() for p in model.parameters())), indent=2))
        return
    if args.bank is None or args.output is None:
        parser.error('--bank and a new --output directory are required')
    if not 1 <= args.updates <= 2000 or min(args.batch_size, args.eval_every, args.seconds, args.threads) < 1:
        parser.error('Initial attempts allow 1–2000 updates and positive batch/time settings')
    if not 0 <= args.preservation <= 16:
        parser.error('Initial preservation subset must contain 0–16 blocks')
    torch.set_num_threads(args.threads)
    bank = Bank(args.bank)
    if bank.manifest['vocab_size'] != config.vocab_size:
        raise ValueError('Experiment vocabulary differs from bank')
    validation = Bank(args.validation) if args.validation else None
    if validation:
        if validation.splits != {'validation'}:
            raise ValueError('Development bank must contain validation prompts only')
        bank.require_disjoint(validation)
    if args.command == 'evaluate':
        if args.checkpoint is None:
            parser.error('Evaluation requires --checkpoint')
        trainer, metadata = Trainer.load(args.checkpoint, device=args.device)
        if trainer.mode != spec['conditioning'] or asdict(trainer.model.config) != spec['model']:
            raise ValueError('Checkpoint differs from experiment configuration')
        if bank.splits == {'train'}:
            if metadata['bank_sha256'] != bank.fingerprint:
                raise ValueError('Training evaluation must use the checkpoint bank')
            evaluated = [bank[i] for i in metadata['selected_indices']]
        elif bank.splits in ({'validation'}, {'test'}):
            if bank.prompts & set(metadata['train_prompts']) or bank.families & set(metadata['train_families']):
                raise ValueError('Held-out evaluation overlaps training')
            if bank.manifest['models'] != metadata['models']:
                raise ValueError('Evaluation model identities differ')
            evaluated = bank
        else:
            raise ValueError('Mixed evaluation splits')
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output / 'evaluation.json', evaluate(trainer.model, evaluated, trainer.mode))
        return
    if bank.splits != {'train'}:
        raise ValueError('Only training-split examples may update weights')
    repairs, preservation = [], []
    for index, row in enumerate(bank):
        if row['draft'][:2] != row['labels'][:2]:
            raise ValueError('Initial fit bank contains an unreachable fixed seed')
        (preservation if row['draft'] == row['labels'] else repairs).append(index)
    if len(repairs) < args.repairs or len(preservation) < args.preservation:
        raise ValueError('Insufficient examples for the requested nested fit subset')
    selected = repairs[:args.repairs] + preservation[:args.preservation]
    training = [bank[i] for i in selected]
    metadata = dict(bank_sha256=bank.fingerprint, train_prompts=sorted(bank.prompts),
                    train_families=sorted(bank.families), models=bank.manifest['models'],
                    bank_origin=bank.manifest['origin'], validation_sha256=validation.fingerprint if validation else None,
                    batch_size=args.batch_size, selected_indices=selected)
    if args.checkpoint:
        trainer, saved = Trainer.load(args.checkpoint, device=args.device)
        if (saved != metadata or trainer.mode != spec['conditioning'] or
                asdict(trainer.model.config) != spec['model'] or trainer.lr != spec['learning_rate'] or trainer.seed != spec['seed']):
            raise ValueError('Resume must preserve data, configuration and batch size')
    else:
        trainer = Trainer(config, spec['conditioning'], seed=spec['seed'], lr=spec['learning_rate'], device=args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    write(args.output / 'run.json', dict(experiment=spec, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                       python=platform.python_version(), torch=str(torch.__version__), git_revision=revision,
                                       device=args.device, parameters=sum(p.numel() for p in trainer.model.parameters()),
                                       metadata=metadata, sampling='Uniform blocks with replacement; same RNG sequence in both variants'))
    checkpoint = args.output / 'checkpoint.pt'
    write(args.output / 'initial-evaluation.json', evaluate(trainer.model, training, trainer.mode))
    started = time.monotonic()
    outcome = 'update budget reached'
    with (args.output / 'updates.jsonl').open('w') as log:
        while trainer.step < args.updates:
            if time.monotonic() - started >= args.seconds:
                outcome = 'wall-time budget reached'
                break
            examples = [training[trainer.rng.randrange(len(training))] for _ in range(args.batch_size)]
            synchronize(args.device)
            start = time.perf_counter()
            row = trainer.update(examples)
            synchronize(args.device)
            row['update_seconds'] = time.perf_counter() - start
            log.write(json.dumps(row) + '\n')
            log.flush()
            if trainer.step % args.eval_every == 0:
                result = dict(train=evaluate(trainer.model, training, trainer.mode))
                if validation:
                    result['validation'] = evaluate(trainer.model, validation, trainer.mode)
                write(args.output / f'evaluation-{trainer.step:06d}.json', result)
                trainer.save(checkpoint, metadata)
                print(json.dumps(dict(step=trainer.step, ce=result['train']['teacher_forced_ce'],
                                      exact_suffix=result['train']['free_running_exact_suffix'])), flush=True)
    trainer.save(checkpoint, metadata)
    write(args.output / 'final-evaluation.json', evaluate(trainer.model, training, trainer.mode))
    write(args.output / 'summary.json', dict(outcome=outcome, updates=trainer.step, elapsed_seconds=time.monotonic() - started,
                                           scope='Training feasibility only; no native acceptance or speedup claim'))


if __name__ == '__main__':
    main()
