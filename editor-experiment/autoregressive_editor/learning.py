"""Readable training mechanics and free-running evaluation; no allocation on import."""
from dataclasses import asdict
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from .data import MODES, teacher_forcing
from .model import Config, TokenEditor


def tensors(model, example, mode):
    ids, segments, start, labels = teacher_forcing(example, mode)
    device = model.token.weight.device
    return (torch.tensor([ids], dtype=torch.long, device=device),
            torch.tensor([segments], dtype=torch.long, device=device), start,
            torch.tensor(labels, dtype=torch.long, device=device))


def progress(tokens, labels):
    count = 0
    for actual, expected in zip(tokens[1:], labels[1:]):
        if actual != expected:
            break
        count += 1
    return count


@torch.no_grad()
def evaluate(model, examples, mode):
    was_training = model.training
    model.eval()
    rows, loss_sum, token_count, correct = [], 0.0, 0, 0
    try:
        for example in examples:
            ids, segments, start, labels = tensors(model, example, mode)
            logits, _ = model(ids, segments, logit_start=start)
            loss_sum += float(F.cross_entropy(logits[0].float(), labels, reduction='sum'))
            edited = model.propose(example['prefix'], example['draft'], mode)
            expected, original = example['labels'], example['draft']
            a, b = progress(original, expected), progress(edited, expected)
            matches = sum(x == y for x, y in zip(edited[2:], expected[2:]))
            correct += matches
            token_count += len(expected) - 2
            rows.append(dict(prompt_id=example['prompt_id'], anchor=example['anchor'],
                             original_progress=a, edited_progress=b, union_progress=max(a, b),
                             repaired=b > a, damaged=b < a, wrong_first=original[1] != expected[1],
                             exact_suffix=edited[2:] == expected[2:]))
    finally:
        model.train(was_training)
    if not rows:
        raise ValueError('Cannot evaluate an empty bank')
    return dict(teacher_forced_ce=loss_sum / token_count,
                free_running_token_accuracy=correct / token_count,
                free_running_exact_suffix=sum(r['exact_suffix'] for r in rows) / len(rows),
                repaired=sum(r['repaired'] for r in rows), damaged=sum(r['damaged'] for r in rows),
                wrong_first=sum(r['wrong_first'] for r in rows), blocks=len(rows), rows=rows,
                scope='Saved-context coverage; not native target acceptance or generation speedup')


class Trainer:
    def __init__(self, config, mode, *, seed=0, lr=1e-3, device='cpu'):
        if mode not in MODES or lr <= 0:
            raise ValueError('Invalid conditioning mode or learning rate')
        torch.manual_seed(seed)
        self.model = TokenEditor(config).to(device)
        self.mode, self.seed, self.lr = mode, seed, lr
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.0)
        self.rng = random.Random(seed)
        self.step = 0

    def update(self, examples):
        examples = list(examples)
        if not examples or any(e['draft'][:2] != e['labels'][:2] for e in examples):
            raise ValueError('Initial fit requires nonempty examples with a reachable fixed seed')
        total = sum(len(e['labels']) - 2 for e in examples)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        # Microbatch one, normalized across all suffix tokens in the effective batch.
        for example in examples:
            ids, segments, start, labels = tensors(self.model, example, self.mode)
            logits, _ = self.model(ids, segments, logit_start=start)
            loss = F.cross_entropy(logits[0].float(), labels, reduction='sum') / total
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()  # Compute gradients. This does not change weights.
            loss_sum += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()  # AdamW changes only the editor-owned weights.
        self.step += 1
        return dict(step=self.step, ce=loss_sum, gradient_norm=float(norm), suffix_tokens=total)

    def save(self, path, metadata):
        payload = dict(schema=1, config=asdict(self.model.config), mode=self.mode,
                       seed=self.seed, lr=self.lr, step=self.step, model=self.model.state_dict(),
                       optimizer=self.optimizer.state_dict(), sampling_rng=self.rng.getstate(),
                       torch_rng=torch.get_rng_state(), metadata=metadata)
        path = Path(path)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, *, device='cpu'):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['schema'] != 1:
            raise ValueError('Unsupported checkpoint schema')
        trainer = cls(Config(**payload['config']), payload['mode'], seed=payload['seed'],
                      lr=payload['lr'], device=device)
        trainer.model.load_state_dict(payload['model'], strict=True)
        trainer.optimizer.load_state_dict(payload['optimizer'])
        trainer.rng.setstate(payload['sampling_rng'])
        torch.set_rng_state(payload['torch_rng'])
        trainer.step = payload['step']
        return trainer, payload['metadata']
