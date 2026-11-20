"""Matched optimization, actual-token evaluation and atomic resumable checkpoints."""
from collections import defaultdict
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import statistics

import torch

from autoregressive_editor.data import teacher_forcing
from autoregressive_editor.learning import progress
from autoregressive_editor.model import Config, TokenEditor
from copy_action_fit.model import CopyEditor, losses


def weights_hash(model, *, shared=False):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if shared and name.startswith('copy_head.'):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def inputs(model, row):
    ids, segments, start, labels = teacher_forcing(row, 'draft')
    device = model.token.weight.device
    return (torch.tensor([ids], dtype=torch.long, device=device),
            torch.tensor([segments], dtype=torch.long, device=device), start,
            torch.tensor(labels, dtype=torch.long, device=device),
            torch.tensor(row['draft'][2:], dtype=torch.long, device=device))


class PilotTrainer:
    def __init__(self, config, variant, *, device='cpu', lr=.0001, seed=0):
        if variant not in ('baseline', 'copy'):
            raise ValueError('Unknown pilot variant')
        torch.manual_seed(seed)
        self.variant, self.seed, self.lr = variant, seed, lr
        self.model = (CopyEditor(config) if variant == 'copy' else TokenEditor(config)).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=.01)
        self.step = 0

    def warm_start(self, path):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['mode'] != 'draft' or payload['config'] != asdict(self.model.config):
            raise ValueError('Warm-start architecture/conditioning mismatch')
        missing, unexpected = self.model.load_state_dict(payload['model'], strict=False)
        expected = ['copy_head.weight', 'copy_head.bias'] if self.variant == 'copy' else []
        if missing != expected or unexpected:
            raise ValueError('Warm start changed shared state')
        if self.optimizer.state or self.step:
            raise ValueError('Warm start must use a fresh optimizer')
        return payload['step']

    def update(self, examples):
        if not examples or any(r['draft'][:2] != r['labels'][:2] for r in examples):
            raise ValueError('Training requires reachable, nonempty native blocks')
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        count = sum(len(r['labels']) - 2 for r in examples)
        action_total = token_total = 0.
        for row in examples:
            ids, segments, start, labels, original = inputs(self.model, row)
            logits, _ = self.model(ids, segments, logit_start=start)
            action, token = losses(logits[0], labels, original, copy_enabled=self.variant == 'copy')
            loss = action / count
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite action loss')
            loss.backward()
            action_total += float(action.detach()) / count
            token_total += float(token.detach()) / count
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        head_norm = None
        if self.variant == 'copy':
            head_norm = float(self.model.copy_head.weight.grad.norm())
        self.optimizer.step()
        self.step += 1
        return dict(step=self.step, action_ce=action_total, token_ce=token_total,
                    gradient_norm=float(norm), copy_head_gradient_norm=head_norm,
                    suffix_tokens=count)

    def save(self, path, metadata):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + '.tmp')
        payload = dict(schema='copy-action-pilot-v1', config=asdict(self.model.config),
            variant=self.variant, seed=self.seed, lr=self.lr, step=self.step,
            model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
            torch_rng=torch.get_rng_state(), metadata=metadata)
        if self.model.token.weight.is_cuda:
            payload['cuda_rng'] = torch.cuda.get_rng_state_all()
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, *, device='cpu'):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['schema'] != 'copy-action-pilot-v1':
            raise ValueError('Unsupported pilot checkpoint')
        trainer = cls(Config(**payload['config']), payload['variant'], device=device,
                      lr=payload['lr'], seed=payload['seed'])
        trainer.model.load_state_dict(payload['model'], strict=True)
        trainer.optimizer.load_state_dict(payload['optimizer'])
        trainer.step = payload['step']
        torch.set_rng_state(payload['torch_rng'])
        if trainer.model.token.weight.is_cuda and 'cuda_rng' in payload:
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
        return trainer, payload['metadata']


@torch.no_grad()
def evaluate(model, examples, variant):
    if not examples:
        raise ValueError('Cannot evaluate an empty bank')
    rows, action_sum, token_sum, matches, copied = [], 0., 0., 0, 0
    was_training = model.training
    model.eval()
    try:
        for row in examples:
            ids, segments, start, labels, original = inputs(model, row)
            logits, _ = model(ids, segments, logit_start=start)
            action, token = losses(logits[0], labels, original, copy_enabled=variant == 'copy')
            action_sum += float(action); token_sum += float(token)
            if variant == 'copy':
                edited, actions = model.propose_actions(row['prefix'], row['draft'])
            else:
                edited = model.propose(row['prefix'], row['draft'], 'draft')
                actions = edited[2:]
            a, b = progress(row['draft'], row['labels']), progress(edited, row['labels'])
            matches += sum(x == y for x, y in zip(edited[2:], row['labels'][2:]))
            copied += sum(x == model.config.vocab_size for x in actions)
            rows.append(dict(prompt_id=row['prompt_id'], anchor=row['anchor'], kind=row['kind'],
                original_tokens=row['draft'], edited_tokens=edited, label_tokens=row['labels'],
                actions=actions, original_progress=a, edited_progress=b, union_progress=max(a,b),
                repaired=b > a, damaged=b < a, wrong_first=row['draft'][1] != row['labels'][1],
                exact_suffix=edited[2:] == row['labels'][2:]))
    finally:
        model.train(was_training)
    count = sum(len(row['labels']) - 2 for row in examples)
    if not math.isfinite(action_sum + token_sum):
        raise ValueError('Nonfinite evaluation loss')
    return dict(teacher_forced_action_ce=action_sum/count, teacher_forced_ce=token_sum/count,
        free_running_token_accuracy=matches/count,
        free_running_exact_suffix=sum(r['exact_suffix'] for r in rows)/len(rows),
        repaired=sum(r['repaired'] for r in rows), damaged=sum(r['damaged'] for r in rows),
        wrong_first=sum(r['wrong_first'] for r in rows), blocks=len(rows), rows=rows,
        mean_original_progress=statistics.mean(r['original_progress'] for r in rows),
        mean_edited_progress=statistics.mean(r['edited_progress'] for r in rows),
        mean_union_progress=statistics.mean(r['union_progress'] for r in rows),
        copy_actions=copied, editable_tokens=count,
        scope='Saved-context coverage; no native acceptance or full-generation speedup')


def aggregate(result, examples):
    output = {k:v for k,v in result.items() if k != 'rows'}
    groups = defaultdict(list)
    for row, metric in zip(examples, result['rows']):
        groups['source/' + row['source']].append(metric)
        groups['kind/' + row['kind']].append(metric)
        groups['first/' + ('wrong' if metric['wrong_first'] else 'correct')].append(metric)
    output['groups'] = {name:dict(blocks=len(items),
        mean_original_progress=statistics.mean(r['original_progress'] for r in items),
        mean_edited_progress=statistics.mean(r['edited_progress'] for r in items),
        repaired=sum(r['repaired'] for r in items), damaged=sum(r['damaged'] for r in items),
        prefix_positions_gained=sum(max(0, r['edited_progress']-r['original_progress']) for r in items),
        prefix_positions_lost=sum(max(0, r['original_progress']-r['edited_progress']) for r in items),
        exact_suffixes=sum(r['exact_suffix'] for r in items)) for name,items in groups.items()}
    return output


def passed(result):
    return result['free_running_exact_suffix'] == 1. and result['teacher_forced_ce'] < .05
