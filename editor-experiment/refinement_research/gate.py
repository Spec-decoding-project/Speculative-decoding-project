"""A frozen autoregressive generator with a small trainable binary decision."""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from autoregressive_editor.data import condition
from autoregressive_editor.model import TokenEditor


def generator_state(model):
    return {k:v for k,v in model.state_dict().items() if not k.startswith('gate_head.')}


def generator_hash(model):
    digest = hashlib.sha256()
    for name, value in generator_state(model).items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class GateEditor(TokenEditor):
    """The head chooses COPY or the frozen vocabulary argmax, at each step."""
    def __init__(self, config, kind):
        super().__init__(config)
        self.requires_grad_(False)
        self.kind = kind
        if kind in ('scores', 'linear'):
            self.gate_head = nn.Linear(8 if kind == 'scores' else config.width+8, 1)
            nn.init.zeros_(self.gate_head.weight); nn.init.zeros_(self.gate_head.bias)
        elif kind == 'mlp':
            self.gate_head = nn.Sequential(nn.Linear(config.width+8, 64), nn.GELU(), nn.Linear(64, 1))
            nn.init.zeros_(self.gate_head[-1].weight); nn.init.zeros_(self.gate_head[-1].bias)
        else:
            raise ValueError('Unknown gate kind')

    def decision(self, features):
        return self.gate_head(features[..., -8:] if self.kind == 'scores' else features).squeeze(-1)

    def readout(self, ids, segments, *, past=None, use_cache=False, logit_start=0):
        if ids.shape[0] != 1:
            raise ValueError('Token prefix execution uses microbatch one')
        if past is None:
            positions = torch.nonzero(segments[0] == 1).flatten()
            if not positions.numel():
                raise ValueError('Whole-draft segment is required')
            first, end = int(positions[0]), int(positions[-1])+1
            original, seed, kv = ids[:, first:end], end+1, None
        else:
            original, seed, kv = past['original'], past['seed'], past['kv']
        offset = 0 if kv is None else kv[0][0].shape[2]
        hidden, updated = self.features(ids, segments, past=kv, use_cache=use_cache,
                                        logit_start=logit_start)
        start = 2+offset+logit_start-seed
        count = hidden.shape[1]
        if start < 2 or start+count > original.shape[1]:
            raise ValueError('Readout does not align to editable proposal positions')
        source = original[:, start:start+count]
        logits = F.linear(hidden, self.token.weight).float()
        best = logits.argmax(-1)
        top = logits.topk(2, dim=-1).values
        original_score = logits.gather(-1, source.unsqueeze(-1)).squeeze(-1)
        normalizer = logits.logsumexp(-1)
        position = torch.arange(start, start+count, device=ids.device)[None].expand_as(best)
        context_length = seed-original.shape[1]-1
        scores = torch.stack((top[...,0]/10, original_score/10,
            (top[...,0]-original_score)/10, (top[...,0]-top[...,1])/10,
            (normalizer-top[...,0])/10, (best == source).float(),
            position.float()/(original.shape[1]-1),
            torch.full_like(original_score, math.log1p(context_length)/7)), dim=-1)
        features = torch.cat((hidden.float(), scores), dim=-1)
        gate = self.decision(features)
        cache = dict(kv=updated, original=original, seed=seed) if use_cache else None
        return dict(logits=logits, normalizer=normalizer, gate=gate, features=features,
                    best=best, source=source, cache=cache)

    def forward(self, ids, segments, *, past=None, use_cache=False, logit_start=0):
        result = self.readout(ids, segments, past=past, use_cache=use_cache, logit_start=logit_start)
        # A proper mixture likelihood for reporting token CE. The trained binary
        # decision is separate: choose COPY at p>=.5, otherwise vocabulary argmax.
        vocabulary = (result['logits']-result['normalizer'].unsqueeze(-1)
                      + F.logsigmoid(-result['gate']).unsqueeze(-1))
        copying = F.logsigmoid(result['gate']).unsqueeze(-1)
        return torch.cat((vocabulary, copying), dim=-1), result['cache']

    @torch.no_grad()
    def propose_actions(self, prefix, draft, conditioning='draft', *, cached=True, record_actions=True):
        if conditioning != 'draft':
            raise ValueError('Frozen-gate models read the complete proposal')
        ids, segments = condition(prefix, draft, conditioning)
        if len(ids)+len(draft)-2 > self.config.max_length:
            raise ValueError('Editor context budget exceeded')
        device = self.token.weight.device
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        types = torch.tensor([segments], dtype=torch.long, device=device)
        source = torch.tensor([draft], dtype=torch.long, device=device)
        metadata = dict(kv=None, original=source, seed=len(ids)-1)
        past, emitted, actions = metadata, [], []
        was_training = self.training
        self.eval()
        try:
            for position in range(2, len(draft)):
                result = self.readout(tokens, types, past=past if cached else metadata,
                                      use_cache=cached, logit_start=tokens.shape[1]-1)
                copying = result['gate'] >= 0
                action = torch.where(copying, self.config.vocab_size, result['best'])
                chosen = torch.where(copying, source[:, position:position+1], result['best'])
                emitted.append(chosen)
                if record_actions:
                    actions.append(action)
                if cached:
                    tokens, types, past = chosen, torch.full_like(chosen, 2), result['cache']
                else:
                    tokens = torch.cat((tokens, chosen), dim=1)
                    types = torch.cat((types, torch.full_like(chosen, 2)), dim=1)
        finally:
            self.train(was_training)
        proposal = list(draft[:2])+torch.cat(emitted, dim=1).flatten().tolist()
        return proposal, torch.cat(actions, dim=1).flatten().tolist() if actions else []

    def propose(self, prefix, draft, conditioning='draft', *, cached=True):
        return self.propose_actions(prefix, draft, conditioning, cached=cached, record_actions=False)[0]


def supervision(original, generated, labels):
    """Only decisive choices on the reference prefix reachable by either action."""
    copy_correct, gen_correct = original == labels, generated == labels
    reachable = (copy_correct | gen_correct).long().cumprod(dim=-1).bool()
    eligible = reachable & (copy_correct ^ gen_correct)
    length = reachable.sum(dim=-1, keepdim=True)
    position = torch.arange(labels.shape[-1], device=labels.device)
    remaining = (length-position).clamp(min=0).float()
    return copy_correct.float(), eligible, remaining*eligible


def simulated_progress(gates, original, generated, labels):
    chosen = torch.where(gates >= 0, original, generated)
    return 1+(chosen == labels).long().cumprod(dim=-1).sum(dim=-1)
