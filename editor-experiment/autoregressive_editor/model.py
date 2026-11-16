"""Small causal decoder with editor-owned embeddings and a per-round KV cache."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    vocab_size: int = 151936
    width: int = 256
    layers: int = 2
    heads: int = 4
    max_length: int = 1088

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError('Model dimensions must be positive integers')
        if self.width % self.heads or self.vocab_size < 2:
            raise ValueError('Invalid attention dimensions or vocabulary')


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.norm1 = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)
        self.norm2 = nn.LayerNorm(config.width)
        self.ff = nn.Sequential(nn.Linear(config.width, 4 * config.width, bias=False),
                                nn.GELU(), nn.Linear(4 * config.width, config.width, bias=False))

    def forward(self, x, past=None, use_cache=False):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = [t.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
                   for t in (q, k, v)]
        offset = 0 if past is None else past[0].shape[2]
        if past is not None:
            k, v = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)
        # A cached query starts at offset, not position zero of a square mask.
        query_positions = offset + torch.arange(length, device=x.device)
        key_positions = torch.arange(k.shape[2], device=x.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
        x = x + self.out(h.transpose(1, 2).contiguous().view(batch, length, width))
        x = x + self.ff(self.norm2(x))
        return x, (k, v) if use_cache else None


class TokenEditor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.max_length, config.width)
        self.segment = nn.Embedding(3, config.width)  # context, draft, correction
        self.layers = nn.ModuleList([Layer(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, ids, segments, *, past=None, use_cache=False, logit_start=0):
        if ids.ndim != 2 or ids.shape != segments.shape or ids.shape[1] == 0:
            raise ValueError('Token and segment tensors must have equal nonempty [B,L] shape')
        if ids.dtype != torch.long or segments.dtype != torch.long:
            raise ValueError('Tokens and segments must be int64')
        if past is not None and len(past) != len(self.layers):
            raise ValueError('Cache must contain every decoder layer')
        offset = 0 if past is None else past[0][0].shape[2]
        if past is not None and any(k.shape[2] != offset or v.shape != k.shape for k, v in past):
            raise ValueError('Inconsistent cache lengths')
        if offset + ids.shape[1] > self.config.max_length:
            raise ValueError('Editor context budget exceeded; no silent truncation')
        if not 0 <= logit_start < ids.shape[1]:
            raise ValueError('Readout must retain at least one position')
        positions = torch.arange(offset, offset + ids.shape[1], device=ids.device)
        x = self.token(ids) + self.position(positions)[None] + self.segment(segments)
        cache = []
        for i, layer in enumerate(self.layers):
            x, kv = layer(x, None if past is None else past[i], use_cache)
            if use_cache:
                cache.append(kv)
        # Read out only needed suffix rows. Input and output embeddings are tied.
        logits = F.linear(self.norm(x[:, logit_start:]), self.token.weight)
        return logits, cache if use_cache else None

    @torch.no_grad()
    def propose(self, prefix, draft, conditioning, *, cached=True):
        from .data import condition
        ids, segments = condition(prefix, draft, conditioning)
        suffix_length = len(draft) - 2
        if len(ids) + suffix_length > self.config.max_length:
            raise ValueError('Editor context budget exceeded')
        device = self.token.weight.device
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        types = torch.tensor([segments], dtype=torch.long, device=device)
        proposal = list(draft[:2])
        was_training = self.training
        self.eval()
        past = None
        try:
            for _ in range(suffix_length):
                logits, past = self(tokens, types, past=past, use_cache=cached,
                                    logit_start=tokens.shape[1] - 1)
                chosen = logits[:, -1].argmax(-1, keepdim=True)
                proposal.append(int(chosen.item()))
                if cached:
                    tokens, types = chosen, torch.full_like(chosen, 2)
                else:
                    tokens = torch.cat((tokens, chosen), dim=1)
                    types = torch.cat((types, torch.full_like(chosen, 2)), dim=1)
        finally:
            self.train(was_training)
        return proposal
