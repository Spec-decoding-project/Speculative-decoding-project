"""Untrained cost prototype: post-normalization residual, no cloud side effects.

Low-rank refinement is prior art. This module tests implementation feasibility,
not novelty, teacher compression, or inference speedup.
"""
import torch
from torch import nn

class CompactSecondPass(nn.Module):
    def __init__(self, hidden_size, rank=256, block_size=16, heads=4):
        super().__init__()
        if rank % heads or block_size < 3: raise ValueError('Invalid shape')
        self.hidden_size=hidden_size;self.block_size=block_size
        self.state_in=nn.Linear(hidden_size,rank,bias=False)
        self.token_in=nn.Linear(hidden_size,rank,bias=False)
        self.position=nn.Parameter(torch.zeros(block_size,rank))
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(rank,heads,2*rank,dropout=0.,batch_first=True,norm_first=True) for _ in range(2)])
        self.up=nn.Parameter(torch.zeros(hidden_size,rank))

    def coefficients(self, normalized_states, first_guess_embedding):
        if normalized_states.shape[1:]!=(self.block_size,self.hidden_size): raise ValueError('States must cover full first-pass block')
        if first_guess_embedding.shape!=(normalized_states.shape[0],self.hidden_size):raise ValueError('Exactly one visible guess embedding required')
        x=self.state_in(normalized_states)+self.token_in(first_guess_embedding)[:,None]+self.position[None]
        for layer in self.layers:x=layer(x)
        return x[:,2:]

    def direct_scores(self, normalized_states, first_guess_embedding, output_weight):
        z=self.coefficients(normalized_states,first_guess_embedding)
        return (normalized_states[:,2:]+z@self.up.T)@output_weight.T

    def folded_weight(self, output_weight):
        # Differentiable for training; detach and cache only for frozen serving.
        return output_weight@self.up

    def folded_scores(self, normalized_states, first_guess_embedding, original_suffix_scores, folded_weight):
        z=self.coefficients(normalized_states,first_guess_embedding)
        if original_suffix_scores.shape[:2]!=z.shape[:2]:raise ValueError('Editable scores must start at original logit index1, corresponding to state index2')
        return original_suffix_scores+z@folded_weight.T
