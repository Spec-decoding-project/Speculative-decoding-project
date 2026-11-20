"""An explicit COPY action; all decoder inputs remain actual vocabulary tokens."""
import torch
from torch import nn
from torch.nn import functional as F

from autoregressive_editor.data import condition
from autoregressive_editor.model import TokenEditor


class CopyEditor(TokenEditor):
    def __init__(self, config):
        super().__init__(config)
        self.copy_head = nn.Linear(config.width, 1)
        nn.init.zeros_(self.copy_head.weight)
        nn.init.zeros_(self.copy_head.bias)

    def forward(self, ids, segments, *, past=None, use_cache=False, logit_start=0):
        hidden, cache = self.features(ids, segments, past=past, use_cache=use_cache,
                                      logit_start=logit_start)
        vocabulary = F.linear(hidden, self.token.weight)
        return torch.cat((vocabulary, self.copy_head(hidden)), dim=-1), cache

    @torch.no_grad()
    def propose_actions(self, prefix, draft, conditioning='draft', *, cached=True,
                        record_actions=True):
        if conditioning != 'draft':
            raise ValueError('The COPY pilot requires whole-draft conditioning')
        ids, segments = condition(prefix, draft, conditioning)
        suffix_length = len(draft) - 2
        if len(ids) + suffix_length > self.config.max_length:
            raise ValueError('Editor context budget exceeded')
        device = self.token.weight.device
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        types = torch.tensor([segments], dtype=torch.long, device=device)
        source = torch.tensor([draft], dtype=torch.long, device=device)
        proposal, actions = list(draft[:2]), []
        was_training = self.training
        self.eval()
        past = None
        try:
            for position in range(2, len(draft)):
                logits, past = self(tokens, types, past=past, use_cache=cached,
                                    logit_start=tokens.shape[1] - 1)
                action = logits[:, -1].argmax(-1, keepdim=True)
                # COPY is an output action, never an embedding index. Future
                # cached/full steps receive exactly the token we just emitted.
                chosen = torch.where(action == self.config.vocab_size,
                                     source[:, position:position + 1], action)
                proposal.append(int(chosen.item()))
                if record_actions:
                    actions.append(action)
                if cached:
                    tokens, types = chosen, torch.full_like(chosen, 2)
                else:
                    tokens = torch.cat((tokens, chosen), dim=1)
                    types = torch.cat((types, torch.full_like(chosen, 2)), dim=1)
        finally:
            self.train(was_training)
        return proposal, torch.cat(actions, dim=1).flatten().tolist() if actions else []

    def propose(self, prefix, draft, conditioning='draft', *, cached=True):
        return self.propose_actions(prefix, draft, conditioning, cached=cached,
                                    record_actions=False)[0]


def losses(logits, labels, original, *, copy_enabled):
    """Return supervised action NLL and induced emitted-token NLL, both sums."""
    if logits.ndim != 2 or labels.ndim != 1 or original.shape != labels.shape:
        raise ValueError('Require suffix logits [L,A] and aligned token vectors [L]')
    if logits.shape[0] != labels.numel():
        raise ValueError('Logits and editable suffix lengths differ')
    if not copy_enabled:
        token_loss = F.cross_entropy(logits.float(), labels, reduction='sum')
        return token_loss, token_loss
    copy_action = logits.shape[-1] - 1
    targets = torch.where(original == labels, copy_action, labels)
    action_loss = F.cross_entropy(logits.float(), targets, reduction='sum')
    log_probabilities = logits.float().log_softmax(-1)
    token_log_probability = log_probabilities.gather(-1, labels[:, None]).squeeze(-1)
    # Either action can emit the same correct token. Their probability masses
    # add; evaluating COPY alone would not be comparable to baseline token CE.
    token_log_probability = torch.where(original == labels,
        torch.logaddexp(token_log_probability, log_probabilities[:, copy_action]),
        token_log_probability)
    return action_loss, -token_log_probability.sum()
