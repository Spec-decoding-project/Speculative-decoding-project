"""Token-only autoregressive suffix drafting; no target hidden states are used."""

import torch
from torch import nn


def load_secondary_model(model_id: str, target: nn.Module, target_tokenizer):
    """Load pretrained weights with the same token-ID meaning as the target."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if (
        tokenizer.get_vocab() != target_tokenizer.get_vocab()
        or set(tokenizer.all_special_ids) != set(target_tokenizer.all_special_ids)
    ):
        raise ValueError("The secondary and target tokenizers must have identical token IDs")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="sdpa", dtype=target.dtype,
    )
    for name in ("get_input_embeddings", "get_output_embeddings"):
        if getattr(model, name)().weight.shape[0] != getattr(target, name)().weight.shape[0]:
            raise ValueError("The secondary and target models must have matching vocabulary sizes")
    return model.to(target.device).eval().requires_grad_(False)


@torch.inference_mode()
def redraft_suffix(
    model: nn.Module,
    accepted_ids: torch.LongTensor,
    block_ids: torch.LongTensor,
) -> torch.LongTensor:
    """Keep [anchor, x1], replace x2 onward using accepted context + [anchor, x1].

    Each block gets a fresh secondary cache. Its speculative state never enters
    either DFlash's or the target's cache. Only the target decides when to stop;
    secondary EOS proposals do not shorten the candidate block.
    """
    if accepted_ids.ndim != 2 or accepted_ids.shape[0] != 1:
        raise ValueError("Secondary drafting requires one accepted token sequence")
    if block_ids.ndim != 2 or block_ids.shape[0] != 1 or block_ids.shape[1] < 1:
        raise ValueError("Secondary drafting requires one nonempty candidate block")
    candidate = block_ids.clone()
    if candidate.shape[1] <= 2:
        return candidate

    # accepted_ids ends BEFORE the anchor, so neither seed token is duplicated.
    next_input = torch.cat((accepted_ids, candidate[:, :2]), dim=1)
    cache = None
    for column in range(2, candidate.shape[1]):
        output = model(
            input_ids=next_input,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=False,
        )
        next_input = output.logits[:, -1:].argmax(dim=-1)
        candidate[:, column:column + 1] = next_input
        cache = output.past_key_values
    return candidate
