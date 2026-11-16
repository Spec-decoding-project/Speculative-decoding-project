"""Explicit first-round cached contract matching pinned DFlash spec_generate.

Each saved prefix starts a fresh native round. This does not assert equality
with the older uncached replay or with cache histories from full generation.
"""
LEGACY = 'uncached-first-round-v1'
NATIVE = 'native-cached-first-round-v1'
CONTRACTS = (LEGACY, NATIVE)


def prefill(target, prefix_ids, contract):
    import torch
    from transformers import DynamicCache
    if contract == LEGACY:
        return target(prefix_ids, use_cache=False, output_hidden_states=True, logits_to_keep=1), None
    if contract != NATIVE:
        raise ValueError('Unknown replay contract')
    cache = DynamicCache()
    output = target(prefix_ids, position_ids=torch.arange(prefix_ids.shape[1], device=prefix_ids.device)[None],
                    past_key_values=cache, use_cache=True, logits_to_keep=1, output_hidden_states=True)
    return output, cache


def continuation(target, prefix_ids, labels, cache, contract):
    import torch
    if contract == LEGACY:
        inputs = torch.cat((prefix_ids, torch.tensor([labels[:-1]], device=prefix_ids.device)), dim=1)
        return target(inputs, use_cache=False, logits_to_keep=16).logits.argmax(-1)[0].tolist()
    if contract != NATIVE or cache is None or cache.get_seq_length() != prefix_ids.shape[1]:
        raise ValueError('Invalid native prefix cache')
    anchor = prefix_ids.shape[1]
    output = target(torch.tensor([labels], device=prefix_ids.device),
                    position_ids=torch.arange(anchor, anchor + 16, device=prefix_ids.device)[None],
                    past_key_values=cache, use_cache=True, output_hidden_states=True)
    # The separately checked prefill predicts the anchor. Native verification
    # row i predicts label i+1; the final bonus row is outside this saved block.
    replay = [labels[0]] + output.logits[0, :15].argmax(-1).tolist()
    return replay


def draft_kwargs(target, features, noise, anchor, contract):
    import torch
    from transformers import DynamicCache
    result = dict(target_hidden=features,
                  noise_embedding=target.model.embed_tokens(torch.tensor([noise], device=features.device)),
                  position_ids=torch.arange(anchor + 16, device=features.device)[None],
                  use_cache=contract == NATIVE, is_causal=False)
    if contract == NATIVE:
        result['past_key_values'] = DynamicCache()
    elif contract != LEGACY:
        raise ValueError('Unknown replay contract')
    return result


def compare_released_round(target, draft, extract, prefix_ids, features, states, original):
    """Regression at the real released call site, using its ordinary proposal."""
    import torch
    observed = []
    normalized = []
    def target_hook(module, inputs, output):
        if not observed:
            observed.append(extract(output.hidden_states, draft.target_layer_ids).detach().clone())
        else:
            observed.append(inputs[0].detach().clone())
    target_handle = target.register_forward_hook(target_hook)
    draft_handle = draft.norm.register_forward_hook(
        lambda module, inputs, output: normalized.append(output.detach().clone()))
    try:
        draft.spec_generate(target, prefix_ids, max_new_tokens=1, stop_token_ids=None, temperature=0)
    finally:
        target_handle.remove()
        draft_handle.remove()
    if len(observed) != 2 or len(normalized) != 1:
        raise ValueError('Unexpected released native first-round call sequence')
    checks = dict(target_features_equal=torch.equal(features, observed[0]),
                  draft_normalized_states_equal=torch.equal(states, normalized[0]),
                  actual_proposal_equal=torch.equal(original, observed[1]))
    if not all(checks.values()):
        raise ValueError(f'Collected round differs from released spec_generate: {checks}')
    return checks


def validate_contract(audit, manifest):
    contract = audit.get('replay_contract', LEGACY)
    if contract not in CONTRACTS or manifest.get('replay_contract', LEGACY) != contract:
        raise ValueError('Unknown or inconsistent replay contract')
    if contract == NATIVE:
        expected = {('gsm8k:train:3821', 38), ('ultrachat_200k:train_sft:164610', 19),
                    ('ultrachat_200k:train_sft:61759', 22)}
        regressions = audit.get('native_regressions', [])
        if {(x['id'], x['anchor']) for x in regressions} != expected:
            raise ValueError('Native fixed-fixture regressions incomplete')
        keys = ('target_features_equal', 'draft_normalized_states_equal', 'actual_proposal_equal',
                'saved_labels_match_native_verification')
        if any(any(x.get(k) is not True for k in keys) for x in regressions):
            raise ValueError('Native fixed-fixture regression failed')
    return contract
