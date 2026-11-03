import copy
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import dflash.model as decoding
from dflash.secondary import load_secondary_model, redraft_suffix
from dflash.validation import assert_same_tokens, check_greedy_output


@pytest.fixture
def models():
    torch.manual_seed(17)
    torch.set_num_threads(1)
    common = dict(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=128, bos_token_id=0, eos_token_id=30,
        pad_token_id=0, attention_dropout=0.0, tie_word_embeddings=False,
    )
    target = Qwen3ForCausalLM(Qwen3Config(num_hidden_layers=2, **common)).eval()
    config = Qwen3Config(num_hidden_layers=1, **common)
    config.num_target_layers = 2
    config.target_layer_ids = [0]
    config.mask_token_id = 31
    config.block_size = 4
    config.is_causal = False
    draft = decoding.DFlashDraftModel(config).eval()
    secondary = Qwen3ForCausalLM(Qwen3Config(num_hidden_layers=1, **common)).eval()
    return target, draft, secondary


class RecordingSecondary:
    """Predict token+1 and require a cache to carry its own speculative history."""

    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, past_key_values, **kwargs):
        self.calls.append((input_ids.clone(), past_key_values, kwargs))
        logits = torch.full((1, 1, 32), -100.0)
        logits[0, 0, (input_ids[0, -1] + 1) % 32] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=object())


def test_suffix_uses_exact_seed_then_its_own_tokens_and_fresh_cache():
    model = RecordingSecondary()
    accepted = torch.tensor([[4, 5]])
    block = torch.tensor([[6, 7, 29, 29, 29]])
    result = redraft_suffix(model, accepted, block)
    assert result.tolist() == [[6, 7, 8, 9, 10]]
    assert block.tolist() == [[6, 7, 29, 29, 29]]
    assert accepted.tolist() == [[4, 5]]
    assert [call[0].tolist() for call in model.calls] == [[[4, 5, 6, 7]], [[8]], [[9]]]
    assert model.calls[0][1] is None
    assert all(call[1] is not None for call in model.calls[1:])
    assert all(call[2]["output_hidden_states"] is False for call in model.calls)
    redraft_suffix(model, accepted, block)
    assert model.calls[3][1] is None


@pytest.mark.parametrize("length", [1, 2])
def test_short_block_does_not_call_secondary(length):
    model = Mock(side_effect=AssertionError("No suffix to replace"))
    block = torch.tensor([[6, 7]])[:, :length]
    assert torch.equal(redraft_suffix(model, torch.tensor([[4, 5]]), block), block)


def test_secondary_eos_does_not_shorten_candidate():
    model = RecordingSecondary()
    # Even an EOS-like ID (30) is just a proposal until target verification.
    assert redraft_suffix(model, torch.tensor([[4]]), torch.tensor([[28, 29, 1, 1]])).tolist() == [[28, 29, 30, 31]]
    assert len(model.calls) == 2


@pytest.mark.parametrize("budget", [1, 2, 3, 4, 5, 11])
@pytest.mark.parametrize("block_size", [3, 4, 8])
def test_real_tiny_models_match_independent_greedy(models, budget, block_size):
    target, draft, secondary = models
    prompt = torch.tensor([[1, 2, 3]])
    ordinary = decoding.dflash_generate(draft, target, prompt, budget, None, block_size=block_size)
    refined = decoding.dflash_generate(
        draft, target, prompt, budget, None, block_size=block_size, secondary_model=secondary,
    )
    assert refined.shape[1] == prompt.shape[1] + budget
    assert torch.equal(ordinary, refined)
    check_greedy_output(target, prompt, refined, budget, None)


def test_verifier_receives_replaced_suffix_with_preserved_seed(models, monkeypatch):
    target, draft, secondary = models
    prompt = torch.tensor([[1, 2, 3]])
    blocks = []
    verified = []

    def record_redraft(model, accepted, block):
        replacement = redraft_suffix(model, accepted, block)
        blocks.append((accepted.clone(), block.clone(), replacement.clone()))
        return replacement

    def record_target(module, args, kwargs):
        verified.append(args[0].clone())

    monkeypatch.setattr(decoding, "redraft_suffix", record_redraft)
    handle = target.register_forward_pre_hook(record_target, with_kwargs=True)
    output = decoding.dflash_generate(draft, target, prompt, 9, None, secondary_model=secondary)
    handle.remove()
    assert blocks
    assert any(not torch.equal(old[:, 2:], new[:, 2:]) for _, old, new in blocks)
    for accepted, old, new in blocks:
        assert torch.equal(old[:, :2], new[:, :2])
        assert torch.equal(accepted, output[:, :accepted.shape[1]])
        assert any(torch.equal(new, seen) for seen in verified[1:])
    check_greedy_output(target, prompt, output, 9, None)


def test_eos_and_rejection_match_target(models):
    target, draft, secondary = models
    prompt = torch.tensor([[1, 2, 3]])
    full = decoding.dflash_generate(draft, target, prompt, 12, None, block_size=1)
    tokens = full[0, prompt.shape[1]:].tolist()
    assert len(set(tokens)) > 1  # Exercise a later EOS, not just a prefill stop.
    for stop in dict.fromkeys(tokens):
        output = decoding.dflash_generate(
            draft, target, prompt, 12, [stop], secondary_model=secondary,
        )
        assert output.shape[1] == prompt.shape[1] + tokens.index(stop) + 1
        check_greedy_output(target, prompt, output, 12, [stop])


@pytest.mark.parametrize("reject_first,stop_early", [(False, False), (True, False), (False, True)])
def test_controlled_acceptance_rejection_and_mid_block_eos(models, monkeypatch, reject_first, stop_early):
    target, draft, _ = models
    secondary = copy.deepcopy(target)  # Perfect suffix predictor with independent caches.
    prompt = torch.tensor([[1, 2, 3]])
    oracle = prompt.clone()
    with torch.inference_mode():
        for _ in range(11):
            logits = target(oracle, use_cache=False).logits
            oracle = torch.cat((oracle, logits[:, -1:].argmax(dim=-1)), dim=1)
    positions = None

    def record_positions(module, args, kwargs):
        nonlocal positions
        width = kwargs["noise_embedding"].shape[1]
        positions = kwargs["position_ids"][:, -width + 1:]

    def controlled_proposals(hidden, head):
        proposals = oracle[:, positions[0]].clone()
        # Keep x1 correct unless testing first-proposal rejection; corrupt the rest.
        proposals[:, 0 if reject_first else 1:] = (proposals[:, 0 if reject_first else 1:] + 1) % 32
        return torch.nn.functional.one_hot(proposals, 32).float() * 100

    handle = draft.register_forward_pre_hook(record_positions, with_kwargs=True)
    monkeypatch.setattr(draft, "compute_logits", controlled_proposals)
    monkeypatch.setattr(decoding, "_cuda_time", time.perf_counter)
    stops = [int(oracle[0, prompt.shape[1] + 2])] if stop_early else None
    result = decoding.dflash_generate(
        draft, target, prompt, 11, stops, secondary_model=secondary, return_stats=True,
    )
    handle.remove()
    expected = oracle[:, :prompt.shape[1] + 3] if stop_early else oracle
    assert torch.equal(result.output_ids, expected)
    if stop_early:
        assert result.acceptance_lengths == [2]
    elif reject_first:
        assert result.acceptance_lengths == [1] * 10
    else:
        assert result.acceptance_lengths == [4, 4, 2]


def test_stochastic_secondary_fails_before_model_work(models):
    target, draft, secondary = models
    with pytest.raises(ValueError, match="greedy"):
        decoding.dflash_generate(draft, target, torch.tensor([[1]]), 8, None,
                                temperature=0.7, secondary_model=secondary)


def test_secondary_requires_a_replaceable_block(models):
    target, draft, secondary = models
    with pytest.raises(ValueError, match="block_size"):
        decoding.dflash_generate(draft, target, torch.tensor([[1]]), 8, None,
                                block_size=2, secondary_model=secondary)


@pytest.mark.parametrize("vocab,special", [({"a": 1, "b": 0}, [0]), ({"a": 0, "b": 1}, [1])])
def test_loader_rejects_incompatible_token_ids_before_weights(monkeypatch, vocab, special):
    target_tokenizer = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1}, all_special_ids=[0])
    tokenizer = SimpleNamespace(get_vocab=lambda: vocab, all_special_ids=special)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda _: tokenizer)
    weights = Mock(side_effect=AssertionError("Must reject before loading weights"))
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", weights)
    with pytest.raises(ValueError, match="identical token IDs"):
        load_secondary_model("unused", Mock(), target_tokenizer)
    weights.assert_not_called()


def test_loader_preserves_pretrained_weights_and_freezes_model(models, monkeypatch):
    target, _, secondary = models
    tokenizer = SimpleNamespace(get_vocab=lambda: {str(i): i for i in range(32)}, all_special_ids=[30])
    before = secondary.lm_head.weight.clone()
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda _: tokenizer)
    loader = Mock(return_value=secondary)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", loader)
    loaded = load_secondary_model("pretrained-secondary", target, tokenizer)
    assert loaded is secondary
    assert torch.equal(before, loaded.lm_head.weight)
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    loader.assert_called_once_with("pretrained-secondary", attn_implementation="sdpa", dtype=target.dtype)


def test_loader_rejects_incompatible_model_vocabulary(models, monkeypatch):
    target, _, secondary = models
    secondary.lm_head = torch.nn.Linear(16, 33, bias=False)
    tokenizer = SimpleNamespace(get_vocab=lambda: {"a": 0}, all_special_ids=[])
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda _: tokenizer)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *a, **k: secondary)
    with pytest.raises(ValueError, match="matching vocabulary sizes"):
        load_secondary_model("unused", target, tokenizer)


@pytest.mark.parametrize("actual,expected,position", [([1, 2], [1, 3], 1), ([1], [1, 2], 1)])
def test_output_check_reports_token_or_length_mismatch(actual, expected, position):
    with pytest.raises(RuntimeError, match=f"position {position}"):
        assert_same_tokens(torch.tensor([actual]), torch.tensor([expected]), "Test")
