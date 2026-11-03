from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import dflash.benchmark as benchmark
import dflash.model as decoding
from dflash.cli import main


@pytest.fixture
def runner(monkeypatch):
    cpu = torch.device("cpu")
    # Only the CLI's CUDA device selection is replaced; generators are stubbed.
    monkeypatch.setattr(torch, "device", lambda _: cpu)
    target = SimpleNamespace(device=cpu, generation_config=SimpleNamespace(eos_token_id=30))
    draft = SimpleNamespace(block_size=4)
    tokenizer = SimpleNamespace(
        chat_template="enable_thinking", eos_token_id=30,
        apply_chat_template=lambda *a, **k: "formatted prompt",
        encode=lambda *a, **k: torch.tensor([[1, 2, 3]]),
        decode=lambda *a, **k: "decoded output",
    )
    monkeypatch.setattr(benchmark, "load_transformers_models", lambda *a: (target, draft, tokenizer))
    secondary = object()
    loader = Mock(return_value=secondary)
    monkeypatch.setattr("dflash.secondary.load_secondary_model", loader)
    calls = []

    def generate(model, target, input_ids, max_new_tokens, stop_token_ids, *args, **kwargs):
        calls.append(kwargs)
        output_ids = torch.cat((input_ids, torch.tensor([[4, 5]])), dim=1)
        if not kwargs.get("return_stats"):
            return output_ids
        refined = kwargs.get("secondary_model") is secondary
        bs = kwargs["block_size"]
        return SimpleNamespace(
            output_ids=output_ids, num_input_tokens=input_ids.shape[1],
            time_per_output_token=0.03 if refined else (0.04 if bs == 1 else 0.02),
            acceptance_lengths=[2] if bs > 1 else [1, 1],
        )

    monkeypatch.setattr(decoding, "dflash_generate", generate)
    return SimpleNamespace(calls=calls, secondary=secondary, loader=loader, generate=generate)


def test_generate_wires_secondary_and_independent_check(runner, monkeypatch, capsys):
    checker = Mock()
    monkeypatch.setattr("dflash.validation.check_greedy_output", checker)
    main(["generate", "transformers", "--model", "target", "--draft", "draft",
          "--secondary-model", "secondary", "--check-greedy", "hello"])
    assert runner.calls[0]["secondary_model"] is runner.secondary
    runner.loader.assert_called_once()
    checker.assert_called_once()
    output = capsys.readouterr()
    assert "decoded output" in output.out
    assert "Greedy check passed" in output.err


@pytest.mark.parametrize("enabled", [False, True])
def test_benchmark_runs_and_reports_requested_paths(runner, monkeypatch, capsys, enabled):
    monkeypatch.setattr(benchmark, "load_and_process_dataset", lambda _: [{"turns": ["one", "two"]}])
    argv = ["benchmark", "transformers", "--model", "target", "--draft", "draft", "--dataset", "gsm8k"]
    if enabled:
        argv.extend(["--secondary-model", "secondary"])
    main(argv)
    warmups = [call for call in runner.calls if not call.get("return_stats")]
    measured = [call for call in runner.calls if call.get("return_stats")]
    assert len(warmups) == (3 if enabled else 2)
    assert len(measured) == (6 if enabled else 4)
    assert sum(call.get("secondary_model") is runner.secondary for call in measured) == (2 if enabled else 0)
    assert runner.loader.call_count == int(enabled)
    output = capsys.readouterr().out
    assert "DFlash throughput:" in output
    assert ("DFlash + secondary throughput:" in output) is enabled
    if enabled:
        assert "Speedup vs ordinary DFlash: 0.67" in output
        assert "Greedy token checks passed for all 2 turns." in output


def test_benchmark_fails_on_refined_output_mismatch(runner, monkeypatch):
    monkeypatch.setattr(benchmark, "load_and_process_dataset", lambda _: [{"turns": ["one"]}])

    def bad_generate(*args, **kwargs):
        result = runner.generate(*args, **kwargs)
        if kwargs.get("return_stats") and kwargs.get("secondary_model") is runner.secondary:
            result.output_ids[0, -1] = 6
        return result

    monkeypatch.setattr(decoding, "dflash_generate", bad_generate)
    with pytest.raises(RuntimeError, match="Secondary drafting: greedy token mismatch"):
        main(["benchmark", "transformers", "--model", "target", "--draft", "draft",
              "--secondary-model", "secondary", "--dataset", "gsm8k"])
