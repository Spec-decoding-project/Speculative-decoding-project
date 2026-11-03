# Test a token-only secondary drafter

This opt-in experiment implements the first approach: a pretrained autoregressive model rewrites the suffix of a DFlash candidate block. The secondary model receives token IDs only. DFlash itself still uses target hidden states as before.

```text
Accepted context P (ends before the anchor)
DFlash candidate:                   [anchor, x1, x2, x3, ...]
Secondary receives P + [anchor, x1]:             -> y2 -> y3 -> ...
Target verifies:                    [anchor, x1, y2, y3, ...]
```

`x1` is still an unverified DFlash proposal. If the target rejects it, the regenerated suffix cannot improve that round. The target remains responsible for accepting tokens, choosing corrections/bonuses, and stopping at EOS. The secondary model never sees future target answers. The insertion runs after either DFlash or DFlash2 proposes a block; the initial checkpoint example below uses original DFlash.

## Weights and model size

| Role | Checkpoint | Initialization |
| --- | --- | --- |
| Target | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) | Load pretrained target weights |
| Parallel drafter | [`z-lab/Qwen3-4B-DFlash-b16`](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16) | Load pretrained DFlash weights |
| Secondary drafter | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | Load all pretrained weights, freeze, use evaluation mode |

The secondary checkpoint has about 0.6 billion parameters, 28 layers, and hidden width 1,024, compared with the target's width of 2,560. Those widths can differ because this approach shares only token IDs. It has its own embeddings, transformer layers, output head, and key/value cache. No weights are copied from the 4B model, no new projection is initialized, and no training is needed for this experiment. The loader uses the target's dtype (BF16 in the existing CLI) and device. All three models must fit on the CUDA device together with their caches and activations.

The loader requires identical token-to-ID mappings and special-token ID sets, plus matching model vocabulary dimensions. Matching vocabulary size alone is insufficient. It uses the target's prompt template and passes the resulting IDs directly to the secondary; it does not apply a second chat template or retokenize the text. The intended first model pair is Qwen3-4B / Qwen3-0.6B; other architectures are not validated by these tests.

## Run one prompt on Linux with CUDA

From your clone on the GPU machine, update to `main` and install this checkout in your active Python environment:

```bash
git pull --ff-only origin main
python -m pip install -e '.[local]'
dflash generate transformers \
  --model Qwen/Qwen3-4B \
  --draft z-lab/Qwen3-4B-DFlash-b16 \
  --secondary-model Qwen/Qwen3-0.6B \
  --block-size 16 --max-new-tokens 128 \
  --temperature 0 --reasoning off --check-greedy \
  "Explain speculative decoding briefly."
```

The first run downloads the checkpoints if they are not cached. `--check-greedy` reruns the target independently, one token at a time with a fresh cache, and compares complete output IDs. It prints `Greedy check passed: exact target-only token match.` to stderr on success; a mismatch raises an error with the first differing position. The check uses raw argmax, matching DFlash's verifier, rather than checkpoint-specific sampling or repetition penalties. It adds a second generation pass, so use the benchmark below for timings.

Omit `--secondary-model` to run ordinary DFlash. This experiment currently requires the Transformers backend, temperature zero, and block size at least three. The final partial block can be shorter; blocks with only an anchor or anchor plus `x1` skip the secondary. MLX and server backends do not implement the new path.

## Compare all three paths

```bash
dflash benchmark transformers \
  --model Qwen/Qwen3-4B \
  --draft z-lab/Qwen3-4B-DFlash-b16 \
  --secondary-model Qwen/Qwen3-0.6B \
  --dataset gsm8k --max-samples 4 \
  --block-size 16 --max-new-tokens 128 \
  --temperature 0 --reasoning off
```

This warms up and measures target-only decoding (block size 1), ordinary DFlash, and DFlash with the secondary model on the same input for each turn. With the secondary option enabled it also asserts both speculative outputs equal the target-only output IDs. It prints throughput, acceptance summaries, speedup relative to the target, and the secondary path's speedup relative to ordinary DFlash. A speedup below 1 means slower. Dataset and checkpoint downloads happen before timing.

All secondary work is inside decode timing. This initial implementation creates a fresh secondary cache per block, prefills the entire accepted context plus the two seed tokens, and then generates the suffix autoregressively. That is deliberately simple to validate and may be slower, especially with long contexts. Reusing a secondary cache safely across accepted/rejected blocks is a future optimization. Better acceptance does not by itself establish a speedup.

The inherited timing summary averages per-turn time per output token and then takes its reciprocal; it is not aggregate request throughput. Initial target prefill is excluded from decode timing, while the output-token denominator includes the first token selected during prefill. Acceptance lengths include a target correction/bonus and may be truncated by EOS or the output budget. See [tour 5](../.tours/05-benchmark-and-modify.tour) for those conventions.

## Read the implementation and run CPU checks

1. [`dflash/cli.py`](../dflash/cli.py): the optional checkpoint flag and greedy/backend validation.
2. [`dflash/secondary.py`](../dflash/secondary.py): pretrained loading, tokenizer compatibility, seed construction, and sequential suffix generation.
3. [`dflash/model.py`](../dflash/model.py): `redraft_suffix` runs immediately before the existing target verification call. Target and DFlash cache cropping stay in the original loop; the secondary cache is separate and discarded after each block.
4. [`dflash/validation.py`](../dflash/validation.py): independent target-only argmax comparison.
5. [`dflash/benchmark.py`](../dflash/benchmark.py): warmup, three-way comparison, and token checks.

```bash
python -m pip install -e '.[test]'
HF_HUB_OFFLINE=1 python -m pytest -q
python scripts/validate_tours.py
```

Tests use tiny randomly initialized Qwen3 models and controlled proposals on CPU; they do not download pretrained weights. They check seed preservation, secondary cache isolation, real model/cache integration, complete target-output agreement, EOS, rejection, short budgets, tokenizer mismatches, and CLI restrictions. Random weights are only test fixtures. These checks do not establish pretrained-model acceptance or GPU performance; run the commands above on your CUDA machine to measure those.
