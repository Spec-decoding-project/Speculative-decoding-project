# Two token-only autoregressive experiments

These are new models trained from scratch, not the historical hidden-state residual editor or the frozen Qwen3-0.6B baseline. Both variants share all model/training code, initialization, optimizer and token labels. The branch-default experiment file changes only the conditioning mode and its descriptive name.

| Branch | Input to the autoregressive decoder |
| --- | --- |
| `work/ar-context-redrafter` | Committed context, then anchor and first DFlash proposal |
| `work/ar-draft-editor` | Committed context, entire DFlash proposal, then anchor and first proposal as the correction seed |

Both produce the remaining suffix sequentially. They preserve the anchor and first proposal. If the first proposal is wrong, editing the suffix cannot rescue that round; evaluation counts these cases. The whole draft can be visible because it exists at serving time. Correct future target tokens cannot be visible when predicting an earlier correction.

## Model

The starter configuration has **40,750,336 trainable parameters**: width 256, two causal transformer layers, four heads, feed-forward width 1,024, learned position embeddings for 1,088 positions, and three segment embeddings (context, original draft, correction). Input/output token embeddings are tied. The 151,936 x 256 embedding table alone has 38,895,616 parameters. All these weights belong to the editor; no target/DFlash weights or hidden activations are inputs to this model. It has its own internal representations and a per-round KV cache. No dropout is used.

This size is a configurable first baseline, not a measured optimum. The original 3,024,384-parameter editor could be smaller because it reused rich drafter states and a frozen large-model vocabulary head. The new model must learn its own token representations. Compare trainable parameters, total resident weights, and measured time separately.

Segment IDs mark the draft/correction boundary without introducing new token IDs. Both modes use the same model dimensions. The whole-draft mode has a longer prefill and different correction position offsets; their cost is part of the treatment, not hidden outside timing. There is no reuse of the editor cache across separate speculative rounds yet.

## How training works

For a corrected suffix `[y2, y3, y4]`, the model predicts `y2` after the seed, `y3` after the seed plus correct `y2`, and `y4` after the seed plus correct `y2,y3`. This is teacher forcing. A causal mask lets us compute those training predictions in one forward pass without exposing later correct tokens. At evaluation, the decoder must use its own predictions one at a time; the evaluator reports free-running accuracy as well as teacher-forced loss.

The actual update is in `learning.py`:

1. Clear gradients from the preceding optimizer update.
2. Run the editor on the training inputs.
3. Compute cross-entropy on the corrected suffix only. Higher probability for the correct token means lower loss.
4. `loss.backward()` computes how each weight affects that loss. It does not update weights.
5. Clip excessively large gradients, then `optimizer.step()` applies AdamW's weight update.
6. Repeat on sampled examples; separately measure performance with no weight updates.

The runner uses FP32 weights and computations, microbatch one, accumulation normalized by all suffix tokens in the effective batch, AdamW with zero weight decay, and gradient-norm clipping at 1. It samples blocks uniformly with replacement using a saved RNG. This is deliberately a readable memorization baseline. It is not yet a tuned mixed-precision trainer for 50k responses.

## Run locally

From the repository root, use an existing environment with the dependencies in `requirements.txt`:

```sh
PYTHONPATH=editor-experiment python -m autoregressive_editor.run describe
PYTHONPATH=editor-experiment python -m pytest -q editor-experiment/autoregressive_editor/test_editor.py
```

`describe` counts parameters on the meta device; it downloads nothing and does not allocate full model weights. Tests use tiny synthetic models and token fixtures. Passing them does not establish useful DFlash corrections.

## Data and the unresolved replay gate

The existing 50k target responses provide committed prefixes and target continuations. They do not contain the DFlash proposal for every sampled context. We still need faithful proposals and a justified replay contract before a real training bank is approved.

For the initial pilot, the exporter reuses the existing independent full-bank checker and requires **128 repairable plus 16 preservation blocks**. It authenticates the original response journal, checks its record hashes against the bank, and writes tokens and provenance only. The failed bank and one-block smoke bank cannot satisfy this gate.

```sh
PYTHONPATH=editor-experiment python -m autoregressive_editor.export_bank \
  --source /path/to/verified-128-plus-16-feature-bank \
  --responses /path/to/qwen_responses_50k \
  --output /path/to/new-token-bank
```

This adapter provides a correctness-preserving pilot route, not a requirement that the eventual token-only collector store hidden features. A scalable token-only collector and a natural-distribution held-out collector remain follow-up work. Reusing the old bank for the first pilot does not make its contents the intended 50k evaluation distribution.

The exporter preserves the source's explicit replay contract. The historical
`uncached-first-round-v1` check and the new `native-cached-first-round-v1` check
are distinct; the latter must include all three fixed-fixture native regressions.
The native contract checks a fresh cached first round at each saved prefix. It
does not establish agreement with the old uncached replay or a full-generation
cache history. No bank is approved merely because this option exists.

The token-bank reader indexes JSONL offsets instead of retaining all tensors in memory. It checks checksums, completed status, per-example replay status, token ranges, duplicates, anchor alignment, and prompt/family split ownership. A separate held-out bank must have matching model identities and disjoint prompt/family IDs. The current exporter produces training data only. Raw responses and banks belong outside Git.

## Train only after the data gate passes

```sh
PYTHONPATH=editor-experiment python -m autoregressive_editor.run train \
  --bank /path/to/verified-token-bank --output outputs/ar-pilot-run1 \
  --device cuda --repairs 1 --preservation 0 \
  --updates 200 --batch-size 16 --eval-every 25 --seconds 1800
```

The default fit selects one repairable block from the fully validated bank. Use `--repairs 16` then `--repairs 128 --preservation 16` for larger nested subsets, restarting from the same seed. Selected indices are saved and checkpoint reload evaluates the same subset.

Output directories must be new. The runner rejects held-out training records and wrong fixed first proposals for the initial fit. It allows at most 2,000 updates per invocation; reaching that cap is not a capacity conclusion. The wall-time limit is checked between optimizer updates; setup and initial evaluation are outside that limit, and an evaluation/checkpoint can overrun it. Use a hard SLURM allocation limit as well. No automatic large-data run or budget extension is scheduled.

Outputs include configuration, code revision, model/data identities, update CE and gradient norms, free-running train metrics, optional validation metrics, and an atomic model/optimizer/RNG checkpoint. Supply `--validation` only for a separately validated prompt-disjoint development bank. The current reader supports `verified-feature-bank` and explicitly labeled `synthetic-test` provenance; a future collector needs its own verified producer contract.

To continue a bounded attempt, use `--checkpoint old-run/checkpoint.pt` and a new output directory. `--updates` is the total target step count, not additional steps. Resume rejects changed conditioning, model dimensions, learning rate, seed, data identity, selected subset or batch size. Model, optimizer and example-sampling state are restored. CPU continuation is tested exactly; cross-hardware/kernel reproducibility is not claimed.

```sh
PYTHONPATH=editor-experiment python -m autoregressive_editor.run evaluate \
  --bank /path/to/verified-token-bank --checkpoint outputs/ar-pilot-run1/checkpoint.pt \
  --output outputs/ar-pilot-reload --device cuda
```

Evaluation reports teacher-forced CE, free-running token/exact-suffix accuracy, original/edited/union consecutive saved-label coverage, direct damage and wrong-first cases. It does not claim native verification or speedup. Replacement/union integration and full-generation timing remain separate required experiments.

## What the first results would mean

- Tiny training examples cannot be reproduced: inspect masks, shifts, labels, gradients and optimization before blaming capacity.
- Tiny training examples fit but held-out examples do not: investigate data coverage and generalization.
- Held-out repairs improve but runtime increases more: the model does not repay its cost.
- Both useful progress and complete cost improve under native verification: this supports a larger controlled test.

Run the small 1/16/128-block checks before a learning curve toward 50k distinct responses. An untrained timing experiment can establish cost; it cannot establish learned repair quality. More data and more parameters are hypotheses to test, not guaranteed fixes.
