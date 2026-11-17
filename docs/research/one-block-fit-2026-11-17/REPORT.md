# Both full-size editors memorize one real block; held-out proposals worsen

November 17, 2026. **Both 40,750,336-parameter token-only autoregressive variants
passed the one-block memorization criterion after 100 updates. Neither repaired
any of the 32 evaluated blocks from eight excluded validation prompts.** Their
revised proposals shortened correct prefixes on 19 and 20 blocks respectively.
This establishes a working real-data training path, not useful generalization
or an end-to-end speedup. No training on additional blocks has started.

## What ran

The user's explicit instruction authorized one real block and excluded-prompt
evaluation before larger training. [ADR 0003](../../adr/0003-authorize-one-block-memorization.md)
superseded the previous requirement to finish 144 examples before the first fit.
The [predeclared protocol](PROTOCOL.md) fixed selection, budgets and stopping.
Executed experiment code is commit `b300dad`; existing model/training code came
from the whole-draft branch at `4565132`, identical in implementation to the
context branch at `f5b50fa`. Runtime conditioning mode alone distinguishes them.

The full model has two causal layers, width 256, four heads and a tied
151,936-by-256 token embedding/readout. Exactly 40,750,336 editor-owned parameters
are optimized. This is the actual full-size model, not the earlier width-16,
20-token synthetic test. Qwen and DFlash were unloaded before fitting; they are
not optimizer members. Neither editor consumes their hidden activations.

Both fits started from identical weight hashes and used seed zero, FP32,
AdamW at 1e-3, no weight decay/dropout, microbatch one and gradient clipping at 1.
Training used teacher forcing; evaluation generated all 14 editable tokens
sequentially from the model's own predictions. The anchor and first proposal
remain fixed. Only one real block updated either model.

## Training result

Training example: `ultrachat_200k:train_sft:37680`, absolute anchor 24. Its actual
native DFlash proposal matched five consecutive target tokens after the anchor;
both trained editors matched all 15. It was the first repairable block in the
fixed candidate order, without reference to either editor's predictions.

| Measurement | Context plus seed | Whole-draft conditioning |
| --- | ---: | ---: |
| Initial teacher-forced cross-entropy | 11.903689 | 11.898078 |
| Cross-entropy at update 25 | 0.539554 | 0.610003 |
| Cross-entropy at update 50 | 0.004187 | 0.003601 |
| Cross-entropy at update 75 | 0.002046 | 0.001772 |
| Cross-entropy at update 100 | 0.001662 | 0.001466 |
| Final free-running editable suffix | 14/14 exact | 14/14 exact |
| Original → edited correct proposal prefix | 5 → 15 | 5 → 15 |
| Updates | 100 | 100 |
| Training loop including periodic evaluation | 1.32 s | 1.16 s |
| Mean optimizer-update time | 11.88 ms | 10.33 ms |

Both already generated the exact suffix at update 25, but its loss exceeded
0.05. Updates 50, 75 and 100 satisfied both requirements, triggering the fixed
three-consecutive-pass stop. Neither approached 2,000 updates or 600 seconds.
The short training times reflect repeatedly fitting one short sequence; setup,
model hashing, checkpoint IO and held-out evaluation are outside those times.

Optimizer ownership, finite gradients, actual weight changes and future-label
exclusion checks passed. Cached and full-sequence editor generation matched.
Reloaded checkpoints reproduced the weights and training metrics. The initial
weight SHA256 was identical for both variants; final hashes differ as expected.

## Excluded-prompt result

Eight prompts were selected before fitting from the project's existing validation
split: four GSM8K and four UltraChat. Eligible IDs were ordered by SHA256(ID),
with exact normalized message-family duplicates and training overlap excluded.
There were 997 eligible records. The underlying source IDs can contain `train`
or `train_sft`; the experiment's authenticated `dataset_split` is `validation`.
These are a small held-out validation probe, not a new official benchmark split.

Each prompt supplied its first four complete native verification blocks within
a 64-token generated window, giving 32 blocks. All were retained regardless of
whether the first proposal was correct. There were 26 repairable blocks and six
wrong-first-proposal blocks; **no completely correct preservation block occurred**.
The six wrong-first cases cannot be rescued by this preserved-first architecture.
No test metric was computed until stopping and checkpoint selection were complete.

Correct-prefix length below counts matching proposals after the target anchor,
with a maximum of 15. It is saved-context coverage, not a live generation rate.

| Held-out measurement | Original DFlash | Context editor | Whole-draft editor |
| --- | ---: | ---: | ---: |
| Mean consecutive correct proposals | 2.5000 | 0.84375 | 0.81250 |
| Total consecutive correct proposals | 80 | 27 | 26 |
| Blocks with a longer correct prefix | — | 0/32 | 0/32 |
| Blocks with a shorter correct prefix | — | 19/32 | 20/32 |
| Exact editable suffixes | — | 0/32 | 0/32 |
| Correct editable token positions | — | 4/448 (0.89%) | 2/448 (0.45%) |
| Mean union coverage, retaining original | 2.5000 | 2.5000 | 2.5000 |

Both untrained editors averaged 0.8125 correct proposals, reflecting mostly the
preserved first guess. After fitting, the context variant added one correct-prefix
token across the entire held-out set; the draft variant added none. Teacher-forced
held-out loss worsened from 11.993 to 17.332 and from 11.977 to 16.596 respectively.
Memorizing one example therefore did not produce useful transfer in this probe.
The one-block difference between variants is not evidence that one architecture
is generally better. Damage here concerns earlier correct prefixes; preservation
of completely correct blocks remains untested by this sample.

## Editor cost and interpretation

Median editor time across held-out blocks was **19.43 ms** for context conditioning
and **19.46 ms** for whole-draft conditioning. The training block took 19.43 and
19.46 ms. Each block had two warmups and ten CUDA-synchronized timing samples.
Timing includes context/draft prefill, the editor's per-round cache and 14 sequential
token outputs in FP32. It uses the existing eager implementation without serving
optimization. Target and drafter were unloaded for this isolated measurement.

These weights provide no observed held-out coverage gain to repay that added
cost. Replacing the proposal worsens measured prefix quality; retaining the
original as a union fallback preserves its coverage but adds no useful tokens.
**No native edited full-generation throughput or latency benchmark ran.** The
existing evaluator measures fixed contexts; it does not execute edited trajectories
or charge native union verification. Do not turn the isolated timings into a
claimed percentage slowdown or speedup for the complete system.

## Collection, resources and evidence

The existing invariant vLLM reference independently generated and verified 64
tokens for four training candidates and eight held-out prompts. All **768 emitted
tokens and corresponding full-vocabulary logit hashes matched**. Captured proposals
were the actual native verification inputs, not injected labels. Every retained
block's prefix, labels and native acceptance counts passed alignment checks.

There were 215 captured native blocks; 48 were retained across the twelve prompts
(four each). Exclusions were 126 beyond the four-block budget and 41 without a
complete reference window. Four training candidates produced 16 available blocks,
but only the single selected block was used for optimization. The 32 held-out
blocks include wrong-first cases. Original response files were read only.

Job **8453776** completed with exit 0 in **5m19s** on one RTX A5000, four CPUs and
32 GiB on tron61. Collection used **262 target forwards and 16.6142 target-call
seconds**, below the 500/120 limits. Maximum host RSS was 21,294,324 KiB;
collection peak GPU allocation was 20,761,271,296 bytes. The editor process's
reported peak GPU allocation was 991,079,936 bytes (0.92 GiB). The user queue
was empty afterward. No package or model downloads occurred.

The pinned environment remained vLLM 0.34.0, Torch 2.14.0+cu130 and Python 3.12.12.
Package freeze matched the preceding pilot and `pip check` passed. Seventeen
local tests passed. A separate CPU audit rechecked all 768 emitted prefixes,
all 33 used training/evaluation records, every reported prediction metric and
executed source/provenance hashes. Both checkpoint file hashes were independently
recomputed on Nexus. This audit used saved evidence and did not rerun GPU work.

Durable artifacts and checkpoints:

```
/fs/classhomes/fpajak/speculative-decoding-project/research/one-block-fit-8453776/
  collection/{inputs,greedy,native,comparison,bank}.json
  training/context/checkpoint.pt
  training/draft/checkpoint.pt
  training/{context,draft}/updates.jsonl
  training/{context,draft}/{training-evaluations,final-training,held-out,summary}.json
```

Each checkpoint is 489,039,878 bytes and includes model/optimizer/RNG state.
Raw token examples, complete traces and checkpoints stay on Nexus; a local audit
copy excluding weights is under `/tmp/one-block-fit-8453776/`. This report's
`result.json` and `evidence/` preserve aggregate results, hashes, training curves,
timing summaries and the independent audit without publishing a response corpus.

## Decision

The one-block training-pipeline question is answered positively for both variants.
Their present weights are not useful editors on the excluded prompts. This small
negative transfer result does not establish a capacity limit or justify a larger
model. The next useful training milestone is a bounded 16-distinct-block fit with
preservation controls, followed by evaluation explicitly treated as development
now that these eight prompts have been inspected. It has not started. Broader
training, new data acquisition and full-generation deployment remain separate work.
