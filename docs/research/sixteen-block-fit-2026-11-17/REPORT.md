# Sixteen-block fit: both models memorize, excluded proposals still worsen

Both autoregressive variants learned all sixteen real training blocks exactly,
passing the fixed criterion at 125 updates. On the eight excluded development
prompts, editing still reduced correct-prefix length substantially. This confirms
that the implementation can memorize multiple real corrections; it does not
establish useful generalization, a need for more parameters, or a speedup.

The run completed successfully on November 17, 2026: Slurm **8454166**, one RTX
A5000, 90 seconds total. The actual optimization loops took about 13 seconds per
variant. No target inference, numerical-determinism investigation, regeneration,
download, 128-block fit or 50k training ran in this phase.

## Exactly what was trained

Each model has **40,750,336 trainable parameters**, width 256, two causal decoder
layers and four attention heads. The token embedding and vocabulary readout are
tied. These are token-only models; neither receives the target's or drafter's
hidden activations. Context mode reads the accepted context plus the preserved
anchor and first proposal. Whole-draft mode also reads the complete original
DFlash proposal. Both generate fourteen revised tokens sequentially.

Both started from identical random seed-zero weights, rather than continuing
from the one-block checkpoints. Target and DFlash were not loaded during training.

We reused actual native DFlash blocks captured in the previous experiment.
CPU re-audit authenticated all 768 emitted reference tokens/full-vocabulary logit
hashes and all 174 complete native blocks. The saved training pool contained 45
repairable blocks across four prompts. Four blocks spread across each prompt's
64-token continuation were selected before training. **Sixteen blocks means four
training prompts, not sixteen independent prompts.** Their 224 editable-token
presentations cover 215 distinct target positions.

Each update followed this sequence:

1. Zero accumulated gradients and visit the same sixteen blocks in fixed order.
2. Predict each suffix token using context and earlier correct suffix tokens
   during training. The causal mask prevents access to later correct tokens.
3. Sum cross-entropy on the fourteen editable positions per block and divide by
   224 across the effective batch. Context and preserved tokens carry no loss.
4. Call `backward()` for each microbatch to accumulate gradients. This computes
   weight derivatives; it does not change the weights.
5. Clip the global gradient norm to one and call AdamW's `step()` once to update
   only the editor. Learning rate 0.001, FP32, no weight decay or dropout.
6. Every 25 updates, generate each suffix using the editor's own previous
   predictions. Stop after three consecutive evaluations with all 224 tokens
   correct and mean CE below 0.05. Limits were 2,000 updates or 600 training-loop
   seconds per variant; neither limit was approached.

At 125 updates, each example had been visited 125 times: 2,000 block
presentations and 28,000 editable-token presentations per model. The full
[predeclared protocol](PROTOCOL.md) records data selection, limits and controls.

## Training result and reload

| Measurement | Context plus seed | Whole-draft conditioning |
|---|---:|---:|
| Initial mean CE | 11.9636 | 11.9549 |
| CE at update 25 | 2.7285 | 2.5167 |
| Exact training suffixes at update 25 | 1/16 | 4/16 |
| CE at update 50; all suffixes already exact | 0.1551 | 0.07845 |
| Three passing evaluations | 75, 100, 125 | 75, 100, 125 |
| Final mean CE | **0.0043345** | **0.0039672** |
| Final free-running suffix accuracy | **224/224** | **224/224** |
| Final exact training suffixes | **16/16** | **16/16** |
| Mean correct proposals before → after | **3.4375 → 15** | **3.4375 → 15** |
| Training-loop seconds | 13.255 | 13.106 |
| Mean duration of first 100 updates | 92.57 ms | 91.00 ms |

Correct-prefix length counts consecutive correct proposals after the anchor,
including the preserved first proposal; its maximum is 15. An exact editable
suffix contains fourteen correct tokens.

Optimizer ownership, finite gradients, weight changes, causal future-label
exclusion, cached versus full-sequence generation on every training block and
checkpoint reload all passed. Reloaded weights and training metrics matched.

## Excluded-prompt results

The same 32 blocks from eight excluded prompts were evaluated only after fixing
each checkpoint. They include 26 repairable and six wrong-first-proposal blocks;
the latter cannot be repaired by changing only the suffix. These eight prompts
were already inspected in the one-block experiment, so they are development data,
not an untouched final test.

| Measurement on the 32 development blocks | Context plus seed | Whole-draft conditioning |
|---|---:|---:|
| Original DFlash mean correct prefix | 2.5000 | 2.5000 |
| Untrained editor mean correct prefix | 0.8125 | 0.8125 |
| Previous one-block editor mean correct prefix | 0.84375 | 0.8125 |
| **Sixteen-block editor mean correct prefix** | **0.8750** | **0.9375** |
| Blocks with longer correct prefix | 0/32 | 1/32 |
| Blocks with shorter correct prefix | 20/32 | 19/32 |
| Exact editable suffixes | 0/32 | 0/32 |
| Correct editable tokens | 6/448 (1.34%) | 8/448 (1.79%) |
| Mean CE before → after training | 11.9930 → 11.3735 | 11.9768 → 11.2699 |
| Union saved-context coverage | 2.5000 | 2.5625 |
| Median isolated editor latency | 19.12 ms | 19.32 ms |

The single whole-draft repair was `gsm8k:train:6867`, absolute anchor 75: correct
prefix 2 → 4. Across all 32 blocks, original coverage totaled 80 positions;
edited coverage totaled 28/context or 30/draft. Keeping both proposals gives
80/context or 82/draft positions of union coverage. The two additional positions
do not demonstrate that verification plus editing would repay its cost.

Both sources worsened. On sixteen UltraChat blocks, mean prefix fell from 2.6875
to 0.75 for both editors. On sixteen GSM8K blocks, it fell from 2.3125 to 1.0 for
context mode and 1.125 for whole-draft mode. These counts are descriptive results
from eight prompts, not 32 independent prompt-level trials.

## Preservation control and limits

The saved captures contain **zero fully correct training blocks**. The earlier
proposed paired repair-plus-preservation training arm therefore did not run.
We did not fabricate native examples or move validation prompts into training.

One fully correct native validation block existed outside the fixed development
set: `gsm8k:train:5868`, absolute anchor 104. Both trained editors reduced its
correct prefix from **15 to 1**; neither generated a correct editable token.
The untrained editors also scored 1 on this block. This is a concrete example
of damage from editing, not evidence that training newly caused the damage or
an estimate of the population-wide preservation rate.

Training prefixes range only up to 86 tokens; development prefixes reach 448.
The models were initialized from scratch and trained on 215 unique target
positions from four prompts. Limited content and context-length coverage are
plausible contributors to poor transfer, not separately proven causes. A
memorization pass cannot distinguish useful editing from memorized continuations.

Latency includes editor prefill, a fresh per-round cache and fourteen sequential
outputs, with two warmups and ten synchronized measurements per block. It is an
FP32 eager prototype with target and drafter unloaded. No edited native full-
generation benchmark ran; neither a complete-system speedup nor its exact slowdown
has been measured. The small union gain is not a speedup result.

## Decision and next useful milestone

The sixteen-block memorization question passes for both variants. Their current
weights remain unsuitable as proposal replacements on these excluded prompts.
There is no evidence here that increasing parameter count is the right response.

The next useful experiment should increase **distinct training prompts** and
include genuine already-correct training proposals, while keeping the architecture
fixed and preserving prompt-level separation. First determine their yield in a
bounded collection; then compare repair-only and repair-plus-preservation fits.
Use the current fixed development set to measure damage, and reserve new prompts
for a later untouched test. This follow-up has not been run or queued.

## Supporting artifacts

- Executed source/protocol: commit `2bcdc06e1b7c688be918d1336ffad46951b1d434`,
  `editor-experiment/sixteen_block_fit/{data,train}.py` and `run.sbatch`.
  Existing editor source is unchanged from branch `work/ar-draft-editor` at
  `4565132`; both modes use that same implementation.
- [Machine-readable result](result.json), [selection/provenance](evidence/dataset-selection.json),
  [training configurations and curves](evidence/), and
  [independent metric recount](evidence/independent-audit.json).
- Twenty-one local tests passed before execution. CPU audit independently
  recounted 164 final/untrained block predictions, stop criteria, update logs and
  timing summaries. Checkpoint SHA256 values were independently verified on Nexus.
- Job completed exit 0 in 90 seconds; max host RSS 2,670,488 KiB, peak allocated
  GPU memory 1,146,663,424 bytes (1.07 GiB). Python/Torch package freeze matches
  the previous run. Twenty uploaded source files passed checksum verification.
- Dataset SHA256: `a7bae8e7f1a068591aafe395bf827376e051662e6508a20a22cbc9ab66ec1747`.
- Checkpoints, each 489,040,774 bytes including optimizer/RNG:
  context `1176ce2d83524297c94d92e403e84b75459d35545308b59f036b5b0723f5ffd3`;
  draft `a74e04a85fc7ceed43ec0d5140333d2b70697e13cacc05d9962d8a274831b701`.

Private artifacts on Nexus:

```
/fs/classhomes/fpajak/speculative-decoding-project/research/sixteen-block-fit-8454166/
  dataset.json
  training/{context,draft}/checkpoint.pt
  training/{context,draft}/{run,summary,training,development,preservation}.json
  training/{context,draft}/{training-evaluations.json,updates.jsonl}
```

Local private audit copy excluding checkpoints: `/tmp/sixteen-block-fit-8454166/`.
Raw token arrays, responses and model weights are not committed to Git.
