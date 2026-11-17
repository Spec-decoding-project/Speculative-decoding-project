# Full-prompt-corpus training protocol

The user authorized this phase after reviewing the sixteen-block result. It
replaces the proposed next small fit. Both token-only autoregressive variants
will be trained from scratch on the same full training-prompt inventory.

## Authenticated inputs

Preflight job 8454304 verified every record and source journal: 50,000 training
prompts (44,000 UltraChat, 6,000 GSM8K), 1,024 validation prompts (512 each source),
zero exact normalized cross-split family overlaps. All records are generated;
2,318 training responses and 23 validation responses had historical length caps.
These old response labels are not used as new canonical labels, so historical
capping does not remove their prompts. Original journals are read-only.

Prompt lengths: median 93, mean 171.22, 99th percentile 1,130, maximum 3,241.
The pinned target receives complete prompts with context capacity 4,096. The
editor receives at most the latest 1,024 accepted tokens, with its original
1,088-position capacity and unchanged parameter count. Store absolute anchors
and window offsets; validate full native context before taking the editor window.

## Collection, validation and retention

First run 128 training prompts covering input-length strata, all long outliers
where available, and the four prior canonical fixtures. Independently generate
64 tokens using native DFlash and greedy target-only execution. Require exact
emitted tokens and full-vocabulary logit hashes, plus unchanged prior fixture
continuations. Measure native throughput by prompt-length stratum and FP32
editor updates on the resulting real examples. A failure prevents full launch.

For the full run, process every one of the 51,024 prompts using the same native
engine, in chunks of 64 requests (at most four active sequences). Save atomic,
compressed raw chunks with request identity, canonical emitted tokens, actual
16-token candidate inputs, positions, raw target argmax/prefix evidence and
native accepted/rejected counts. Hash completed chunks and re-audit them on
resume and bank construction. Never inject future correct labels into DFlash.

Authenticate every emitted token against its target argmax at the exact prefix.
Require every retained block's complete reference window, native prefix,
positions, anchor and acceptance counts to agree. No mismatch filtering. The
128-prompt independent greedy check is a scoped configuration validation; the
full corpus uses native canonical generation and per-record acceptance checks,
not a second full greedy pass or a proof about arbitrary future inputs.

Keep up to four blocks per training prompt with a correct preserved first
proposal. Include one naturally correct full proposal whenever available, then
spread remaining choices across eligible native anchors. Keep wrong-first cases
in natural development/test selection (up to four spread blocks per prompt).
Report every prompt's coverage and reasons for zero eligible blocks. No artificial
preservation controls, response rewriting or validation-to-training transfer.

Build an immutable indexed SQLite bank from the re-audited chunks. Store actual
examples privately; publish aggregate counts, provenance and metrics only.

## Training and excluded evaluation

Both 40,750,336-parameter models use the unchanged causal decoder and own tied
token embedding/readout, width 256, two layers and four heads. Context mode reads
context plus anchor/first proposal. Draft mode additionally reads the complete
original proposal. Both preserve the two seed tokens and generate fourteen
editable suffix tokens sequentially. Target and drafter are not optimizer members.

Reset both to seed zero and initial hash
`672080cdb5b25a32ae824198f3549e43979352bab0f621e75379dd9c46d03ab8`.
Three complete epochs, deterministic epoch shuffling, microbatch one/effective
batch 32, FP32, AdamW peak LR 0.001, weight decay 0.01, clip norm one. Linear
warmup for the first 2% of optimizer steps, then cosine decay to LR 0.0001.
Cross-entropy only over editable suffix tokens, normalized over all valid tokens
in each effective batch; teacher forcing behind the causal mask. Partial final
batches use their actual token denominator. No overfit-exactness stopping gate
on the full corpus and no automatic hyperparameter sweep.

Preserve model, optimizer, RNG, epoch/cursor, best-checkpoint decision and source/
dataset fingerprints for resume. Periodic checkpoints limit lost work. Restart
must neither silently change data/configuration nor skip the unfinished epoch.

Use 512 development and 512 untouched final-test prompts, with 256 of each
source per split. All eight previously inspected prompts are development.
Evaluate after each epoch; select the highest development mean correct-prefix
checkpoint, using lower CE to break ties. Evaluate its final-test behavior once
after all three epochs. Include naturally correct and wrong-first cases and
report prompt-level coverage, source-specific results and preservation separately.

Report token/suffix accuracy, CE, repairs/damage, original/edited/union saved-
context coverage and isolated editor latency. This run does not integrate an
edited native generation loop, so it cannot establish complete-system speedup.

## Resource and estimate

One existing RTX A5000 at a time, four CPUs and an initial 16 GiB host reservation;
the pilot measures whether that fits the available node memory. Keep the pinned
local environment and compare package hashes. No downloads or paid resources.
Maximum full pipeline wall time: 24 hours, with resumable collection/checkpoints.
Calibration and full-run output directories are separate and never overwrite
the previous one-/sixteen-block experiments. Calibrated training weights are
discarded; actual models restart from the declared initialization.

Estimate collection from measured per-length-stratum throughput weighted by the
full inventory. Estimate training from measured effective-batch-32 updates and
the actual retained example count, for three epochs per mode. Add initialization,
bank validation, checkpoint I/O and evaluation overhead; keep scheduler delay
separate. Update the estimate from actual full-run progress before reporting it.
