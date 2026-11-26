# Autoregressive DFlash editor: research history

This is the running, plain-English record of what we tried, why we tried it, and
what the evidence actually supports. New experiments are described before launch
and updated when results arrive. Failed attempts stay in this record.

## Current state — November 25, 2026

Nexus login succeeded and no old job was running. The next diagnostic is being
prepared on `work/ar-refinement-research`: determine whether COPY hides a useful
token prediction or whether fine-tuning weakened the underlying generator. Then
use that result to choose the next training approach. The user has authorized
successive experiments and asked to stay informed; no additional data is needed.

Our objective is an autoregressive module that improves DFlash's proposals enough
to repay its own computation. The target still verifies all accepted output.
Memorizing training examples proves that training works; it does not prove useful
performance. A promising result must improve excluded-prompt prefix quality, and
ultimately improve measured generation throughput with correctness checks.

## November 15–17: replace the hidden-state design with token-only autoregression

The earlier prototype used target/drafter hidden states. The user wanted an
autoregressive editor, so two token-only variants were implemented: one reads
context and the seed; the other also reads the complete DFlash proposal. They
generate fourteen editable tokens sequentially and preserve the anchor and first
proposal. Each has about 40.75 million parameters, mostly its own vocabulary
embedding/readout table. Target and DFlash weights stay frozen.

The target's sequential and block executions sometimes disagreed because of
numerical execution differences. A separately versioned invariant-vLLM reference
was validated and adopted. Original response journals were preserved. That issue
is not an explanation for the later editors' poor transfer.

## November 17: prove that real examples can be learned

Both variants memorized one real repairable block in 100 updates, then sixteen
blocks in 125 updates. Reload and causal/cache checks passed. Excluded proposals
got worse, so we had evidence of working training but not generalization.

## November 17–19: train on the full prompt inventory

Both variants completed three epochs using 199,205 eligible training blocks from
49,953 of the 50,000 training prompts. The other 47 had no eligible complete block.
Development and final-test prompts were excluded from optimization.

On the reserved test, mean correct-prefix length was 3.5422 for DFlash, 2.0176 for
the context-only editor, and 3.3446 for the whole-draft editor. Reading the proposal
helped, but destructive edits outweighed useful repairs. This motivated explicit
preservation rather than automatically adding parameters or training longer.

[Full-corpus report](docs/research/full-corpus-training-2026-11-17/REPORT.md).

## November 20: make copying an explicit action

We added a 257-parameter COPY head to the whole-draft model. COPY emits the
original token at the current position; generation still proceeds autoregressively
using the actual emitted tokens. A matched control and COPY variant started from
the same full-trained checkpoint and learned the same 64 authentic blocks: half
repairable, half already correct, with distinct prompts.

Both memorized everything: the control at 100 updates and COPY at 250. On 512
development prompts, DFlash scored 3.5776 correct-prefix tokens, the refitted
control 3.4736, and COPY 3.4683. COPY reduced damaged blocks from 273 to 94 but
also reduced useful repairs from 267 to 38. Its isolated inference cost was about
20 ms. This was not a useful correction or a measured generation speedup.

The saved action traces explain the immediate failure: 231 of 244 lost control
repairs occur because COPY repeats DFlash's first wrong token. We do not yet know
whether the COPY model's vocabulary head had a correct alternative available.

[COPY pilot report](docs/research/copy-action-2026-11-20/REPORT.md).

## November 25, approach 1: inspect the frozen generator before retraining

**Question:** Is COPY overruling a good answer, or did joint fine-tuning weaken
the generator that would otherwise supply it?

**Method:** Reproduce the saved checkpoint outputs. Inspect the COPY and vocabulary
scores at the same actual prefix for those 231 skipped repairs. Also generate with
COPY disabled, keeping every learned weight fixed. Compare with the full-trained
starting checkpoint and the matched control. No target inference or optimizer
updates are needed. The existing final test stays unused.

**Budget and checks:** one A5000, four CPUs, 5 GiB, at most 15 minutes; immutable
input hashes, exact replay, prefix/action alignment, and an independent recount.

**Decision rule:** available correct vocabulary predictions point toward improving
the COPY decision. Lost vocabulary predictions point toward protecting the trained
generator during fine-tuning. A mixed result will be reported as mixed. This
diagnostic does not itself establish a deployable selection rule.

**Status:** 15 local diagnostic/editor tests passed. Frozen source `54cef7d` was
submitted as Slurm job **8579035**, with a 15-minute cap. Results are pending.

## How to read future entries

Each entry will state the hypothesis, the one main change, data and compute budget,
checks, training result, excluded-prompt result, inference cost, and resulting
decision. Development results are exploratory because we have inspected them
repeatedly. A later success claim needs fresh confirmation prompts and complete
generation timing. Neither copying everything nor an oracle choosing the better
proposal counts as a working editor.
