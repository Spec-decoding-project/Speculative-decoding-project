# Autoregressive DFlash editor: research history

This is the running, plain-English record of what we tried, why we tried it, and
what the evidence actually supports. New experiments are described before launch
and updated when results arrive. Failed attempts stay in this record.

## Current state — November 25, 2026

Nexus login succeeded and no old job was running. The frozen-model diagnostic finished on `work/ar-refinement-research`. It found
both bad COPY choices and damage to vocabulary predictions. The next training
approach will freeze the original full-trained generator and learn only a small
decision layer, using a broader dataset. The user has authorized
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

**Result:** job **8579035** completed in 4m01s with exit 0. All three checkpoint
replays, prefix alignment and independent ablation checks passed. Of the 231
skipped repairs, COPY overruled a correct vocabulary choice in 86, while the
vocabulary choice was itself wrong in 145. Of 211 cases the full-trained starting
generator answered correctly, 131 became incorrect after joint COPY fine-tuning.
Disabling COPY produces only 1.0796 correct-prefix tokens on development, down
from 3.4683 with COPY enabled. Thus changing the decision threshold alone would
leave a substantially weakened generator.

**Decision:** preserve the original full-trained generator exactly. Train only a
small decision layer, and use substantially more distinct examples plus a separate
calibration split. This tests a different learning strategy without increasing
the generator's size.

## November 25, approach 2: protect the generator and learn the decision

**Question:** Can a small learned decision layer retain the original editor's
useful repairs while copying tokens it would otherwise damage?

**Main change:** freeze every weight of the original full-trained whole-draft
editor. Only a small gate learns to choose between the original proposal token
and that frozen editor's next-token prediction. Generation remains autoregressive:
its next step receives the actual token selected. Inputs remain tokens; any
internal features come from this editor, never from target/drafter hidden states.

**Comparison:** a score-only linear gate (9 parameters), a linear gate that also
reads the editor's own representation (265), and a small nonlinear gate (17,025).
All use the same frozen generator, data, objective and training budget. These are
three specified comparisons, not an open-ended hyperparameter search.

**Data:** 2,048 authentic training-origin blocks, balanced across source and
repair/preservation status, all distinct prompts/families. Use 1,536 for gate
optimization and 512 for internal calibration. Then evaluate frozen selections
on the existing 512 development prompts. No final-test labels enter training.

**Objective:** learn which candidate is correct at positions where exactly one
is correct, while the preceding target prefix is reachable using these two
choices. Weight earlier decisions by the remaining attainable prefix length.
When both candidates are correct, the decision does not change the emitted token;
when neither is correct, the gate cannot repair that position. Reference labels
are used for training/diagnosis only, never to make inference decisions.

**Selection and budget:** initialize to copying; use AdamW LR 0.001, weight decay
0.01, clip 1, batch 256, seed 0; at most 2,000 gate updates or 200 epochs per arm.
Check internal calibration every five epochs; select highest mean correct prefix,
then lower weighted decision loss. Stop after 50 epochs without improvement.
One A5000/four CPUs/5 GiB, two hours total including feature extraction and all
three arms. The generator hash must remain identical before/after every fit.
Reload and cached/full checks are required. Passing training memorization is not
the selection criterion for this stage.

**Status:** implementation and tests in progress; no training job submitted yet.

## How to read future entries

Each entry will state the hypothesis, the one main change, data and compute budget,
checks, training result, excluded-prompt result, inference cost, and resulting
decision. Development results are exploratory because we have inspected them
repeatedly. A later success claim needs fresh confirmation prompts and complete
generation timing. Neither copying everything nor an oracle choosing the better
proposal counts as a working editor.
