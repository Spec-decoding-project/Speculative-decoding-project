# Autoregressive DFlash editor: research history

This is the running, plain-English record of what we tried, why we tried it, and
what the evidence actually supports. New experiments are described before launch
and updated when results arrive. Failed attempts stay in this record.

## Current state — November 25, 2026

Nexus login succeeded and no old job was running. The frozen-model diagnostic finished on `work/ar-refinement-research`. It found
both bad COPY choices and damage to vocabulary predictions. Three frozen-generator gates have now completed training with small positive
development gains. The completed shorter-edit comparison did not retain a useful quality/cost
tradeoff. The 9,392-prompt gate comparison is complete and does not reliably improve the
result. An exact-output CUDA graph cost comparison is now running. The user has authorized
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

**Status:** submitted Slurm job **8579153** on November 25 from frozen source
`08a4fff91e5810d22c1fc54ad6219820352c9781`. All 47 local tests passed; the job
runs pinned-environment tests and full-size COPY/forced-generator controls before
optimization. Remote output is `research/frozen-gates-20261125/`. The job runs
independently of SSH, writes progress/checkpoints/results, and has a two-hour cap.
Archive SHA256: `5bd18c7efa70ee331fd2d6a3660c4f30884aa167ddbc5ae5eae88eb8de9931a0`.
**Result:** job completed in **6m44s**, exit 0. Score/linear/nonlinear gates produce
mean development prefixes **3.6558 / 3.6997 / 3.6768**, versus **3.5776** for
DFlash. Their improved/damaged block counts are **246/136, 269/187, 286/212**.
Selected updates are 80/180/220; total updates run are 280/380/420. All generator,
reload, full-size integration, calibration simulation and cache checks passed.
Independent Mac recounts and artifact hashes agree. The linear gate's +0.1221
prefix tokens has a descriptive paired 95% interval [0.0410, 0.2041]; development
remains exploratory. All gates cost roughly **26 ms**, so no speedup is shown.
[Detailed results](docs/research/refinement-2026-11-25/GATES.md).

## November 25, approach 3: spend fewer autoregressive steps

**Reason:** the first two frozen-gate arms improve development prefix quality only
modestly, while their fourteen-step inference costs about 26 ms. Before scaling
training, test whether most of the useful change can be retained with less work.

**Predeclared method:** keep all three selected gate checkpoints fixed. For each,
edit only the first one, two or four editable tokens, then copy the untouched tail.
Compare all nine combinations with their corresponding fourteen-step versions.
These are global fixed budgets; no reference labels or oracle per-example choices
enter inference. Use the same internal calibration and development populations;
no final-test or new prompts. No optimizer updates or model-size changes.

**Checks:** limited outputs must exactly match the corresponding first outputs of
the saved full autoregressive execution, with the rest copied. Check all 512
calibration and 2,048 development blocks for each short budget, action emission,
frozen hashes, and cached/full agreement on 64 fixed development blocks. Measure
all four budgets with two warmups/ten synchronized repeats per block. Report
paired-by-prompt quality changes and isolated FP32 latency, including negative
results. One A5000/four CPUs/5 GiB, twenty minutes total.

**Interpretation:** a shorter loop has value only if its quality/cost tradeoff
improves. As a screening calculation, report the baseline round time required to
repay the measured editor cost under unchanged verification cost and the observed
prefix distribution: editor_ms × (original_prefix+1) / prefix_gain. This is not an
end-to-end benchmark; a gain cannot be claimed as actual serving speedup.

**Status:** Slurm job **8579195** submitted from frozen source
`d2d3a54041039b83b326324b39b485bc136eeab0`, with a twenty-minute cap. All 51 local
tests passed. Output: `research/short-edits-20261125/`. Source archive SHA256:
`2c691e2a054795cef40dfb10f303fb6cf6ae789c2fef87448e8377fca8bb2b63`.
**Result:** job completed in **4m30s**, exit 0. Eight of nine shortened
configurations worsen development prefix quality. The four-token linear gate
scores **3.5952**, only +0.0176 over DFlash, at **7.58 ms**; its interval spans zero.
One/two-step loops cost about 2.1/4.0 ms but lose quality. All 23,040 short-proposal
replays, timing/cache checks, generator/parent hashes and independent Mac recounts
passed. Thus early stopping alone does not repay editing. [Full comparison](docs/research/refinement-2026-11-25/SHORT_EDITS.md).

## November 25, approach 4: increase decision-training coverage

**Evidence:** freezing the generator gives small positive development gains, but
shortening its output loop loses most or all of them. The first fit contained only
991 decisive training positions after excluding choices that cannot affect the
reachable prefix. Before changing capacity, test substantially more examples.

**Predeclared comparison:** train the same three gate architectures from their
original zero-output initialization, around exactly the same frozen full-trained
generator. Use **9,392 distinct training prompts**: 600 natural preservation and
4,096 repair blocks per source. Preserve the exact original 512-prompt gate
calibration set, exclude those prompts and families from optimization, and keep
development unchanged. No final-test access, new labels or native collection.
Selection is deterministic and authenticated against the immutable bank.

Natural preservation prompts are scarce. To preserve equal source/kind weighting
rather than silently changing the objective, multiply each training block's
remaining-prefix decision weights by N/(4 × group_count): 3.913333 for preservation
and 0.573242 for repair groups. Keep all other optimizer, stopping, selection,
feature-extraction, reload and evaluation rules from approach 2. At most 2,000
updates/200 epochs per gate, one A5000/four CPUs/5 GiB, two hours total; expected
runtime roughly ten to fifteen minutes. Calibration remains the sole checkpoint
selection population. Report every arm, including deterioration.

Fit SHA256: `8527af5b2b4e610217cba9b702bf78cea3d7019d7e5e4add1a9ea575bed36ded`.
Calibration SHA256 remains `1a356888d96ee15f5e282fd979c34129f8d6eff9e6fb9fe0f7411ae0fb7f815e`.
The local selection check passed with no split/prompt/family overlap.

**Decision rule:** larger gains would justify fresh confirmation and investigation
of execution cost. No improvement would argue against simply scaling this gate
objective; reconsider the decision features/objective and measure execution costs.
This is an exploratory learning-curve comparison, not independent confirmation.

**Status:** Slurm job **8579313** submitted from frozen source
`b3c4ce451beb3afb93335549e2f41f478d3d2186`. All 53 local tests passed. The job runs
pinned GPU tests, selection/provenance checks, extraction, all three gate fits and
evaluation automatically. Output: `research/broader-gates-20261125/`. Source archive
SHA256: `b018c78513e243bf0b8088010c42d6d34234135f08021283f78f42848aee00ab`.
**Result:** job completed in **10m33s**, exit 0. All three arms reached 2,000
updates. Mean development prefixes are **3.6895 / 3.6797 / 3.6353** for
score/linear/nonlinear gates. Relative to their smaller fits, changes are
+0.0337 / −0.0200 / −0.0415; the latter two intervals span zero. None beats the
earlier smaller linear fit. The larger bank supplies 8,300 meaningful decisions,
but data expansion alone has not produced a material gain. All GPU checks and
independent audits passed, including bitwise-identical calibration features and
exact group-weight reconstruction. All artifacts are backed up. [Detailed results](docs/research/refinement-2026-11-25/BROADER_DATA.md).

## November 25, approach 5: test whether GPU graph replay cuts the full-loop cost

**Evidence:** broader gate training helps one arm but worsens the other two; all
full-loop variants still cost about 26 ms. Shortening the loop mostly destroys
the gain. Test execution overhead before another data or capacity increase.

**Predeclared method:** freeze the broader-fit linear gate and generator. Separate
tensor-only decoding from host input/output handling, and capture the exact
fourteen-step computation in a CUDA graph for each tested prefix length. Follow
[PyTorch's capture workflow](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs):
three warmup iterations on a side stream, persistent input/output buffers, and
in-place input copies before replay. No arithmetic precision or weights change.
This is a fixed-shape prototype, not a deployed variable-length graph cache.

**Comparison and checks:** use the same 64 fixed development timing blocks. Require
eager output to match saved parent output, cached/full agreement, graph agreement
on original and altered proposal inputs, and restoration after the altered input.
Use two warmups and ten interleaved eager/graph timing repeats per block. Include
CPU input construction, transfers and returned output in warm call time. Report
cold graph preparation time and extra live allocation separately. Keep every
repeat, hash checks and any failed controls. Attempt a three-call eager CUDA/CPU
profile to investigate GPU kernel time; missing profiler support will be explicit.

One A5000/four CPUs/5 GiB, twenty minutes, no optimizer updates or target inference.
A warm latency reduction would support execution engineering; it would not prove
native generation speedup or amortize graph preparation automatically.

**Status:** Slurm job **8579567** submitted from frozen source
`11423488bcc3066eb12b2df0be92d9cab72559da`, twenty-minute cap. All 54 local tests
passed. Remote output: `research/graph-replay-20261125/`. Source archive SHA256:
`9bab36c445544a8281c1adfb1ea0c90f176747839bfff86584dea5ce51b99368`.
Results pending.

## How to read future entries

Each entry will state the hypothesis, the one main change, data and compute budget,
checks, training result, excluded-prompt result, inference cost, and resulting
decision. Development results are exploratory because we have inspected them
repeatedly. A later success claim needs fresh confirmation prompts and complete
generation timing. Neither copying everything nor an oracle choosing the better
proposal counts as a working editor.
