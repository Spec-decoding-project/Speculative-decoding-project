# Autoregressive DFlash editor: research history

This is the running, plain-English record of what we tried, why we tried it, and
what the evidence actually supports. New experiments are described before launch
and updated when results arrive. Failed attempts stay in this record.

## Completed state — November 27, 2026

The user authorized a new bounded comparison to improve the editor that reads
the entire DFlash block. This is already the stronger existing conditioning
variant, so the new experiment tests three full-generator training objectives:
ordinary cross-entropy, early-position weighting, and extra protection of the
correct original prefix plus emphasis on its first error. Each starts from the
same original checkpoint and uses the same 8,192 training-origin blocks, including
1,024 naturally perfect proposals. After checkpoint selection, each generator
gets a separately fitted linear COPY gate. All inputs remain tokens and all
fourteen suffix predictions remain autoregressive. Target/DFlash stay frozen.

The [fixed protocol](docs/research/whole-draft-training-2026-11-27/PROTOCOL.md)
records three 768-update fits, calibration-only selection, reload/causal/cache
checks, dual token-accuracy/prefix criteria, independent recount and a two-hour
GPU-job cap. The [whole-block training investigation](https://github.com/Spec-decoding-project/Speculative-decoding-project/issues/28)
is complete. All three fits, gate fits, development evaluations, graph checks,
independent recounts and verified backups finished. Original responses,
checkpoints and prior evidence stay unchanged.

Launch: source `dd31481`, Slurm 8600245 on tron61/A5000, four CPUs/16 GiB/two hours.
All 18 local and remote implementation checks pass; package pins match. Local and
remote authenticated training selection hashes agree. Logs/results are in
`research/whole-draft-8600245/`; selected and resumable checkpoints use its scratch
symlink. The GPU job completed in 24m07s. Independent CPU recounts on Nexus and
the Mac agree. All 22 targeted local checks pass, including three auditor tests
and a regression for storage export.

| Development candidate | Correct prefix | Editable-token accuracy |
|---|---:|---:|
| DFlash | 3.577637 | 32.4289% |
| Earlier gated editor | 3.699707 | 30.6571% |
| Ordinary continuation + gate | 3.617676 | 28.8086% |
| Early-position emphasis + gate | 3.672363 | 30.0502% |
| Preserve/repair, selected original + gate | 3.699707 | 30.6571% |

None beats the earlier editor. Ordinary training selects update 640; early
weighting selects 256; preservation/repair selects unchanged step zero because
no trained checkpoint improves its calibration prefix. All arms nevertheless
complete 768 updates. The last row reproduces the old editor, not a new trained
winner. Both newly selected models lose on both sources relative to that editor.
Training loss falls, but calibration improvements do not transfer to development.
The calibration distribution and fixed-first-proposal limitation remain concrete
issues for a future design, rather than reasons to hide the negative result.

Warm isolated editor time remains about 6.9ms, with roughly 100ms graph setup.
Input/output copies are included. No new native throughput claim follows from
this measurement. These are repeatedly inspected development prompts, not fresh
confirmation; the old reserved test was not rerun. The existing whole-draft
input was retained throughout, all generator parameters trained, and no target
or drafter inference/training ran in this comparison.

Postprocessing 8600354 passed its numerical audit, then ran out of Nexus home
write space while staging redundant checkpoints. Direct scratch-to-Mac streaming
8600808 recovered all six selected files; only the three new verified home
duplicates were removed. The exporter now inventories scratch without copies;
CPU inventory 8600874 passes. All 62 output/checkpoint files and 59 frozen source
hashes verify. Optimizer states remain on node scratch; selected pairs and raw
evidence are backed up privately under
`~/.local/share/wayfinder-artifacts/whole-draft-8600245/`. Queue empty. No automatic
larger sweep or further training was started.

[Complete comparison, process, source results, limitations and artifacts](docs/research/whole-draft-training-2026-11-27/REPORT.md).

## Completed state — November 26, 2026

The cheap-selector experiment is complete. A rule learned on training/calibration
data simplifies exactly to editing only the first proposal. With warm graphs it
improves native GSM8K throughput by 2.73% and leaves UltraChat effectively unchanged
(−0.08%), across 64 development prompts and three repeats. All tested outputs
match; empty-cache generation still loses. All jobs, audits and backups are done;
larger editor training remains paused. See the [selector result](docs/research/refinement-2026-11-25/CHEAP_SELECTOR.md)
and the concise
[project update](docs/research/refinement-2026-11-25/REPORT.md).

November 26 explanation audit: the latest editor is the original 40,750,336-
parameter whole-draft generator plus a 265-parameter gate, not a 265-parameter
generator. Later gate fits do not update the generator. The full generator's
development results improved through its third and final epoch; the experiments
do not establish that better or longer training cannot help. Native timing
rejects the current always-on cost/acceptance trade-off, not all possible editors.
No experiment, training run or remote job was started for this explanation.
Primary-source architecture/training alternatives and their limitations are
recorded in [the research notes](docs/research/refinement-2026-11-25/EDITOR_DESIGN_SOURCES.md).

Evaluation wording correction: the original reserved test was evaluated after
full-corpus training on November 19. Later references to it as "unused" or
"untouched" mean it was not evaluated again in those later experiments, not that
it has never been inspected. A new confirmatory result requires fresh prompts.
The original full-corpus report already records this limitation.

November 27 accuracy clarification: a hash-verified recount of the existing
linear-gate development predictions finds 32.4289% editable-token accuracy for
DFlash and 30.6571% for the gated editor. Mean correct prefix improves from
3.5776367 to 3.6997070 on those same blocks, with 876 prefix positions gained
and 626 lost. Thus the positive result concerns consecutive usable proposals,
not all-position token accuracy or final-answer correctness. No model inference,
training or remote job ran for this recount. Details are in
[the gate report](docs/research/refinement-2026-11-25/GATES.md).

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
updates are needed. This diagnostic does not re-evaluate the existing final test,
which was already inspected after full-corpus training.

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
**Result:** job completed in **1m01s**, exit 0. All 64 original outputs, 64
changed-input controls, cache checks and 1,280 timed calls pass. Warm editor time
falls **25.45 → 6.73 ms (3.78×)**; every case is faster. Median capture plus
warmup is **101.57 ms per shape**, and 52 prefix lengths are tested. The profile
records about 6.97 ms of GPU kernel work per ordinary call, supporting substantial
launch/scheduling overhead. Independent timing/profile recounts and all backup
hashes pass. This remains a fixed-shape prototype with no native generation
speedup claim. [Detailed results](docs/research/refinement-2026-11-25/GRAPH_REPLAY.md).

## November 25 — native replacement and complete-generation timing

**Question:** does the fixed broader linear gate repay its execution cost when
inserted before native DFlash verification? The user explicitly requested this
measurement. No training, data expansion or target determinism investigation.

**Predeclared protocol:** eight development prompts: four with 1–128 input tokens
selected by stable hash, and the four longest available in 512–1,900; 64 generated
tokens maximum. Before any GPU inference, the data-only preflight found just two
prompts in the initially proposed 1,024–1,900 stratum. The revised selection uses
input lengths only, without inspecting candidate results, and retains both.
Use the canonical invariant vLLM 0.34 environment, frozen target/drafter, and
broader linear checkpoint `b6f1c27a…`. Batch one, unchanged target/drafter eager
execution, 2 GiB native KV cache. Original DFlash, identity replacement and FP32
editor must first match independent target-only greedy output and termination.
Audit actual complete prefixes, native acceptance counts and successive rounds.
Include forced first rejection/all acceptance, short budgets and stop-token tests.
These oracle controls test the verifier only and never count as learned results.

Then measure five alternating original/editor repeats per prompt, including the
complete generation call and native round CUDA events. An empty editor graph cache
run is separate from warm repeats. Cache at most 64 exact prefix lengths with LRU
eviction; record hits, misses, preparation and memory. Keep the complete target
prefix; only the editor receives its trained last-1,024-token window. Separately
check BF16 generator/FP32 gate native output, after the FP32 benchmark. Preserve
all raw results privately and publish aggregate results, including slowdowns.

**Budget and scope:** one A5000, four CPUs, 32 GiB host RAM, 45-minute job limit.
The original pinned node has insufficient schedulable RAM; copy its unchanged
environment and checkpoint to available tron59 scratch using CPU-only staging.
Both timing arms run on the same GPU. This measures native replacement; union
verification, a natural-EOS fixture, non-invariant speed and a fresh final-test
claim remain separate requirements. Do not close all of issue #8 from this subset.
If correctness fails, stop timing and retain the concrete failure. If the editor
does not repay its cost, report that plainly before more training is proposed.

**Status:** 23 relevant local tests passed. Slurm **8579825** submitted with source
`140b0550b171eb0f001011422d5a82760ef8c763`; archive SHA256
`9bea99592cc682312a6b60fffc3bfbb731807092aa290a81de2a0320e30b38dc`.
Output: `research/native-speed-8579825/`. CPU-only staging completed; environment
package freeze and original checkpoint hash match on tron59. No native result yet.

**Result:** completed in 9m14s, exit 0. Native replacement, identity and stopping
checks pass. All warm repeats match greedy output. Initial FP32 integration is
32.9% slower overall (13.267 → 17.626 seconds for the 512-token workload), despite
196 → 193 total native rounds. Warm native editor latency is materially higher
than the isolated graph prototype, motivating the bounded kernel diagnosis below.

## November 25 — diagnose native editor kernel overhead

The first native correctness gate passed. During its unchanged timing phase,
warm graphed editor calls take about 23.5 ms rather than the earlier isolated
6.7 ms. The pinned vLLM source installs process-wide ATen matrix multiplication
overrides on Ampere when batch invariance is enabled; these include the editor.
This is a concrete unresolved integration issue, not a reason to train more.

Ranked explanations: (1) invariant ATen kernels slow the editor, predicting a
large reduction when only editor dispatch changes; (2) input/output handling
dominates, predicting a fast isolated graph despite slow native calls; (3) GPU
execution conditions differ, predicting both isolated kernel modes stay slow.

**Bounded follow-up, no training:** after the initial job succeeds, compare eight
native input blocks under invariant → ordinary → restored invariant dispatch,
with identical weights/GPU/inputs and five timings each. Restore checks must pass.
If ordinary dispatch is faster, run paired native FP32 and BF16 editor benchmarks
with the original invariant target. Reuse the authenticated reference; do not
regenerate labels. Keep all eight prompts, five alternating repeats and cold cost.

The adapter temporarily removes vLLM's private ATen override library only while
capturing/eager-checking the editor, then restores it before target execution.
The recorded graph retains its editor kernels. This is restricted to the pinned
synchronous, batch-one worker; it is not a concurrent serving implementation.
Assert invariant target dispatch before every native round and compare full
target-logit hashes on all 512 emitted positions between original/identity/editor
correctness runs, in addition to existing greedy-token/cache/stop checks. A failed
restoration/hash check prevents timing. Target weights, kernels and reference
definition are unchanged during every target call.

One sequential follow-up job, one A5000/four CPUs/32 GiB, at most 30 minutes;
180-second isolated probe and 720-second bound per precision. Original job and
its source remain immutable. All results, including failures, will be retained.

Follow-up Slurm **8579890** is dependent on successful completion of **8579825**.
Source `40b5260ce98c40bd7a1dcba4de9af6d6bcc16bfc`; source archive SHA256
`3b335bd8151902c7da894ac6c3e9a2ea1b5dee50489b08e1a5472dece16e7421`.
Twenty-four relevant local tests pass, including restoring target dispatch after
an editor exception. Output: `research/native-kernels-8579890/`.

**Result:** completed in 12m22s, exit 0. Controlled dispatch changes reduce
isolated editor cost 24.44 → 8.81 ms with unchanged outputs. All 173 native FP32
proposals also remain identical. Corrected FP32 is 11.9% slower overall; BF16 is
7.5% slower. BF16's short group is effectively tied (0.5% faster), longer inputs
15.2% slower. One 94-token input improves 1.269 → 1.022 seconds (1.24×); two of
eight prompts improve. All 394 candidate generations across the three arms match
the greedy reference; 2,560 full target-logit comparisons pass. BF16 changes some
unaccepted proposal tokens but preserves measured native output and round counts.
The empty-cache BF16 workload takes 27.05 seconds versus 13.24 for DFlash.

CPU-only job 8579976 verifies original generator/target/drafter pinned files in
17 seconds. Independent Mac audits and all 37 output-file backup hashes pass.
All jobs are finished. No final-test data or additional training was used.
The always-on design does not pass aggregate speed; test cheap round selection
and broader graph reuse before scaling. Replacement's correctness subset passes;
union and natural EOS remain open. [Complete report and evidence](docs/research/refinement-2026-11-25/NATIVE_SPEED.md).

## How to read future entries

Each entry will state the hypothesis, the one main change, data and compute budget,
checks, training result, excluded-prompt result, inference cost, and resulting
decision. Development results are exploratory because we have inspected them
repeatedly. A later success claim needs fresh confirmation prompts and complete
generation timing. Neither copying everything nor an oracle choosing the better
proposal counts as a working editor.

## November 25 — matched native benchmark of the earlier best linear gate

**User request:** finish this comparison before any more training. Use the earlier
265-parameter gate (development prefix 3.69970703125), not the broader gate used
in the first native pilot. The gate still calls the frozen 40,750,336-parameter
autoregressive generator; 265 is not the complete editor size.

**Registered method, before execution:** exact gate SHA256
`5ab048b622beffb10ab8d282027cb2510f5439e4621793d5a82e687e696d679b`, FP32 generator
and gate, unchanged original generator and canonical invariant target/drafter.
Ordinary editor kernels are captured under the tested temporary dispatch scope;
invariant dispatch is restored before every target call. Batch one, one A5000,
4 CPUs/32 GiB, 80-minute maximum allocation, no optimizer updates.

Select 128 development prompts, 64 per source. Within each source's 256 development
prompts, rank by prompt length/hash into four 64-prompt quartiles; choose 16 per
quartile by a fixed hash independent of outputs. Interleave prompt order by a
second fixed hash. Preserve the final test split. This is exploratory development
evidence, not fresh confirmation. Use 64 output tokens maximum and normal EOS.

Eight target-only reference controls cover each source/quartile. Every selected
prompt gets baseline/editor correctness runs checking exact output/stopping,
native accepted counts, full target-logit hashes and editor graph/eager agreement.
Then an unedited warmup and an empty-cache editor run precede five timing pairs.
Pair order alternates by prompt and repetition. Every timed output must match;
any warm graph miss fails the benchmark. No precision sweep or training.

End-to-end timing includes the whole generation call, all input/output copies,
editor work, engine scheduling, and a final worker synchronization/RPC to drain
last-round GPU work. Also save ordinary generate-call latency and drain/RPC cost.
A synchronized boundary immediately before the first DFlash proposal separates
prefill/request work from decode, identically in both arms. Decode includes the
first proposal and any unused final proposal. Average decode round wall time is
this complete interval divided by actual speculative verification calls. Commit
counts exclude the prefill's first token and clip final-block overshoot against
tokens actually returned. This prevents confusing correct draft prefix with
emitted tokens or counting chunked prefill as verification.

Warm graph reuse and cold setup are reported separately; exact-length graphs are
kept in the existing 64-entry cache and cleared for each cold prompt. Warm results
are an optimistic reusable-shape case, not a claim that setup amortizes across a
fresh arbitrary workload. Report means/aggregate throughput, five paired repeat
ratios, paired-prompt bootstrap intervals, separate source and length-quartile
results/counts. Break-even arithmetic uses measured visible commits and round
cost, with the assumption of fixed noneditor round work made explicit.

The prepared tron59/tron61 hosts lack 32 GiB allocatable memory. CPU-only staging
allocation 8580160 copies the identical environment and original generator to
tron50; packages and checkpoint hashes must pass before GPU execution.

Implementation tests: 27 passed locally. Source commit
`30f1850`, archive SHA256
`7639be0e3517dc5558bb89109479cf00660b30c6764bb6bab9e6c02ba2eb564c`.
CPU staging completed successfully and its allocation was released. Native job
**8580225** submitted to tron50; outputs are written per request under
`research/native-best-8580225/benchmark/records/`. The job is self-contained and
continues if SSH disconnects. All later report audits are CPU-only.

**Timing-accounting observation during the run (no GPU rerun):** early records
show ~8.45 ms of post-return drain/RPC for unedited DFlash versus ~0.38 ms for the
edited path. Synchronization inside the editor can wait for preceding drafter
kernels; therefore its inclusive callback wall time must not be labeled entirely
as incremental editor computation. The requested primary break-even calculation
uses the observed complete-round difference instead: if baseline/edited round
costs are B/E and baseline/edited visible commits are c0/c1, the affordable added
cost is B*(c1/c0-1), and the extra commits needed at measured edited-round cost
are c0*(E/B-1). These quantities follow directly from measured decode throughput.
The raw runner's callback-based estimate is retained for provenance but is not
the headline economics. A separate CPU audit reconstructs counts and this
calculation from raw records; its regression check reproduces the user's
50 ms /3.4% example (1.7 ms affordable). No data, model, timing or sampling setting
was changed during the GPU run.

**Completed result:** Slurm **8580225** finished successfully in **57m24s**;
CPU-only model pin audit **8580512** passed in **17s**. No jobs remain queued.
All 128 development prompts completed five paired timings (640 pairs); all paired
outputs/stops match. Independent Mac reconstruction checks 1,792 native runs,
eight target-only controls, 16,750 raw argmax positions and 8,119 paired full-logit
hashes. Four UltraChat cases terminate naturally; both paths agree. Every warm
editor call reuses a graph: 11,995 hits, zero captures. Local checks pass 28 tests;
the GPU host passed 10 benchmark tests. All 1,820 backup files match remote hashes.

**UltraChat:** 37.383→34.069 end-to-end tokens/s (**8.86% lower**);
69.683→78.428ms per complete decode round; 2.68407→2.74169 actual committed
output tokens per round. Verification rounds over all five repeats fall
7,375→7,220 (1,475→1,444 in one pass). The observed acceptance gain permits
only 1.496ms extra work, versus 8.745ms measured extra round cost. At that cost,
required extra commits are 0.33686, compared with 0.05762 observed.

**GSM8K:** 63.300→57.860 tokens/s (**8.59% lower**);
69.757→78.427ms/round; 4.56625→4.67207 committed tokens/round; rounds
4,415→4,315 over five repeats (883→863 per pass). Affordable extra cost 1.617ms;
measured extra cost 8.670ms. Required extra commits 0.56752; observed 0.10582.

All five repeat ratios lose on both datasets; aggregate throughput ratios are
0.91136 and 0.91406, with paired-prompt 95% intervals 0.89678–0.92807 and
0.89280–0.93564. Every source/length quartile loses on average. Four UltraChat
and twelve GSM8K prompts improve individually. This does not establish a
population-wide impossibility; it rejects this always-on implementation on the
measured workload. No further training was started.

Cold runs capture 2,399 graphs, taking 230.07 s for capture plus warmup across 128
requests (1.797s/request). Full empty-cache generation takes 413.90 s versus 172.32 s
for one warm DFlash pass. This is an optimistic warm-cache speed test plus a
separate per-prompt empty-cache measurement, not an inferred production hit rate.

The raw runtime archive remains immutable at 30f1850. Its callback-based cost
estimate is retained in private raw results; the independent audit reports the
correct complete-round economics. Reusable reporting code is now also corrected
to avoid attributing pending drafter work entirely to the editor callback.
Model initialization and ordinary generate-call-only timing are separately
reported; the latter also loses on both datasets.

[Complete report and aggregate evidence](docs/research/refinement-2026-11-25/NATIVE_BEST.md).
Private raw outputs, source, logs and hashes:
`~/.local/share/wayfinder-artifacts/native-best-8580225/`. Target/drafter/initial
editor/gate remain unchanged, original workspace edits are preserved, and final
test responses were not evaluated. Serving-fidelity remains open for union,
concurrency and fresh confirmation; four replacement natural-EOS cases now pass.

## November 25: test selection before paying for the editor (registered)

The user requested the cheap-selector follow-up. A new tiny decision tree will
look at DFlash's top-two score gaps, proposal repetition, prompt/generated length,
and the previous round's committed-token count. These are available before any
editor step. DFlash already computes the vocabulary scores; we will measure the
additional reduction/copy cost, rather than treating these features as free.
The existing confidence buffer is uninitialized in our configuration and is not
used. The selector receives no target labels, hidden states, or editor features
at inference. Target, drafter, 40.75M generator and earlier 265-parameter gate
remain frozen; the only newly fitted object is a depth-three CPU decision tree.

Collect baseline trajectories on 96 training-origin fit prompts and 32 disjoint
training-origin calibration prompts (balanced by source/length quartile). Run
the existing editor in shadow mode: save its proposals but return DFlash's
original proposal to the verifier. The completed canonical continuation labels
both candidates. Discard windows without all 15 future reference tokens, and
retain damage/zero-gain/wrong-first examples. Fit maximum depth 3, minimum leaf
64, leaf shrinkage 32. Pick one threshold from 0/.25/.5/.75/1/1.5/2/never using
only calibration net prefix gain minus 0.45 tokens per edit; ties favor fewer
edits. This fixed prior approximates the earlier complete-round break-even cost;
native timings, not the proxy objective, decide success.

Freeze the selector before evaluating 64 development prompts excluded from the
previous 128 native prompts: 32 per source, eight per source/length quartile.
These are new native trajectories, not untouched final confirmation (saved
development blocks have previously been evaluated). Three balanced repeats
compare baseline, selector-only/always-skip, always-on and selective editing,
batch one, 64-token cap. All feature extraction, copies and editing are timed.
Warm graph reuse is required; both editor arms get separate empty-cache runs.
Check all final tokens/termination, and raw target logits plus independent greedy
references on one prompt per stratum. No final-test responses or larger training.
One GPU, maximum 80 minutes. A selector that cannot beat baseline is a negative
result, not a reason to silently increase training or change the frozen editor.

Tracked in [Can a cheap pre-editor selector recover native DFlash throughput?](https://github.com/Spec-decoding-project/Speculative-decoding-project/issues/27).

The first launch (8580929, source 444ebc3) stopped after one shadow example:
all 64 emitted tokens passed the native audit, but the last native round had
computed six tokens beyond the output cap. The label builder checked an
unreturned prefix before discarding its incomplete window. Fixed that ordering:
validate the entire known prefix, retain the alignment/anchor guards, and omit
any window without 15 observed future tokens. A synthetic overshoot regression
test and the saved real failure fixture now pass. No selector fit or development
evaluation occurred in that failed launch; its source/log/output remain intact.

Collection job 8580943 then completed 96 fit and 32 calibration prompts. There
are 1,397 fully labelled fit rounds (143 repairs, 37 damage, 1,217 unchanged) and
455 calibration rounds. The frozen threshold is 0.5. Crucially, its classification
simplifies exactly to **edit only the first proposal of a response**: that branch
scores 1.0546875; every later-round leaf scores below 0.5. Calibration selects 32
edits with net prefix gain 46 (31.6 after the fixed 0.45/edit charge). Eight greedy
reference controls also completed. No native development evaluation request had
started when the job was stopped at 9m47s to avoid measuring needless features.

Before native evaluation, added a fifth arm that compiles the frozen classifier
to its equivalent first-proposal-only policy. The compiler proves all discarded
branches give the same false decision. It adds no fitting or threshold changes.
Once the first edit has run, a host-side counter skips before confidence reduction,
GPU scalar reads or context copies. The general selector remains as a measured
control; both implementations must edit exactly once per prompt and follow the
same native round trajectory. Reuse the existing collection, selector and greedy
references byte for byte; do not repeat them. The replacement evaluation job has
a 68-minute cap, keeping combined allocation time below 80 minutes. This change
is prompted by the frozen rule itself, before any native evaluation outcome.

## November 26: first-block selection gives a small warm-cache win

The bounded experiment finished. The new component is a tiny fitted tree that
compiles to a flag: call the existing autoregressive editor on the first proposal,
then skip it on subsequent rounds. The editor still generates its fourteen
editable tokens sequentially when called. Its 40.75M generator and earlier
265-parameter gate remain frozen; no neural-network training ran in this phase.

On the same 64 development prompts, compare baseline, confidence features with
editing always skipped, always-on editing, the general selector, and its proven
equivalent cheap implementation. Each gets three rotated timing repeats on one
A5000, batch one, 64-token output cap. Timing includes all editor/feature work
and token copies, and drains the GPU before stopping. All warm calls reuse graphs.

| Source, 32 prompts each | Baseline tokens/s | First-block tokens/s | Change |
|---|---:|---:|---:|
| UltraChat | 38.372 | 38.339 | −0.08% |
| GSM8K | 64.865 | 66.633 | +2.73% |

GSM8K's three speed ratios are 1.02716/1.02721/1.02741; its paired-prompt 95%
interval is 1.00971–1.04654. UltraChat's interval spans 0.99046–1.00992, so it is
effectively unchanged. Across both sources the workload throughput improves
0.94%, whereas always-on editing loses 8.21%. Confidence extraction with no edits
alone costs 1.316ms/round and loses 1.80%; eliminating those unused features is
essential. The general selector loses 0.75% overall despite identical edits.

The cheap rule makes 192 editor calls instead of 3,573 across all timed repeats,
94.63% fewer. In one pass it saves 19 of 1,177 baseline verification rounds:
UltraChat 742→739, GSM8K 435→419. It reduces rounds on 18 prompts, increases them
on four, and leaves 42 unchanged. The very first timed case is a loss: baseline
1.31968s/19 rounds, cheap selector 1.39541s/20 rounds. We retain that example in
the report rather than selecting only a favorable pair.

This is a warm-cache result. Empty-cache first-only editing captures one graph
per request, about 101ms/request, versus 1,191 captures across the always-on
64-prompt pass. Cold first-only throughput is still 5.76% below baseline overall
(−5.23% UltraChat, −6.64% GSM8K). Since there is only one edit per response, useful
graph reuse must come from compatible shapes across requests. No production hit
rate was measured. Inputs span 24–576 tokens for UltraChat and 46–120 for GSM8K;
all outputs reach the 64-token cap. This does not establish long-response,
long-context or fresh final-test performance.

Independent reconstruction passes all 960 timed generations and 1,576 evaluation/
reference sequences, checks 3,584 emitted-token positions and 3,072 aligned full
target-logit hashes, and proves general/cheap trajectories identical. Eighteen
local and eighteen GPU-host checks pass. Model-file pins are unchanged; all 1,901
backup files match remote hashes. Original data and workspace changes are intact.
Evaluation 8581045 completed in 39m56s; CPU pin audit 8581059 in 25s. Total GPU
allocation across attempts was 51m01s, below the 80-minute budget. Queue empty.

Retain first-block editing as an experimental option, especially for math prompts
with graph reuse. The next useful milestone is fresh confirmation on longer
responses with measured cache behavior or cheaper shape reuse, before more
editor training. That follow-up has not started. Serving-fidelity remains open
for union/concurrency and fresh confirmation.

[Full measurements, economics, length groups, setup and artifacts](docs/research/refinement-2026-11-25/CHEAP_SELECTOR.md).
Runtime source: e210c22. Raw evidence: `~/.local/share/wayfinder-artifacts/selector-8581045/`;
completed collection: `selector-8580943/`; preserved failed startup: `selector-8580929/`.
