# Whole-draft training comparison

The user explicitly requests training/testing ways to improve the editor while
giving it the complete DFlash-generated block. The existing whole-draft variant
already has that input; this experiment changes its learning objective. It
supersedes the previous pause on editor training for this bounded comparison.

Every arm starts from the identical original full-trained 40,750,336-parameter
whole-draft checkpoint. Target and DFlash remain frozen and are not loaded for
this experiment. Inputs stay token-only: up to 1,024 accepted context tokens,
the entire original 16-token block, then the anchor/first-proposal seed. Fourteen
suffix tokens are generated autoregressively. The preserved-first-proposal
restriction remains unchanged so the objective comparison has a matched scope.

## Fixed comparisons

1. **uniform:** continue ordinary token cross-entropy; matched training control.
2. **early:** multiply suffix-position loss by exp(-j/7), j=0..13.
3. **protect_repair:** use the early weights, multiply correct original-prefix
   positions by two and the original first-error position by four. Later
   positions retain their early weights. A naturally perfect proposal has all
   suffix positions counted as correct-prefix positions.

Normalize by the sum of weights over the effective batch. Labels and the first
error determine weights only during training, never inference. All generator
parameters train, unlike the earlier gate-only experiments. No capacity increase.

Use 8,192 distinct training-origin prompt/family blocks: 4,096 per source,
including 512 naturally perfect proposals per source. Choose scarce preservation
examples first, deterministically hash-ranked, then repairable examples. Exclude
all 512 prior training-origin gate-calibration prompt/families. Validate the
authenticated bank SHA, block alignment, provenance and split separation.
The source has enough distinct preservation prompts for this fixed allocation.

FP32 AdamW, peak LR 1e-4, weight decay .01, clip norm one, effective batch32,
microbatch1, seed0; 16-update warmup then cosine decay to 1e-5. Each arm receives
768 updates (three passes over the same 8,192 blocks), with identical ordering.
Score all 512 calibration blocks at initialization and every128 updates; select
highest free-running correct prefix, then higher editable-token accuracy. The
unchanged checkpoint is eligible, so deterioration is not hidden by forced
selection of a trained checkpoint. Save resumable checkpoints every128 updates.

After generator selection, fit a fresh 265-parameter linear COPY gate for each
selected generator using the same 1,536 gate-fit blocks and512 calibration blocks
as the earlier gate study. Use its existing decisive/reachable-prefix supervision,
weighted binary loss, AdamW .001/decay .01/clip1/batch256, at most2,000 updates or
200 epochs; calibrate every five epochs and stop after50 without improvement.
The generator stays frozen during this gate fit. Calibration was part of the
historical base-model training inventory, but is excluded from this fine-tuning
and gate optimization; development is excluded from all gradient updates.

## Evaluation and checks

Evaluate each selected generator both alone and with its newly calibrated gate
on the unchanged2,048 development blocks/512 prompts. Compare with original
DFlash, the original generator, and the exact earlier linear gate. Report
editable-token accuracy, prefix length, exact suffixes, gains/damage, initially
perfect and wrong-first cases, and UltraChat/GSM8K separately. Preserve raw
predictions privately and recount independently. These development prompts have
been repeatedly inspected; this is exploratory, not fresh confirmation. Do not
load the old reserved test, which was already inspected in November.

Require causal-mask/no-future-label checks, finite gradients, correct optimizer
ownership, cached/full inference agreement, checkpoint reload equality, and
gate-simulation/free-running prefix equality. Record source/dataset/checkpoint
hashes and retain every arm, including failures. Select weights only by calibration.

A useful result must improve correct-prefix length over the existing gated
editor (3.699707 on development) while at least matching original DFlash's
32.4289% editable-token accuracy. Report source-specific regressions explicitly;
do not call a one-source win a broad improvement. Provide paired prompt-bootstrap
intervals versus both incumbent and DFlash, with the multiple-comparison caveat.
If no arm meets both criteria, report that outcome without automatically
increasing parameters or launching a larger sweep.

Measure selected editor inference with warm CUDA graph reuse on fixed examples,
checking exact graph/eager outputs and separating setup. This measures editor
cost, not complete native throughput. A new native comparison would be needed
before claiming these trained weights accelerate DFlash.

## Resource and execution

One existing RTX A5000, four CPUs, at most16GiB host RAM, no model downloads.
Slurm wall cap two hours, internal experiment budget110minutes. Measure initial
update speed and report an ETA; stop cleanly with a resumable checkpoint if the
budget prevents completion. Job state, logs, checkpoints and results are stored
independently of SSH. Original corpora/checkpoints and earlier experiments stay
immutable. No heavy computation on the login node.
