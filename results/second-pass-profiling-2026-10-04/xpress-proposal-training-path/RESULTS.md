# A trainable second pass through the released refiner is locally validated

The stronger-teacher preparation now has a working gradient path, including released refiner weights and saved real states. This is implementation evidence, **not** evidence of a learned correction advantage. No GPU or real-model training was launched.

## What the teacher would do

Keep a released xPress first drafter and its small refiner frozen. A separate trainable copy of the backbone processes the initial draft again. Compare two matched copies: one sees only the original first guess; the other sees the whole original proposal. Both use the same frozen six-step refiner after their backbone calculation and preserve the first guess in the resulting proposal. The target, vocabulary weights and initial draft remain unchanged.

This tests whether processing the completed proposal produces useful corrections beyond extra training and first-token conditioning. The unchanged model's existing cheap refinement remains a required control. A positive result would supply a possible full second-pass teacher to compress. Merely training or compressing the existing xPress head is not the objective.

The strongest counterargument remains that repeated guesses add no useful information for this checkpoint, encourage copying wrong tokens, or need much more training data. The local checks do not resolve it.

## Why the training calculation is valid

The refiner makes discrete greedy token choices at each step. Ordinary autograd does not differentiate through those integer choices. For a loss on the sixth step's scores, its first five step graphs can therefore be discarded. The final score still depends differentiably on the backbone states through both the original vocabulary scores and the refiner's hidden-state input.

The implementation computes all six steps, preserves the same greedy decisions, and differentiates the exact final score calculation. It introduces neither soft token feedback nor a straight-through gradient estimator. This gradient is piecewise: it does not predict discontinuous changes at an argmax boundary.

## Checks completed

Eleven tests passed in 20.84 seconds, including:

- Exact score, token and complete input-gradient equality with the literal full autograd calculation at one, two and six steps, in FP64 and BF16.
- Agreement with the existing author-math clamped refiner and a finite-difference check inside a region with unchanged intermediate choices.
- Actual native-backbone updates under both input conditions while frozen first-pass/refiner weights remain unchanged.
- Reference-answer perturbations leave predictions unchanged; frozen shared-weight requirements are enforced.
- Two matched updates through the new collected-data training loop, equal supervision, and rejection of incorrectly substituted old-DFlash proposal provenance.

The initial test run had one failure: the random tied-embedding model guessed the mask ID at every later position, so the two test conditions were identical. The fixture now explicitly supplies a non-mask synthetic proposal and asserts that the input intervention is nondegenerate. No production computation or tolerance changed to repair the test.

## Released-weight check

Four contexts were selected by context-identity hash from the completed sixteen-context screen, without selecting repair outcomes. Using the released BF16 refiner, actual Qwen vocabulary weights and saved second-pass states:

| Context | Reproduces saved output | Exact full-score match | Exact full input-gradient match |
|---|---|---|---|
| GSM8K 5091, anchor 8 | yes | yes | yes |
| GSM8K 3387, anchor 8 | yes | yes | yes |
| UltraChat 12468, anchor 24 | yes | yes | yes |
| UltraChat 12677, anchor 8 | yes | yes | yes |

Gradients were finite and nonzero; the refiner and output weights stayed unchanged. This checks the gradient entering the backbone, not a released full-backbone optimizer update. Native CUDA behavior and the complete training/resource path remain unverified.

The calculation took **1,938.89 seconds (32m19s) on CPU**. Do not repeat model-scale BF16 checks on this CPU by default: they are materially slower than the tiny tests and should be folded into a bounded GPU attempt. No paid compute was used.

While the released check was running, the module gained a separate `fit_collected` helper and an unused `time` import. The measured `refine_training_scores` function and the refiner math were unchanged. The check's module hash is of the final on-disk file; it is not a complete snapshot of the earlier imported module. The later eleven-test run covers the final module, including the new helper. Preserve this provenance distinction.

## Next executable comparison

The native fitting loop now accepts only authenticated initial proposals from the stronger released checkpoint and matching metadata. It uses the existing optimizer and weighted suffix loss. It rejects old DFlash proposals, development metadata in training, and trainable shared readout/refiner weights.

Before allocating a GPU, finish one controller/worker bundle with:

1. Fresh external job inventory and a reservation within the existing **$3.50** balance.
2. Collection of this checkpoint's training proposals from saved committed target features; no new target generation needed. Keep the original train/development split.
3. Matched first-guess/full-proposal fits, unchanged and extra-cheap-refinement controls, correct-prefix preservation and the established stronger proposal comparisons. Choose a fixed dose before seeing outcomes; a short unfinished learning curve cannot establish mechanism failure.
4. A measured admission check for the complete two-fit path, including collection, evaluation, checkpoint saving, recovery and shutdown. The old $3.366 extrapolation does not establish that this refiner-aware path fits.
5. No automatic retries; interrupted execution must preserve evidence and cannot silently restart the fits.

This is preparation for a useful teacher, not an already justified new student fit. Learning benefit, competitive correction headroom, native output-preserving generation and the requested 5% latency improvement remain unproven. Existing failed recipes and the stronger-reference gap remain preserved. Research goal active; no spending or reservation; $3.50 unchanged.

Evidence: `PROTOCOL.md`, `released-gradient.json`, `initial-test-failure.json`; implementation `experiments/xpress_full_proposal_teacher.py`; tests `experiments/test_xpress_full_proposal_teacher.py`; released check `analysis/check_xpress_released_training_gradient.py`.
