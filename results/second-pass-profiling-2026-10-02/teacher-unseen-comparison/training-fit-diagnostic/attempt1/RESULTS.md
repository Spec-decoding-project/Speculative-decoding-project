# Useful training corrections transfer poorly to fresh responses

The same frozen compact editors add approximately 15% coverage on a sample of their training responses and approximately 3% on the completed fresh panel. They learned useful corrections on training examples. This makes weak transfer a substantial obstacle, without proving that memorization, dataset size, capacity or optimization alone explains it.

All 64 hash-selected training responses and 2,048 contexts completed. No optimizer updates or new text generation occurred. All six saved-prediction controls reproduced exactly, checkpoint hashes remained unchanged, and an independent local audit recounted every prediction against the authenticated native references.

## Matched checkpoints on two different panels

| Training method | Training sample coverage gain | Fresh coverage gain | Training repairs | Training damaged edited prefixes |
|---|---:|---:|---:|---:|
| Correct tokens only | 15.06% | 2.79% | 646 | 269 |
| Teacher-correct weighting | 14.67% | 2.91% | 622 | 230 |
| Teacher probabilities | 15.43% | 2.77% | 615 | 240 |

Coverage is the additional consecutive matching prefix from retaining both the original and edited proposals. It is not generation speed. Each panel has 32 math and 32 instruction responses; all original and edited predictions remain in the results.

For teacher probabilities, training coverage is 12.84% on math and 18.02% on instructions, versus 3.36% and 2.19% on the fresh panel. The supplementary domain-stratified prompt-bootstrap interval for its training coverage is 14.00–16.91%. The interval for the training-minus-fresh difference is 11.13–14.22 percentage points. These intervals are conditional on one fitted checkpoint, not training-seed uncertainty. Training and fresh prompts were resampled independently because they are different responses.

## Training success is not complete teacher compression

The full second-pass teacher adds 9.96% coverage on the training sample and 10.92% on the fresh panel. The compact editor can exceed the teacher on training examples because correct-token supervision also teaches corrections outside the teacher's own successful continuations.

The probability-trained editor adds 1,419 positions on this training sample. Of those, 447 overlap the teacher's 1,094 added positions; 972 lie outside that teacher extension. It therefore retains only 40.9% of the teacher's added positions even here. The corresponding retention is 38.4% for ordinary training and 43.1% for weighted labels.

Consequently, a 15.43% training gain must not be described as faithful compression of the teacher, proof of sufficient capacity, or proof that only generalization remains. Useful fitting and faithful reproduction of the teacher's corrections are different measurements. The student both misses teacher corrections and learns other training-specific improvements.

## What changes next

The current evidence does not justify a larger editor merely because fresh performance is weak. It also does not justify claiming that the architecture or input information is sufficient for all useful teacher corrections. This diagnostic evaluated existing weights; it did not vary capacity or expose a solution to every missed case.

Similarly, the training/fresh gap does not prove that collecting more examples, adding regularization or extending training will solve it. The completed equal-size old/new-data comparison already failed to improve fresh coverage materially. On exposed development data, the new probability-trained editor's three epoch results were 3.06%, 2.93% and 3.46%, so a claim of a clear monotonic validation collapse would also be unsupported.

Preserve the negative fresh comparison and this diagnostic. The next learning choice must distinguish a specific explanation for poor transfer or incomplete teacher capture; another unmotivated size, loss or duration change is not established by these results. No follow-up training or speed test was launched.

## Scope and recovery

The sample was chosen by a fixed hash without consulting its editor outcomes and contains 64 of the 384 new training responses. The training and fresh panels differ in prompt difficulty and available corrections. The similar teacher coverage provides context, not a matched causal experiment. These measurements remain fixed-context observations with original proposals retained, not generated output fidelity or latency measurements.

Execution through recovery and cleanup took 4.20 minutes. All 73 artifacts, totaling 8,319,996 bytes, were recovered and authenticated. Compute is stopped with zero tasks, and the owned scratch volume was deleted. The original research archive remains retained. The full $1.25 allowance is conservatively held pending billing, leaving approximately $15.29 unallocated and no active reservation.

Evidence: [independent analysis](analysis.json), [descriptive uncertainty](descriptive-uncertainty.json), [execution and shutdown](execution.json), [recovery identities](recovery.json), [frozen protocol](../frozen/PROTOCOL.md), and [completed fresh comparison](../../final-attempt1/RESULTS.md). Reproduce the main token audit with `analysis/analyze_teacher_editor_fit_diagnostic.py` from the repository root.
