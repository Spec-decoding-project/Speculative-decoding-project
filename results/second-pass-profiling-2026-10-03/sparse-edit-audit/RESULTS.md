# How many changes earn the second pass's useful progress?

Completed locally for $0 on 2,048 saved contexts from 64 reused development responses. Independent enumeration verified every top-four single-edit oracle, teacher edit-distance count and label-free policy outcome.

## Observed structure

The full second pass adds 1,136 matching positions in 353 repaired contexts. Correcting only one originally wrong token would preserve 706 of those positions (62.15%). Two corrections would preserve 891 (78.43%). In 222 of the 353 repaired contexts, one correction is sufficient to preserve the entire teacher gain. This is a hindsight minimal-edit calculation, not evidence the model can identify that correction before verification.

When the correction must come from the original drafter's top four choices, one edit still covers 662 teacher-added positions (58.27%). On the complete panel, the best such single edit chosen with knowledge of the answer gives +18.72% coverage. That oracle is not a deployable result or proof of learnability.

## Label-free references

All retain the original branch and keep its first guess fixed.

| Second proposal | Extra matching-prefix coverage over original |
| --- | ---: |
| Existing mixture editor | 3.46% |
| Change the smallest-margin position to its runner-up | 2.50% |
| Change one position using the expected-progress proxy below | 7.05% |
| Full trained second-pass teacher | 10.21% |
| Ordinary top-four 30-node tree | 14.16% |

The expected-progress rule exceeds the editor by +3.59 percentage points (descriptive paired interval +2.67 to +4.53), with positive differences in both domains. Its own coverage interval is +6.02 to +8.12%. These are reused development observations and multiple comparisons, not a fresh confirmation. No runtime is measured.

For edit position j, the rule scores the probability of reaching j along the original guesses, choosing the runner-up there, and continuing along the unchanged suffix. It chooses the largest score. Under independent per-position probabilities this exactly maximizes expected added prefix length within the one-substitution family. We verified the identity on 30 exhaustively enumerated small distributions. Original drafter marginals need not equal those conditional target probabilities; the formula is a model-based proxy, not an actual calibrated probability or a proof the heuristic generalizes.

The rule receives no correct labels, validity mask, teacher outputs or editor outputs. Only its subsequent evaluation uses the recorded continuation. Changing an arbitrary later position after a missed earlier token provides no progress, which is why merely choosing the smallest margin can fail.

## Research implication and next diagnostic

This establishes that much useful second-pass behavior admits a sparse description. It does not establish that sparse behavior is easy to learn, that one edit suffices for every useful case, or that restricting the current editor to one of its own choices solves the problem: even a perfect selector among its existing edited tokens gives only 4.70% coverage / 4.38% free-cost modeled savings.

Before training a new selector, ask whether the expensive teacher actually supplies a better deployable one-edit score than the original drafter. Compare original versus teacher probabilities on the identical original top-four action set, with the same expected-progress construction and output budget. The saved teacher states permit native readout without new teacher transformer passes. A positive result would establish a useful teacher decision to try compressing; it would not establish student transfer. A negative would reject this particular teacher-policy target before another fit. This keeps the focus on replacing useful second-pass computation rather than another arbitrary head or loss sweep.

## Prior art and limits

[OPT-Tree](https://arxiv.org/abs/2406.17276) already selects draft structures using expected acceptance; this restricted two-chain calculation is not claimed as a novel objective. [D²SD](https://arxiv.org/abs/2606.04446) uses confidence-guided recovery prefixes and a second drafter for alternative continuations. Its existence supports retaining an ordinary resampling reference, not assuming our sparse edits are novel. [DEdit](https://arxiv.org/abs/2609.38510) already studies repairing drafts while preserving correct predictions; sparse preservation alone is not a novelty claim.

[AdaFlash](https://arxiv.org/html/2607.19223v1) uses clipped reverse-KL plus hard labels in on-policy target adaptation. That result does not establish that changing divergence will make our offline compact student recover useful teacher decisions. Our previous retained-prefix and teacher-only imitation experiments remain closed; this audit supplies no automatic justification to repeat them.

[Results](results.json) · [Independent verification](verification.json) · [All context outcomes](rows.json). No GPU run or additional spending in this audit.
