# Does explicitly teaching the second-pass change create a different learning problem?

## Decision

Do not launch a new fit solely under the rationale "teach the correction rather than the answer." Our editor already adds a learned score correction to frozen first-pass scores. Squared-error regression on the teacher's score change is exactly final-score regression with this parameterization. It is different from the current probability KL objective, but it supplies no additional teacher information by itself.

This closes a formulation-only justification, not all regression-based learning. A changed metric or weighting could change optimization; that requires evidence for the particular change. It does not establish that regression would fail, or that the current probability objective is optimal.

## Exact calculation

Let b be frozen first-pass vocabulary scores, r(theta) the learned correction, and t the full second-pass scores. Student scores are b+r(theta).

Residual regression: Lr = ||r(theta) - (t-b)||^2.

Final-score regression: Lf = ||b+r(theta)-t||^2.

These are the same expression, so they have the same parameter gradients, provided masks, weighting, normalization and frozen inputs match. Subtracting the vocabulary mean from both errors preserves this equality and removes sensitivity to a common score offset. Matching the underlying hidden-state change is a different metric and representation constraint; it must not be called equivalent to score matching.

Checks on the actual CompactSecondPass module at reduced dimensions covered eight initializations, nonzero output projections, mixed valid masks, and centered/uncentered variants. Maximum parameter-gradient discrepancy was 1.78e-15 in float64. This is a numerical check of an algebraic identity, not evidence about real-model learning.

## Probability of a score change is not the teacher's next-token probability

Applying softmax to t-b creates a different distribution from softmax(t). Example: b=[6,0], t=[5,1]. The teacher assigns 98.20% to token 1; softmax(t-b) assigns 88.08% to token 2. Token 2 improved relative to its original score but is still not the teacher's preferred token.

At zero correction, KL on final probabilities has gradient [0.01551,-0.01551]; KL on residual probabilities has [0.38080,-0.38080]. Thus residual-softmax matching changes the training geometry substantially. With unlimited ability and exact residual distribution matching, adding b can still recover teacher probabilities up to a constant; the counterexample does not show it is intrinsically wrong. It shows that residual probabilities must not be interpreted as correct-token probabilities or a guaranteed better acceptance objective.

## Prior work checked

- [ResKD, Li et al.](https://arxiv.org/html/2006.04719v2), section 3.2, explicitly trains an added lightweight network against the logit residual using L2 and sums its logits with the earlier student. Its image-classification results establish precedent, not a speculative-decoding guarantee. A residual student alone is not our novelty.
- [Concrete Score Distillation, official implementation](https://github.com/aailab-kaist/CSD), describes matching pairwise logit differences with weights while respecting common-offset invariance. This establishes prior art for score-gap supervision. It does not demonstrate improvements for our frozen residual editor or prefix-progress objective.

These sources were inspected on 2 October 2026. The claims here concern the specified formulations, not a comprehensive novelty clearance of every possible conditional editor.

## Connection to our results

Our current module already computes b+r(theta); the three-arm comparison changed its objective, not this residual structure. The earlier projected-teacher audit also found that decreasing centered-score reconstruction error by 3.95% did not improve prefix progress on that fixed sample. That oracle projection experiment did not train a network and is not a rejection of learned score regression. It nevertheless cautions against treating a smaller score error as the practical goal.

The actionable research requirement is to name what changes beyond algebra: which deployment-relevant predictions receive different supervision, why that should repair early errors while preserving already-correct prefixes, and what matched comparison isolates it. A new title for the same target or a generic loss swap does not answer those questions.

No model training, GPU job or spending occurred. Evidence: checks.json; analysis/check_residual_distillation_identity.py; experiments/compact_second_pass_prototype.py; experiments/controlled_second_pass_distillation.py.
