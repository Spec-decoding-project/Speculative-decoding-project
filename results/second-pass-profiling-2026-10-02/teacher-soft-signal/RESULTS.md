# Did probability distillation provide a different teaching signal?

Yes, but the added signal was not concentrated at the original first error. This rejects the blanket explanation that the teacher-KL arm was identical to the weighted-label arm; it also limits how strongly that arm tested richer first-error supervision.

The audit uses the previously hash-selected 64 training prompts and all64development prompts, two anchors each: 1,692 training and1,763development editable rows. Teacher states and readout were authenticated. CPU FP32 probabilities are reconstructed approximations, not recovered original CUDA BF16 probabilities. Native saved correctness masks remain fixed.

## Exact objective interpretation

On teacher-correct positions, the implemented objective is `CE(label,p) + 0.5 KL(q||p)`. Up to a parameter-independent constant this is1.5times cross-entropy against `(one_hot(label)+0.5q)/1.5`. Its target probability on all wrong alternatives together is `(1-q_label)/3`. On teacher-incorrect positions the objective stays ordinary CE.

The actual loss implementation passes an independent float64 gradient check, including mixed eligible/ineligible and invalid rows. Relative to the eligible-CE control, each eligible row contributes the logit-gradient difference `0.5(one_hot(label)-q)/N`. Its L1 magnitude before the all-valid normalization is `1-q_label`. These formulas are exact for supplied scores; they do not make reconstructed CPU probabilities exact historical training scores.

## Sampled training evidence

| Position category | Valid rows | Teacher-correct rows | Mean teacher probability on correct label, when eligible | Mean alternative mass in blended target, when eligible |
|---|---:|---:|---:|---:|
| Earlier correct prefix |260|260|98.01%|0.66%|
| Original first error |88|87|91.07%|2.98%|
| Later position |876|569|76.26%|7.91%|
| Block with uneditable first guess wrong |390|55|44.15%|18.62%|
| Originally complete block |78|78|98.71%|0.43%|

At eligible first errors, median teacher probability is96.27%; only31.0% exceed99%. Thus near-certain targets are common, but it would be wrong to claim that every first-error target collapsed to a hard label. Across all eligible training rows, mean teacher-label probability is82.86%, giving5.71% mean alternative mass in the blended target.

Summing per-row teacher-specific logit-gradient L1 differences in this sample, before actual minibatch normalization, places4.32%at first errors,75.15%at later positions,17.09%in blocks whose fixed first guess is wrong,2.88%on earlier prefixes and0.56%on originally complete blocks. These are not parameter-gradient contributions, optimizer update shares or causal estimates. Shared parameters, gradient cancellation and changing batch denominators prevent that interpretation. Later-position supervision can still help earlier corrections through shared weights; it is not automatically wasted.

## Development contrast and numerical scope

On sampled development first errors, the teacher is correct at29/100positions, versus87/88in training. Among eligible development first errors, its mean correct-label probability is73.88%, leaving8.71%alternative mass in the blended target. This shows a substantial change in the teacher's correctness and uncertainty on these sampled first-error cases; it does not isolate memorization, data difficulty or a causal training failure.

CPU winning tokens differ from saved teacher winners at30/1,692training and25/1,763development rows. There are zero such disagreements at the88training first errors and two at the100development first errors. Rounding reconstructed FP32 logits to BF16 changes the correct-label probability by0.0023on average over training rows and0.0021on development rows. This simple rounding sensitivity is not a simulation of every CUDA numerical difference.

## Decision

Do not dismiss the KL comparison as having no distinct signal, and do not automatically increase its temperature. The evidence says that much of its additional signal concerned alternatives after the first error, while its immediate repair targets were usually close to ordinary correct-label training. That is a useful specification of what we tested, not proof that first-error weighting or softer targets will solve the learning problem.

The previous retained-prefix objective comparison already tested one way of reallocating label supervision and did not advance performance. A proposal to reweight richer teacher supervision would need to explain what additional decision-relevant information becomes available, and distinguish that effect from merely changing gradient magnitude. No new loss sweep, fit or GPU run is selected by this audit. Existing failed recipes remain closed; research remains active.

All256contexts completed locally in253.9seconds. Additional cloud spending:$0. `verification.json` records the exact implemented-gradient check and independent probability recount. `gradient-difference-allocation.json` preserves the scoped allocation calculation; `rows.json` and `results.json` preserve the observations. Reproduce with `analysis/audit_teacher_soft_signal.py` and `analysis/verify_teacher_soft_signal.py`.
