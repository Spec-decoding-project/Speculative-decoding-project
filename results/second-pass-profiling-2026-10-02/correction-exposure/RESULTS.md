# Compact-editor correction exposure audit

## Question and result

Does the apparent training volume mostly repeat the same corrections, and is weak transfer simply a lack of previously seen token substitutions? The saved evidence does not support either simple account.

All 448 response banks were read and authenticated against the feature manifest. Labels and original predictions were checked against historical metadata. No model was trained, no new prompts were exposed, and no paid work ran. Elapsed analysis: 52.47 seconds.

| Quantity | Training | Development |
|---|---:|---:|
| Responses | 384 | 64 |
| Draft contexts | 12,288 | 2,048 |
| Valid editable label occurrences per epoch | 165,351 | 27,580 |
| Distinct response positions covered | 119,876 | 20,397 |
| Repeated-occurrence fraction | 27.50% | 26.04% |
| Teacher-repaired contexts | 8,587 | 353 |
| Distinct response positions at their first rejection | 7,984 | 342 |
| Distinct wrong-token → correct-token pairs at those rejections | 6,462 | 321 |

These editable-position counts exclude the known root and immutable first guess. They must not replace the historical 132,238 count of generated response tokens, which measures a different quantity. Three training epochs repeat this fixed supervision; they do not create additional distinct response positions. Responses, rather than individual overlapping labels, remain the natural unit for held-out evaluation.

The ten most frequent substitution pairs account for only 2.53% of training teacher-repaired contexts. Thus those repairs are not dominated by a handful of literal edits. Identical substitutions can still occur in very different contexts; distinct substitutions can share a learnable linguistic rule.

## What literal overlap predicts descriptively

Of the 353 teacher-repaired development contexts, 140 use a wrong-token → correct-token pair that occurred at a teacher-repaired training first error; 213 do not. This definition does NOT mean the constituent tokens or linguistic patterns were absent from training.

| Development subgroup | Contexts | Probability-trained editor repairs | Teacher extra positions captured |
|---|---:|---:|---:|
| Pair observed among training teacher repairs | 140 | 47 (33.57%) | 97 / 532 (18.23%) |
| Pair not observed there | 213 | 47 (22.07%) | 121 / 604 (20.03%) |

The editor repairs unseen literal substitutions, so a claim that it only memorizes a table of token replacements would be false. The seen subgroup has a higher block repair rate, but not higher retained-position capture. These groups differ in difficulty and repair length; the contrast is neither a causal estimate of data coverage nor proof of compositional generalization.

## Implication for the next training decision

Do not propose deduplication as a demonstrated remedy: repeating a response position at another anchor changes its visible context and draft predictions, so it is not necessarily a duplicate input. Do not propose a token-pair lookup as the research method. Do not infer a sufficient dataset size or a scaling law from one training size.

The verified conclusion is narrower and useful: the student learned across thousands of distinct correction sites yet retained little of the teacher's additional progress, including on seen substitution pairs. More varied data remains a hypothesis, not an established fix. To justify a data-scaling intervention, compare held-out correction gain versus distinct training responses while separately controlling repeated update exposure. Reusing the same teacher/data pairing would still leave teacher generalization as a confound. No such fit is authorized or launched by this audit.

## Verification and limitations

An independent metadata-only recount reproduces all editable-position totals. Every repeated response position has a consistent label. All feature files pass stored hashes and byte counts. Aggregate teacher progress matches historical 54,132 training and 1,136 development extra positions. The saved native student evaluation join reproduces 94 repaired teacher-repair contexts and 218 captured teacher positions.

This is a post hoc audit of reused data, not a fresh evaluation or a statistical estimate of effective sample size. No claim of runtime benefit, novelty, or a causal diagnosis follows. Source hashes and rows are preserved in results.json and rows.json; independent counts and student strata are in verification.json. Reproduce the primary audit with analysis/audit_compact_correction_exposure.py.
