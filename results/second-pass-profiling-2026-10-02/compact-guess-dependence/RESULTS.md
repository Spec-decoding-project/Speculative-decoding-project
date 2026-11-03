# Frozen editor: dependence on explicit first-token feedback

## Finding

The saved probability-trained compact editor did not gain useful prefix progress from its explicit first-guess input on this sample. Removing or swapping that input changes some output tokens, but preserves the total 33 added prefix positions. This is adverse evidence for attributing this editor's measured benefit to the explicit feedback path. It is not evidence that token feedback cannot help a properly trained editor.

## Comparison

128 fixed contexts, two anchors from each of 64 repeatedly exposed development prompts. One frozen checkpoint, original first-pass states and original output first guess held fixed. Three input conditions: actual first-guess embedding, a zero embedding, and another prompt's first-guess embedding. Donors were assigned by a within-domain permutation independent of outcomes, preserving the input-token distribution. The substituted token changed in 125 of 128 contexts.

| Explicit input | Added prefix positions | Repaired blocks | Damaged edited prefixes | Equal-prompt union progress gain |
|---|---:|---:|---:|---:|
| Actual first guess | 33 | 13 | 28 | 5.451% |
| Zero input | 33 | 14 | 28 | 5.494% |
| Shuffled first guess | 33 | 13 | 28 | 5.592% |

Union progress retains the original branch as fallback. Damage counts concern the edited branch; they are not subtracted from union coverage. This calculation does not charge the additional verification or editor runtime and is not generation speedup.

Actual-minus-zero equal-prompt contrast: -0.044 percentage points, descriptive prompt-bootstrap interval [-0.293, +0.162]. Actual-minus-shuffled: -0.141 points, interval [-0.586, +0.162]. Identical aggregate added positions can coexist with different prompt-weighted gains because the affected prompts have different denominators.

Zeroing changes 41 of 1,763 valid editable tokens in 34 contexts; shuffling changes 50 in 35 contexts. In each comparison, actual input improves union progress on one context, harms it on one, and ties on 126. The network responds to the input, but that response has no demonstrated useful advantage here. Do not describe the input as mathematically ignored.

## Numerical scope and verification

CPU FP32 reconstruction differs from saved native BF16 editor outputs at 55 of 1,763 tokens. Union progress differs on two contexts, adding four positions overall. All intervention conditions use the same CPU model, arithmetic and fixed first-pass states; original GPU results remain unchanged. Runtime was 123.35 seconds, with zero cloud cost.

An independent saved-token recount verified every original/edited prefix length, repair, damage, union count, unchanged first output positions, donor mapping and within-domain token-frequency preservation. Input artifacts and the checkpoint were hash-checked. These checks establish internal consistency, not native CUDA equivalence.

## Interpretation and next decision

1. The full teacher's matched training comparison previously supported a benefit from explicit feedback. That does not establish that this small student learned the same mechanism.
2. The student's weak capture of feedback-model-only teacher extensions is now accompanied by no observed progress advantage from its correct explicit input. This sharpens the missing step: learning useful conditional corrections, rather than merely generating different corrections.
3. First-pass states still encode the predictions from which the first guess was selected. Removing the explicit embedding does not erase that information. The proposed feedback path is a computational shortcut, not necessarily an additional independent source of information.
4. Zero and shuffled inputs are inference interventions on a model trained with real inputs. A matched trained no-feedback control could learn differently. These results do not establish general necessity, sufficiency, an information limit, or equivalence between training procedures.
5. No new training, architecture change or GPU replication is automatically justified. Before another fit, its training design must specify how correct explicit feedback should change useful predictions, and include a matched way to test that contribution. A generic increase in token accuracy alone cannot validate that mechanism.

Evidence: [results.json](results.json), [rows.json](rows.json), [donors.json](donors.json), [verification.json](verification.json). Implementation: `analysis/audit_compact_guess_dependence.py`; independent recount: `analysis/verify_compact_guess_dependence.py`.
