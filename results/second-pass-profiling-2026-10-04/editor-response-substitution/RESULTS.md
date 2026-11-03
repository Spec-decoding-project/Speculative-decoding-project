# Perfect input-response imitation alone does not repair this editor

The small editor's near-absent response to an alternative first guess is real, but correcting that response alone does not fix its competitive proposal deficit. **Do not fund paired-input training solely on the previously measured response mismatch.** The experiment leaves the second-pass compression objective active; it does not establish that a better-trained editor cannot work.

## What was changed

Use all 2,048 saved development contexts from 64 responses. The existing teacher and student each have scores for the original first guess and for the original drafter's second-ranked first-token candidate. Separate each model's original-input scores from its score change after that input changes.

In the decisive intervention, keep the student's original-input scores and substitute the teacher's exact score change at the alternative branch. The reciprocal intervention keeps the teacher's original-input scores but supplies the student's score change. The candidate set, probability mass, original-prefix preservation, and existing 30-node proposal procedure stay fixed. This procedure is a measurement control, not a new tree research direction.

Substituted teacher computations are privileged expensive information. These artificial combinations are not trained students, realizable performance bounds, or predictions of what paired training will learn.

## Results

Positive modeled savings below means more economical progress **assuming every condition has the same complete-round cost**. No runtime or generation was measured.

| Scores used | Total covered positions | Modeled savings vs ordinary same-budget proposals |
|---|---:|---:|
| Ordinary first-draft scores | 12,818 | reference |
| Existing student, both inputs | 12,351 | -3.825% |
| Student original-input scores + exact teacher input response | 12,390 | -3.487% |
| Teacher original-input scores + student input response | 13,246 | +2.839% |
| Full teacher, both inputs | 13,398 | +4.107% |
| Ordinary first-draft scores + teacher input response | 12,872 | +0.337% |

Relative to the existing student, replacing only its response provides 0.318% modeled benefit, descriptive 95% interval -0.021% to +0.660%. Replacing its original-input scores instead provides 6.346%, interval +5.331% to +7.376%. The first intervention remains below the ordinary comparison in both math and instructions; its overall deficit interval is -4.353% to -2.607%.

The complete teacher's response still matters: restoring it after the teacher-original/student-response hybrid gives 1.291% additional modeled benefit. It is not enough to fix the student's original-input prediction errors in this intervention. The effects interact and are not an additive decomposition of causes.

The unchanged-tail controls confirm the pattern. Reusing the student's original scores on both branches covers 12,350 positions; its actual alternative-input computation adds just one pooled position. Reusing the teacher's original scores covers 13,247; its alternative-input computation increases that to 13,398.

## Repairs, losses and economics

The student gains 224 positions and loses 691 relative to ordinary proposals. Giving it the exact teacher response changes those figures to 279 gained and 707 lost. Thus it recovers some useful choices but also adds losses; it does not restore the ordinary scorer's strengths. The complete teacher gains 973 and loses 393. All conditions preserve every original draft prefix by construction: these losses are relative to the ordinary alternative-proposal set, not rejection of an original prefix that was previously retained.

At the observed progress ratios, the student with perfect teacher response would need complete rounds about **8.20% cheaper** than ordinary proposals to model the goal's 5% savings. The complete teacher would need about **0.93% cheaper** rounds. No matching runtime data establish either condition. Historical two-chain editor timings cannot be assigned to this different proposal policy.

## What this explains—and what it does not

This rules out a specific sufficient explanation for the current failure: "the student would be competitive if only it reacted to the alternative first guess like the teacher." Under the controlled score substitution, that change alone is insufficient. It provides a concrete reason to defer that proposed training experiment.

The larger measured deficit is reproducing useful teacher predictions at the original input. Calling that a deficit does not explain its underlying cause. Inputs, pretrained computation, representation, fitting and generalization remain possible causes. Earlier selected-case fitting establishes that some corrections are jointly learnable; earlier loss, context, optimization, fresh-data and candidate-distillation studies did not produce reliable competitive transfer. Do not repeat those as if untested. The new result also does not prove paired training could not improve original-input predictions; its wider effects were not simulated here.

**Next research decision:** retain learned second-pass compression, but require the next proposed fit to address original-input useful imitation/preservation with new evidence. Do not select paired-input teaching, another generic loss, or a wider random head just from this test. The pending full-proposal-teacher option remains a separate untested teacher-design question, not automatically justified by the student's failure. Neither another fit nor a new spending request is selected by this report.

## Checks and scope

- Student original/alternative score differences reproduce the prior CPU diagnostic across all 2,048 contexts; weights remain unchanged.
- Complete teacher, teacher unchanged-tail, and ordinary endpoints exactly reproduce the previous full panel.
- An independent explicit-frontier enumeration reproduces 1,024 trees: all eight conditions on 128 hash-selected contexts. All 16,384 endpoints are independently recounted, and original-prefix inclusion and prefix closure are checked.
- Sources and output scores are retained for replay. Student arithmetic is CPU FP32; teacher scores are historical native BF16. Prior CPU/native mismatches remain; no native-equivalence claim.
- Development responses have been repeatedly exposed. Bootstrap resamples responses within domains; its intervals are descriptive and do not cover training-seed uncertainty.
- Initial local invocation stopped before completing its first context because a comparison reconstructed saved values in the wrong dtype. Explicit float64 reconstruction repaired the check without changing tolerances or conditions; the failure is preserved.

Successful calculation completed in **90.79 seconds on CPU**, exit 0. No GPU work, training or additional spending. **$3.50 remains** under the existing authorization. Goal active and unmet.

Evidence: `PROTOCOL.md`, `results.json`, `rows.json`, `scores.json`, `per-prompt.json`, `sources.json`, `attempt1-failure.json`. Reproduce with `analysis/audit_editor_response_substitution.py` using the existing local artifacts.
