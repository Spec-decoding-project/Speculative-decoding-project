# Centered-score projection does not improve useful progress

The fixed comparison completed on all 128 contexts from 64 previously exposed development prompts. It took 257.7 seconds locally; additional cloud spending was $0. No training or generation ran.

The teacher, rank256 output basis, original states and vocabulary readout were unchanged. Only the coefficients used to project the teacher's state change differed: Euclidean hidden-state projection versus least-squares centered vocabulary-score projection. Both receive the teacher's per-example state and are therefore oracle diagnostics, not cheap deployable editors.

| Outcome | Hidden-state projection | Centered-score projection |
|---|---:|---:|
| Extra matching-prefix positions with original retained | 46 | 43 |
| Repaired blocks | 21 | 21 |
| Damaged edited prefixes | 10 | 12 |
| Full-teacher extra positions retained | 39/71 | 38/71 |
| Equal-prompt union progress gain | 9.09% | 8.82% |
| Modeled maximum total round overhead for 5% savings | 1.12% | 0.91% |

Centered vocabulary-score squared error fell 3.95%, with the rowwise least-squares improvement checked. Nevertheless, the paired progress contrast was -0.267 percentage points, with a descriptive prompt-bootstrap interval of [-3.23, +2.57]. Five prompts improved, five worsened and 54 were unchanged. This does not establish equivalence or general inferiority; it supplies no demonstrated advancement of useful progress.

## Verification

The 256-dimensional normal system had condition number21.36 and relative solve residual1.05e-15. Two independently accumulated Gram calculations agreed to1.98e-8 relative error. Small fixtures agree with direct least squares, improve its prescribed objective and preserve invariance to a common vocabulary-score shift. No regularization or development-selected damping was introduced.

Every token sequence in the five unchanged controls reproduced the earlier CPU evaluation exactly. A separate implementation recounted all prefix lengths, repair/damage counts, overlap and prompt-level gains from the saved sequences. This establishes repeatability of the CPU comparison, not agreement with native BF16 GPU execution. The preceding original/teacher native readout differences remain limitations.

## Decision

Close this exact same-basis projection-metric change. Do not launch coefficient training or claim that improved hidden/logit reconstruction implies useful decoding. The next proposal needs evidence about actual correction decisions and cost; an automatic sequence of reconstruction objectives would repeat the same unresolved premise.

The result does not bound all coefficients in this basis. Least-squares coefficients are optimal for their own score objective, not accepted-prefix length. It also does not reject learning a compact editor, decision-sensitive projection, or improved output bases generally. No basis, model or historical outcome was overwritten, and research remains active.

Evidence: `results.json`, `verification.json`, `rows.json`, and `projection.pt`. Reproduction: `analysis/audit_score_metric_projection.py` and `analysis/verify_score_metric_projection.py`.
