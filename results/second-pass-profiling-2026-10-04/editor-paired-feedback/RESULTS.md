# The existing editor does not reproduce the teacher's conditional response

The frozen 3,024,384-parameter editor responds weakly and in largely different directions when the first draft guess changes. This identifies an unlearned part of the second-pass teacher's function. It does **not** prove that missing paired examples caused the historical failure, or that supplying them will solve it.

No training or GPU work occurred. The complete local calculation took 76.62 seconds; independent scalar recount and artifact checks passed. The existing $3.50 allowance is unchanged.

## Controlled comparison

All 2,048 saved development contexts from 64 responses are included. For each context, keep the original first-pass hidden states, student weights and readout fixed. Supply either the original first-token guess or the original drafter's second-ranked candidate. The previously recorded full teacher also received those two inputs under the same committed context and weights.

At every valid editable position, compare the two input conditions' score changes over the same original 16 candidates. Centering removes the arbitrary common log-probability offset. These are changes in relative preferences, rather than differences caused by normalization. The zero-response control predicts no change at all.

| Measurement, with responses weighted equally | Result |
|---|---:|
| Student response magnitude / teacher response magnitude | 3.15%, descriptive 95% interval 3.02–3.28% |
| Directional alignment, cosine | 0.0023, interval −0.0028 to +0.0075 |
| Squared response error / error of predicting no response | 1.00080, interval 1.00043–1.00115 |

Thus the student's response is small, and it is not consistently aligned with the teacher's. It is marginally worse on this particular score-change objective than predicting no response. **3.15% is not teacher-correction retention, accepted-token improvement, or speed.** The response can differ on vocabulary entries outside this candidate set.

The comparison does not show that the student ignores its input: in the 128-context full-vocabulary check, changing the guess changes 45 valid suffix tokens across 35 contexts.

## Useful teacher decisions

The prior full-panel teacher experiment found 71 contexts where recomputing after supplying a reference-correct second candidate improves the copied-tail continuation. At 68 of their first blocking positions, both the old wrong token and corrected token are in the recorded candidate support. These 68 cases span 44 responses.

Under the same input change, the student increases the correct-versus-old-wrong score margin in 33 cases and decreases it in 35. Every recorded teacher margin change in this group is positive, ranging from 1.0 to 33.36. The student's margin being positive against this one competitor does not imply that it wins against the full vocabulary or preserves its preceding prefix.

The separate full-vocabulary check uses the same 128 contexts selected by an earlier hash rule, without selecting new examples for this diagnostic. Nine have a reference-correct second first-token candidate. On these nine, teacher recomputation improves three continuations by four positions in total and worsens none; student recomputation changes no matching-prefix lengths compared with copying the new first token into its old tail. This small subset is explanatory evidence, not a useful-effect estimate for new prompts.

## What this means for compression

Earlier tests asked whether the student copies teacher outputs on the original input, or whether removing/shuffling its guess helps. This test asks a different question: **when the teacher changes its prediction because its draft input changes, does the student reproduce that change?** For this model, the answer is largely no on the measured candidate scores.

That makes paired-input supervision a specific candidate to assess. A training example could hold the saved first-pass computation fixed, show two allowable draft inputs, and provide the teacher's corresponding responses. Unlike renaming final-score regression as residual regression, this supplies an additional teacher evaluation at a different input. It could teach sensitivity that a model can avoid learning from one input per context.

However, three counterarguments remain decisive:

1. **This is partly outside the student's training support.** The original student saw the original guess for each saved state. Its failure on an alternative guess may be expected distribution shift rather than the cause of its failure on the original input. The original guess is itself determined by the saved states; a successful editor for that one-input policy need not use the explicit guess path at all.
2. **Conditional fidelity is not useful prediction.** The teacher also changes incorrect predictions. Better matching its input response could still leave original-input repairs, preservation and unseen-response transfer unimproved. Prior global-imitation and focused-rehearsal negatives remain relevant.
3. **The current teacher has limited competitive headroom.** Its two-condition policy previously gave 4.107% modeled savings over ordinary matched-budget expansion at zero extra cost. This diagnostic does not increase that opportunity. It does not justify spending the remaining budget simply to reproduce the same insufficient 5% point target more accurately. Smaller measured gains could still be useful, but must be compared with strong alternatives and charged for all work.

The result therefore selects a **function to test learning**, not a new architecture or a funded training run. It does not justify a model-size sweep or another generic loss change. A learning comparison would need training-only paired examples, the same student and matched teacher-call/update exposure, an ordinary correct-token control, and separate evaluation of original-input performance and paired-input fidelity. It must retain the teacher-performance comparison rather than calling successful imitation the end result.

## Next decision

Keep the research on learned second-pass compression. The next design must specify a teacher and deployment policy with useful competitive margin, and decide whether paired-input supervision addresses a behavior that that policy actually requires. Do not start a tree allocator or cache optimization as a substitute. Do not reuse this repeatedly exposed development panel as new training data or a fresh final test.

A current check of the [official D²SD repository](https://github.com/catnanami/D2-SD) still finds its README directing users to train local redrafter checkpoints and promising pretrained weights for a later release. That page does not supply a ready stronger teacher to swap in. This is a repository check, not proof that no checkpoint exists anywhere. No model download or paid execution followed.

## Numerical scope and verification

The student is reconstructed in CPU FP32; teacher scores are the previously recorded native BF16 scores. The 128 original-input student token vectors reproduce the earlier CPU audit exactly. They retain the same 55 valid-token disagreements with historical BF16 student outputs. No tolerance or alternate precision was selected to make decoding outputs agree. Candidate-response magnitudes near rounding scales should not be treated as native CUDA measurements.

Input identities and candidate ordering were checked against the original recovery records. Independent scalar arithmetic recounts all 2,048 response-energy records, all 128 full-vocabulary prefix comparisons and all 68 repair-margin records. Frozen student parameter bytes match before and after. The reported bootstrap resamples responses within domains; it is descriptive on exposed data and conditional on one checkpoint, not training-seed uncertainty.

An initial local invocation failed before evaluation because inference-created tensors do not expose version counters. The check was replaced by before/after parameter-byte hashes. The successful invocation exited normally; no historical artifacts or scientific thresholds were changed.

Evidence: PROTOCOL.md, results.json, rows.json, per-prompt.json, sources.json and verification.json. Reproduction: analysis/audit_editor_paired_feedback.py and analysis/verify_editor_paired_feedback.py. Existing teacher/student outcomes remain preserved. The research goal remains active and unmet.
