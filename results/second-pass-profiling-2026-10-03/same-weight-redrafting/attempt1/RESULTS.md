# Same weights, one draft versus two — completed

## Finding
Conditioning a frozen model on its own first guess supplies complementary proposals on these saved contexts. It does not consistently make the second draft a better standalone replacement. The old full-teacher coverage gain also includes the value of a separately trained alternative proposal; it cannot all be attributed to token feedback.

This supports the existence of useful second-pass work, not successful cheap compression, generation speedup, or novelty. Another small-editor fit is not justified by this result alone.

## Fixed setup
All 64 reused development responses, 32 GSM8K and 32 UltraChat; four fixed anchor indices 4,12,20,28 each, 256 contexts. Frozen Qwen3-4B target and original DFlash, plus the existing feedback and no-feedback epoch-three full checkpoints. No training, new generation or fresh evaluation exposure. Each model first predicts from a masked future, then the same weights run with their own predicted x1 visible; output x1 stays unchanged. Only committed target features enter either pass.

The table reports equal-response mean percentage changes in root-inclusive matching-prefix coverage. These are means of per-response ratios, not ratios of pooled mean lengths. Keeping both proposals is a two-branch opportunity measurement; it is not a free deployable selector or a runtime claim.

| Frozen checkpoint | First draft vs released q0 | Second draft alone vs released q0 | Keep both vs released q0 | Keep both vs its own first draft |
|---|---:|---:|---:|---:|
| Released q0 | 0% | -0.48% | +5.37% | +5.37% |
| Trained feedback | +2.98% | +4.59% | +6.94% | +3.95% |
| Trained no-feedback | +0.32% | -0.40% | +5.06% | +5.33% |

Response-bootstrap 95% intervals, 10,000 shared resamples:
- Released q0 keep-both improvement: +3.03% to +8.13%.
- Trained feedback first-only change: -2.30% to +8.21%; a better single-pass drafter is not established.
- Trained feedback keep-both versus q0: +1.26% to +12.56%; versus its own first: +2.11% to +6.16%.
- Trained no-feedback keep-both versus q0: -0.22% to +10.76%; versus its own first: +2.50% to +9.01%.

Second-pass prefix repairs/damage: q0 27/28, feedback 21/17, no-feedback 20/29. The preserved first guess was correct at 216/256 q0 contexts, 213/256 feedback contexts and 211/256 no-feedback contexts. Relative to q0, feedback first guesses repaired five contexts and damaged eight; no-feedback repaired four and damaged nine. Retaining the first branch is therefore material, and own-x1 errors remain unrepairable by this construction.

## Supplemental same-weight conditioning contrast
Independently recounted after completion from the already saved proposals; no new inference. Hold the original q0 branch, the second proposal's output x1, and trained weights fixed. Compare an unconditioned tail with a tail computed while x1 is actually visible. This separates copying a token into the candidate from letting it affect the neural calculation.

For the trained feedback checkpoint:
- q0 plus unconditioned trained tail, with q0 x1 copied into the candidate: +7.72% [4.69,11.17].
- q0 plus x1-conditioned trained tail: +9.02% [5.89,12.43].
- Paired conditioning contrast: +1.30 percentage points [-0.97,+3.80].

Thus a substantial coverage opportunity was present without x1 conditioning. This does not establish that conditioning has no effect, that fine-tuning alone explains a fixed fraction of the benefit, or that a trained first pass replaces the two-model procedure. Both proposals cost computation, and the original branch remains available. The 9.02% is the old teacher configuration on this particular 256-context subset; it does not replace the previous 10.21% full development or 10.92% fresh-panel results.

The no-feedback checkpoint's corresponding same-weight contrast is +5.95% to +8.03%, or +2.08 points [0.19,4.34]. This is supplemental, and its visible-x1 input mode was not the mode it was trained on. The released q0 unconditioned tail is identical to q0, as expected; conditioning creates its +5.37% retained-branch opportunity.

## What this changes
The same-model result supplies direct evidence that an explicit token guess can change a second pass into a useful alternative at fixed weights and context. However, describing our full teacher as simply repairing its own first draft was too strong: our main historical procedure combined released q0 with separately trained weights. The new one-pass comparisons do not establish a free improved replacement drafter.

For the cheap-editor research, the relevant target is useful additional prefixes after accounting for what the original branch already covers. Global teacher imitation and assumed uniform improvement are inadequate success criteria. This is not a new proposal to repeat prefix weighting, targeted rehearsal, or eligible-token training: those previous variants already failed their advancement tests. The remaining missing justification is a specific reason a different student/data/training choice would transfer these complementary corrections more effectively. No new fit, generation run, or loss sweep is initiated from this diagnostic.

## Limits
Both teachers were trained only on editable positions after original q0 x1 and only where that x1 was correct. They were not directly trained on their own first guess. The feedback model's fully masked first pass and the no-feedback model's visible-x1 second pass change their training mask conditions. Negative or modest results here do not rule out properly matched one/two-pass training. These are one-checkpoint, exposed-development observations, with four contexts per response; no population-wide claim or complete generation fidelity/latency test.

All main proposals use the same 15-row readout. Historical 16-row q0 and 14-row teacher controls were checked separately. All 768 historical controls reproduced exactly; there were zero candidate-token readout-shape differences. Initial repeats for all three models were exact. This is numerical fidelity of this probe, not end-to-end serving validation.

## Execution and verification
One A100-SXM4-80GB attempt completed all 256 contexts. Independent token-based analysis checked all 768 model/context rows, first-token preservation, reference alignment and complete pairing. Trained deployment digests matched the historical audit and remained unchanged; q0 digest and frozen parameter versions passed final checks. All eight evidence files (1,566,443 bytes) were recovered with remote and local hashes matching. The app has zero active tasks and is stopped; the owned temporary volume is absent. The retained research archive was untouched.

Elapsed through cleanup: 11 minutes 6 seconds. Conservatively retain the full $1.50 attempt allowance, not a settled invoice. $11.28506668 remains unallocated; no reservations or running jobs.

## Reproduction
- Protocol: ../PROTOCOL.md; frozen inputs/sources: ../frozen/ and ../local-bundle.json.
- Raw observations: recovered/predictions.json; controls: recovered/preflights.json; model audit: recovered/final-audit.json.
- Independent primary analysis: analysis.json, produced by analysis/analyze_same_weight_redraft.py.
- Supplemental calculation: conditioning-supplement.json, reproduced by analysis/supplement_same_weight_conditioning.py.
- Integrity/accounting checks: report-check.json, analysis-source-manifest.json, recovery.json, execution.json.

The research objective remains open. This diagnostic establishes a limited mechanism and corrects attribution; it does not establish a successful small editor.
