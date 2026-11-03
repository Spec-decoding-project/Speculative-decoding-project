# Where the trained second-pass recipe fails

The preceding goal turn made scientific progress: both matched teacher fits, all four evaluation panels, independent counts, recovery and shutdown completed. This continuation uses only their saved predictions. No new model runs, GPU spending, training or parameter changes.

## Main finding: learning mostly restored damage already avoided by the first pass

Training improved the completed-draft model's raw chain in108contexts, worsened27, and tied121. Positive changes added440matching positions and negative changes lost58, for a net382. Of those440added positions,389(88.4%) merely regained positions the original draft already got right;51went beyond the original chain. Against the stronger existing conditional-refinement coverage,428of440(97.3%) added positions were already available, and only12went beyond it.

This explains why genuine improvements in the trained model produced almost no competitive proposal gain. It learned to undo part of the disruption from the new input format, while adding little beyond the strong method already available. These are descriptive comparisons of actual before/after outputs, not proof of an internal mechanism or a learning limit.

## Locking the first guess does not explain the overall failure

The locked first guess is wrong in23of256contexts. Those cases account for55of341pooled missing coverage positions versus the stronger control. In the233contexts with a correct first guess, the full-proposal teacher still has11.30%lower coverage averaged by prompt. Unlocking that guess alone is not an evidence-backed remedy for most of the observed deficit. The older protocol already established a large privileged oracle opportunity under the locked-first rule; we did not rerun that oracle.

## Raw-chain damage is not final-proposal damage

There are62initially complete blocks. Full-proposal editing damages39of their raw chains and loses234raw positions. However, the proposal union retains the original draft and preserves all992positions across those62blocks. Therefore39damaged edited chains must not be described as39failed accepted proposals or final-output errors. The problem is insufficient additional useful proposals and their computation cost, not loss of these protected original paths.

## Training and the second-pass input have separate observed deficits

| Output | Raw matching positions | Conditional/retained proposal coverage |
|---|---:|---:|
| Released original |2049|2700|
| First-guess-trained weights used as an ordinary first pass |1973|2437|
| Those weights used as their intended second pass |2000|2386|
| Completed-draft-trained weights used as an ordinary first pass |2025|2489|
| Those weights used as their intended completed-draft second pass |1657|2359|

The completed-draft weights nearly retain the original raw-chain total when used on ordinary masked input, but perform much worse in their trained redrafting configuration. The first-guess condition behaves differently: its raw second pass is slightly better than its own single pass, while its retained proposals still underperform the stronger control. A single label such as 'training failed' hides these distinctions.

These are useful controls, not a causal decomposition of neural internals. Input format, seeded refinement and retained-proposal construction differ between first and second pass. The trained single-pass evaluation was not separately trained for that input. Neither a weight-capacity limit nor an optimization cause follows.

## Next discriminating question

Does the trained backbone already produce useful corrections that the frozen xPress refinement head subsequently undoes, or are the useful corrections missing before that head acts?

The current predictions save only the final refined output, so they cannot resolve this. Do not infer that unfreezing the head will fix it. The old four-case channel experiment concerned an untrained first-guess rerun, not these newly trained full-proposal checkpoints; its narrow interactions do not settle the new question.

A bounded inference-only trace can compare the same trained backbone's direct vocabulary readout with its current six-step frozen refinement, on the same256development contexts in both trained conditions. Preserve first-guess clamping, original proposals, committed-only features and all saved references. Reproduce the saved final predictions before interpretation. Record per-step guesses and correct-token ranks, but predeclare direct-readout versus current-final as the primary comparison; no selecting a best intermediate step as a validated method. Compare available correct continuations with the released conditional control, not just with the weak second-pass output.

Decision-changing outcomes: a direct readout with substantial additional useful corrections would identify a specific teacher-output path worth assessing for compression; harmful frozen refinement without sufficient better-than-control corrections would not. If direct readout is already weak and the head partly rescues it, do not blame the readout or launch another student fit. Mixed or unmatched traces remain inconclusive. Removing a component the backbone was trained through is an inference intervention; it does not show joint training would succeed.

No runner, live cost estimate, launch or reservation is created here. Any paid diagnostic must fit the existing $3.75 unallocated allowance and retain complete recovery. The objective remains compressing useful second-pass computation. The failed recipe and all original metrics remain unchanged.

Evidence: ../xpress-teacher-completion-path-repaired/attempt1/failure-strata.json; training-gain-location.json; repair-anatomy.json; analysis.json. Analysis source: analysis/analyze_teacher_failure_strata.py. Strata use reference answers and are explanatory, not deployable selection rules.
