# Existing students do not recover the teacher's useful ranking advantage

## Outcome

All 2,048 saved contexts from 64 reused development responses completed. Both frozen final checkpoints reproduced all 4,096 historical edited-token predictions exactly; original predictions also matched, repeated full-score checks passed, and final model hashes were unchanged. No training, new generation, checkpoint selection or parameter sweep occurred.

The primary test applies the same single-edit and 30-node tree construction to each model's full-vocabulary-normalized scores. Original candidate IDs, first guess, original chain and node budget remain fixed. Percentages below are additional matching-prefix coverage over the original draft, including the known root; they are not generation speedups.

| Scoring model | One-edit proposal | 30-node tree |
| --- | ---: | ---: |
| Original drafter | 7.048% | 14.158% |
| Full second-pass teacher | 8.725% | 17.075% |
| Frozen mixed-loss student | 5.362% | 11.329% |
| Frozen teacher-only student | 6.457% | 12.573% |

The teacher-only student's tree trails original scoring by **1.585 percentage points**, descriptive paired interval **[-2.176, -1.002]**. The mixed-loss student's difference is **-2.829pp [-3.552, -2.139]**. Both have adverse tree point differences in math and chat. The teacher-only tree gains 148 matching positions but loses 322 relative to original scoring; the mixture gains 172 and loses 473.

Teacher-only training nevertheless improves on mixture training under tree selection: **+1.244pp [0.443, 2.057]**. Its one-edit contrast is +1.095pp [0.202, 1.993]. Thus the old argmax-only result did not capture every effect of the training objective. It remains incorrect to conclude teacher-only training recovered an economical benefit: both trees remain below the no-editor score reference even before charging any additional computation. Historical argmax coverage remains exactly 3.460% for mixture and 3.235% for teacher-only.

## Secondary comparison: keep the original top-four probability mass

Before inspecting student outcomes, a secondary scoring rule was recorded. At each position, preserve the original drafter's total probability assigned to its top four candidates, but distribute that mass using the editor's relative preferences within those four tokens. Vocabulary probabilities outside the set remain original. The first guess and candidate set do not change.

| Scoring model under this rule | One-edit proposal | 30-node tree |
| --- | ---: | ---: |
| Original drafter | 7.048% | 14.158% |
| Full second-pass teacher | 8.908% | 16.903% |
| Mixed-loss student | 5.306% | 11.026% |
| Teacher-only student | 6.536% | 12.513% |

The teacher retains **94.10% of its observed extra tree coverage over original scoring** (2.745pp rather than 2.917pp), a ratio of point estimates, not an equivalence finding. The teacher-only student still trails original tree scoring by **1.645pp [-2.249, -1.061]**. Holding candidate-set probability mass fixed therefore does not repair the existing students' candidate preferences in this test.

This suggests a cheaper output calculation worth preserving for a future learning comparison. In exact arithmetic, the student's top-four relative scores can be computed from its coefficient network and the corresponding rows of the folded output matrix. The full-vocabulary residual projection and new student vocabulary normalizer cancel from this rule. It still requires the network, gathering and selection; total runtime and BF16 shape-dependent decision fidelity are untested. It is a deliberately changed score distribution, not an exact implementation of the original full student distribution.

The first local secondary analysis stopped on a numerical validity check: rounded saved log probabilities gave original top-four probability mass slightly above one in 543 editable rows, by at most 1.876281e-6. Mass is capped at its mathematical maximum of one before redistribution. No primary data or threshold was changed. Original-score edit and tree outputs still reproduce exactly on every context. The repair is recorded separately in candidate-mass-precision-note.md and candidate-mass-numerical-audit.json.

## What this changes in the research question

The useful second-pass ranking signal exists on this panel, including under a four-candidate readout. Our existing students have not learned it well enough. Global probability imitation improved relative to the mixed objective but still damaged the stronger original-score reference. Removing the full-vocabulary output cost alone cannot rescue those checkpoints because their coverage is already worse.

The next defensible learning question is narrower: **does training specifically to match the teacher's relative preferences within the original candidate set transfer better than whole-vocabulary imitation?** The measured teacher result supplies a useful target; it does not prove that focused training will generalize. Its maximum observed economic margin is modest: about 2.29% added round cost versus original tree scoring if this secondary teacher benefit were preserved completely. Claims of a large practical gain would require stronger evidence.

No new fit is selected by this result alone. A concrete next protocol must keep a matched full-vocabulary-distillation control, compare ordinary correct-token supervision where applicable, and identify how outside-candidate labels are handled. It must price collection and training, preserve response-level splits, include fully correct drafts, and evaluate the original-score tree rather than only the old editor. Do not repeat the earlier 64-unit target-candidate head experiment as though candidate restriction were new; the proposed distinction is learning a full second-pass teacher's preferences with the current representation, not merely adding a candidate head. TAPS and related learned tree selection remain relevant prior art. An economical or original contribution is still unestablished.

## Verification and execution

Independent analysis reconstructed 4,096 original/teacher trees before the new outcome analysis and independently checked each single-edit choice. All recovered files are hash-verified. One hundred probability-algebra cases and 57,344 real student candidate rows pass the independent mass-redistribution calculation.

The attempt completed, recovered evidence and cleaned up in **3m35s**. Five files totaling 11,082,066 bytes were recovered. App `ap-EUC3w1w36szzm5NTFy3abT` is stopped with zero tasks; owned volume `vo-0Ff2pHvjDMJLFisLJmbQ5K` is absent. The full **$1.25** cap remains a conservative unsettled allowance, not a measured charge. **$5.53506668** remains unallocated; no jobs or reservations are active.

Evidence: [primary results](analysis.json), [secondary results](candidate-mass-analysis.json), [verification](verification.json), [execution](execution.json), [recovery hashes](recovery.json). All uncertainty intervals are descriptive because the development prompts have been reused. No timing run or follow-up training has been launched.
