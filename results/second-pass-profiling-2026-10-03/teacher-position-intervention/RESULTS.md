# Teacher benefits are partly local, but perfect sparse selection has limited headroom

All 2,048 contexts from 64 reused development responses completed locally in 86 seconds for $0. This directly intervenes on saved scoring vectors. It does not train an editor or measure generation speed.

## What changed

At each of the 14 editable positions, substitute the teacher's relative preferences among the same four tokens, keeping the original probability mass on that set. All other positions retain their original scores. Separately, start from the teacher's complete score table and revert each position to original scores. The original chain, first guess and 30-node checking budget remain fixed.

## Findings

The complete teacher score table improves 293 contexts over the original-score tree, adding 624 positions before counting its damage elsewhere.

- In **138/293** teacher-beneficial contexts (47.1%), a single-position substitution reaches or exceeds the complete teacher's progress.
- In **236/293** (80.5%), at least one single-position substitution improves the original result. In the other **57**, none does: more than one teacher position is needed for any improvement under these interventions.
- A reference-informed choice of one position captures **338/624** (54.2%) of the teacher's extra positions in those 293 contexts, with capture capped at the full teacher benefit per context.
- Reverting one teacher position improves its result in **222 of all 2,048** contexts. This includes contexts outside the teacher-beneficial subset.
- In **80 contexts**, a single-position substitution exceeds both the unchanged original and complete teacher. Teacher score changes can interfere through the tree ranking; copying all changes is not always preferable.

These are effects of score substitution on this deterministic selector. They do not explain the teacher's internal reasoning. Several-position effects can arise from selection thresholds and need not imply a particular linguistic dependency.

## Economic opportunity, assuming free computation and perfect choices

| Fixed options available to a reference-informed selector | Modeled time saved versus original-score tree | Extra retained positions |
|---|---:|---:|
| Original or any one-position teacher substitution | 3.182% [2.754, 3.618] | 429 |
| Original or complete teacher | 4.436% [3.823, 5.059] | 624 |
| Original, any one-position substitution, or complete teacher | 5.080% [4.447, 5.713] | 715 |

The last point estimate leaves only about **0.084% uniform added round cost** while maintaining 5% modeled savings, even with perfect reference-informed choices. It has no convincing margin for teacher computation, a student, or selection errors. No result in this table is a measured speedup. Equal round costs are assumed; intervals resample prompts within domains and remain exploratory after repeated development use.

## Decision

This supports a narrower mechanistic statement: some useful second-pass scoring changes are sparse, while other changes can counteract them. It does **not** justify training a sparse gate next. Even its perfect-choice opportunity is too small for the current performance goal under the modeled cost assumptions.

Keep the cheap-second-pass research question active. Do not repeat an all-position imitation fit, call a sparse-score policy novel, or claim that another loss will recover a speedup. A new learning proposal needs evidence that it can produce materially better useful continuations than the frozen score variants studied here. Candidate expansion with teacher reranking remains untested; the separate support bound rules out unchanged-score widening in many cases, not teacher-assisted widening.

## Checks and reproduction

All 61,440 resulting trees matched an independent frontier-list implementation using the frozen Python 3.11 operation order. All 2,048 original/teacher endpoints match the historical record. Input, protocol and script hashes are in [results.json](results.json). Full per-position gains, losses, and context records are retained, including unfavorable outcomes.

Run `analysis/audit_teacher_position_intervention.py` with `.venv-profile-launch/Scripts/python.exe`. No GPU, model call, learning, fresh prompt exposure or paid execution occurred.
