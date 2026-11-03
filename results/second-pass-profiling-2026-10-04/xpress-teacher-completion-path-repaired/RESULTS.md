# Completed stronger-teacher comparison: no useful compression target from this recipe

## Decision

Do not train a small editor to imitate these new checkpoints. The completed-draft teacher fails the frozen advancement screen and supplies little additional proposal value over released xPress's existing conditional refinement. Keep the research objective fixed on compressing useful second-pass computation; this result closes this tested teacher recipe, not that objective or the possibility of compression.

## What ran

The repaired staging check passed. Reused all 448 authenticated responses: 384 training responses (12,288 contexts), 64 previously exposed development responses (256 contexts). Both matched conditions completed 384 updates from the same released initialization with the frozen original schedule, loss, optimizer, training exposure and seed. One exposes only the first guess, the other the completed draft. Both preserve the first guess when evaluated as redrafters. Each final checkpoint was also evaluated as an ordinary single-pass drafter, with no prior draft input. All four 256-context panels completed. This is one training seed, recorded-prefix coverage, not native generation.

## Main results

All percentages below weight the 64 responses equally. Matching coverage includes the same known root in every condition. Every primary proposal set has 64 nodes. Intervals are descriptive prompt-bootstrap intervals on reused development data.

| Comparison | Matching coverage difference | 95% interval |
|---|---:|---:|
| Trained completed-draft pass vs released conditional refinement | -13.38% | -16.78% to -10.04% |
| Trained first-guess pass vs released conditional refinement | -12.73% | -15.89% to -9.77% |
| Trained completed-draft vs trained first-guess pass | -0.77% | -2.20% to +0.58% |
| Trained completed-draft pass vs its own weights used in one pass | -6.39% | -9.80% to -3.10% |
| Trained first-guess pass vs its own weights used in one pass | -3.52% | -7.06% to -0.02% |
| Trained completed-draft vs the same input configuration before training | +1.00% | -0.29% to +2.80% |

The completed-draft pass is +1.08% versus the weaker shared64 control, but that interval crosses zero and it loses substantially to conditional64. Do not headline a weak-control comparison as success. No support for added completed-draft feedback was established under this training recipe. This is not a proof of zero feedback value.

Pooled coverage totals, a different aggregation from the response-weighted percentages: released conditional control 2700; trained first-guess second pass 2386; trained completed-draft second pass 2359; trained first-guess weights in one pass 2437; trained completed-draft weights in one pass 2489. On this ordinary-drafting evaluation, both trained weight sets also underperform the released model. A failed masked-input evaluation does not replace a separately trained single-pass baseline, nor establish general damage to model capacity.

## Repair and damage anatomy

These counts compare each raw edited chain to the initial raw chain, before tree selection.

| Trained input | Prefixes repaired | Prefixes damaged | Fully correct blocks damaged | Positions gained / lost |
|---|---:|---:|---:|---:|
| First guess | 35 | 41 | 13 | 118 / 167 |
| Completed draft | 25 | 94 | 39 | 74 / 466 |

Thus the failure is not only tree allocation: the raw completed-draft output also has an adverse repair/damage balance. Of its 25 repairs, 14 are shared with first-guess conditioning and 11 are unique to it. First-guess conditioning has 21 unique repairs. Unique repairs are not evidence of useful economical complementarity by themselves.

Among 1798 originally correct editable token predictions, completed-draft conditioning damages 451 versus 240 for first-guess conditioning. Among 1771 originally wrong editable predictions, it corrects 163 versus 243. Of 171 editable original first errors, it predicts the correct token at 34 versus 39. These token counts are dependent observations and are not treated as independent samples.

A simple wrong-token-copying explanation is insufficient: the completed-draft model repeats the exact original wrong first-error token in 67 cases versus 88 for first-guess conditioning, while still correcting fewer blockers. Across all wrong editable positions, repetition counts are similar (605 versus 600), with more changes to another wrong answer (1003 versus 928). This locates the behavioral failure; it does not identify an internal computation or prove a causal optimization mechanism.

## Is there still a worthwhile target hidden among its successes?

Against the released conditional control, the completed-draft proposal set wins on 9 contexts, loses on 93 and ties on 154. It gains 18 coverage positions while losing 359. Allowing free hindsight choice between those two recorded proposal sets, without charging computation or limiting their union, gives 2718 rather than 2700 positions: +0.67% pooled, +0.97% averaged by response. The raw teacher chain alone goes beyond that control in only 3 contexts, adding 12 positions (+0.59% averaged by response).

This is an upper bound only for combining these saved proposal sets on these contexts. It is not a limit on a newly trained student, other teachers, further training, or language-model information. It does show that merely preserving this checkpoint's rare extra paths is insufficient motivation for another compression fit.

## What this resolves and what remains open

The matched comparison is now tested rather than blocked by infrastructure. At this dose, exposing and training on the complete draft does not yield a stronger teacher than first-guess conditioning or existing refinement. Both the full teacher's quality and a future student's cost must be justified; the small editor cannot be blamed for a teacher deficit that is already present before compression.

One epoch does not establish convergence. Training-input distribution, adaptation of the backbone with a frozen refinement head, optimization, and suitability of this conditioning route remain possible causes, not established explanations. These results do not justify picking one and launching another fit automatically.

Next research decision: use the retained checkpoints and evidence to distinguish degradation from changing the trained weights from damage caused by the second-pass input configuration. The ordinary-input controls already show both are relevant at the level of observed behavior. Any additional diagnostic must offer a discriminating explanation and a credible path to teacher advantage over conditional64; another arbitrary head, loss change or training extension is not supported. Do not pivot to tree/caching engineering or restart old student recipes.

## Correctness, recovery and accounting

Frozen-weight and complete-exposure audits passed. The main analyzer independently recounted every proposal tree; a separate direct chain recount matched both trained panels' repair/damage counts. Both complete checkpoint hashes match their saved attestations and recovery manifest. All 545 artifacts (2,206,687,707 bytes) are recovered and verified.

The GPU worker completed both fits in about 13.5 minutes each. App lifecycle was 33.48 minutes; total through recovery and cleanup was 41.65 minutes. A transient DNS failure interrupted checkpoint recovery after the GPU had stopped. Recovery-only continuation succeeded; no training, GPU allocation or generation was retried. 512 already-authenticated inherited files were reused locally after matching the remote manifest. An empty local partial-download file is preserved.

App ap-w329Eb3PlqurOYctR59Oa2 is stopped with zero tasks; temporary volume vo-AXIblHXMJnhBlQqF7LWJ6c is verified absent. Reported compute is $1.69615441, not a settled invoice. The $6 all-in hold remains; $3.75 is unallocated within the existing $40 authorization. No further paid work has launched.

## Evidence

- attempt1/analysis.json: frozen primary analysis and source hashes.
- attempt1/repair-anatomy.json: independent descriptive repair analysis.
- attempt1/saved-proposal-opportunity.json: scope-limited free-combination calculation.
- attempt1/recovered/final-audit.json: matched exposure and unchanged frozen weights.
- attempt1/recovery.json, execution.json, metering.json: artifacts, shutdown and accounting.
- launch-snapshot/: exact submitted sources, plan and authorization.
- secondary-analysis-plan.json: descriptive questions recorded before reading trained prediction panels.
