# Completed native-preserving teacher context diagnostic

The intervention does not support changed context selection as a necessary explanation for most of the 14 feedback-only repairs on this exposed panel. Restoring the unconditioned within-context preferences preserves 11 repairs completely and loses three. All fourteen cases have identical prefix outcomes under FP32 and FP64 auxiliary arithmetic. The three losses do establish case-specific dependence under the specified intervention; they do not identify the small editor's root cause or prove context access is generally dispensable.

## Completed comparison

All 256 saved contexts from 64 development responses completed on an A100-SXM4-80GB, torch 2.9.1+cu128, native SDPA. The teacher stayed frozen. No training, fresh target generation or model-selection sweep occurred. Initial identities, every historical native proposal, complete native observation scores, all neutral modes in both precisions and the final parameter audit passed. The prior failed case was first and passed. Previous failure evidence remains unchanged.

The intervention replaces context-related attention quantities in layers 2-5 while retaining the conditioned draft computation and output first guess. Selection-only restores unconditioned preferences among context rows while preserving current total context mass. Mass-only replaces the context score log-partition while retaining current preferences and current draft-side scores. Combined restores context scores in both respects. It does not fix the entire context contribution to its unconditioned value: the current draft partition still participates in normalization, and first-layer changes remain. These are residual interventions added to original native attention outputs, not exact alternative fused-kernel executions.

## Primary group: 14 feedback-only repairs across 12 prompts

| Intervention | Teacher repairs fully retained | Repairs entirely lost | Teacher-added positions retained |
|---|---:|---:|---:|
| Selection-only | 11 / 14 | 3 | 30 / 35 |
| Mass-only | 12 / 14 | 1 | 30 / 35 |
| Combined context-score replacement | 9 / 14 | 4 | 25 / 35 |

These counts match between precisions. Mass-only partly shortens a second repair; combined partly shortens a fifth. Additional positions beyond the original teacher endpoint are separate: selection +2, mass +4, combined +2. They are not included in retained teacher positions.

Selection dependence is particularly concrete in three cases: gsm8k-6119/28, gsm8k-7323/20 and ultrachat-12633/28. Their original blocking-token margins change from +2.5, +1.125 and +0.75 under conditioning to -0.5, -1.5 and -1.5 under selection replacement. Mass-only preserves each complete repair. Conversely, mass-only harms gsm8k-3387/4 and gsm8k-5128/4 while selection-only preserves them. The combined losses are precisely those two groups; no additional feedback-only failure appears solely under the combined intervention on this panel.

The nine repairs surviving combined replacement still run through all five teacher layers, pretrained projections/MLPs, draft-to-draft attention and first-layer conditioning. Their survival cannot be credited to a tiny head or interpreted as proof that raw context is unnecessary.

Prefix lengths below include the common root, which cancels in differences:

| Prompt / anchor index | Original | Unconditioned teacher | Conditioned teacher | Selection | Mass | Combined |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k-2541 / 20 | 6 | 6 | 7 | 7 | 7 | 7 |
| gsm8k-3387 / 4 | 12 | 10 | 16 | 16 | 10 | 10 |
| gsm8k-5128 / 4 | 7 | 7 | 9 | 9 | 8 | 8 |
| gsm8k-6119 / 28 | 10 | 10 | 13 | 10 | 13 | 10 |
| gsm8k-6953 / 20 | 3 | 3 | 5 | 5 | 5 | 5 |
| gsm8k-7323 / 20 | 3 | 3 | 4 | 3 | 4 | 3 |
| gsm8k-7354 / 20 | 2 | 2 | 3 | 5 | 5 | 5 |
| ultrachat-12035 / 28 | 3 | 3 | 5 | 5 | 5 | 5 |
| ultrachat-12117 / 28 | 2 | 2 | 3 | 3 | 3 | 3 |
| ultrachat-12334 / 20 | 2 | 2 | 13 | 13 | 15 | 13 |
| ultrachat-12633 / 4 | 2 | 2 | 3 | 3 | 3 | 3 |
| ultrachat-12633 / 28 | 3 | 3 | 4 | 3 | 4 | 3 |
| ultrachat-12684 / 20 | 2 | 2 | 3 | 3 | 3 | 3 |
| ultrachat-12684 / 28 | 3 | 3 | 7 | 7 | 7 | 7 |

## Controls and numerical sensitivity

Among 27 repairs already available without conditioning, selection preserves 24/27 in FP32 and 25/27 in FP64; mass 23/27 and 24/27; combined 24/27 in both. Thus interventions can disrupt shared teacher repairs too; a lost repair does not establish a uniquely feedback-specific pathway. The additional one-case conditioning extension is retained by selection/combined and shortened by mass. All 214 nonrepair contexts remain in the report.

Across the full panel, selection retains 111/122 teacher-added positions in FP32 and 113/122 in FP64; mass 104/122 and 107/122; combined 104/122 in both. Four contexts change prefix outcome across precisions: gsm8k-4924/20 and ultrachat-12248/4 (selection), ultrachat-12373/12 (mass), ultrachat-12436/12 (combined). They remain unresolved and appear in both complete recounts. None belongs to the feedback-only group. Full token-vector differences are more frequent: 46 selection, 33 mass, 44 combined; many occur after rejection and do not alter these prefix counts.

The conditioned teacher itself has shorter standalone prefixes than the original in 31 contexts. Selection, mass and combined have 29, 32 and 32 such cases in FP32; newly introduced cases relative to the conditioned teacher are 1, 3 and 5. These are edited-branch counts, not damage after retaining the original fallback. Reported union losses and gains are separate in analysis.json. No tolerance or favorable precision was chosen after seeing results.

All counts are descriptive, from reused development contexts with clustered observations. No population rate, generation speed, information limit, or deployable cached implementation is established.

## Research decision

Do not use "the small editor cannot reread the right context" as the main explanation or launch another context-access student on that assumption. Changed within-context preferences contribute to three specific feedback repairs but are unnecessary for the other eleven under this intervention. Even combined context-score replacement leaves nine intact.

The next design question is which remaining teacher computation carries those surviving corrections: the direct interaction with the first guessed token, subsequent draft interactions, and the teacher's existing nonlinear transformations remain coupled. A distinguishing intervention must preserve the context path and the rest of the model while separating those possibilities, with neutral controls. This is a question for design, not a completed mechanism result or authorization for another run. General teacher compression remains a separate issue: 27 of 42 repaired contexts already have equivalent unconditioned teacher predictions.

This is progress in narrowing the explanation, not completion of a reliable cheaper editor. No new architecture or training recipe is justified from survival alone.

## Execution and accounting

Execution through recovery and cleanup took 8.07 minutes. Twelve artifacts totaling 92,807,258 bytes were recovered and checked against remote hashes. GPU stop and zero active tasks were independently confirmed; the owned temporary volume is absent. There were no retries.

User approved $1.50 additional funding by replying "continue" to the explicit one-attempt request, bringing the aggregate campaign authorization to $175.94. The full $1.50 all-in allowance remains retained pending billing; unallocated balance is $0.0050666768450267075. The provider returned no metering rows at the first post-run query, so actual reported compute is unknown, not zero. Historical caps are not expenses. No further paid work is authorized or launched.

## Evidence

- analysis.json: independent FP32/FP64 token-level recount and sensitive-case list.
- recovered/predictions.json: every context, proposal, prefix, blocker margin and numerical audit.
- recovered/first-case-validation.json and final-audit.json: initial failure replay and final frozen-weight checks.
- recovered/*-attention.pt: four predeclared attention snapshots.
- verification.json, recovery.json, execution.json and metering.json: provenance, recovery, shutdown and accounting.
- ../runner-plan.json, local-bundle.json and authorization.json: frozen scope, sources and approval.
