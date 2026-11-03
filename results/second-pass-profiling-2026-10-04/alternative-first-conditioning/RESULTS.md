# Alternative-token feedback: useful causal effect, insufficient competitive margin

## Decision

Do not fund a generation run or another compression fit for this two-condition, fixed-budget policy. The run demonstrates a useful feedback effect in the existing teacher, but its point-estimated advantage against ordinary branching remains below the5% goal even before added teacher computation. This does not reject cheap conditional redrafting generally or prove a limit on other teachers, policies or training.

Next analysis should use these new paired outputs to distinguish misplaced verification budget from missing useful predictions. In particular, check whether the useful recomputed continuations can coexist in the fixed budget before proposing a selector. Do not assume a learned selector will recover oracle opportunity, add another condition merely to raise a coverage number, or call existing conditional branching novel. No follow-up paid work selected yet.

## What changed and why this is a causal test

The same frozen full teacher receives exactly the same committed context and masked future positions. Only the visible first guess changes: original top choice versus second-ranked candidate. Both inputs are chosen by the original drafter, without target labels. The critical control copies the second candidate into the original teacher's unchanged tail. Differences between that control and recomputation therefore measure the downstream effect of the changed input for this checkpoint, not a benefit caused solely by substituting the first word.

All2,048 development contexts/64 reused responses completed. No training, new generation, target-label selector, or runtime speed comparison. Original-conditioned tokens AND candidate scores reproduced historical records exactly throughout; first case both complete score matrices repeated exactly; final frozen-weight/version audit passed.

## Frozen whole-panel comparisons

| Comparison | Modeled equal-round-cost saving | Descriptive95% prompt interval | Math / instructions |
|---|---:|---:|---:|
| Fixed30-node conditioned tree vs ordinary30-node tree |4.107%|3.107 to5.089%|4.650 /3.564%|
| Same conditioned tree vs reused-tail scores |1.292%|0.955 to1.652%|1.035 /1.548%|
| Same conditioned tree vs previous single-condition teacher tree |0.784%|0.492 to1.097%|0.533 /1.036%|
| New-token chain unions vs ordinary tree at matching node count |−0.581%|−1.546 to0.365%|2.032 /−3.194%|
| New-token chain unions vs reused-tail union |0.993%|0.645 to1.385%|0.959 /1.028%|

These figures use the same round-time assumption, not measured latency. The primary point estimate permits no nonnegative extra round cost while reaching5%; even free teacher computation gives4.107%. Its upper descriptive interval barely exceeds5%; this is neither evidence of a robust5% gain nor proof that a population effect cannot exceed5%. Two full teacher calls are additional work and remain unpriced here.

Against ordinary30-node branching, the primary policy gains973matching positions across440contexts and loses393 across275. Against unchanged-tail reuse, recomputation gains168 across83contexts and loses17 across14. Correct original-chain prefixes are always retained; these losses concern progress the competing tree would otherwise cover. Independent recount reproduced all8,192 primary endpoints.

In the chain-union comparison, raw pooled positions rise by80 versus ordinary expansion, while equal-prompt modeled savings is adverse. That is not a contradiction: aggregation weights differ, and the instruction domain loses. Preserve the frozen equal-prompt definition rather than choosing whichever summary is favorable.

## Post-result explanation, not a new success criterion

Among147contexts where the second first-token candidate matches the target, changing only that token yields381matching chain positions; recomputing under it yields587. These counts include the anchor and chosen first token in each context. Recomputed chains improve71, worsen4 and tie72; gains212positions versus6lost. This subset is selected with reference answers for explanation only, not available to a decoder beforehand.

The computation can matter. However,55of147 recomputed chains still fail immediately at the following token, and only6match every valid position. Just15matched positions use tokens absent from the original top16 candidate support. Most achieved corrections are choices that an ordinary candidate tree can already represent; the earlier optimistic new-token oracle was not realized by this frozen teacher. These observations do not prove the remaining errors require a larger model or identify the cause of student failure.

## Validity and cost

Four local tests cover native input/output behavior, rejected contracts, exhaustive short-tree selection and matched union budget/prefix closure. All2,048 reference alignments passed before launch. Eight recovered artifacts/49,688,727bytes match remote hashes;43 frozen sources unchanged;8,192 primary endpoint checks independently pass. Secondary uses frozen synthetic checks and independent ordinary-tree controls, without a second independent complete reconstruction. Development prompts have been repeatedly exposed; intervals are descriptive, not fresh confirmation.

One A100-80GB PCIe attempt.6.09minutes from reservation through recovery and cleanup. Provider app is stopped/zero tasks, owned temporary volume absent. Full1.50USD held conservatively against the newly authorized30USD;28.50USD remains. This hold is not measured usage or a settled invoice. No retry, training or follow-up job.

Evidence: attempt1/analysis.json, verification.json, execution.json and recovery.json; frozen/ contains protocol and source snapshots. Prior studies unchanged. D2SD, PCTree and DARTree already establish conditional branch generation; this result is evidence about our checkpoint and controls, not a new algorithm claim.

## Additional fixed-budget and training-support analysis

A disclosed post-result oracle selects among five actual30-node proposal sets: ordinary tree, old conditional tree, new two-condition tree, original-plus-first-teacher chain, and original-plus-second-teacher chain. Each chain retains only its longest prefix that fits while preserving the original chain; remaining nodes use ordinary scores. Perfect answer-informed selection yields8.094% modeled savings before all extra computation, versus6.704% for the old three-set family. New outputs expose192additional oracle positions across66contexts. Even this perfect new selector allows only3.367% uniform added round cost to retain5% modeled savings. This is not a deployable selector, a global bound on trees, or evidence a new small model can learn it. Blindly preserving the whole second-choice teacher chain loses13.38% modeled efficiency versus ordinary allocation. Thus useful proposals exist, but blindly spending node budget on them is damaging.

Code and metadata reveal a specific support mismatch worth examining before another fit. The full teacher's training objective uses cut1 and masks suffix supervision whenever the original first guess is wrong. That was appropriate for the old fixed-first-token experiment: those continuations could not be accepted. The newly considered alternate-first branch makes them relevant.2,274of12,288 training contexts were excluded this way (458math,1,816instruction); all have valid next-token labels. Evaluation contains331such original-first errors, of which147have the correct token in the second candidate slot.

This proves an exposure difference, not causation of failure. The current checkpoint already transfers useful corrections to many such contexts. The next research question is whether including those recorded contexts, conditioned on their correct first token, adds competitive corrections beyond equal-cost continued training on the old support. A matched same-data/no-feedback control is needed to separate data coverage from explicit token feedback. This is a candidate explanation, not a ready fit or novelty claim. Inspect prior training/conditional-branch work and expected effect/cost before implementation. No further GPU spending.

Supplemental evidence: attempt1/budget-opportunity.json and training-support-audit.json.
