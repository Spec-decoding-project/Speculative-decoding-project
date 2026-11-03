# Full-history projection reuse: small saving, adoption screen failed

## Decision

Close further optimization of this specific projection-reuse shortcut. It reduced measured complete-round cost by **0.506%** (response-bootstrap95% interval **0.305–0.703%**), below the predeclared1% engineering adoption screen. Only29/32 packed proposal trees were identical, also failing the exact-proposal condition. No automatic training, width sweep, generation retry, or kernel follow-up. Preserve the code and all evidence; do not silently replace the established conditional runner.

The research5% end-to-end correctness-preserving goal remains unmet. This is neither a novel algorithm nor a full-generation result.

## Completed comparison

Same32 exposed fixed contexts/16 responses, six paired repetitions after two warmups, oneA100-SXM4-80GB. Same frozen target/backbone/head, causal full-history mixer, candidate64/top16 policy, width8, horizon16 and64-node verification budget.

| Method | Actually accepted positions | Mean complete-round ms |
|---|---:|---:|
| Strong shared-score CPU heap |248|41.238|
| Original full-history conditional |303|41.399|
| Incremental full-history conditional |303|41.191|

Timings include first draft, proposal preparation and construction, target verification, acceptance, retained features and cache compaction. Prefix reset/setup are excluded and recorded separately. This initial-draft implementation is uncaptured in all three arms, so results are not a production-engine speed claim. Earlier study timings were on another execution; do not compare absolute milliseconds across runs.

## Useful progress and regressions

Incremental versus original conditional:0 repaired positions,0 lost positions,0 damaged fully accepted contexts; all32 accepted prefixes and bonus tokens match. Gains are from reduced execution cost on this panel, not additional corrections. Both domain mean savings are positive: GSM8K0.606%, UltraChat0.406%.15/16 responses have lower estimated round cost. The small bootstrap interval is descriptive of this exposed panel and six within-run repetitions, not broad deployment confirmation.

Versus shared-score construction, both conditional variants gain56 positions and lose1, with11better/1worse/20tied contexts and no fully covered block damaged. Incremental's equal-response round cost per position is 14.82% lower [7.00,23.37]. This largely reproduces the established conditional-construction benefit. It must not be attributed to the new cache optimization.

## Numerical scope

29/32 packed trees match exactly. Two remaining cases have identical node sets in different order. One substitutes a single unaccepted candidate with another. Across every common node in the three changed trees, target argmax winners agree. All32 actual accepted prefixes and bonus tokens agree. All recorded causal-path, following-token, masks/positions and cache-compaction checks pass; no numerical reference disagreement occurs on this fixed panel. These facts do not erase the prior full-generation fidelity failures or prove arbitrary-context equivalence. The exact operation responsible for changed tree scores was not isolated; changed projection matrix shapes remain a plausible numerical explanation, not a demonstrated repair.

## Mechanism and what this changes

The preceding controlled intervention established that older branch tokens support useful predictions in this frozen head: last-token-only retained48.7% of net extra progress;34 failed next-token winners flipped when older choices were restored at fixed prefixes. This experiment keeps that information and reuses already-computed prefix projections, cutting projected token rows1808→114 per proposal. Yet the complete-round gain is only0.51%. Operation counts alone do not identify a valuable optimization, and this run does not isolate whether gathers/launches or unaffected operations account for the limited gain.

This is a different question from whether a small trained editor can learn a full second pass. No student was trained. Do not use this result as evidence against compression capacity, better training data, or a different correction mechanism.

## Next boundary

Retain the original conditional implementation as the reference and stop this micro-optimization branch. Do not spend the remaining allowance merely chasing this0.5% effect. The useful conditional-history gain remains supported on fixed contexts, but decisive unresolved requirements are faithful consecutive generation and an algorithmic advantage over existing conditional-tree approaches. Any next proposal must change one of those decisions. Recheck existing evidence before funding a new diagnostic; do not repeat the closed batch-invariance feasibility review or unfinished generation panel unchanged.

## Validation, artifacts and accounting

16local tests passed; released16context CPU prototype parity passed before launch. Independent accepted-path reconstruction passes96trees. All73 recovered files/2,371,156bytes match receipt hashes; analysis regenerates exactly; weights unchanged. The analysis-only correction removed an irrelevant inherited report field and normalized JSON node-list serialization; all progress/timing/comparison metrics remained identical. Frozen launch source is preserved under frozen/.

Worker299.58sec; total staging through cleanup444.41sec. Original controller82191 exited0; app ap-CCjf5bebNTeEYiIih2PwEm stopped with0active tasks; owned volumevo-EngIIoghE0KpcBnaYFZEoC absent. Full2USD cap retained conservatively; actual settled billing not established. Existing30USD goal allowance has3.50USD remaining, no live reservation or new authorization.

Evidence: [analysis](analysis.json), [parity anatomy](packed-differences.json), [validation](validation.json), [protocol](PROTOCOL.md), [execution](attempt1/execution.json), [recovery receipt](attempt1/recovery.json).
