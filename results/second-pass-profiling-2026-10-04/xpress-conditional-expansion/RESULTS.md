# Conditional expansion completed: useful repairs, no competitive speed result

**Decision:** do not run full generation or train a replacement for this implementation. The one-batch candidate improves over ordinary proposals, but adds too much proposal work and has almost no net coverage advantage over the stronger shared-score control. A saved-data audit identifies a concrete support limitation to address locally before considering another paid test.

All16 contexts and all six methods completed. Recovery, independent verification and cleanup passed. The2USD allowance remains conservatively held, not billed usage; **22.50USD remains** from the newly authorized30USD. No live jobs or temporary volumes remain.

## What was measured

One released Qwen3-4B xPress checkpoint, its own backbone, and frozen target; no training. Eight exposed development responses, two saved contexts each. Every proposal contains32 nodes including the committed anchor. These are candidate sets checked against recorded target continuations, not executed target trees or full generated responses.

The candidate builds a128-node ordinary tree, includes the published six-pass refined chain, then scores each existing parent's concrete history in one batch. It adds one generation of children and prunes to32 nodes. New children cannot themselves be expanded in the same batch.

| Method | Total matching positions | Mean complete proposal time |
|---|---:|---:|
| Ordinary marginal tree |149|0.634ms|
| Published refined chain + ordinary tree, filled to32 |157|2.294ms|
| Tree from shared scores conditioned on the refined spine |169|3.084ms|
| One-batch immediate-parent input intervention |169|11.369ms|
| **One-batch full-history candidate** |**170**|**11.560ms**|
| Depth-wise full-history expansion, width8 |177|13.939ms|

Times include all proposal preparation, refinement, device/host transfers, path scoring, child generation and pruning. Head computations are CUDA-graph captured. One warmup and three alternating-order repeats ran per context/method. Setup took0.189seconds and is separate. These timings use reference Python tree construction; they do not establish the cost of an optimized serving implementation.

The common first drafting pass, target verification and cache work are excluded. Therefore none of these numbers is an end-to-end speedup. Shared-spine scoring includes an additional score evaluation after six-pass refinement; it is not mislabeled the published six-pass-only algorithm.

## Repairs and regressions

Relative to ordinary32-node proposals, the candidate gained21 matching positions across7 contexts and lost none. Relative to the published-chain union it gained13 across7 and lost none. However:

- Against shared-spine scores: **7 gained,6 lost**, across5 better/1 worse contexts. The candidate costs8.476ms more per proposal on average. Its mean equal-response, equal-round-cost modeled savings is5.28%, with a descriptive95% interval of−6.62% to15.63%, **before its added cost**.
- Against immediate-parent input: **1 gained,0 lost**, in just one context, for0.191ms more. This does not establish a substantial value from longer history on this panel.
- Against depth-wise expansion: **5 gained,12 lost**, across2 better/3 worse contexts, but2.379ms cheaper. Its modeled tradeoff depends on common costs. At an assumed40ms common cost, the mean modeled saving is0.28%, interval−5.62% to6.02%. That40ms is a scenario, not a measured complete round.

No method is a newly trained weight control: these are execution/input interventions on the same released refiner. Immediate-parent inputs retain the refined spine at older positions; they are not a separately trained DSpark model.

## What the saved evidence explains

The candidate's initial pool covers165 target positions. Its new conditional children expand coverage to173; pruning retains170. The immediate-parent condition also expands the pool to173 and retains169. Both place the first missing correct child in their top16 on8 contexts and first in4. So adding conditioned predictions does create useful support, but this panel provides little additional value from full history beyond the immediate-parent intervention.

Crucially, **all12 positions lost to depth-wise expansion are absent from the candidate's expanded pool**. These are structural support failures, not merely a selector overlooking an available path:

| Context | Original seed coverage | After one conditional child | Selected candidate | Depth-wise |
|---|---:|---:|---:|---:|
|gsm8k-6150, anchor8|10|11|11|16|
|ultrachat-12677, anchor24|3|4|4|5|
|ultrachat-12468, anchor8|7|8|8|14|

This follows from the one-generation construction. Once the right parent chain leaves the seed tree, the batch can add its next token but cannot recursively condition on that new token. It is not proof that serial execution is fundamentally necessary, or that a better initial tree would fix the method.

A subsequent **local, answer-informed** audit rebuilt the128-node seed from the already recorded shared-spine scores instead of original backbone scores. Including the same refined chain, its coverage rose from165 to180 positions. This is an opportunity measurement, not a selected32-node result: probabilities on1,032 new parents remain unmeasured. It does not justify copying the oracle path or counting180 as achieved progress. No thresholds or methods in the completed GPU comparison were changed.

The next local question is whether this better-supported seed survives actual conditional scoring and selection. The recovered4B head and first-pass features now make a CPU replay possible without regenerating target features. Any CPU replay must first compare its original-policy outputs with the recorded GPU results and disclose precision/tie differences; it cannot stand in for GPU runtime. There is no next paid launch, training sweep or generation experiment selected yet.

## Verification, limitations and novelty

- All89 recovered files,256,319,444bytes, authenticated by recorded hashes.
- All96 selected trees independently recounted against target-reference prefixes.
-48 conditional/depth-wise trees independently reconstructed from saved edge scores; parent-only/full-history input boundaries verified.
-128 prescribed native full-head versus requested-row CUDA comparisons: zero logit difference and zero token disagreements. This is a subset check, not all-query or full-generation equivalence.
- Repeated proposals stable; frozen-model versions and frozen scientific source hashes unchanged.
- No dropped context, new training, checkpoint selection or retry. The run completed in approximately4.84minutes through recovery and cleanup.
- Eight exposed responses are too few for population claims. Long-window caps, recorded-reference numerical differences and complete decoder correctness remain outside this proposal screen.

Conditional branch correction and probability pruning already exist in PCTree, DominoTree and DARTree; JetSpec describes parallel tree-causal computation, and xPress supplies the learned head. This result is not a novelty claim. The local protocol and previous [mechanism report](../xpress-path-mechanism/RESULTS.md) preserve those source checks. The useful new diagnosis concerns this implementation's seed support and value/cost tradeoff; the broader research goal remains unmet.

## Evidence

- [Frozen protocol](frozen/PROTOCOL.md)
- [Complete analysis](attempt1/analysis.json)
- [Independent verification](attempt1/verification.json)
- [Support and selection analysis](attempt1/mechanism-analysis.json)
- [Corrected-score seed audit](attempt1/seed-support-audit.json)
- [Resource closure](attempt1/closure-resource-check.json)
- Raw recorded model inputs, serving head, score rows, selected trees and timing repetitions: `attempt1/recovered/`.
