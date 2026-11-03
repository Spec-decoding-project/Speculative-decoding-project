# Better seed and identical-query sharing: local evidence

The corrected-score seed increases available continuations, but does not establish a reliable improvement over the original expansion policy. No training, GPU run or generation occurred in these replays.

## Executed comparison

Same recovered released 4B xPress head and first-pass features; BF16 CPU, full-vocabulary scoring. Sixteen contexts from eight already exposed development responses. The original-policy replay uses the saved GPU seed. The changed seed uses the saved GPU shared-spine scores. All final trees contain32 nodes. Labels are used only after proposal selection.

| Policy | Matching positions | Available in expanded pool |
|---|---:|---:|
| Original seed, full history |170|173|
| Corrected seed, full history |175|187|
| Corrected seed, immediate parent |175|187|

Changing the seed gains7 and loses2 matching positions, improving2 contexts and worsening2. Equal-response modeled equal-cost savings versus the old policy are only0.24%, descriptive95% interval[-5.36%,7.03%]. Much of the pooled gain comes from one UltraChat context. Full-history and parent-only coverage agree in all16 contexts, although their selected trees agree in only7. This does not show identical computations or that earlier history never matters.

Independent reconstruction of48 trees and all selected/pool coverage counts passes. Original CPU trees and coverage match the recorded GPU on16/16 contexts, but1,674 top16 rows differ numerically; maximum common log-probability difference is0.24999. Therefore original-policy outcome agreement is not general CPU/GPU numerical parity. Cross-device comparisons to saved shared-spine/depth-wise outcomes remain exploratory.

## Cost-relevant finding

In the parent-only intervention, all older token inputs come from the same fixed spine. Consequently two paths with the same depth and immediate-parent token have identical neural inputs. Among1,913 parent requests, only554 are distinct (71.0% redundant). Full-history queries do not have this equivalence.

An implementation now groups these requests and gathers the needed existing-edge scores once per group. A second actual-head CPU replay, reducing padded scorer capacity144 to64, reproduces all16 trees and every compared top16/existing-edge score exactly. The replay took17.94seconds. This is CPU compatibility evidence, not GPU speed evidence. Input preparation is also made linear in tree size rather than scanning the pool once per parent; apply equivalent bookkeeping improvements to the full-history control.

## Decision

Do not train a new editor or claim a full-history advantage. The narrow next test is a recovered-feature GPU comparison of corrected seeds and grouped scoring against shared-spine, published refinement, ordinary trees, and depth-wise expansion. Charge seed creation, six-pass refinement, grouping, all scoring, transfers and selection. A grouped-versus-ungrouped control isolates the execution change. CUDA batch-shape effects remain untested.

The candidate's free-cost margin over shared-spine is small enough that overhead can erase it. The test is justified by observed duplicate work and preserves weights; it is not a new architecture or novelty claim. Conditional trees and parent-state sharing overlap prior work already recorded in the preceding report. Only a favorable complete proposal value/cost comparison warrants a fresh full-round experiment. Neither result here establishes speedup, decoding fidelity, or a publishable contribution.

Evidence: results.json, audit.json, grouped-replay.json, sixteen raw context files; analysis/replay_xpress_corrected_seed.py, analysis/audit_xpress_corrected_seed.py, analysis/replay_xpress_grouped_parent.py. Additional paid execution:$0. Goal allowance unchanged:$22.50 available before any later reservation.
