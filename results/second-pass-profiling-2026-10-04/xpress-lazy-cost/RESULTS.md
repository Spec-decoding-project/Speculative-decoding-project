# Conditional expansion after exact selection optimization

The attempt completed. **Close this fixed conditional-expansion configuration without a generation run or another kernel-only iteration.** The implementation optimization worked, but the extra corrections did not establish a competitive cost advantage over the simpler shared-score tree. The broader research goal remains active and unmet.

## Measured result

All nine policies used the same frozen released 4B xPress weights and sixteen saved contexts from eight exposed development responses. Final verification trees contain 32 nodes. No training, fresh generation or target verification ran.

| Policy | Total matching positions | Mean proposal time, ms |
|---|---:|---:|
| Ordinary first-draft tree | 149 | 0.505 |
| Published refined-chain union | 157 | 1.991 |
| Shared-spine scores | 169 | 2.595 |
| Original seed, full-history expansion | 170 | 6.237 |
| Corrected seed, full-history expansion | 175 | 6.749 |
| Corrected seed, parent-only, ungrouped | 175 | 7.050 |
| Corrected seed, parent-only, grouped and lazy | 175 | 5.478 |
| Same grouped method, eager selection | 175 | 8.771 |
| Depth-wise expansion | 177 | 11.054 |

Times include complete proposal construction, including host work, transfers, vocabulary scoring and selection. They exclude the common first-draft backbone, target verification and cache work; setup is reported separately. Each context has one warmup and five alternating timing repetitions. Matching positions include the common anchor and are measured against saved references, not actual generated tokens.

Lazy selection saved 3.293 ms versus the same-run eager control and preserved all sixteen selected trees. Grouping saved 1.572 ms versus ungrouped execution and also preserved the trees. Three top-16 score rows nevertheless differed numerically (maximum compared score difference 0.13602): this is not bitwise CUDA score equivalence.

## Why the candidate does not advance

Against shared-spine scoring, the candidate adds six matching positions across five contexts and loses none, but costs another 2.883 ms per proposal. Under an **assumed**, unmeasured 40 ms of common work, the equal-response modeled saving is only 0.544%, with a descriptive prompt-bootstrap interval of -5.157% to +8.009%. At 20 ms it is 5.028% worse. These are sensitivity calculations, not latency measurements or evidence of a general upper bound.

The candidate looks better against expensive depth-wise expansion, but that comparison cannot erase the stronger shared-spine alternative. Likewise, posthoc assessment of shared-spine itself gives only 3.089% modeled savings versus the ordinary tree in the 40 ms scenario, with a wide interval spanning harm. Do not promote a new winner merely because one comparison failed. Generic trees built from refined scores are already contemplated by [xPress](https://arxiv.org/html/2608.02438v1).

The evidence rejects advancing this configuration on the present panel. It does not prove conditional expansion can never help, establish performance on a wider workload, or justify another student fit.

## Does the older full teacher expose a missing opportunity here?

A separate posthoc local join authenticated the same saved banks, anchors, labels and valid masks against the old full-teacher records. An answer-informed best choice between its original-first and alternative-first chains adds just two matching positions beyond shared-spine, and zero beyond the candidate, on these sixteen contexts.

This is an optimistic opportunity check, not a deployable selector, matched-compute comparison, architecture ablation or general ceiling. The teacher and xPress have different backbones and training, and chains and trees differ. It supplies no reason to return automatically to compressing that teacher. Evidence: `cross-teacher-opportunity.json`.

## Validation and closure

- All 24 recovered artifact hashes and 42 frozen source snapshot hashes checked.
- Independently reconstructed 96 policy trees and recounted 144 coverage endpoints.
- All sixteen lazy/eager trees matched; original-policy trees reproduced the previous GPU run.
- No target generation fidelity or end-to-end speed claim is established by these checks.
- Execution through cleanup took 77.48 seconds. Independent follow-up confirms app `ap-lqtlsmCjz18cYvUPrjccUY` stopped with zero tasks and its owned temporary volume absent.
- The full $1.50 allowance remains held conservatively, not invoiced spending. The new $30 allowance has **$19.50 remaining**; no new authorization or reservation was added by this analysis.

Primary evidence: `attempt1/analysis.json`, `attempt1/verification.json`, `attempt1/execution.json`, `attempt1/closure-resource-check.json`, authenticated files in `attempt1/recovered/`, and `frozen/`. Posthoc shared-spine sensitivity is in `shared-spine-secondary.json`; it is not a changed primary criterion.
