# Conditional coverage improves; exact completion is not the contribution

All 256 saved contexts from 64 previously exposed GSM8K/UltraChat responses completed locally in 1,439.66 seconds. No training, target transformer, generation, GPU job or paid execution. Weights stayed unchanged. Every conditional score uses the same single-row BF16 CPU path; the initial refined chain was recovered from historical GPU evidence. Independent reconstruction of all final and intermediate trees and exact analysis regeneration passed.

| Construction, 64 verification nodes | Matching positions | Gained / lost positions vs shared | Better / worse contexts | Fully correct blocks damaged |
|---|---:|---:|---:|---:|
| Shared scores from refined chain | 2,350 | — | — | — |
| Exact conditional search | 2,697 | 356 / 9 | 92 / 9 | 1 |
| Depth-wise width 8 conditional search | 2,696 | 356 / 10 | 91 / 8 | 1 |
| One parallel completion wave | 2,469 | 120 / 1 | 120 / 1 | 0 |
| Two parallel completion waves | 2,536 | 187 / 1 | 106 / 1 | 0 |

These are matching-prefix positions on fixed recorded continuations, including the shared known root. They are not measured accepted generation tokens or speedups. Target labels were used only after proposal construction. The one/two-wave policies are exploratory development configurations suggested by the earlier small panel; this is not fresh confirmation.

## Decision

Do not pursue exact tree completion as a proposal-quality improvement over depth-wise construction: the two have essentially identical aggregate coverage. The equal-response free-extra-work comparison is only +0.048% with a descriptive 95% prompt-bootstrap interval of -1.047% to +1.075%. Different contexts favor different trees. Exactness maximizes the head's own path score, not necessarily target acceptance.

The narrower open question is scheduling: can a short, initial-draft-seeded construction recover enough conditional coverage at much lower cost than sequential depth-wise construction? Two waves retain 186/347 = 53.6% of the exact method's net improvement over shared scores. They lose meaningful progress relative to the strong structural control, so a small proposal-only timing reduction will not suffice.

Under the explicitly simplified model of equal complete-round costs and equal response weighting:

| Candidate | Free-work modeled savings vs shared | 95% interval | Maximum extra round cost to retain 5% modeled savings vs shared | Required round-cost reduction to beat depth-wise by 5% |
|---|---:|---:|---:|---:|
| Exact | 13.65% | 10.68% to 16.82% | 10.02% | 4.95% |
| One wave | 6.24% | 5.31% to 7.22% | 1.32% | 14.10% |
| Two waves | 8.98% | 7.45% to 10.51% | 4.38% | 11.10% |

These are spending-screen scenarios, not achievable latency predictions. Complete rounds include original drafting, seed refinement, candidate gathering, selection, target verification and cache work. Fixed node count does not guarantee identical verifier time. Actual GPU score-shape effects remain untested.

## Why shallow completion misses useful progress

A scoring wave can extend a known prefix by one token. It cannot recursively evaluate a newly created child's continuation during that same parallel wave. The companion raw-seed test shows this constraint matters: two waves from the raw tree cover only 168 positions versus 178 for the refined shared tree on the 16-context subset, with a free-node structural ceiling of 173. Removing seed refinement is not currently supported.

The probability-mass stopping bound is also insufficient as a practical acceptance certificate. After one/two waves the mean remaining model-mass upper-bound gap is 30.06%/18.91%; actual model-mass gaps are 11.55%/7.19%. Relative to exact completion, the shallow trees miss 264/197 target positions but include 36 positions exact ranking loses. Do not equate model-mass optimality with target usefulness or tune a stopping threshold on these outcomes.

## Next action and prior art

Prepare a matched, GPU-resident construction comparison before another paid run. Cache invariant features and candidate weights, capture both policies comparably, and include all seed and selection work. A candidate-restricted scoring distribution must be an explicit new condition; it is not numerically equivalent to this full-vocabulary-normalized experiment. Include both proposal coverage and runtime so a fast but less useful tree cannot pass on timing alone. No training or generation is justified yet.

Parallel best-first search and threshold certification already appear in SpecExec; conditional trees and batched depth-wise construction already appear in DominoTree/DARTree. No novelty is claimed for those primitives. The possible distinction is using initial-draft prefixes across depths to reduce sequential construction, and it still needs a useful measured tradeoff against equally optimized alternatives. The released DARTree uses a different trained head and candidate normalization; our width-8 control is a matched structural control, not its official reproduction.

Evidence: `summary.json`, `analysis.json`, `mass-bound-analysis.json`, all per-context JSON files; scripts `analysis/replay_xpress_exact_tree.py --panel`, `analysis/analyze_xpress_exact_tree.py --panel --verify`, `analysis/analyze_xpress_mass_bound.py --panel`. Companion studies: `../xpress-exact-tree-local/BOUND_AND_SEED.md` and `../xpress-exact-tree-local/SERVING_COMPARISON_REQUIREMENTS.md`.

Remaining allowance is $12 of the existing $30 authorization. No new reservation. The overall research goal remains unmet.
