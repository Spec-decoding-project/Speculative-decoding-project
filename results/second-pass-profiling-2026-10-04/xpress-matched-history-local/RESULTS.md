# Branch-specific feedback survives a matched-search comparison

## Question and counterargument

The earlier conditional-versus-shared comparison changed both the score source and search width. Its improvement could have been caused by a different search schedule rather than useful feedback from the candidate branch. We tested that explanation using saved evidence, without new model calls or spending.

## Intervention

Construct both trees with width eight, horizon sixteen, sixty-four retained nodes, identical probability accumulation, and identical tie ordering. In the control, every branch at a position uses the frozen initial chain's score row. In the conditional arm, each branch uses the row computed from its own token prefix. The frozen head, initial hidden features, underlying candidate policy, contexts and reference continuations stay fixed.

All 256 contexts from 64 previously exposed responses are included. The score rows use the earlier canonical single-row CPU arithmetic and full-vocabulary normalization. This is not the later candidate-64 GPU policy. Target labels are consulted only after construction. The independent constructor exactly reproduced all 256 prior conditional trees and all 256 wider shared-score trees. Source hashes match the previous independent audit and recovered feature/proposal receipts.

## Results

| Condition | Total matching positions |
|---|---:|
| Shared scores, width 64 | 2,350 |
| Shared scores, width 8 | 2,341 |
| Branch-conditioned scores, width 8 | 2,696 |

Against the matched width-eight control, conditional scoring gains 361 positions and loses six, a net 355. It improves 94 contexts, worsens six, and ties 156. No previously fully covered context loses coverage. The aggregate increase is 15.2%; weighting responses equally gives 19.3%, with a development bootstrap interval of 14.6–24.5%. These are different averaging conventions, not different experiments. Both domains improve on average.

The wider shared control loses only nine positions when narrowed to width eight. Thus narrowing that control does not explain away the large conditional-scoring difference.

## What this establishes

Under this fixed search, replacing shared score rows with branch-specific rows changes useful proposal coverage. This is a direct intervention on the score source, not merely a correlation between difficult tokens and errors. It strengthens the reason to reuse first-draft information to evaluate candidate continuations.

It does **not** isolate older history from the immediately preceding token: both can change. It does not prove these rows are calibrated target probabilities, that a learned smaller head can reproduce them, or that the extra computation earns its cost. Conditional trees already appear in prior work, including DARTree and DominoTree; this analysis is not novelty.

## Next decision

Do not launch another generation or imitation run from this result. The remaining mechanism question is narrower: how much of the coverage gain needs the entire branch prefix, versus only its most recent changed token? A matched frozen-head input intervention can answer that locally using existing features. This would determine whether a simpler reusable scoring computation has an evidence-based target. It must retain the same search, examples, scoring precision and fully correct controls; no tuned cutoff or favorable-case selection.

Before any end-to-end speed claim, the separate numerical-consistency problem still needs a demonstrated solution. An incomplete standalone batch-invariance flag is not such a solution.

Evidence: `analysis.json`; implementation: `analysis/analyze_xpress_matched_history.py`. Additional GPU cost: $0. Existing allowance remains $5.50. The research goal is unmet.
