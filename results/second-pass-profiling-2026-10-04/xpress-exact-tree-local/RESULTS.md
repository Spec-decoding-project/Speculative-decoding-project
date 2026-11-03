# Conditional-tree value is real; exact completion is expensive

Completed 16 saved contexts from eight exposed responses using the frozen released xPress head, authenticated recovered features and a canonical single-row BF16 CPU scorer. No GPU spending, training, new target execution or generation. The second invocation completed the remaining 15 contexts in 46.9 seconds; the first context took 2.55 seconds of analysis. These are local computation times, not serving measurements.

| Equal 64-node proposal | Matching positions |
|---|---:|
| Scores shared across branches | 178 |
| Exact conditional best-first tree | 188 |
| Conditional depth-wise width8, pruned to64 | 188 |

The two conditional builders have identical coverage on all16 contexts, though this does not require identical trees. Exact conditional search improves four contexts by ten positions and worsens none. At equal zero additional cost, the equal-response modeled saving is 8.09% [1.56%, 16.02%]. This is not actual latency or a bound on another workload.

## What the new schedule did

The initial shared64 tree supplies guessed prefixes to score together. Subsequent batches score unknown prefixes that could still alter the exact final tree. All16 completed trees and scores match the sequential heap using the identical cached conditional score function. Forty-four synthetic tests also compare against exhaustive enumeration, including ties and incomplete query caps.

However, completion needs 4–14 batches and89–178 prefix rows. Sequential heap uses50–63 rows; depth-wise width8 uses113 rows in15 batches. The warm seed therefore gives a limited reduction in barriers while often doing more work. There is no measured GPU cost advantage. Do not optimize this exact configuration solely to obtain an exact-search certificate.

## Exploratory anatomy of intermediate proposals

One batch gives184 matching positions, two give186, and exact completion gives188. Twelve of16 contexts reach their final coverage in the first batch, though their selected trees continue changing. Late score-certification work often does not affect useful coverage. This is retrospective development evidence: stopping after one or two batches was not the original primary success test, and intermediate coverage need not improve monotonically.

A broader saved-panel comparison is justified to distinguish a real low-depth opportunity from this small panel's composition. It must retain both shared64 and the conditional depth-wise comparator, count damage, and preserve the difference between incomplete model-score search and valid target-verifiable proposals. It must not claim that two batches always suffice or that incomplete search has the same tree as exact search.

## Prior art and novelty

[DominoTree](https://arxiv.org/html/2607.08642) already describes conditional best-first scoring, restricted candidates, and a graph-captured depth-wise serving builder. Its width-equals-budget construction can recover the same node set as the heap under the stated conditions. Beating an unoptimized Python heap would not be an adequate comparison.

[DARTree](https://arxiv.org/html/2608.13524) uses fixed-width depth-wise conditional expansion followed by global pruning. Our width8 comparison tests that structural alternative; it is not an official DARTree reproduction, which uses additional specified settings including candidate restriction and a depth bonus.

The possible distinction under investigation is speculative scheduling across depths using the initial draft's guessed prefixes. Conditional tree scoring and the nonpositive-score stopping bound themselves are established. No novel contribution or serving improvement is established.

[SpecExec (NeurIPS2024)](https://papers.neurips.cc/paper_files/paper/2024/file/1d91d5689e251d27993a3c2182dddcf7-Paper-Conference.pdf), Algorithm2 and AppendixA, already performs parallel shortest-path search with a cumulative-score stopping threshold. Thus batching a best-first frontier and certifying completion are also prior art. The only possible remaining distinction here is using already available parallel-draft prefixes to speculate across several depths before their ancestors would normally be expanded, and obtaining a measured benefit from a shallow completion schedule. That distinction is a research question, not yet a contribution.

Evidence: `summary.json`, individual context records, `analysis.json`. Code: `analysis/replay_xpress_exact_tree.py`, `analysis/analyze_xpress_exact_tree.py`, `experiments/xpress_batched_tree_search.py`.
