# Retained-prefix opportunity: leaf-only continuation is insufficient

All 256 saved contexts from 64 exposed development responses were analyzed locally. No new model execution, training, target generation or GPU spending was required. Inputs were authenticated against recovery hashes.

## Finding

The matched eager-score 64-node tree fully covers 73 continuations. Of 183 incomplete continuations, the last correct prefix is a leaf in 20 and an internal node in 163. An internal node can have several children while still missing the correct child.

Providing the known-correct retained prefix and following the frozen head's greedy conditional predictions for up to six more positions adds 272 matching positions in 83 contexts. Nine useful starts are leaves; 74 are internal nodes. This is privileged information, with extra nodes and computation treated as free.

| Rule | Additional matching positions | Equal-response free-work modeled savings |
|---|---:|---:|
| Known-correct retained prefix | 272 | 11.01% [8.66%, 13.52%] |
| Known-correct leaf only | 17 | 0.74% [0.31%, 1.26%] |
| Two leaves selected by existing probability | 3 | 0.08% [0.00%, 0.25%] |

Intervals resample responses, not individual tokens. Even perfect leaf selection leaves very little room on this panel. Close the leaf-only continuation configuration; do not fund a leaf selector.

## Mechanism and limits

In two strict-margin cases, changing earlier history while retaining the same immediately preceding token changes the next prediction from wrong to correct. This isolates an earlier-history influence in those two cases. The other cases do not establish that earlier history is irrelevant: the immediate predecessor often changes too.

The correct next token is within the original top16 candidates in 77 of the 83 useful cases. Candidate presence is necessary support, not evidence that a deployable method selects or preserves it.

The tree contains an average of 58.94 nonterminal prefixes. Scoring every one, then recursively scoring newly promising prefixes, can be expensive. The correct prefix is unknown before verification. Conditional probabilities can also rank the wrong histories highly. These are the next obstacles, not solved by the oracle.

This is a greedy causal-continuation diagnostic under matched eager arithmetic. It is not a bound on all finite Jacobi iterates, other models or retrained heads. It does not preserve a 64-node budget after extension and measures no generation speedup. The earlier exploratory native/eager comparison (82 contexts/271 positions) mixed arithmetic; the matched result here is primary.

## Next decision

Assess whether exact conditional tree construction materially improves useful coverage over an equally sized shared-score tree, and how many sequential scoring rounds a batched implementation still needs. Conditional trees, prefix-probability ranking and depth-wise batching already have prior art. A new schedule must earn its place against those alternatives; exact search alone is not novelty.

Evidence: `analysis.json`; implementation: `analysis/analyze_xpress_retained_prefixes.py`. The existing campaign retains $12; no new reservation or paid work.
