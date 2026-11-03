# More tree coverage is partly consumed by construction and verification cost

All 16 hash-selected contexts from eight reused development responses completed, with 12 paired repetitions for each of four conditions: 768 timed rounds. No training or new response generation. All 101 artifacts (268,619,473 bytes) were recovered and authenticated. The app is stopped and the owned temporary volume is absent.

## Measured complete rounds

| Condition | Mean of context-median round times | Total committed positions across 16 contexts |
| --- | ---: | ---: |
| Original DFlash chain | 30.95 ms | 111 |
| Captured compact editor plus original | 31.53 ms | 114 |
| Ordinary top-four tree, first guess fixed | 33.38 ms | 120 |
| Ordinary top-four tree, first guess editable | 33.21 ms | 123 |

The editor adds approximately 0.59 ms to the original round; the fixed-first-guess tree adds 2.44 ms. Both include all live proposal computation, host transfers, tree/packing work, actual target verification, acceptance, feature gathering and cache compaction. The editor uses its existing CUDA graph; all target verifiers use CUDA graphs. Initial model loading, graph setup and measurement-reset copies are outside steady-round timing and are reported separately.

The fixed-first-guess tree's response-weighted time per committed position is 0.29% lower than the editor's, with descriptive response-bootstrap interval from 5.86% lower to 4.31% higher. The editable-first-guess tree is 4.48% lower than the editor, with interval from 12.77% lower to 2.93% higher. Neither comparison establishes superiority or equivalence.

Versus original DFlash, the editor, fixed tree and editable tree have response-weighted time-per-position changes of -3.46%, -3.40% and -7.02%. All intervals cross zero. These are **fixed-context round measurements, not generation speedups**. The summary weights eight responses equally; it is not obtained by dividing the totals in the table. The pending bonus token is not credited twice; the known root is included consistently.

## Correctness and numerical scope

Native first-pass capture checks passed; editor eager/captured scores and decisions matched exactly. All timed repetitions reproduced their respective audited outputs. All tree cache gathers matched exactly at every layer; following-token checks passed in every condition/context.

Numerical differences from fresh causal scoring occurred in 2 original, 1 editor, 4 fixed-tree and 6 editable-tree contexts. Most concerned unselected paths. One original and one editable-tree context changed the selected accepted decision relative to causal scoring. The editor and fixed tree had no such selected-path difference in this panel. All cases remain in the report, including score tensors; no agreeing-only result is substituted. This does not establish generally lossless decoding or make numerical disagreements harmless.

The GPU was an A100 80GB PCIe. Earlier score collection used A100 SXM4; all timing comparisons here share the same PCIe device. Hardware/prefix construction differs from the saved-feature calculation, so old coverage and new timing cannot be combined into an observed speedup.

## Relation to the saved coverage study

A $0 post-hoc overlap check found five changed original token guesses across five of these 16 contexts when comparing saved features with live reconstructed prefix/cache execution. On these selected contexts, saved and live ordinary-tree response-average coverage gains happen to agree: fixed tree +14.09%, editable tree +21.38%. The editor's gain changes from +4.06% saved to +6.25% live. Its folded BF16 serving path and live reconstructed context are not the same scoring computation as the saved training evaluation; this is not evidence that graph capture improved model quality. Eager/captured parity within the serving path was exact.

The small eight-response sample is inadequate for a population claim about these differences. The overlap analysis is descriptive and does not isolate which numerical/context change caused each token difference.

## Decision for the research direction

Do not describe the conventional tree as having already defeated the editor in practical performance. It yields more coverage, but this CPU tree implementation costs more, and the direct cost comparison is inconclusive. Conversely, beating a single original draft still does not establish an editor contribution. The ordinary tree must remain a strong comparison.

The positive systems observation is narrow: the existing captured editor adds only about 0.59 ms per round here. Useful prediction quality remains the larger unresolved issue. That is a reason to keep studying whether better corrections can be learned at this cost, not a reason to promise another fit will succeed. Optimizing the conventional tree could also narrow the editor's cost advantage; no claim of an optimally implemented tree is made.

The full teacher's residual opportunity beyond the ordinary tree, measured in the previous larger saved panel, was only about 4.4–5.1% additional coverage before paying for any new work. Therefore simply reproducing more teacher argmax tokens does not establish a path to a 5% advantage over this stronger reference. A next learning proposal needs to identify useful improvements beyond that comparison, or a measured way to obtain comparable progress more cheaply. No new training recipe, loss sweep or numerical-repair run is automatically justified by this result.

Research remains active. The original goal is not complete: no reliable cheap replacement with an established advantage over strong methods has been demonstrated.

## Execution and evidence

8 minutes 20 seconds including staging, execution, recovery and cleanup. App `ap-2QjHpFI0W3GJPcviWBjl6t` stopped with zero tasks; owned volume `vo-vQImvBZNghK4zlZqsEQXg3` is absent. No retries. Full $1.50 retained conservatively for unsettled compute/storage/recovery; this is not a settled invoice. $8.78506668 remains unallocated, no active jobs or reservations.

[Protocol](../PROTOCOL.md) · [Authenticated analysis](analysis.json) · [Saved/live overlap](saved-live-overlap.json) · [Execution](execution.json) · [Recovery](recovery.json) · [Setup costs](recovered/setup.json) · [All context audits](recovered/contexts.json).
