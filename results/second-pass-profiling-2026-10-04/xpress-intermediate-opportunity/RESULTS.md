# What additional refinement can still recover

The completed CPU replay found **zero additional matching-prefix positions** in intermediate drafts beyond keeping the initial and sixth-pass drafts. This closes whole-chain intermediate recycling on this panel; it does not cover shifted n-grams, other workloads, or other models.

## Evidence and scope

All 16 contexts from eight exposed development responses completed in 665.29 seconds, using the recovered released 4B xPress head and its own backbone features. Input hashes, unchanged weights, and the manual trace against the author's CPU callback passed. Fourteen final chains matched the earlier CUDA result exactly; two differed by one token. These are BF16 CPU results, not a reproduction of every fused CUDA trajectory. No training, new generation, or GPU spending occurred.

Results: `results.json`; reproducible calculation: `analysis/audit_xpress_iteration_opportunity.py` at the repository root. Intermediate chains also add zero coverage to the recorded published-union, shared-spine, full-history, and depth-wise controls; these cross-device comparisons are exploratory.

## Why unfinished refinement can be useless

With backbone features and weights fixed, prediction at position k uses only token choices before k. If two successive iterations agree before their first changed position d, the new prediction at d is fixed too: its inputs can no longer change. All positions through d are therefore fixed for subsequent identical iterations. An incorrect token inside that prefix cannot be corrected by waiting longer.

The implementation shifts token IDs by one position and uses a lower-triangular mixer. The certificate was checked over 4,096 transitions of exhaustive small deterministic causal maps, alongside inspection of the released head. This is a computation-specific invariant, not a statement that the fixed prediction is correct.

Applying it to the saved trace:

| Outcome after six passes | Contexts |
| --- | ---: |
| All valid tokens already match | 6 |
| First mistake is fixed by the causal map | 8 |
| More identical passes could still help | 2 |

Nine entire blocks were still changing, but seven of those had already fixed their first wrong token. Counting changing blocks would therefore exaggerate useful remaining work.

Even granting perfect continuation in both unresolved cases, another seven positions are available. With free extra computation and retention of both existing endpoint drafts, the equal-response modeled savings ceiling is 3.98% on this panel. This is not measured latency. Against the recorded GPU shared-spine trees, the exploratory cross-device ceiling is 2.60%; that must not be called a GPU guarantee.

Certificate and counts: `followup-analysis.json`; calculation: `analysis/analyze_xpress_iteration_opportunity.py`.

## What this changes

Do not fund a longer-iteration-only experiment on these contexts or claim that intermediate complete chains provide new support. To escape a fixed wrong preference, something must change: a supplied token choice, the model's inputs, or its weights. That requirement does not establish which intervention is useful or economical.

The saved CUDA scores provide only partial evidence about changing choices: among six fixed-error cases with matching CPU/GPU final chains, the correct token ranks 3, 3, 2, 9, 2, and outside the top 16. Only two are top-two alternatives. Historical conditional rows are incomplete and selected by beam search; they cannot justify a new branch policy by themselves.

## Completed oracle-restart intervention

`oracle-restart.json` completed in 64.49 seconds locally. It evaluates the same head on correct preceding token IDs, with its original fixed features and logits. Strict causality makes the first error in these scores exactly the first error of a converged greedy sequence. If an oracle supplies the correct token there, the next erroneous score locates the next failure of that conditionally generated continuation. Scores after that failure do not describe the generated path. An independent exhaustive check over 1,024 small causal maps and target sequences confirmed this prefix equivalence. The recovered mixer has zero entries above its diagonal; two actual-head future-input perturbations preserved earlier scores exactly.

For the eight cases with an already-fixed first mistake:

| After supplying the correct token at the first mistake | Cases |
| --- | ---: |
| Conditional continuation improves on copying the old suffix | 3 |
| Same correct-prefix length as copying the old suffix | 5 |
| Conditional continuation is worse | 0 |

Conditional continuation adds four positions beyond the single-token patch across those eight cases. Six correct replacements rank within the head's top four; the other two rank 10th and 28th under exact CPU tie ordering. Both figures concern privileged first-error locations, not a discovered selector. One CPU/GPU top-16 rank differs from the historical score check; device precision remains material.

The conditional examples are specific: UltraChat 12677 at anchor 8 goes from two matching tokens to three after the patch, then four after conditioning; its anchor 24 goes from one to two to three. UltraChat 12067 at anchor 8 goes from eight to nine to eleven, but its required replacement ranks tenth. In GSM8K 6150 at anchor 24, a single replacement already preserves the entire 16-token block; more conditional computation adds nothing.

This separates two mechanisms: **finding a better token** and **updating later predictions because that token changed**. The first can be valuable without the second. These interventions are conditional on frozen first-pass features; they do not identify the linguistic cause of the model's preference or a way to find the correct token cheaply.

An oracle union of a converged, once-repaired chain and the GPU shared-spine tree adds 20 positions. The resulting free-work, cross-device modeled number is 13.34%, versus 23.39% against the weaker CPU endpoint union. Neither is achievable progress or a speedup: the oracle knows the error and replacement, can converge for free, retains fallback paths without charging a node budget, and spans different execution backends. It would be misleading to use either number as a method's headline.

Next: assess the prevalence of both mechanisms on the already-saved 64-response development panel, before designing or training a new selector/editor. The small panel contains enough distinctions to define that test, not enough to endorse an implementation. No new GPU work or funding reservation has occurred; research remains active.

## Prior-work boundary

[Lookahead Decoding](https://github.com/hao-ai-lab/LookaheadDecoding) already reuses Jacobi trajectories. [xPress](https://arxiv.org/html/2608.02438v1) already evaluates refinement iterations and composition with draft trees. Neither trajectory reuse nor running more iterations is claimed as a new contribution.
