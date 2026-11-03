# Older branch feedback supplies about half the measured gain

All 256 saved contexts from 64 development responses completed locally in 5 minutes 40 seconds. No training, target generation, GPU allocation, or additional spending occurred. The original process exited successfully.

## Matched result

The model, hidden features, first-draft seed, numerical scoring path, width-eight search, sixteen-position horizon, and sixty-four-node verification budget were fixed. The only intervention was which branch tokens entered the score function.

| Feedback supplied to the frozen head | Total matching positions | Net gain over shared scores |
|---|---:|---:|
| Initial chain only | 2,341 | — |
| Branch's last token, earlier tokens from initial chain | 2,514 | +173 |
| Complete branch prefix | 2,696 | +355 |

The last-token substitution retains 48.7% of full history's net gain over the shared control. That is a ratio of aggregate gains, not a fraction of independent corrections or a speedup.

Compared with shared scores, last-token feedback gains 179 positions and loses six, improves 88 contexts and worsens six, and damages no fully covered contexts. Compared with full history, it gains only three positions and loses 185: three contexts improve, fifty worsen, and fifteen contexts lose full coverage. Full and last-token trees are identical in only nine of 256 contexts.

The equal-response coverage change of last-token relative to full history is −6.77%, with a descriptive development interval of −8.81% to −4.90%. Both domain averages are negative. These are reused data, not fresh confirmation.

## What older history actually changes

For all fifty contexts where full history covers more, we inspected the first correct edge omitted by the last-token tree. We compare scores for the **same branch prefix**, keep the immediately preceding token and first-draft hidden features fixed, and change only earlier token guesses.

- In **34 cases**, supplying the actual older branch tokens changes a strictly preferred wrong next token into a strictly preferred correct next token. Neither side is an exact top-score tie. This directly establishes an older-history effect on the frozen head's decision in those cases.
- The correct token's local rank improves in forty cases and ties in ten; it worsens in none.
- Under last-token scoring, eight omitted correct edges fall outside the top sixteen candidates, thirty are removed by the width-eight search, and twelve survive that stage but are removed by the final sixty-four-node budget.

The latter categories explain *where* correct paths disappear. They do not mean a larger tree is the fix: changing scores affects both the useful branch and competing branches. Nor do the thirty-four local winner flips individually account for every retained or lost continuation token.

## Consequence for the method

Close the assumption that the current frozen head's useful branch corrections can be recovered mainly by substituting only the last token into the initial chain. Do not fund a last-token-only generation run on this evidence.

This is more specific than saying “the editor needs more information.” The first draft already supplies the hidden features in both arms. What matters in these cases is making its later decisions consistent with earlier **changed choices along this particular branch**. Two branches with the same last token can therefore require different next-token scores.

This does not prove every older token is needed, that every branch needs the same amount of recomputation, or that a separately trained last-token model would fail. The tested hybrid prefix may be inconsistent, and the head was trained to use full causal history. We have identified a computation a cheaper implementation must preserve or approximate, not demonstrated that a new smaller network can learn it.

## Cost and prior work

Last-token scoring needs 14,680 distinct position/token rows for 28,928 requested prefix scores. That exposes possible reuse, but losing half the useful gain makes row-count reduction insufficient justification. No GPU time or complete-round cost was measured here.

[LiLiCorr](https://research.nvidia.com/labs/nemotron/lilicorr/) already precomputes adjacent-candidate scores; its candidate representations use the full lattice. Pinned [DARTree code](https://github.com/VILA-Lab/DARTree/blob/5bb3240e806a9ccdc2c65e98d3bc40998a9845f1/utils/correction.py) already caches projections and carries a branch-specific GRU state. Neither last-token scoring nor caching branch state is a new contribution by itself. The difference between our frozen-head ablation and those separately trained systems must remain explicit.

The next research question is how to preserve the useful older-history contribution at lower cost. Before another implementation or paid run, use the actual frozen head's algebra and measured cost breakdown to determine what work can be reused and what nonlinear scoring still must run. Do not introduce another head, window-length sweep, or learned gate merely because the last-token shortcut lost coverage. The separate target-output consistency blocker still prevents an end-to-end speed claim.

## Verification and evidence

All source feature/proposal and historical score hashes passed. Root and deeper seed rows reproduced exactly in all contexts. Frozen weight versions remained unchanged. An independent heap-based constructor reproduced all 768 shared/full/last-token trees and their coverage; first-missing-edge classifications were independently reconstructed. The complete result regenerates exactly.

- `analysis.json`: panel comparisons and source hashes.
- `miss-anatomy.json`: all fifty failure cases and fixed-prefix score comparisons.
- `summary.json` and per-context files: full recorded rows and trees.
- `prior-work-check.json`: source scope.

Existing campaign allowance: $5.50 remains, no new reservation. The goal is active and unmet.
