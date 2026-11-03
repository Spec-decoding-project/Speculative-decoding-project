# Can weaker versions of the learned corrections solve the deficit?

## Result

Not on this sampled panel under the stated fixed-context model. An answer-informed per-block strength between0and1 raises added prefix positions from33to42, improving six of128contexts. Even if producing and verifying those corrections were free, the equal-prompt modeled saving is4.073%, below the5%point target. This supplies adverse evidence for simple down-scaling as the main remedy, without claiming a population-level impossibility.

| Correction rule | Added positions over recorded original | Equal-prompt progress gain | Zero-added-cost modeled saving |
|---|---:|---:|---:|
| No correction, reconstructed readout | 9 | 1.051% | 0.768% |
| Original learned strength1 | 33 | 5.451% | 3.266% |
| Best answer-informed block strength in[0,1] | 42 | 6.771% | 4.073% |

The oracle permits a different strength for every context and knows the correct labels. It is more permissive than any global shrinkage calibration. It does not choose separate strengths at each position, amplify above1, change correction direction, alter model weights, introduce candidates or modify the immutable first guess. No claim about those other interventions follows.

## Computation and checks

Same frozen probability-trained checkpoint and128previously sampled contexts, two per64exposed development prompts. Reconstruct first-pass vocabulary scores b and learned score changes d in the same FP32 chunked readout used by the previous input audit. Its alpha1 token outputs reproduce that audit exactly. For each desired token y versus every vocabulary competitor j, intersect alpha*(d_y-d_j)>=b_j-b_y across all earlier required positions and0<=alpha<=1. Require strict inequality when a lower-index competitor would win a tie. Select an interior witness for the longest feasible prefix and explicitly verify its full-vocabulary winner sequence.

These inequalities are interpreted in float64 on the reconstructed FP32 component values. The endpoint alpha1 has zero token differences between float64 and FP32 addition in this sample. This is not native BF16 CUDA equivalence: the prior actual-input audit already recorded55valid-token discrepancies against the historical GPU model. Alpha0's nine extra positions likewise reflect numerical rereading of the original scores, not learned corrections. They remain visible as a control and are not credited to the method.

The vectorized interval implementation matches the previously checked scalar implementation on100random multirow problems and explicit open/closed endpoint and permanent-tie fixtures. Independently recounted all128witness prefix lengths,256endpoint lengths, interval membership and economic aggregates. All pass. The6contexts with longer union prefixes contribute9extra positions in total. Execution took146.18seconds, CPU only, no cloud cost.

## Why6.77%progress does not imply6.77%speed

Each prompt has its own original progress P and edited-union progress U. Equal-prompt progress averages U/P-1. The free-computation saving instead averages1-P/U. These inverse ratios are not interchangeable, especially with gains concentrated in a few prompts. With added round cost c, modeled relative time is(1+c)*mean(P/U). The oracle requires negative0.967%added cost to reach5%on this sample, meaning it misses that target even at c=0.

## Decision

Do not fund global down-scaling or a context-dependent scalar selector as the next substantial-improvement method based on these saved directions. The optimistic oracle already falls short on the point estimate and would still require a learnable selection rule. This is a sampled development result, not a general statistical rejection or a claim that modest calibration gains have no use.

The direction remains cheap useful second-pass correction. Its next learning change must produce different useful prediction changes or a materially stronger model, rather than merely weakening all existing changes together. No specific architecture, new loss or training dose is established by this negative result. Source code and artifacts: analysis/audit_correction_shrinkage.py, checks.json, results.json, rows.json and verification.json.
