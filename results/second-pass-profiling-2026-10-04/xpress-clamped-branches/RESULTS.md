# Conditional continuation works on real selected alternatives; present cost margin is insufficient

## Decision
Keep conditional redrafting as the mechanism under investigation. Do not advance this fixed late-branch configuration to generation or another student fit. This completed test shows that recomputing the continuation, after changing a model-selected token, adds useful proposal support beyond both copying the old suffix and merely doing more unconditioned refinement. It has not shown a competitive 5% end-to-end latency reduction.

The next unresolved scheduling question is whether useful branch computation can replace some of the initial refinement work rather than being appended after all six original updates. An earlier decision uses less reliable scores and possibly an unsettled prefix; any test must measure that loss. This is a candidate investigation, not a justified method or novelty claim. No new paid attempt is reserved.

## Completed comparison
All 256 contexts from 64 previously exposed responses completed, all seven conditions using exactly 64 ancestor-closed nodes. Frozen released xPress weights; no training, new generation, target transformer or draft backbone execution. Root tokens are included in the coverage counts, consistently across all arms.

| Proposal | Matching positions, all 256 contexts | Added positions versus shared64 | Mean of context-median proposal time, separate 16-context timing subset |
|---|---:|---:|---:|
| Ordinary original-score tree | 2,153 | -197 | 0.747 ms |
| Published-style six-pass chain/tree union | 2,310 | -40 | 2.017 ms |
| Shared-spine corrected-score tree | 2,350 | 0 | 2.003 ms |
| Two alternative tokens, old suffix copied | 2,340 | -10 | 2.305 ms |
| Two alternatives, two clamped updates | 2,411 | +61 | 2.858 ms |
| Two alternatives, six clamped updates | 2,475 | +125 | 3.560 ms |
| Two additional ordinary updates | 2,353 | +3 | 2.777 ms |

These are saved-context coverage and measured **proposal** times, not accepted generation tokens or whole-request latency. The timing subset is eight predetermined responses, two contexts per response, eight rotated paired repetitions. Do not silently combine its timing with full-panel coverage as a measured full-panel latency result.

## What caused the observed difference?
The copy and conditional arms use the same selected sites, alternatives, initial branches, weights and core tree. Two conditional updates add 71 net positions versus copying; six add 135. Two unconditioned extra updates add only three versus shared64. Thus the particular intervention—recomputing after an alternative token—has value that simply copying suffixes or continuing the settled main chain does not capture here. The eight-pass control is cheaper than two branched passes; it is not a matched-FLOP claim.

The two-update and six-update variants improve the same 36 contexts versus shared64 and damage the same 16. Two updates gain 78 positions and lose 17; six gain 142 and lose 17. Six improves over two in 19 contexts, loses in zero, for 64 additional positions (42 math, 22 general instruction). Each of these 19 cases selected the actual first-error replacement. This localizes the extra progress to successful propagation after a correct alternative, rather than a new selector or more training.

Causal interpretation: the frozen head predicts a position from preceding token identities and fixed first-pass features. Clamping through an alternative gives later updates a changed, stable prefix. One update guarantees convergence one further position into the suffix; later positions may settle sooner, but need not. Six passes are not assumed globally converged. This deterministic structural property explains why two updates can stop short without implying that later tokens are intrinsically unpredictable. The intervention establishes effects for these saved contexts; it does not explain all language dependencies or establish generalization to new generation trajectories.

The selector identified the true first-error site in 126 contexts and selected its correct replacement in 71. In 35 of those 71, shared64 already covered at least as much as the six-update conditional branch; only 36 contributed additional support. Both conditional variants' 17 lost positions in 16 damaged contexts were recoverable by restoring shared64 nodes: they are fixed-budget displacement losses. This is hindsight diagnosis, not a deployable oracle that can keep only helpful branches.

## Economic result
Relative to shared64, two updates add 0.855 ms proposal work on the timing subset; six add 1.557 ms. The 64-response equal-response **free-work** modeled savings are 2.64% [0.77, 4.55] and 4.71% [2.22, 7.39], respectively. Those scenarios charge zero additional work and assume fixed-context progress translates into fewer rounds. Neither is an actual speedup; the six-pass mean is already below 5% before charging its extra work.

On the actual eight-response timing subset, adding an illustrative common 40 ms per round gives modeled savings versus shared64 of **-1.43% [-3.68, 1.41]** for two updates and **-0.24% [-5.34, 8.33]** for six. Positive means faster. At common 80 ms, six gives +1.50% [-3.53, 9.94]. Common work was not measured here. These noisy scenarios do not support a performance claim or a population-wide rejection. Better-looking comparisons against ordinary/published unions must not replace the stronger shared64 comparison.

Graph setup took 5.75 seconds and is not amortized into the table. A real serving claim must account for setup as appropriate, complete target execution, output agreement, stopping behavior and fresh workload performance.

## Additional local counterfactual
Because displacement explained every loss, tested one fixed rule using saved branches: keep only the prefix guaranteed settled by the number of clamped updates (through cut + updates), then fill with ordinary corrected-score nodes. This is label-free, not a threshold sweep. It loses two aggregate positions for the two-update arm and nine for six updates versus full-branch packing. Equal-response differences versus the original policies have intervals crossing zero. It does not provide a clear packing-only repair. Full details: [diagnosis.json](diagnosis.json). It was not GPU timed or prospectively confirmed.

## Validation and resources
15 local tests passed. Staged source/input checks passed. All eight GPU author-wrapper comparisons for six and eight passes were exact; repeat trees were exact. Eager/fused two-update branches matched at all eight checked cases. All 256 initial native chains reproduced the preceding saved native chains exactly. Weights stayed unchanged. These checks do not test native target-generation fidelity.

All nine recovered artifacts authenticated. An independent implementation reconstructed all 1,792 trees and recounted all prefixes, then regenerated analysis exactly. The Windows/Linux exponential calculation differed by one ULP in a selection probability; validation allows 1e-14 relative/1e-16 absolute error in that scalar only and still requires exact sites and token choices. Frozen GPU evidence and selection were unchanged.

App `ap-TRODhk16lXqNlYOv0D8QW3` stopped with zero tasks; owned volume absence independently confirmed. Total through cleanup: 175.08 seconds (2m55s). Full $1.50 cap held conservatively pending billing. $16.50 remains in the new $30 authorization; no active jobs or reservations.

## Reproduction
Run `analysis/analyze_xpress_clamped_branches.py --verify` with `.venv-profile-launch` to authenticate and reproduce the primary analysis. Run `analysis/diagnose_xpress_clamped_branches.py` for the posthoc mechanism/packing diagnosis. Frozen sources and plan are under `frozen/`; original GPU evidence under `attempt1/recovered/`. Re-running the controller is not part of reproduction and is blocked by the existing attempt directory.
