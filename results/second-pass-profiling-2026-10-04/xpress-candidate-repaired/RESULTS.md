# Candidate restriction preserves most branch value, but competitive headroom remains small

## Decision
Keep restricted scoring as a useful implementation option. Do not advance this fixed two-branch configuration to a5% generation claim, another kernel-only experiment, or automatic candidate-count sweep. It adds useful coverage, but its modeled advantage over the equally optimized control is small and uncertain. The next substantive question concerns better conditional continuations, not further shaving this same proposal's cost.

## Complete paired results
All256contexts from64 exposed development responses completed, all six methods timed with8 rotated paired repeats each. Same64 verification nodes. Positions include the known root and are matched-reference coverage, not emitted tokens.

| Method | Matching positions | Mean context-median proposal ms |
|---|---:|---:|
| full_shared | 2350 | 2.062 |
| full_conditional6 | 2475 | 3.653 |
| full_seed_restricted_branches | 2465 | 3.181 |
| restricted_shared | 2341 | 1.596 |
| restricted_conditional6 | 2454 | 2.644 |
| author_restricted_shared | 2339 | 1.813 |

Holding the full main draft, selector and packing scores fixed, restricted branches lose10positions across6contexts (no gains), saving0.472ms/proposal. Net progress beyond the same full shared-tree baseline falls from125 to115positions:92% of that net increment retained. This is an implementation comparison, not an exact numerical isolation of candidate exclusion. Seven lost positions are in four cases containing an absent correct suffix candidate; three losses in two cases are not explained by that necessary-support test. Intermediate states and projection/tie arithmetic also change.

Applied to both main and branches, restriction gives2454positions versus2341for the equivalently restricted shared tree:131repairpositions minus18lost through packing.34contexts improve and16worsen; one previously full-match context loses coverage. Added proposal time1.048ms. Its equal-response free-additional-work modeled saving is4.104% [1.825,6.541]; even the point estimate with branch work assumed free is below5%.

Charging measured proposal time with ASSUMED40ms common round work gives1.689% modeled savings [-0.642,4.184] versus restricted_shared. MATH2.457%[-0.157,5.490]; chat0.920%[-2.922,4.904]. Versus the separate small-tile author fused control,2.322%[-0.019,4.803]. These are conditional cost models, not actual target verification or generation. The author all-in-one kernel is slower here than the staged restricted path; report both instead of choosing the weaker control.

## What this explains
The shortcut preserves much of the conditional continuation benefit. Cost reduction alone does not supply enough additional useful progress for this configuration's5% aim. The restricted main has197incomplete contexts: selector misses the first-error site in67, chooses a wrong alternative in56, and chooses the correct alternative in74. The correct token is in the originaltop16 in164 and the correctedtop16 in185. These are main-chain errors, not errors of the stronger baseline tree.

All74 selected correct replacement prefixes already exist in the baseline tree. The measured extra benefit therefore concerns extending those alternatives with conditionally appropriate suffixes, not merely inserting a missing alternative token. The branch union adds131positions; forcing it into64nodes loses18through displacement. The full baseline union is an over-budget oracle and cannot be retained forfree.

A decision-relevant next LOCAL question: can already-retained alternative tree prefixes provide better starting points for conditional continuation than the present two uncertainty-selected edits? Use saved paths/scores and existing causal-oracle evidence to assess available opportunity and uncertainty before designing any new paid test. Counterarguments: useful prefixes cannot be recognized from scores; deeper prefixes consume budget and repeat prior conditional-tree work; additional continuation work can erase gains. This is not a recommendation for a new allocator yet, nor a novelty claim.

## Compatibility and verification
The previous attempt failed before measurements and stays preserved. Changing author launch tiles from CT128/HT64 to CT16/HT32 keeps its kernel body and16candidate set, removes unsafe undersized candidate accesses, and compiled onCPU forsm80 at156672bytes shared memory, below166912. CUDA then ran successfully. Reduction order changes: restricted staged vs all-in-one main chains differ in43/256contexts; restricted vs full main in120; suffixes in146. Eight direct author checks preserve originalfull6 outputs; restricted/eager discrepancies recorded rather than hidden. Repeated proposal trees are identical.

12CPU tests, all recovery hashes,1536independent tree reconstructions and primary analysis exact regeneration pass. Full shared/full conditional trees reproduce512prior control trees exactly. Models unchanged; no training, target transformer execution or new generation. Graph preparation12.19seconds reported separately, not amortized as a serving result.

## Resources and evidence
Worker113.87seconds;216.97seconds through staging/recovery/cleanup. All10artifacts recovered and hash-checked. App ap-QRZcbZyWnIe1nZMYePbc2M independently stopped with0tasks; ownedvolumeabsent. Conservative1.50USDcap held (not a settled bill);12USD remains in the existing30USDallowance, no active paid jobs.

Primary: analysis.json; mechanism-diagnosis.json; attempt1/recovered/contexts.json.gz; attempt1/recovered/checks.json; attempt1/recovered/compiled-resource.json; attempt1/execution.json. Reproduce with analysis/analyze_xpress_candidate_repaired.py --verify. Generic restricted scoring and conditional branching have prior art; useful algorithmic contribution and actual generation speedup remain unestablished. Goal active/unmet.
