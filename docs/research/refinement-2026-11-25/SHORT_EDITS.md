# Shorter editing loses most of the useful correction

Eight of the nine shortened configurations make DFlash proposals worse. The remaining four-token linear gate gains only 0.0176 prefix tokens at about 7.6 ms, with an uncertainty interval spanning zero. **Simply stopping after one, two or four edits does not solve the computation problem.** Keep this negative result; do not choose a different budget per example using reference labels.

Job **8579195** completed November 25 in **4m30s**, exit 0, using one A5000/four CPUs/5 GiB; peak RSS 1,621,164 KiB. Frozen source `d2d3a54041039b83b326324b39b485bc136eeab0`. Zero optimizer updates. All three earlier gate checkpoints and the generator remain unchanged.

Every configuration reads the same context and whole proposal, edits the first configured number of editable positions autoregressively, then copies the rest. The first two seed tokens remain fixed. The short outputs match the corresponding prefix of full editing exactly, with untouched tails copied. This can change correct-prefix quality in either direction: an early repair can require later repairs to remain useful, or stopping can avoid a later destructive edit.

Development has 2,048 blocks from 512 previously inspected prompts. DFlash mean correct prefix is **3.577637**. All three gates and all four budgets were declared before running this comparison; none was selected using development labels.

| Gate | Edited positions | Mean prefix | Change and descriptive 95% interval | Editor ms | Screening break-even baseline round ms |
|---|---:|---:|---|---:|---:|
| scores | 1 | 3.510742 | -0.066895 [-0.101074, -0.035645] | 2.15 | No positive gain |
| scores | 2 | 3.500000 | -0.077637 [-0.125000, -0.033691] | 4.00 | No positive gain |
| scores | 4 | 3.538086 | -0.039551 [-0.101074, +0.018555] | 7.66 | No positive gain |
| scores | 14 | 3.655762 | +0.078125 [-0.000012, +0.155273] | 25.95 | 1521 |
| linear | 1 | 3.550293 | -0.027344 [-0.067383, +0.011719] | 2.14 | No positive gain |
| linear | 2 | 3.552246 | -0.025391 [-0.074707, +0.022461] | 3.95 | No positive gain |
| linear | 4 | 3.595215 | +0.017578 [-0.047363, +0.080078] | 7.58 | 1974 |
| linear | 14 | 3.699707 | +0.122070 [+0.041003, +0.204102] | 25.70 | 964 |
| mlp | 1 | 3.509277 | -0.068359 [-0.112793, -0.026843] | 2.20 | No positive gain |
| mlp | 2 | 3.507324 | -0.070312 [-0.127441, -0.016113] | 4.08 | No positive gain |
| mlp | 4 | 3.559082 | -0.018555 [-0.091797, +0.051758] | 7.75 | No positive gain |
| mlp | 14 | 3.676758 | +0.099121 [+0.009277, +0.189453] | 26.29 | 1214 |

The break-even column assumes unchanged verification cost and the same prefix distribution, adding one target token to progress per round. It is editor_ms × (baseline_prefix+1) / prefix_gain. It is a screening calculation, not a measurement of native generation speed. A negative quality change cannot repay a positive editor cost under these assumptions. No native end-to-end run or fresh confirmation was performed.

All 51 local tests and twelve pinned GPU tests passed. Each of the nine short configurations reproduced all 512 calibration and 2,048 development outputs/actions expected from the saved causal prefix, totaling 23,040 checked proposals. All twelve configurations passed cached/full agreement and two-warmup/ten-repeat timing on 64 fixed development blocks. Generator hashes and parent artifacts stayed unchanged. A separate Mac recount verified every saved prefix/repair/damage metric and all backup hashes.

The next bounded learning comparison expands gate optimization to 9,392 distinct training prompts, retaining the generator and exact original calibration population. The first fit contained only 991 decisive positions; broader coverage tests a plausible limitation without adding model capacity. The protocol is predeclared in [HISTORY.md](../../../HISTORY.md). This does not erase the unresolved full-loop cost.

Evidence: [complete results](evidence/short-edits/complete.json), [run contract](evidence/short-edits/run.json), [independent recount](evidence/short-edits/mac-audit.json). Private artifacts: remote `research/short-edits-20261125/`; Mac `~/.local/share/wayfinder-artifacts/short-edits-20261125/`.
