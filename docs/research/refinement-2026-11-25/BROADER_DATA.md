# Broader gate training does not reliably improve the result

Increasing decision-training coverage helps the score-only gate slightly, but the two representation gates have lower development point estimates. Their differences from the smaller fits have intervals spanning zero. **More data alone has not produced a material improvement or justified a larger model.** Stop expanding the dataset blindly and investigate full-loop execution cost.

Job **8579313** completed November 25 in **10m33s**, exit 0, from source `b3c4ce451beb3afb93335549e2f41f478d3d2186`. One A5000/four CPUs/5 GiB, peak RSS 2,250,676 KiB. All three arms ran 2,000 updates, reaching the predeclared cap. The selected score/linear/nonlinear checkpoints are updates **1,980 / 1,980 / 990** (epochs 60/60/30). No final-test access, new native collection, or target/drafter updates.

The original 40,750,336-parameter generator stayed frozen. Gate sizes remained 9, 265 and 17,025 parameters, with the same initialization, features, optimizer and per-epoch selection rules. Optimization expanded from 1,536 to **9,392 distinct training prompts**: 600 natural preservation and 4,096 repair examples per source. Source/kind weights preserve the previous equal-group objective despite scarce preservation prompts. The exact original 512 calibration prompts and all their cached features remain unchanged. Development still contains 2,048 blocks from 512 separate prompts.

This yields **8,300 decisive positions**, versus 991 before: 1,170 COPY and 7,130 generate targets before group weighting. The raw number of suffix positions is much larger, but positions where both choices agree or the reference prefix is no longer reachable do not teach a useful decision. Training uses only editor-owned representations and token scores; no target/drafter hidden states enter the editor.

| Gate | Smaller-fit prefix | Broader-fit prefix | Improved / damaged blocks | Broader minus smaller, descriptive 95% interval | Editor ms |
|---|---:|---:|---:|---|---:|
| scores | 3.655762 | 3.689453 | 235 / 110 | +0.033691 [+0.007812, +0.062500] | 26.00 |
| linear | 3.699707 | 3.679688 | 289 / 203 | -0.020020 [-0.075195, +0.033691] | 25.85 |
| mlp | 3.676758 | 3.635254 | 288 / 236 | -0.041504 [-0.102539, +0.017578] | 26.70 |

DFlash mean prefix is **3.577637**. Every broader-fit point estimate remains above DFlash, but no broader arm beats the earlier smaller linear gate's **3.699707**. These are exploratory, repeatedly inspected development results, not fresh confirmation or training-seed estimates. Confidence intervals use paired whole-prompt resampling. The same per-epoch selection schedule means calibration snapshots occur every 165 updates here versus 20 earlier; this is a learning-curve comparison under fixed rules, not an isolated causal estimate of sample count.

Lower loss does not imply higher accepted-prefix length. In particular, the larger linear fit improves its calibration prefix while its development point estimate decreases. The two evaluation populations have different preservation rates, and a small calibration improvement should not be treated as proof of generalization. The full inference loop still costs about 26 ms.

For context, an answer-informed upper bound on the **calibration** candidate pairs is prefix 10.042969 versus DFlash 9.458984; 77 calibration blocks have some attainable improvement. This uses reference labels to choose between candidates and is not a usable inference policy or a development result. It shows remaining decision headroom while also showing that the generator cannot supply a repair everywhere. [Upper-bound calculation](evidence/broader/calibration-headroom.json).

All 53 local tests and thirteen pinned-environment tests passed. Full-size integration controls, unchanged generator hashes, selected-checkpoint reloads, all 512 calibration-prefix simulations per arm, and 64-block cached/full timing checks passed. Independent Mac recounts match every raw calibration/development audit. Selection histories and all 2,000 contiguous update records per arm were checked. Every backup file hash matches Nexus. The calibration feature cache is **bitwise identical** to the earlier fit; group-balanced objective weights were independently recomputed and match exactly.

The next bounded experiment keeps the broader linear model fixed and compares ordinary execution with exact-shape CUDA graph replay. Its purpose is to distinguish avoidable launch/scheduling overhead from unavoidable full-loop computation. Warm timing must include input/output handling; cold capture cost is separate. All outputs must match saved references. This does not itself implement a native serving cache or demonstrate end-to-end speedup.

Evidence: [complete results](evidence/broader/complete.json), [run contract](evidence/broader/run.json), [selection](evidence/broader/selection.json), [independent audit and paired comparison](evidence/broader/mac-audit.json). Private artifacts: remote `research/broader-gates-20261125/`; Mac `~/.local/share/wayfinder-artifacts/broader-gates-20261125/`.
