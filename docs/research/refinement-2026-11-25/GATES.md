# Frozen-generator gates: modest positive transfer, unresolved cost

Freezing the original generator and learning only a decision gate improves the direction of the result. All three predeclared gates increase mean development correct-prefix length relative to DFlash. **These are small exploratory quality gains, not demonstrated generation speedups.** The largest gate does not beat the linear representation gate, so increasing capacity is not indicated by this comparison.

Slurm job **8579153** completed November 25 in **6m44s**, exit 0. Source `08a4fff91e5810d22c1fc54ad6219820352c9781`. One A5000, four CPUs, 5 GiB; peak host RSS 2,229,156 KiB. No target inference, target/drafter training or final-test loading.

## Model and training process

The existing 40,750,336-parameter whole-draft autoregressive editor reads accepted context, the entire DFlash proposal and the two preserved seed tokens. Its weights remain frozen. At each of fourteen editable positions, a new gate chooses the original proposal token or the frozen editor's vocabulary argmax. The selected token becomes the next input. The gate has no access to reference labels during inference; representation features are the editor's own, not target/drafter states.

The score gate uses eight scalar confidence/position features. The linear representation gate also uses the editor's 256-dimensional internal representation. The nonlinear gate adds a 64-unit hidden layer. Both generation and the confidence readout still run, so tiny trainable parameter counts do not imply tiny inference cost.

The fit uses 1,536 distinct training-origin prompts; another 512 training-origin prompts select checkpoints. Each population balances repair/preservation and UltraChat/GSM8K. Development contains 2,048 blocks from 512 separate prompts. Gate calibration is excluded from gate optimization; the generator previously learned from the training corpus. No development labels enter optimization or individual checkpoint selection. These development prompts have been inspected in prior experiments and are not fresh confirmation.

We first run the frozen editor using correct preceding training tokens and cache its internal features. Only positions where exactly one candidate is correct, and the reference prefix remains reachable through either candidate, teach a meaningful choice. There are **991 such training decisions** (369 COPY, 622 generate) and **289 calibration decisions**. This is less information than the raw 21,504 training suffix positions might suggest.

Training minimizes binary decision cross-entropy weighted by remaining attainable correct-prefix length. AdamW uses learning rate 0.001, weight decay 0.01, batch 256, gradient clipping 1, seed 0. Only gate parameters receive gradients or optimizer updates. Every five epochs, calibration selects the highest mean correct prefix, breaking ties by lower weighted loss. Early stopping occurs after fifty epochs without improvement; limits were 2,000 updates/200 epochs per arm and two hours total. All three stopped at the calibration plateau. This stage optimizes transfer, not exact memorization of all suffixes.

| Gate | Updates run | Epochs run | Selected update / epoch | Calibration prefix | Weighted decision CE |
|---|---:|---:|---:|---:|---:|
| scores | 280 | 70 | 80 / 20 | 9.734375 | 0.658929 |
| linear | 380 | 95 | 180 / 45 | 9.628906 | 0.428207 |
| mlp | 420 | 105 | 220 / 55 | 9.660156 | 0.382508 |

Calibration DFlash prefix is 9.458984. Its balanced bank contains many more perfect proposals than natural development; the absolute values should not be compared across those populations.

## Development and cost

DFlash mean correct prefix is **3.577637**. The first proposal is preserved, so all 320 wrong-first blocks remain at zero accepted proposals under every gate.

| Gate | Trainable parameters | Mean prefix | Blocks improved / damaged | Change and descriptive 95% interval | Editor ms |
|---|---:|---:|---:|---|---:|
| scores | 9 | 3.655762 | 246 / 136 | +0.078125 [-0.000012, +0.155273] | 25.92 |
| linear | 265 | 3.699707 | 269 / 187 | +0.122070 [+0.041003, +0.204102] | 25.92 |
| mlp | 17,025 | 3.676758 | 286 / 212 | +0.099121 [+0.009277, +0.189453] | 26.61 |

Intervals use 10,000 paired whole-prompt bootstrap draws, seed 0. They do not account for trying multiple approaches or training-seed variation. The score gate's interval touches zero. The linear gate's +0.1221 tokens is **3.41%** more correct proposals, or approximately **2.67%** more progress if one target token per round is included. Almost all its net gain comes from GSM8K: +244 prefix positions versus +6 for UltraChat. The nonlinear gate worsens UltraChat slightly. Every gate damages 9 of the 44 originally perfect development proposals.

The isolated FP32 measurement uses 64 fixed development blocks, two warmups and ten synchronized cached calls each. Target and drafter are unloaded. The gate loop avoids per-token host synchronization but adds confidence calculations. It still costs about 26 ms. Under unchanged verification cost and the same prefix distribution, the linear gate would need a baseline round of roughly **972 ms** to repay that cost. This is a screening calculation, not a measured native baseline or a serving speedup claim. Prior heavily instrumented correctness-run timings are not a suitable performance baseline.

## Checks and decision

All 47 local tests and eight pinned-environment gate tests passed. Full-size initial COPY preserves proposals exactly; forced generation matches the original generator on 16 controls. All 512 calibration prefixes per arm match actual autoregressive execution. Selected checkpoint reloads preserve metrics. Cached/full execution and timing repetitions agree on 64 development blocks per arm. The generator hash remains `8babb2bfe3f4c11ef283312e22ccd2e20878438c96957302ff2c462748bbd382` throughout.

A separate Mac recount matched every calibration/development raw-token audit, checked all checkpoint-selection histories and contiguous update journals, and verified all backed-up file hashes against Nexus. Gate checkpoints and feature caches are independently backed up. No old checkpoint or bank changed.

The next comparison keeps these checkpoints fixed and tests one, two and four editable tokens before copying the rest. Report every predeclared budget and its latency, without an oracle choosing a budget per example. This tests execution cost before committing to larger training. Full details are recorded before launch in [HISTORY.md](../../../HISTORY.md).

Public evidence: [complete results](evidence/gates/complete.json), [run contract](evidence/gates/run.json), [feature counts](evidence/gates/feature-summary.json), [independent Mac audit](evidence/gates/mac-audit.json). Private complete artifacts: remote `research/frozen-gates-20261125/`; Mac `~/.local/share/wayfinder-artifacts/frozen-gates-20261125/`.
