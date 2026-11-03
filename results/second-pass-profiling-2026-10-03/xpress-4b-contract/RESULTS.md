# 4B xPress preparation: dimensions checked, compression value unproven

No released weight tensors were downloaded or executed. No GPU or paid work occurred. The archived 4B configuration and safetensors header identify a compatible target family, but not a ready replacement for the historical 8B runner.

## Concrete loading difference

The old helper constructs a width-4,096 head with a 512-wide MLP. The released 4B checkpoint uses width 2,560 and a **1,024-wide MLP**. Both must come from the checkpoint configuration. Five parameter shapes would mismatch under the old helper; old sources remain unchanged.

Instantiating both author head definitions on the meta device with the actual configuration matches every released head parameter name and shape. Eight small CPU fixtures checked the training-side to serving-side mapping. Unfolded versus folded FP64 calculations agree within the declared numerical comparison; after both mixers are folded, all 48 BF16 refinement steps produce identical bias tensors and choices. This validates those toy CPU paths, not real-weight CUDA execution, fused kernels, complete generation or native-target fidelity.

The head has 80,150,528 parameters versus 80,543,744 in the previously examined 8B head. The vocabulary embedding/readout account for 77,791,232 parameters, or **97.06%**. Thus the 4B target label does not mean the refiner is half as large. Parameter counts do not determine runtime.

## Why a strong teacher is not enough

Let `f` be the fraction of complete round time spent in the refiner, `u` the fraction of its cost left after replacement, and `r` the new-to-old accepted-progress ratio. Holding the remaining work fixed gives the conditional cost ratio:

`new time per accepted position / old = (1 - f + u*f) / r`.

At unchanged progress, deleting the entire refiner saves only `f`. Our old 8B instrumented-region illustration has 0.946 ms of refinement in a 73.242 ms round, or 1.291%. Even a free perfect replacement in that illustration needs roughly **3.90% more progress than xPress itself** to reach 5% lower time per accepted position. This is not a measured bound for the new 4B checkpoint: different target size, graph coverage, kernels, prefixes and uninstrumented work can change the fractions.

| Assumed refiner share of round time | Perfect removal, same progress | Progress required for 5% savings if replacement costs half as much |
|---|---:|---:|
| 3% | 3% savings | +3.68% |
| 5% | 5% savings | +2.63% |
| 8% | 8% savings | +1.05% |
| 10% | 10% savings | unchanged |

These are scenarios. No 4B time, accepted-prefix gain or student retention is measured. A small trainable parameter count also does not imply lower complete serving cost.

## Next informative measurement

Before a student fit, compare the released refiner with an optimistic cached-output reference on the **same contexts and proposals**. Both still execute the first drafter and target verification; the reference supplies the already-computed refined proposal instead of doing refinement again. This removes the learning question and asks whether enough runtime is removable at all. It is not a deployable decoder or a speedup claim. The detailed local protocol is `PROTOCOL.md`; a GPU runner is not yet prepared or funded.

If removing refinement offers too little benefit, do not train a student merely to imitate it. If sufficient runtime is removable, then useful correction retention against the ordinary-tree and refiner-off controls becomes the next question. Strong published performance, exact CPU mapping, and checkpoint availability cannot substitute for these measurements.

Existing 8B caching, shortlist and speculative-iteration experiments remain closed. This is not authorization to revive them. The broader cheap-redrafting objective remains active and unmet.

Sources: [official head implementation](https://github.com/Supercomputing-System-AI-Lab/xPress/blob/main/refiners/xpress_head.py), [released 4B checkpoint](https://huggingface.co/UIUC-SSAIL/Qwen3-4B-XPress-b16), [historical measured cost audit](../../xpress-recycle-2026-09-22/output-cost-audit.json). Source hashes, exact shapes, fixture results and complete cost scenarios are in `results.json`. Reproduce with `analysis/audit_xpress_4b_contract.py` using the CPU torch environment.
