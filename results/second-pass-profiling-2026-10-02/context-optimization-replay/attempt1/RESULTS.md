# Context optimization replay

The diagnostic reproduced the original learning trajectory through the recorded loss burst. It tests an explanation of a failed fit, not an improved editor.

All 896 losses, step identities and valid-row counts independently match the original log exactly. Both earlier BF16 checkpoint comparisons passed. Per-microbatch supervised CE reconstructs each of the 132 observed update losses.

| Window (epoch 3) | Mean CE | Context attention entropy | Correction RMS | Context residual RMS |
|---|---:|---:|---:|---:|
| early (1–32) | 2.782 | 0.000 | 0.878 | 3.390 |
| before_observed_burst (65–86) | 4.031 | 0.000 | 1.856 | 3.737 |
| observed_burst (87–124) | 22.785 | 0.002 | 8.714 | 7.672 |
| after_burst (125–128) | 9.071 | 0.002 | 4.900 | 7.578 |

Attention statistics reconstruct only the added context lookup in FP32 from actual detached pre-RoPE queries and keys. They are not native BF16 probabilities and do not measure the trunk's self-attention distributions. Query/output summaries exclude positions not supervised by CE. Low entropy alone is not a defect. Temporal associations can nominate a mechanism; no intervention here identifies a cause or demonstrates a remedy. Loss-window definitions come from the earlier failure, not an independent validation panel.

No new generation, speedup, architecture improvement or novelty is demonstrated. Earlier negative development comparisons remain unchanged.

## What the trace changes

The new measurements do not support a sudden collapse of the added attention at the onset of the burst. It was already nearly one-hot during comparatively benign training: epoch-three steps 1–32 average normalized entropy 0.000076 and largest probability 0.99987. During the burst entropy increases slightly. Persistently concentrated attention could still create fragility; this is not evidence that it is harmless.

Instead, early localized excursions become broad damage. At step 75 one microbatch has CE 37.63 while the others are 3.20–4.84. At step 84 one reaches 27.94 while the others are 2.61–4.82. By step 86 every microbatch is elevated (6.08–14.95), and at step 87 all are 12.41–38.35.

At step 86 the context-value and first-state projection gradient norms reach 89.45 and 87.88. The value projection changes by approximately 12% of its weight norm in that update. From steps 86 to 87 correction RMS rises 4.10 to 13.32 and loss 8.30 to 18.51. Other parameters also move; query/key gradients are nearly zero at step 86, but their weights still change through optimizer history. We cannot isolate the value projection, rule out query/key effects, or call one operation the root cause.

Correction RMS is not by itself a probability-quality measure: it includes softmax-invariant shifts. The independently recorded correct-versus-best-alternative margins also deteriorate, confirming harmful relative-score changes. [Temporal locator](temporal-locator.json).

A concrete next comparison is one smaller-update continuation from the saved step-64 full optimizer checkpoint, before the prominent excursions. Preserve the remaining original batches, CE, update count and final checkpoint; use one predeclared learning-rate reduction rather than a sweep. If isolating gradient-update magnitude, preserve per-step AdamW weight decay separately. Judge both stability and final development repairs/damage against the preserved original trajectory. Stable training without a useful development gain would not justify a generation run. A positive result would be an optimization sensitivity finding on this seed, not a new speculative-decoding technique or proof that context was the missing information. This follow-up has not been implemented or launched.

Recovery authenticated 17 artifacts totaling 471,490,167 bytes. The app stopped with zero tasks and its owned temporary volume is absent. Elapsed through cleanup: 17.06 minutes. The $2.50 approved allowance remains conservatively retained, not invoiced spending; $6.54 remains unallocated. No retry occurred.

[Analysis](analysis.json) · [Closure checks](closure.json) · [Execution](execution.json)
