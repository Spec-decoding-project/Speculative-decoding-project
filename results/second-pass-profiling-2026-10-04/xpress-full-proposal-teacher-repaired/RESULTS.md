# Repaired teacher attempt: compatibility passed; training remains untested

The GPU repair passed, but the matched learning comparison did not run. The second complete-work gate stopped the attempt before either 384-update fit. Do not treat this as a negative training result or as completion of the compression goal.

## What completed

Released CUDA refiner scores, token choices and input gradients matched the literal six-step reference exactly. The same released first pass reproduced throughout the control panel after the attention-policy repair. Sixteen calibration contexts were collected; temporary calibration copies took 28 optimizer updates and were discarded. These are not scientific training fits.

All 448 response banks were processed: 384 training responses supplied 12,288 new initial-draft contexts, and 64 reused development responses supplied 256 contexts. Existing proposal controls and both untrained second-pass panels completed. Neither trained model, either trained-weight single-pass control, nor a student fit ran.

The complete collection is authenticated against the original metadata and recorded labels in `collection-reuse-manifest.json`. It can be reused only with the same frozen producer, numerical path, source feature banks and labels. All 37 launched sources are preserved under `launch-snapshot/`.

## Why training did not start

The initial admission projected 275 seconds for collection and the pretraining comparisons. Those stages took 1,294 seconds: 4.71 times the estimate. Kernel/helper timings did not cover the whole prompt pipeline, including data movement, serialization and durable-volume synchronization. This section of the worker performed 640 prompt-level persistence calls; their individual cost was not timed, so the residual must not all be attributed to synchronization.

At the final gate, the remaining fixed work required 1,872 seconds including the original 30% margin and reserves; only 1,061 seconds remained. The worker correctly stopped instead of shortening either fit or bypassing the gate. The preparation error was the incomplete elapsed-time model. Collection's recorded model-call time was only 125 seconds; that number is not complete collection cost.

## Secondary result: untrained feedback inputs

The same released weights were evaluated with only the first guess visible versus the completed initial draft visible. Both retain the same original first guess. No model was trained for these input patterns in this attempt.

| Input to the untrained second pass | Repaired prefixes | Damaged prefixes | Damaged fully correct blocks | Gained / lost prefix positions |
| --- | ---: | ---: | ---: | ---: |
| First guess only | 41 | 23 | 6 | 174 / 93 |
| Completed draft | 19 | 144 | 59 | 48 / 822 |

An independent direct token-by-token recount agrees. These are raw second-draft chains; retaining the original proposal prevents some of this damage from reducing the combined proposal set.

With the fixed 64-node proposal policy, first-guess feedback covered 2,443 recorded positions and completed-draft feedback covered 2,347. The existing conditional refinement control covered 2,700. Giving each response equal weight, completed-draft feedback was 3.93% worse than first-guess feedback (descriptive 95% interval: -5.76% to -2.22%), and 14.20% worse than the existing conditional control. Coverage includes the known anchor consistently. These are fixed-context observations, not generated-token savings or measured speedups.

The narrow causal comparison changes the visible draft input while keeping the released weights and inference procedure fixed. On this panel, exposing the completed proposal degraded the resulting untrained predictions. It does not explain the internal cause, establish that later guesses contain no useful information, or predict the trained comparison's outcome. In particular, these weights have not been trained here to interpret the completed draft as something to revise.

This secondary analysis was added before the panels completed; it does not replace the frozen primary analysis, which correctly returns INCOMPLETE. No positive teacher-headroom, student-compressibility or novelty claim follows.

## What should happen next

Keep the second-pass compression question. Complete the same two trained-teacher fits and their trained-weight single-pass controls using the recovered proposal collection and pretraining comparisons. Do not repeat collection, change the teacher recipe because an operational gate failed, or start another student fit on this result.

A completion runner must authenticate and load these saved inputs, measure whole-prompt training/evaluation and persistence costs, avoid needlessly committing every already-saved proposal again, and include loading, checkpoint writing and recovery in its admission estimate. The unchanged scientific dose and controls remain necessary. That runner and a new paid authorization are not claimed ready here. No retry, training-dose reduction or new spending was launched.

## Closure and accounting

Controller 24309 exited normally after the worker returned not_admitted. App ap-Kd4EyTa1qia6BY46mkmqmu is stopped with zero tasks. All 542 artifacts (108,720,438 bytes) were recovered and independently verified, and owned temporary volume vo-WiHTrMMZeownw7u2m5piC9 was deleted. Overall execution, recovery and cleanup took 32.78 minutes.

Provider-reported compute is $1.44792468, not a settled invoice. The full $2.90 cap remains retained for this closed attempt; the separately approved $0.60 previous-attempt hold remains. The original $30 goal authorization has $0 unallocated, with no active jobs or reserved live-job caps. No automatic refund or further attempt is authorized.

Evidence: `attempt1/analysis.json`, `attempt1/initial-comparison.json`, `closure-audit.json`, `collection-reuse-manifest.json`, `attempt1/recovery.json`, `attempt1/metering.json`, and the preserved launch snapshot.
