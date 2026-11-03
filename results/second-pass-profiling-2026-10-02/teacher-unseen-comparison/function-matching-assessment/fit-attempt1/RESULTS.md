# Literal imitation improved fidelity, without a demonstrated coverage benefit

Both fixed fits completed 1,152 updates. The existing mixture's deployed state digest and all three epochs' development predictions exactly reproduced the historical control. All four native preflights passed. Independent token recount, source-label checks, recovered-file hashes and endpoint checks passed.

## Primary endpoint: 64 reused development responses

| Measurement | Existing gated mixture | Teacher-only imitation |
|---|---:|---:|
| Additional retained-union coverage over original draft | 3.460% | 3.235% |
| Teacher top-1 agreement, equal-prompt mean | 50.171% | 59.484% |
| Teacher KL, equal-prompt mean | 0.90185 | 0.61465 |
| Repaired prefixes | 177 | 163 |
| Damaged alternative prefixes | 461 | 282 |
| Damaged fully correct alternatives | 60 | 51 |
| Teacher-added positions recovered | 218 / 1,136 | 237 / 1,136 |

The paired coverage difference is -0.225 percentage points, with 95% prompt-bootstrap interval [-1.007, +0.550]. It does not establish superiority, equivalence or reliable inferiority. Both domain point differences are negative: math 4.375% versus 4.036%, general instructions 2.546% versus 2.434%. The fixed advancement screen fails coverage, paired difference and domain conditions; teacher-position capture improves slightly.

These are exposed development results. They do not replace the earlier 64-fresh-response result of 2.774% for the mixture, and must not be compared to that number as though the prompts were identical. Neither experiment measured generation speed.

The original draft is retained, so damaged edited prefixes do not directly subtract from this union-coverage metric. Producing and verifying alternatives still costs time. A reduction in damage is useful to report, but cannot by itself establish a runtime advantage.

## What changed in the learned behavior

Direct imitation increased aggregate teacher agreement by 2,575 editable-token decisions. Of that net increase, 2,009 occurred on teacher-wrong tokens or blocks whose unchanged first guess already prevented useful extension. This is a descriptive allocation of agreement changes, not a count of independently learned facts or wasted optimizer work.

At the 373 original first errors where the teacher's local choice was correct, the editor matched that choice 118 times under the mixture and 125 under imitation: 63 gained matches and 56 lost matches. Earlier preservation and continuation are still needed to turn such local matches into retained progress. Global fidelity improved far more than this immediate repair count.

Thus the experiment successfully changed imitation behavior, while useful coverage remained similar. This is evidence against assuming that better overall teacher fidelity will be sufficient at this architecture and dose. It does not identify whether capacity, inputs, optimization, or allocation of training effort is the remaining cause. The two-arm change jointly removed the gold-label term and correctness gate; their individual effects are not isolated.

## Training sample and trajectory

| Measurement on 64 selected training responses | Mixture | Teacher-only |
|---|---:|---:|
| Additional coverage | 15.427% | 4.839% |
| Teacher agreement | 45.772% | 63.894% |
| Teacher KL | 1.06298 | 0.49496 |
| Teacher-added positions recovered | 447 / 1,094 | 385 / 1,094 |

The large training-coverage gain of the mixture was not an indication that it closely reproduced the teacher. Conversely, the teacher-only model improves training fidelity while recovering fewer of the teacher's useful positions on this sample.

Teacher-only development coverage across epochs is 2.768%, 3.187%, 3.235%; development KL is 0.6042, 0.6048, 0.6146. Mean training loss continues down, 0.6469, 0.5948, 0.5468. Training is not proven converged, but the development fidelity trajectory does not supply a reason for an automatic duration extension. Epoch three remains primary; no checkpoint is selected in hindsight.

## Decision

Do not advance this teacher-only recipe to a generation run or automatically extend it. Preserve the specific finding: closer teacher imitation and fewer damaged alternatives did not produce a demonstrated useful-coverage gain. Reweighting the same examples again needs a new, concrete justification; prior prefix-weighting and correction-rehearsal negatives remain relevant.

The broader cheap-second-pass objective remains active. A next learning proposal must explain why it should recover missed useful teacher decisions on other responses, and specify a comparison that distinguishes that mechanism from simply improving global fidelity. This result does not establish that a larger model, more data, or longer fitting would solve the problem.

## Scope, execution and evidence

One training seed, 384 training responses, 64 exposed development responses and 64 training probes; 2,048 contexts per evaluated panel. Two objectives share architecture, inputs, initialization, minibatch order, optimizer and exposure. Teacher scores reconstructed exactly to saved winning tokens across reported evaluations; zero disagreements. Full-probability KL is worker-computed; token agreements and prefix metrics were independently recounted. Repeated blocks are not independent prompts.

Execution through recovery and cleanup took 12.67 minutes. All 35 artifacts, 153,438,246 bytes, were recovered and authenticated. App `ap-BugMbdxmdUhkDWh6H0lXkt` is stopped with zero active tasks; owned scratch `vo-5jQISRVHFTk3i1KsLgw8Gy` is deleted. The retained research archive is preserved.

The full $2.50 cap is conservatively held for unsettled costs, leaving $12.78506668 unallocated and no active reservations. This is not a settled invoice. No retry or extension occurred.

Evidence: `analysis.json`, `agreement-allocation.json`, `historical-control-replay.json`, `recovery.json`, `execution.json`, `recovered/frozen-final-checkpoints.json`; design and checks are in the parent directory. Reproduce the independent recount with `analysis/analyze_editor_function_matching.py`.
