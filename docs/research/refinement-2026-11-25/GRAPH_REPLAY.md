# CUDA graph replay: same tested edits at 3.78× lower warm latency

The full fourteen-step editor takes **6.73 ms with warm CUDA graph replay versus 25.45 ms with ordinary execution**, a **3.78× reduction in editor latency**. Every tested output matches the saved model output. This is a useful execution result; native DFlash generation speedup remains unmeasured.

Job **8579567** completed November 25 in **1m01s**, exit 0, source `11423488bcc3066eb12b2df0be92d9cab72559da`. One A5000/four CPUs/5 GiB; peak RSS 1,411,380 KiB. Zero optimizer updates, no target/drafter inference or final-test access. The checkpoint is the broader-fit **linear gate**, whose ordinary development prefix is **3.679688**, versus DFlash **3.577637**. It is not the earlier smaller-fit linear checkpoint with the highest observed development prefix.

## What changed

The weights, FP32 precision and fourteen-step decision rule remain fixed. The proposal method now separates tensor decoding from host input/output handling. A CUDA graph records the complete GPU operation sequence and replays it using persistent input/output buffers. New context/proposal tokens are copied into those buffers before replay. The method follows [PyTorch's documented CUDA graph workflow](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs), including three warmup iterations on a side stream.

This prototype captures a graph for each tested input length, then releases it after that case. It does **not** implement a production graph cache, padding/bucketing policy, concurrent execution or native verifier integration.

## Measurements and checks

The same 64 development blocks used in earlier latency measurements cover **52 distinct prefix lengths**. Each case uses two warmups and ten timed calls per method, interleaving and alternating the method order. Warm calls include CPU input construction, host-to-device copies, GPU replay and conversion of output tokens back to a host list. Target and drafter are unloaded. The editor-only scripts set `VLLM_BATCH_INVARIANT=0` and instantiate no target engine; these timings do not benchmark the adopted invariant target.

| Measurement | Result |
|---|---:|
| Median ordinary editor call | 25.4533 ms |
| Median warm graph editor call | 6.7302 ms |
| Ratio of those medians | 3.782× |
| Cases where graph median is faster | 64 / 64 |
| Timed calls checked | 1,280 |
| Median graph preparation, including three warmups | 101.5693 ms |
| Maximum additional PyTorch live allocation per case | 16.25 MiB |
| Distinct tested prefix lengths | 52 |

Before timing, ordinary execution reproduces each saved parent output and agrees with full-sequence execution. Each graph also matches ordinary execution on an altered proposal of the same shape; **43 of 64 altered inputs change the output**, showing that replay is not just returning a stale result. Restoring the original input restores the original output. Every warmup and timed call matches. Generator and gate hashes remain unchanged.

All 54 local tests and thirteen tests in the pinned GPU environment passed. A separate Mac audit verified all backup hashes, all 1,280 saved timing samples and aggregate calculations, the 64/64 faster-case count, and the profile totals. Token equality was asserted during GPU execution; the Mac audit does not rerun CUDA inference. [Audit](evidence/graph/mac-audit.json).

## What the profile supports

A separate three-call ordinary-execution profile records **4,365 CUDA kernels**, or 1,455 per editor call, with a summed kernel duration of about **6.97 ms per call**. The largest matrix-vector kernel accounts for about **3.16 ms**; the largest attention kernel about **0.80 ms**. This, together with the exact-output graph result, supports substantial launch/scheduling overhead in the ordinary loop. Profiling is instrumented and uses one case, so the wall-time difference is not an exact exclusive attribution to Python.

The remaining GPU computation still matters. A graph changes how work is issued; it does not remove the full vocabulary readout or fourteen autoregressive decisions.

## Practical limits and next milestone

The roughly **102 ms preparation cost per new shape** must be amortized. Capturing on every generation round would erase the warm advantage. A persistent cache or another validated shape strategy is required, with its memory and cold-request costs measured. The reported extra allocation is a per-case PyTorch allocation delta, not total process memory or the memory of a complete serving cache.

The quality gain for this checkpoint is still small: +0.102051 correct-prefix tokens. Under the same prefix distribution, unchanged verification cost and one target token per round, even the warm 6.73 ms editor requires a baseline round of about **302 ms** to break even. That is a screening calculation, not a measured native baseline. No net generation-speedup claim is justified yet.

The next milestone is bounded native integration/fidelity with the frozen token editor and a documented graph-reuse policy, followed by original-versus-edited full-round timing. Use the canonical invariant target and keep precision changes separate. Measure ordinary and edited paths under the same settings; preserve target outputs, rejection/EOS/budget behavior and following cache state. More data or parameters are not the next justified intervention.

Evidence: [complete results](evidence/graph/complete.json), [run contract](evidence/graph/run.json), [redacted timing samples](evidence/graph/timings-redacted.json), [profile summary](evidence/graph/profile-summary.json). Complete private artifacts and trace: remote `research/graph-replay-20261125/`; Mac `~/.local/share/wayfinder-artifacts/graph-replay-20261125/`.
