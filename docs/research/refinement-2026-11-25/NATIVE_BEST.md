# Exact earlier linear gate: matched native DFlash benchmark

Completed: the earlier gate reduces native verification rounds but does not repay its added cost on either dataset. Both paths return identical tested outputs. No training ran.

## Results

All 128 prompts completed. Each path was timed five times, giving **640 paired comparisons**. All paired final token sequences and termination reasons match. No training ran.

| Dataset | Path | End-to-end tokens/s | Mean decode ms/round | Actual tokens/round | Verification rounds, all 5 repeats |
|---|---|---:|---:|---:|---:|
| UltraChat | DFlash | 37.383 | 69.683 | 2.68407 | 7,375 |
| UltraChat | DFlash + earlier gate | 34.069 | 78.428 | 2.74169 | 7,220 |
| GSM8K | DFlash | 63.300 | 69.757 | 4.56625 | 4,415 |
| GSM8K | DFlash + earlier gate | 57.860 | 78.427 | 4.67207 | 4,315 |

These round totals cover all five repetitions; divide by five for one pass over the 64 prompts in each dataset. End-to-end throughput includes prefill; decode-round metrics exclude prefill and its first output token. Terminal overshoot is excluded from committed tokens.

- **UltraChat:** throughput changes by -8.864%; request time changes by +9.726%. Native commits/round change by +2.147%. 4/64 prompts improve on their five-repeat mean.
- **GSM8K:** throughput changes by -8.594%; request time changes by +9.401%. Native commits/round change by +2.317%. 12/64 prompts improve on their five-repeat mean.

## Repeatability

Ratios below are edited/baseline end-to-end throughput; **1.0 is break-even**. Intervals resample paired prompts, not individual rounds. They describe exploratory development variation.

| Dataset | Repeat 1 | Repeat 2 | Repeat 3 | Repeat 4 | Repeat 5 | 95% paired-prompt bootstrap interval |
|---|---:|---:|---:|---:|---:|---:|
| UltraChat | 0.91146 | 0.91105 | 0.91156 | 0.91125 | 0.91150 | 0.89678–0.92807 |
| GSM8K | 0.91423 | 0.91403 | 0.91388 | 0.91408 | 0.91410 | 0.89280–0.93564 |

## Break-even at the measured complete-round cost

| Dataset | Affordable added ms/round at observed acceptance | Measured added ms/round | Observed extra tokens/round | Required extra tokens/round at measured cost | Required total tokens/round |
|---|---:|---:|---:|---:|---:|
| UltraChat | 1.496 | 8.745 | 0.05762 | 0.33686 | 3.02093 |
| GSM8K | 1.617 | 8.670 | 0.10582 | 0.56752 | 5.13377 |

Required acceptance holds the measured edited-round duration fixed. Increasing acceptance could itself change work and duration; this is a measured break-even condition, not a prediction for another trained model. The measured added cost is the whole-round difference, including copies and scheduling, not the standalone 6.73 ms number or the callback’s native-kernel waiting time.

## By input length

Each row contains 16 prompts selected from one source-specific length quartile. Ties can straddle adjacent quartiles. Every numerical pair below is **DFlash / edited**. Round totals include all five repeats. These subgroups are small; no subgroup is presented as fresh confirmation.

| Dataset | Input tokens | n | Tokens/s | Decode ms/round | Committed tokens/round | Verification rounds | Throughput change |
|---|---:|---:|---:|---:|---:|---:|---:|
| UltraChat | 23–60 | 16 | 35.374 / 32.714 | 69.319 / 77.433 | 2.4889 / 2.5649 | 2,025 / 1,965 | -7.521% |
| UltraChat | 65–102 | 16 | 39.266 / 36.988 | 69.552 / 77.951 | 2.7635 / 2.9129 | 1,755 / 1,665 | -5.804% |
| UltraChat | 106–152 | 16 | 36.381 / 32.949 | 69.347 / 77.838 | 2.5681 / 2.6021 | 1,910 / 1,885 | -9.433% |
| UltraChat | 187–949 | 16 | 38.846 / 34.019 | 70.636 / 80.693 | 2.9674 / 2.9326 | 1,685 / 1,705 | -12.425% |
| GSM8K | 34–55 | 16 | 64.211 / 57.584 | 69.633 / 78.099 | 4.6452 / 4.6452 | 1,085 / 1,085 | -10.321% |
| GSM8K | 55–68 | 16 | 64.209 / 59.286 | 69.529 / 78.120 | 4.6239 / 4.7773 | 1,090 / 1,055 | -7.667% |
| GSM8K | 68–81 | 16 | 65.453 / 60.370 | 69.932 / 78.707 | 4.7324 / 4.8932 | 1,065 / 1,030 | -7.766% |
| GSM8K | 81–139 | 16 | 59.648 / 54.544 | 69.925 / 78.769 | 4.2894 / 4.4017 | 1,175 / 1,145 | -8.557% |

## Graph preparation and cold requests

Warm editor timing contains **11,995 graph replays and zero captures**. Separate empty-cache runs capture 2,399 graphs; total capture plus its warmup is 230.070 s, averaging 1,797.42 ms per prompt. The cache is deliberately emptied for each cold prompt; this does not estimate a production cache hit rate.

| Dataset | Warm baseline seconds, one pass | Empty-cache edited seconds | Capture/warmup seconds within edited time | Empty-cache edited tokens/s |
|---|---:|---:|---:|---:|
| UltraChat | 107.617 | 257.239 | 142.262 | 15.639 |
| GSM8K | 64.708 | 156.663 | 87.808 | 26.145 |

Model initialization was separate from request timing: 45.68 s for the target-only
control engine and 34.03 s for the native engine. The end-of-request drain/RPC
averages 7.661 ms for DFlash and 0.377 ms for editing. Using only the ordinary
`llm.generate` interval also loses: throughput ratios are 0.90826 for UltraChat
and 0.90601 for GSM8K. Thus the sign does not depend on the drain accounting.

## Correctness and audit

The independent Mac audit reconstructs all 1,792 native generations plus 8 target-only controls, checks 16,750 raw-target argmax positions and 8,119 paired full-vocabulary target-logit hashes, and independently recounts returned commits, verification rounds, timings and break-even arithmetic. The runner also checks graph/eager editor equality before timing each prompt.

Natural EOS counts: UltraChat 4/64, GSM8K 0/64. Both paths agree on those terminations. The remaining responses hit the 64-token cap. Equality is established for these tested outputs; this is not an uncapped-response or concurrent-serving validation.

[Audited aggregate data](evidence/native-best/mac-audit.json) and [environment/integrity evidence](evidence/native-best/environment.json). Raw prompts/tokens remain in the private artifact mirror.

## Question and fixed implementation

Does the earlier 265-parameter linear gate improve complete generation throughput?
The exact checkpoint's saved development-prefix score is 3.69970703125 versus
3.57763671875 for its original proposals. That saved-context metric is not native
committed tokens per round. The 265 parameters belong to a decision head on the
frozen 40,750,336-parameter autoregressive generator. Both are executed and timed.
Even COPY decisions require generator inference in this implementation.

The gate reads the generator's 256-dimensional representation plus eight scores;
it chooses the original proposal token or the generator's vocabulary argmax at
each of 14 editable positions. Inputs remain token IDs: accepted context (up to
1,024 tokens), anchor and the complete proposal. The anchor and first proposal
are preserved. No target or drafter hidden states are inputs to the editor.

Gate SHA256 `5ab048b622beffb10ab8d282027cb2510f5439e4621793d5a82e687e696d679b`;
generator SHA256 `c79b8c59c1e2cf73a7adf7aa9738b1b273779d330fe32f8d2c4a15fd56b7e8d2`.
This differs from the broader-data gate in [the earlier eight-prompt native
pilot](NATIVE_SPEED.md). Results from that pilot are not substituted here.

## Matched design

128 development prompts excluded from editor optimization: 64 UltraChat and 64
GSM8K, sampled deterministically without looking at outputs. Each source's 256
available development prompts is divided into four length quartiles, then 16
prompts are selected per quartile by fixed hash. Prompt order is interleaved by
a second hash. These are repeatedly examined development data; the final test
split is not evaluated.

| Dataset | Length group | Selected input tokens | Prompts |
|---|---|---:|---:|
| UltraChat | Q1 | 23–60 | 16 |
| UltraChat | Q2 | 65–102 | 16 |
| UltraChat | Q3 | 106–152 | 16 |
| UltraChat | Q4 | 187–949 | 16 |
| GSM8K | Q1 | 34–55 | 16 |
| GSM8K | Q2 | 55–68 | 16 |
| GSM8K | Q3 | 68–81 | 16 |
| GSM8K | Q4 | 81–139 | 16 |

One RTX A5000 on tron50, four CPU cores, 32 GiB host-memory allocation; batch one,
greedy decoding, 64-new-token cap and normal EOS. Identical Qwen3-8B/DFlash models,
BF16 target/drafter, canonical batch-invariant vLLM 0.34.0, Torch 2.14.0+cu130,
FlashAttention, 2 GiB KV cache, chunked prefill with 256-token budget, no prefix
cache or asynchronous scheduling. Target/drafter retain eager execution. The
FP32 editor reuses exact-length CUDA graphs from a bounded 64-entry cache. Ordinary
editor kernels are captured under the previously tested temporary dispatcher
scope; invariant target dispatch is restored before every target execution.

Each prompt has correctness runs, an unedited warmup and a separate empty-editor-
cache run, then five alternating unedited/editor timing pairs. Initial order is
balanced across prompts. Warm timing fails if a new graph is captured. Eight
independent target-only controls cover both datasets and all length groups.
Every native output must equal its baseline token-for-token with identical
termination; correctness runs also compare full target-logit hashes and check
native acceptance, committed prefixes and graph/eager editor equality.

## Measurement definitions

End-to-end wall time includes request processing/prefill, generation, all editor
input/output copies and execution, scheduling, and a final GPU drain/RPC. The
ordinary `llm.generate` interval and the drain cost are also saved. Draining is
necessary because the baseline may return before its final GPU proposal finishes.
Model loading is outside request timing and saved separately.

Decode wall time starts after a synchronized prefill anchor, immediately before
the first DFlash proposal; it ends at the drained request boundary. It includes
first/last proposal work, editing and engine gaps. Divide it by actual speculative
verification calls for average ms/round. Prefill calls are not verification rounds.
Committed tokens are tokens actually returned during verification: exclude the
initial prefill token and clip the terminal block to the returned sequence.
Thus neither discarded tail tokens nor prefill inflate the committed-token metric.

Report aggregate throughput (total returned tokens/total wall time) and weighted
round averages across all five repeats. Show per-repeat aggregate speedup ratios
and 95% paired-prompt bootstrap intervals (10,000 draws, fixed seed 0). Small length
subgroups are exploratory. Cold runs clear the editor cache per prompt; their
capture/warmup and full request times are reported separately. Warm results are
an optimistic reused-shape case, not proof that setup amortizes in an arbitrary
request stream.

For baseline/edited complete-round times B/E and commits c0/c1:

- Maximum affordable extra cost at observed acceptance: B × (c1/c0 −1).
- Measured extra complete-round cost: E − B.
- Extra commits required at measured edited-round cost: c0 × (E/B −1).

The last two include all execution effects. They do not attribute every change
to a standalone editor kernel. Editor callback wall time can wait for native
drafter kernels; using it entirely as incremental editing would double-count
some existing work. The raw runner preserves that callback-based estimate, but
the independent audit's `round_economics` supplies the headline comparison.

## Execution and evidence

Slurm 8580225; source commit 30f1850; source archive SHA256
`7639be0e3517dc5558bb89109479cf00660b30c6764bb6bab9e6c02ba2eb564c`.
Remote outputs: `research/native-best-8580225/benchmark/` and its parent directory.
Private Mac mirror: `~/.local/share/wayfinder-artifacts/native-best-8580225/`.
No source prompt text or generated token sequences are published in this report.

Methods: `editor-experiment/refinement_research/native_best.py`,
`native_speed_worker.py`, `native_best.sbatch`; independent CPU reconstruction:
`native_best_audit.py`. Experiment rationale and changes are in root `HISTORY.md`.

## Outcome and remaining scope

This always-on FP32 editor fails the native throughput test, including every
prespecified length group. Four UltraChat and twelve GSM8K prompts improve
individually; those wins do not offset the full workload's cost. The data support
a no-go for scaling this execution design on the expectation of a speedup.
A cheaper or selective execution design remains untested by this result.

The GPU job completed successfully in 57m24s. CPU-only pin audit 8580512 completed
in 17s. All 1,820 mirrored files match their remote hashes; the queue is empty.
The local suite passes 28 tests, and ten benchmark-specific tests passed on the
GPU host. The final test split was not evaluated. No further training or
experiment was launched after these checks.

Remaining serving scope includes union verification, concurrency and fresh
confirmation. Four natural-EOS cases now pass for this replacement path; that
updates the earlier pilot's untested EOS status. The temporary private dispatcher
adapter is still restricted to this pinned, synchronous batch-one prototype.
