# Exact earlier linear gate: matched native DFlash benchmark

Status: running; results will be filled only after the complete run and independent
artifact audit. No new training is part of this experiment.

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
BF16 target/drafter, canonical batch-invariant vLLM0.34.0, Torch2.14.0+cu130,
FlashAttention, 2 GiB KV cache, chunked prefill with256-token budget, no prefix
cache or asynchronous scheduling. Target/drafter retain eager execution. The
FP32 editor reuses exact-length CUDA graphs from a bounded64-entry cache. Ordinary
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
and95% paired-prompt bootstrap intervals (10,000 draws, fixed seed0). Small length
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

Slurm8580225; source commit30f1850; source archive SHA256
`7639be0e3517dc5558bb89109479cf00660b30c6764bb6bab9e6c02ba2eb564c`.
Remote outputs: `research/native-best-8580225/benchmark/` and its parent directory.
Private Mac mirror: `~/.local/share/wayfinder-artifacts/native-best-8580225/`.
No source prompt text or generated token sequences are published in this report.

Methods: `editor-experiment/refinement_research/native_best.py`,
`native_speed_worker.py`, `native_best.sbatch`; independent CPU reconstruction:
`native_best_audit.py`. Experiment rationale and changes are in root `HISTORY.md`.
