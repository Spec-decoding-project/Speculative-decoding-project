# Native DFlash replacement: correctness passes, aggregate speedup does not

The best tested configuration takes **7.5% longer overall** than unchanged DFlash,
even with warm editor CUDA graphs. It does help two short prompts, including a
**1.24× generation speedup** on one 94-token input. This supports a conditional
correction signal, not an always-on accelerator or a general speedup claim.

No training ran. The same frozen whole-draft generator and broader linear gate
were used throughout. The target remained the adopted invariant vLLM target.

## Complete generation timing

Eight development prompts, 64 generated tokens each, batch one, five alternating
original/editor repeats per prompt. Each cell below uses the sum of per-prompt
median elapsed generation times. Startup/model loading and correctness auditing
are excluded; prefill, drafting, editing, verification, native state updates and
output handling are included. The editor remains resident during baseline runs.

| Editor execution | Original / edited seconds, all 512 tokens | Overall latency change | Short inputs | Longer inputs |
|---|---:|---:|---:|---:|
| FP32, process-wide invariant kernels | 13.267 / 17.626 | **32.9% slower** | 22.6% slower | 42.8% slower |
| FP32, ordinary editor kernels | 13.286 / 14.864 | **11.9% slower** | 2.4% slower | 21.1% slower |
| BF16 generator, FP32 gate, ordinary editor kernels | 13.244 / 14.237 | **7.5% slower** | 0.5% faster | 15.2% slower |

The best aggregate throughput ratio is **0.9303× DFlash** (about 7.0% lower
throughput). The 0.5% short-group difference is effectively a tie on this tiny
sample, not a meaningful general gain. These are exploratory development results,
not a final-test estimate. All eight selected prompts are from UltraChat.

| Input tokens | DFlash / BF16 edited seconds | Speedup | Native rounds, original → edited |
|---:|---:|---:|---:|
| 102 | 1.674 / 1.592 | 1.052× | 25 → 22 |
| 118 | 1.130 / 1.227 | 0.920× | 17 → 17 |
| 60 | 2.427 / 2.627 | 0.924× | 36 → 36 |
| 94 | 1.269 / 1.022 | **1.242×** | **19 → 14** |
| 1714 | 1.847 / 2.247 | 0.822× | 27 → 30 |
| 1107 | 1.536 / 1.690 | 0.909× | 23 → 23 |
| 953 | 1.510 / 1.737 | 0.869× | 22 → 23 |
| 949 | 1.851 / 2.095 | 0.884× | 27 → 28 |

Thus short input length does not determine whether editing helps. Total native
rounds fall only **196 → 193 (1.5%)**, including chunked prefill. That saving cannot
repay the measured extra work. Across the complete reference windows on the
edited trajectory, 11 blocks improve and 12 are damaged. Those local prefix
deltas are not additive counterfactual generation savings because edits change
subsequent round boundaries and the final output cap truncates some windows.

## Round cost and the integration correction

The first native timing exposed a penalty absent from the earlier editor-only
prototype. Pinned vLLM 0.34 installs process-wide ATen overrides for matrix
multiplication and reductions on Ampere when batch invariance is enabled. The
editor inherited those kernels too.

A controlled eight-input probe changed only editor ATen dispatch, using
invariant → ordinary → restored invariant execution. Median graphed call time
fell **24.44 → 8.81 ms**; every tested proposal matched, and restored invariant
outputs matched the first pass. The adapter captures ordinary editor kernels,
then reinstalls invariant dispatch before any target execution. All **173**
audited native FP32 proposals also match between the two kernel modes.

Native editor-call medians across the eight cases are **26.86 / 9.58 / 6.13 ms**
for the three configurations above. In the BF16 run, median speculative-round
GPU elapsed time rises **68.32 → 74.80 ms** (median of per-case medians). Full
generation wall time, not this component estimate, determines the headline.

This diagnosis does **not** measure the speed of a non-invariant target. The
target and drafter retain their canonical configuration throughout. It explains
the editor integration penalty; it does not establish performance on other GPUs,
batch sizes, target kernels or optimized target/drafter graph configurations.

## Cold graph cost and memory

The implemented cache holds at most 64 exact prefix lengths with LRU eviction.
Warm measurements have **865 editor graph hits and zero misses per configuration**.
They represent best-case reuse of the tested request trajectories.

Clearing the editor graph cache before each prompt requires **132 captures**
across eight runs. Corrected FP32 preparation totals **13.00 seconds**. Complete
empty-cache generation totals **27.63 seconds in FP32** and **27.05 seconds in
BF16**, versus about **13.3 seconds** for DFlash. BF16 therefore takes roughly
**2.04× as long** in this cold scenario. This is an empty-cache-per-prompt bound,
not a measurement of a continuous mixed-request service.

Peak PyTorch allocation is **20.52 GiB FP32 / 20.44 GiB BF16** with ordinary editor
kernels; this is not an accounting of every driver allocation. Short input
lengths change as generation advances, so an exact-length cache still creates
many graphs. Contexts beyond 1,024 tokens reuse one editor-window shape. A
different shape-reuse design remains necessary for practical first-use latency.

## Correctness, precision and scope

- Independent target-only greedy reference: 512 main tokens plus 24 stopping-fixture
  tokens. All **394 candidate generation calls** across the three configurations
  match their reference tokens and stopping behavior, including all timed calls.
- Native original, pass-through identity and edited replacement use the existing
  vLLM rejection sampler and target/draft cache machinery. Audits reconstruct
  complete target prefixes and check native acceptance against target choices.
  Only the editor receives its trained last-1,024-token window.
- Forced all-accepted and first-rejected proposals, output budgets 1/2/17 and a
  stop-token case pass. Oracle proposals are controls only, never learned results.
- Ordinary-kernel runs compare **2,560 full target-logit hashes** in total across
  repeated comparisons of the 512 canonical positions. These include identity,
  FP32 and BF16 checks; every comparison passes. Invariant dispatch is asserted
  before every target round. This is finite validation, not a universal proof.
- BF16 changes 59 proposed tokens in 16 of the 173 inspected blocks relative to
  FP32, with identical contexts and original proposals. Native output and round
  counts remain unchanged on these prompts; precision conversion is not claimed
  to preserve every editor proposal exactly.
- Twenty-four relevant local tests pass, including restoration after an editor
  exception. GPU eager/graph comparisons, kernel restoration controls and
  independent Mac artifact audits pass. Original generator, target and DFlash
  files pass the existing pinned revision/hash audit. All backup file hashes match.

This completes the **replacement subset** and its requested timing measurement.
Union verification, a natural-EOS fixture, concurrency and fresh confirmation
prompts remain untested. The serving-fidelity ticket stays open. The private
dispatcher-library scope is deliberately limited to the pinned synchronous,
batch-one worker; it is not a concurrency-safe production integration.

## Decision

Do not scale training or parameters solely to pursue the current always-on
14-step editor. The idea is not disproven: two prompts repay editing, including
a substantial short-input win. The measured problem is paying for editing on
every round while useful repairs are sparse and some edits cause damage.

The next useful research milestone is a cheap decision to skip unprofitable
rounds, with its own cost included, plus graph reuse across changing context
lengths. Choose its policy using training/calibration data, then require a net
complete-generation gain on fresh prompts, including realistic cold costs,
before a larger training commitment. An oracle choosing winning prompts is not
a deployable policy. No new fit is launched by this recommendation.

## Reproduction and artifacts

One RTX A5000 on tron59, four CPUs, 32 GiB allocation; Torch 2.14.0+cu130,
vLLM 0.34.0, Transformers 5.22.2. Invariant BF16 target, FlashAttention 2,
2 GiB KV cache, maximum model length 4096, synchronous scheduling, chunked
prefill enabled, target/drafter eager execution, no prefix caching. Each run
records its complete configuration. The original pinned environment and
checkpoint were copied unchanged from tron61 using CPU-only staging.

| Job | Purpose | Source commit | Result |
|---|---|---|---|
| 8579825 | Initial native verification/timing | `140b0550b171eb0f001011422d5a82760ef8c763` | completed, exit 0, 9m14s |
| 8579890 | Kernel attribution, corrected FP32/BF16 timing | `40b5260ce98c40bd7a1dcba4de9af6d6bcc16bfc` | completed, exit 0, 12m22s |
| 8579976 | CPU-only original/target/drafter pin audit | existing `vllm_probe.prepare` | completed, exit 0, 17s |

Source archive SHA256s: `9bea99592cc682312a6b60fffc3bfbb731807092aa290a81de2a0320e30b38dc`
and `3b335bd8151902c7da894ac6c3e9a2ea1b5dee50489b08e1a5472dece16e7421`.
Original generator checkpoint: `c79b8c59c1e2cf73a7adf7aa9738b1b273779d330fe32f8d2c4a15fd56b7e8d2`.
Gate checkpoint: `b6f1c27a312a829efedc1db2ac42091f3702ad7d586f9025033508ab2c606882`.

Code: `editor-experiment/refinement_research/native_speed.py`,
`native_speed_worker.py`, `editor_kernels.py`, `kernel_probe.py`, `native_audit.py`
and the two Slurm scripts. Raw tokens, repeated timings, native traces, logs and
frozen sources are preserved remotely under `research/native-speed-8579825/` and
`research/native-kernels-8579890/`, mirrored privately in matching directories
under `~/.local/share/wayfinder-artifacts/`. Originals are unchanged.

Public aggregate evidence: [initial audit](evidence/native/invariant-fp32-audit.json),
[corrected FP32](evidence/native/standard-fp32-audit.json),
[BF16](evidence/native/standard-bf16-audit.json),
[kernel probe](evidence/native/kernel-probe.json),
[precision differences](evidence/native/precision-proposals.json),
[model pins](evidence/native/model-pins.json). No raw prompt or response tokens
are included in these public artifacts.
