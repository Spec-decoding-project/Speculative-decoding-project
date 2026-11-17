# Canonical invariant target: full pilot passed

November 17, 2026. **All four complete responses reached EOS. Independent greedy
and DFlash generation matched all 3,898 emitted tokens and every corresponding
full-vocabulary raw-logit hash.** The new versioned pilot was published separately;
all 197 historical response/metadata files retain their November 10 SHA256 hashes.
No editor training, full-corpus regeneration or feature-bank pass occurred.

The user explicitly adopted invariant vLLM after reviewing the earlier
`employee`/`Employee` difference. [ADR 0002](../../adr/0002-version-the-invariant-target.md)
records that decision. This is a new canonical numerical target, not a claim
that the original reproducible generator was wrong.

## Validation and publication

The [protocol](PROTOCOL.md) kept chunked prefill enabled in every engine,
correcting the unsupported configuration used in the first diagnostic. Both the
short tests and full pilot used context limit 2,304, 2 GiB KV capacity, BF16,
FlashAttention 2, eager synchronous execution, disabled prefix caching, seed zero,
and the pinned checkpoints. No package or model downloads occurred.

Invariant greedy/DFlash pairs passed three fresh-request repeats at each of the
256- and 64-token scheduling budgets. Reversed batch order, the failing prompt
alone, and changed prefill chunk boundaries all passed. Real candidate rejection
was followed by correct continuation. Ordinary kernels still disagreed on the
known fixture in all three comparisons; 54/55, 56/56 and 54/55 shared-prefix
full-logit rows differed. The comparisons never equate logits from different
token prefixes.

Only after this gate passed did full generation begin. Both full paths used the
original 2,048-token cap and natural EOS stopping, with genuine native proposals
and no forced labels. Each returned token was independently aligned to its exact
input prefix and captured argmax. All four outputs end with EOS token 151645.
Their first 16 tokens and shared logit rows also match the short-run results.

| Prompt ID | Original tokens | New tokens | First difference from original, zero-based |
| --- | ---: | ---: | --- |
| `ultrachat_200k:train_sft:37680` | 1,628 | 1,568 | 6: 12946 (`employee`) → 16984 (`Employee`) |
| `ultrachat_200k:train_sft:61759` | 872 | 937 | 16: 3070 → 4124 |
| `ultrachat_200k:train_sft:164610` | 1,005 | 1,196 | 114: 264 → 9760 |
| `gsm8k:train:3821` | 197 | 197 | None; full sequence unchanged |

These changes are recorded, not filtered. All four prompt IDs, messages and train
splits are retained, together with original record hashes. The new records have
`verified_pilot` status and are explicitly not a training bank. The existing CPU
journal reader validated every record/run checksum before atomic publication;
the downloaded publication was checked again against both generated sequences,
the authenticated prompts, EOS IDs and journal/run file hashes.

The durable Nexus data directory is:

```
/fs/classhomes/fpajak/speculative-decoding-project/data/qwen_responses_vllm_invariant_pilot_20261117
```

It contains `run.json`, `shard-00000.jsonl` and `verification.json`. Journal SHA256:
`5948edc85d9384cffe1404319104458effd24b0e185bf62de7b7b642d5fc3c2f`.
The [publication metadata](evidence/pilot-verification.json) records all response
hashes and old/new differences. Full response records and full-generation traces
remain in the protected Nexus experiment/data directories; their hashes and
sizes are listed in [result.json](result.json).

## Measured cost in the supported configuration

Medians of short-run repeats 1 and 2, four prompts × 16 output tokens:

| Measurement | Invariance off | Invariance on | Ratio |
| --- | ---: | ---: | ---: |
| Greedy target-call time | 0.558 s | 1.079 s | 1.93× |
| Greedy complete generation | 0.941 s | 1.528 s | 1.62× |
| DFlash target-call time | 0.289 s | 0.452 s | 1.57× |
| DFlash complete generation | 0.840 s | 1.061 s | 1.26× |

Thus complete instrumented DFlash time increased by about **26%** here. The earlier
40% estimate came from a different configuration and run; do not combine them
into one production estimate. There are only two timed repeats per median. Both
DFlash modes used eight target forwards in each summarized repeat. Ordinary
greedy used 16/17 forwards versus invariant greedy's 17/17, reflecting dynamic
request scheduling. The flag also changes draft arithmetic and candidate work.

The full invariant pilot took 144.56 s for greedy generation (1,568 target
forwards, 105.46 target-call seconds) and 115.23 s for DFlash (564 target forwards,
36.21 target-call seconds). These measurements include synchronization, tracing,
CPU hashing and bookkeeping; DFlash hashes rejected-candidate rows too. They are
correctness-run costs, not serving benchmarks or evidence of an editor benefit.

### Why invariance can cost time

On Ampere, the [pinned implementation](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/model_executor/determinism/batch_invariant.py)
replaces ordinary matrix multiplication with invariant Triton kernels, constrains
reduction organization and disables reduced-precision BF16 reductions. Fixed
tiles can waste work on small batches. Both observed modes already used IEEE FP32
precision, so a TF32 switch does not explain this difference. Target weights
remain BF16; this is not a wholesale FP32 model conversion.

The [attention implementation](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/attention/backends/flash_attn.py)
constrains split/reduction scheduling and uses fixed maximum-sequence metadata.
Those choices can sacrifice parallelism. They are source-supported explanations,
not an exclusive per-kernel attribution: no profiler breakdown was collected.
Single-GPU NCCL restrictions are not a useful primary explanation here.

To argue that an eventual editor pays for itself without this overhead, measure
editor off/on **within each invariance setting**, using the same frozen editor
and held-out prompts. Comparing an invariant baseline to an ordinary-kernel
editor would confound two changes. In a simple steady-round calculation, the
accepted-token ratio must exceed `(round cost + editor cost) / round cost`.
Reducing the target's cost can therefore make an editor harder to justify.
Ordinary-kernel numerical disagreement must still be reported; speed alone does
not establish exact greedy equivalence. No editor was trained or evaluated here.

## Resources, integrity and remaining gate

Job **8453328** completed with exit 0 in **10m55s** on one RTX A5000, four CPUs and
32 GiB on tron61. It used **2,394 target forwards** and **156.6208 synchronized
target-call seconds**, below the 4,800/600 limits. Peak allocated GPU memory was
20,755,220,992 bytes (19.33 GiB); maximum host RSS was 20,804,396 KiB. The user queue
was empty afterward. Original-data verification found all 197 recorded files
unchanged; see [the checksum audit](evidence/original-data-integrity.json).

Runtime remained vLLM 0.34.0, Torch 2.14.0+cu130, CUDA 13.0, Transformers 5.22.2,
Python 3.12.12 and driver 595.91.07. Package freeze matched the isolated setup;
`pip check` passed. Every target attention implementation and the invariant
dispatch flag matched the requested mode. Executed source hashes match local
and remote files. Eight CPU integrity tests pass. Both AR editor branches and
the original workspace changes remain preserved.

This establishes the tested pinned contract on four complete prompts, including
3,898 exact full-generation positions. It is not a universal proof over arbitrary
hardware, versions, graph modes, longer contexts or all inputs; the upstream
feature remains beta. The next task is faithful native token-block export from
the canonical engine, followed by a complete, independently audited
128-repair/16-preservation bank. That gate still blocks fitting. Expanding the
canonical response pool and running 50k regeneration require their own explicit
scope and measured resource budget.
