# vLLM batch invariance: compatibility and cost investigation

November 17, 2026. **Batch-invariant vLLM eliminates the observed target
disagreement in this bounded test. It does not preserve every historical label.**
No editor training, bank collection, label replacement or corpus regeneration ran.
After reviewing this result, the user explicitly chose invariant vLLM as the new
canonical target and authorized a separate four-prompt pilot after supported-
configuration validation. See [the next experiment](NEXT_EXPERIMENT.md).

## Observed result

All six engine configurations completed. With invariance enabled, actual greedy
generation and actual DFlash verification produced the same 16 output tokens for
each of four fixtures, three independent request repeats. Their **full-vocabulary
raw-logit hashes matched at all 64 corresponding prefixes in every repeat**.
This is stronger than matching sampled IDs or top-five logprobs. Input prefixes
were reconstructed from the actual verification inputs, including rollback;
unmatched prefixes were excluded from logit comparisons.

With invariance disabled, the newly failing fixture disagreed at output position
6 in every greedy/DFlash comparison. Reversing batch order also changed the
choice: repeat 0/2 ordinary greedy chose 12946 and DFlash chose 16984; repeat 1
ordinary greedy chose 16984 and DFlash chose 12946. There were respectively
54/55, 51/56 and 54/55 unequal full-logit rows on shared prefixes. These are
cross-path and batch-layout differences, not a new claim that the original
Transformers generator randomly changes its outputs.

With invariance enabled, batch-order comparisons, the failing fixture alone,
and prefill chunk changes all matched in both tokens and full-logit hashes on
their shared prefixes. DFlash trials contained real proposal rejections and
subsequent continuation; this was not forced-token trace replay. The enabled-
chunking greedy/DFlash pair also matched all 64 generated-position logit rows.

The new canonical choice at the old failure is always **16984 (`Employee`)**;
the original saved choice is **12946 (`employee`)**. The other three saved
16-token windows match exactly. Original generation was previously reproduced
three times and is not demonstrated to have been wrong. Changing the engine
changes the numerical target definition; it does not retroactively invalidate
the saved data. The four record hashes match the prior rowwise report, and all
executed harness hashes match their local and remote copies.

### Configuration caveat

vLLM warned that `enable_chunked_prefill=False` is not officially supported for
this model. The principal three-repeat matrix used that setting, so it must not
be adopted as the final contract. The two additional cells used supported
chunked prefill, with a 64-token scheduling budget, and reproduced the same
invariant results. Those cells have **one** repeat each. The next validation
must enable chunked prefill in every cell and vary the scheduling budget to test
whole versus chunked prefill; all four short prompts fit within a 256-token
budget without disabling chunking. Repeat that supported configuration before
any bank collection. The original executed harness is retained unchanged as
evidence, rather than rewriting its configuration after the run.

## Measured cost

Median of repeats 1 and 2, each generating 16 tokens for four prompts (64 output
tokens total); repeat 0 is excluded from this small timing summary:

| Mode | Invariance off | Invariance on | Ratio |
| --- | ---: | ---: | ---: |
| Greedy target computation | 0.588 s | 1.161 s | 1.97× |
| Greedy complete generation | 1.046 s | 1.797 s | 1.72× |
| DFlash target computation | 0.281 s | 0.581 s | 2.07× |
| DFlash complete generation | 0.884 s | 1.238 s | 1.40× |

Target time includes synchronized backbone/head calls and bookkeeping, excluding
subsequent CPU logit hashing. Complete generation includes tracing, hashing,
scheduling and RPC overhead. Both modes use eager execution. These two-repeat
instrumented measurements, in the configuration carrying the warning above,
are **not production throughput estimates or an editor speedup**. Ordinary
greedy used 17 target forwards per batched trial; the two summarized DFlash
trials used eight each in both modes. All actual forwards, including extra
scheduler/profile calls, were counted inside the worker.

The flag increased complete DFlash time by about 40% here. Source inspection
below explains plausible mechanisms; this run does not isolate each kernel's
exclusive contribution. The flag also affects draft arithmetic and proposal
work, so the observed total is not purely a target-kernel penalty.

## Setup, resources and integrity

Initial job 8452972 stopped at my 900-second installation timeout while unpacking
packages, after 15m07s. It performed zero target forwards. All 199 downloaded
wheels survived and passed their resolver-recorded SHA256 hashes. A protocol
amendment was [recorded before recovery](https://github.com/Spec-decoding-project/Speculative-decoding-project/issues/17#issuecomment-6954819602):
one 14-minute recovery allocation, no further downloads, no extra total resource
budget. It reinstalled every dependency to avoid retaining partially unpacked
packages. `pip check` passed.

Recovery job **8453091 completed in 8m28s**, exit 0. Total allocated time was
**23m35s**, below 30 minutes. Both allocations used one RTX A5000, four CPUs,
32 GiB on tron61. The diagnostic counted **206 target forwards** and
**13.8170 synchronized target-call seconds**, below the 300/600 limits. Maximum
allocated GPU memory across cells was 19,142,772,224 bytes (17.83 GiB); maximum
host RSS was 30,571,132 KiB. The user queue was empty afterward.

Runtime: Python 3.12.12, vLLM 0.34.0, Torch 2.14.0+cu130, CUDA 13.0,
Transformers 5.22.2, NVIDIA driver 595.91.07. All target attention layers reported
FlashAttention version 2 and the requested invariance state. The global invariant
dispatch state was also checked. FlashInfer sampling kernels were unavailable;
the logged fallback sampler was used consistently. Greedy sampling explicitly
used temperature 0 and seed 0. Installation time is unrelated to the timing table.

The isolated environment is `/scratch0/fpajak-vllm034-Ry21jsII/venv` on tron61;
it is node-local and subject to scratch cleanup. Resolved artifacts are recorded
in [resolved-packages.json](resolved-packages.json). The existing `.venv-qwen`,
target/draft checkpoint files, original responses and both untrained AR editor
branches remain unchanged. The four CPU comparison tests pass.

Raw numerical evidence is in [result.json](result.json), with per-cell records in
`evidence/`. Full installer logs/reports remain on Nexus under
`research/editor-feature-bank/vllm-invariance-8452972/` and
`research/editor-feature-bank/vllm-invariance-8453091/`.

## Why this implementation is a relevant correction

The prior experiments established repeatable disagreement between cached greedy
and block target execution. Serializing the target's linear layers fixed their
first projected rows but did not fix the final token choice. A complete execution
path with invariant reductions is therefore a better candidate than rounding
logits or patching only matrix multiplication.

[vLLM's batch-invariance documentation](https://docs.vllm.ai/en/stable/features/batch_invariance/)
lists Qwen3-8B among tested models and NVIDIA compute capability 8.0+ among
supported hardware. The feature is beta. The flag must be active before engine
initialization; adding it to the existing Transformers/DFlash script would not
change that script's kernels. Both target generation and verification must use
the new engine.

We pinned vLLM 0.34.0, released November 16, to source commit
`62d4bc3c7e33b1b6c906e233e6af2d3506c482ca` and the SHA256-pinned Linux wheel in
[upstream-pin.json](upstream-pin.json). Its source registers the released DFlash
architecture and contains an [acceptance test for this exact model pair](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/tests/v1/e2e/spec_decode/acceptance_rates/dflash/test_dflash.py).
That upstream support does not replace validation of our checkpoint revisions,
hardware, cache layouts and batch-invariance setting.

## Where a slowdown can come from

The [pinned invariant implementation](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/model_executor/determinism/batch_invariant.py)
installs Triton matrix multiplication overrides on Ampere, because its ordinary
cuBLAS path cannot provide the needed invariance. Fixed reduction organization and
tiles restrict shape-dependent kernel choices. Small batches may leave part of a
tile unused. The implementation also supplies invariant reductions/softmax and
disables reduced-precision BF16/FP16 reductions. These are source-level reasons
for a possible cost, not a measured attribution on our device.

The [FlashAttention implementation](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/attention/backends/flash_attn.py)
uses a fixed maximum sequence length for invariant metadata, disables its usual
ahead-of-time scheduling path and limits attention to one split. Constraining
reduction partitioning can reduce parallelism. The selected FlashAttention
version and activation flags must be read from the run, not inferred from newer
GPU examples. NCCL restrictions are not an appropriate primary explanation for
this single-GPU experiment.

Decimal rounding cannot provide the same contract: two nearby values can still
land on opposite sides of a rounding boundary. Batch invariance addresses the
computation that produces logits; it does not promise to reproduce a different
engine's historical numerical choices. vLLM's [reproducibility scope](https://docs.vllm.ai/en/stable/usage/reproducibility/)
also depends on the fixed hardware and version. Four passing prompts would be a
compatibility result, not proof for all future inputs.

## Experiment and limits

The [diagnostic protocol](../../../editor-experiment/replay_diagnostic/VLLM_INVARIANCE.md)
defines the six engine processes, genuine greedy/DFlash comparison, full-vocabulary
hashes on identical prefixes, original-label comparison, batch/chunk controls,
budget and instrumented timing limits. All checkpoint files are reused in place.
The existing generator environment is preserved. Node-local installation avoids
putting another Torch/CUDA environment in the nearly full class home directory.

The allocated device is an RTX A5000 on tron61, compute capability 8.6, NVIDIA
driver 595.91.07, 24,564 MiB. That clears the CUDA 13 wheel's minimum driver-family
preflight; successful installation and kernel execution remain distinct checks.

## Separating editor benefit from invariance cost

After the correctness and training gates pass, the useful timing matrix is:

| Execution mode | DFlash alone | Same DFlash plus trained editor |
| --- | --- | --- |
| Invariance off | baseline A | edited B |
| Invariance on | baseline C | edited D |

Compare A/B and C/D on the same held-out prompts and frozen editor checkpoint.
Compare C/A separately to estimate the invariance cost without an editor. A
comparison of C/B would confound both changes. Record acceptance, total target
calls, proposal changes, complete wall time and correctness in each mode. The
flag can also change drafter arithmetic, so a wall-time difference need not be
purely target-kernel overhead. An editor gain with invariance disabled would not
by itself establish exact agreement with ordinary greedy decoding in that mode.

No trained editor is part of this run, so it cannot establish an editor speedup.
