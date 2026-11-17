# Bounded vLLM batch-invariance diagnostic

This is an isolated compatibility experiment for the four known first-block
fixtures. It does not enable a collector, modify saved labels, or train an editor.
The executed scripts are retained unchanged for provenance. Do not automatically
rerun this matrix: vLLM warned against its disabled-chunking configuration. The
next matrix must keep chunking enabled and vary its token budget instead.
The original `vllm_probe.sbatch` was submitted from the Nexus project root. It installs the
hash-pinned vLLM 0.34.0 wheel and its resolved dependencies in a unique directory
on the allocated node's `/scratch0`. Existing `.venv-qwen` and model files remain
unchanged. The package report and freeze record the actual dependency versions.

The reviewed source commit is `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`.
See `docs/research/vllm-invariance-2026-11-17/upstream-pin.json` for wheel and
reviewed-source hashes. The wheel's own package version must be 0.34.0.

## Execution and evidence

The principal matrix is batch invariance off/on × ordinary greedy/DFlash, with
three fresh-request repeats per cell. All four fixture prompts are submitted
together; repeat 1 reverses their order. Invariance-on cells also run the newly
failing fixture alone. Two extra engines repeat the invariant comparison with
prefill chunking enabled and a smaller scheduling token budget. These are layout
stress tests, not a backend or precision sweep.

Every engine uses BF16, one GPU, greedy sampling, 16 output tokens, disabled prefix
caching, synchronous scheduling, eager execution, and the FlashAttention backend.
DFlash proposes 15 tokens after the anchor, retaining a 16-query verification
block. Native proposals and native rejection sampling are used unchanged. EOS is
ignored solely to obtain a fixed 16-token diagnostic window. Original fixture
records, checkpoints, tokenizer files and generator hashes are authenticated.

The worker extension installs before model loading. All top-level Qwen3 target
forwards, including profiling and warmup, share a persistent 300-call limit.
Target backbone and vocabulary-head time share a 600-second limit, checked at
call boundaries. The maximum planned generation steps are 256, with up to eight
extra chunk steps and 36 startup calls reserved. The scheduler's actual count is
authoritative. The SLURM allocation is one RTX A5000, four CPUs, 32 GiB, 30 minutes
including installation; there is no automatic retry after an execution failure.

The worker captures full-vocabulary raw-logit hashes before sampling. It rebuilds
the exact input prefix from token IDs and positions, truncating speculative
history when verification rolls back. Only equal prefix hashes are compared;
logits after a differing prefix are not evidence about numerical equality.
Repeated evaluations of one prefix cannot overwrite an earlier disagreement.
Sample counts and rejection counts accompany the trace, so rejection/continuation
coverage must be checked from the result rather than assumed.

`invariant_active` and each target attention implementation's invariance flag
must agree with the requested mode. Explicit FlashAttention selection prevents a
silent backend fallback. This checks dispatch/configuration activation; it is not
a proof that every supported shape or upstream kernel is correct.

## Interpretation and timing

Report canonical greedy-versus-verifier agreement separately from compatibility
with the older saved labels. A passing four-fixture test neither proves universal
agreement nor passes the complete bank gate. Missing shared-prefix comparisons,
missing rejection coverage, installation failure and unexecuted matrix cells are
limitations, never passes. Preserve first failures and partial output.

Target timings synchronize CUDA and exclude subsequent CPU logit hashing. Whole
generation wall time includes hooks, hashing, RPC and scheduling overhead. These
are instrumented eager measurements, not production throughput. Use later repeats
to separate startup/compilation effects. The global invariance flag also affects
the drafter, so changed proposals, accepted-token counts and target-call counts
can change wall time independently of kernel cost.

An eventual editor claim needs a separate held-out editor-off/editor-on comparison
within **each** invariance mode, with one frozen editor checkpoint and matched
settings. Compare the editor against its own mode's baseline and report numerical
correctness separately. An invariant-baseline versus non-invariant-editor timing
comparison confounds two changes. No trained editor benefit is established here.

CPU checks:

```sh
python3 -m unittest discover -s editor-experiment/replay_diagnostic -p test_vllm_probe.py -v
```
