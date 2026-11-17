# Supported canonical target and four-prompt pilot

The user approved the new invariant target and a separately versioned four-prompt
pilot after reviewing the `employee`/`Employee` difference. See ADR 0002 and
[the active task](https://github.com/Spec-decoding-project/Speculative-decoding-project/issues/18).
This is a new, explicitly authorized experiment following the completed first
compatibility test. It does not overwrite historical labels or start training.

One RTX A5000 on tron61, four CPUs, 32 GiB, 30 minutes; shared worker-side limits
of 4,800 target forwards or 600 synchronized target-call seconds. Reuse the pinned
isolated environment and checkpoints with no downloads. Package freeze and
`pip check` must match the completed setup. Validation reserves up to 500 calls;
two full-response paths reserve up to 4,104, leaving initialization/scheduling
margin. All actual calls are counted, including profiling and empty/drain steps.

Every cell keeps chunked prefill enabled. The common contract is vLLM 0.34.0,
the pinned Qwen3-8B and matching DFlash weights/tokenizer, BF16, explicit
FlashAttention, eager synchronous execution, no prefix cache, greedy sampling,
seed zero, four maximum sequences, context limit 2,304 and 2 GiB KV capacity.
The unavailable FlashInfer sampler is explicitly disabled. Invariance is set
before engine startup and verified in both dispatch and target attention layers.

Six short validation cells run first: invariant greedy/DFlash with scheduling
budgets 256 and 64, and ordinary-kernel greedy/DFlash with budget 256. Each has
three fresh-request repeats; repeat 1 reverses prompt order. Invariant budget-256
cells also test the failing prompt alone. Every invariant pair, repeat, chunk
change and companion control must agree exactly on tokens and full-vocabulary
logit hashes at matching prefixes. Ordinary-kernel differences are recorded.
Require real rejections followed by continued generation.

After that gate passes, generate the four complete greedy continuations with the
original 2,048-token output cap and natural EOS stopping. Run genuine DFlash
independently from the same prompts. Require exact complete token sequences and
full-logit hashes on every emitted prefix. A capped response or mismatch prevents
publication as a passing pilot. No label injection or forced-token trace replay.

Only after successful verification, atomically publish a new directory containing
`run.json`, `shard-00000.jsonl` and `verification.json`. Preserve source prompt IDs,
messages and splits, exact new token IDs, original record hashes, full provenance
and checksums. Use `verified_pilot` status, not a training-bank status. The existing
CPU journal reader independently checks the staged record/run hashes before its
directory is published. Original responses and both AR editor branches remain
unchanged. A complete canonical 128-repair/16-preservation bank still precedes
fitting; this pilot does not authorize 50k regeneration.

The runner and hooks are `vllm_contract.py`, `vllm_contract_worker.py` and
`vllm_contract.sbatch` under `editor-experiment/replay_diagnostic/`. The older
executed scripts remain frozen. `audit_vllm_probe.py` checks that each emitted
token is the captured argmax at its actual full prefix, independently of vLLM's
decorated internal request IDs. Eight CPU integrity/comparison tests pass.

Observed cost must be reported as instrumented eager timing. The fixed invariant
reductions and Ampere kernels can add cost; the flag also changes draft arithmetic
and scheduling work. The setup install is a separate expense. Full pilot agreement
is evidence for this pinned contract and these four prompts, not a universal proof
for arbitrary hardware, versions, graph modes or context/batch layouts.
