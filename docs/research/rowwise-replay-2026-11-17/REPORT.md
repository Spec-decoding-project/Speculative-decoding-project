# Single-row linear correction fails; full target arithmetic needs a consistent contract

The proposed correction was implemented and tested. It **does not restore exact saved-token agreement**, and it is not adopted for collection or serving. Processing the target's linear layers one token at a time makes the selected first-layer query, key and value projections match cached decoding, but differences reappear later. The new failing token remains wrong in all three modified-verifier repeats. Instrumented verification time increases roughly ninefold.

This is repeatable disagreement between two executions of the **target**. It is not evidence that either execution randomly changes its answers, and it is not evidence of drafter nondeterminism. The target and drafter are distinct models and are expected to disagree on proposals.

The user's November 17 requirement is target self-consistency across generation and verification. Reducing digits was suggested, and regeneration was raised conditionally if the original responses were produced incorrectly. Original greedy generation again reproduces the recorded first block on every fixture; these tests do not show incorrect source generation. No source response was overwritten or regenerated, and no editor training occurred. Changing the canonical execution contract and finding different valid greedy outputs would be a separate result from discovering a source-generation bug.

## Actual intervention and results

[rowwise.py](../../../editor-experiment/replay_diagnostic/rowwise.py) changes only the target's second call inside an actual released DFlash `spec_generate` first round. All 253 linear modules, including the vocabulary head, evaluate 16 contiguous single-token rows instead of one 16-token matrix, for 4,048 linear calls in the verification block. Prefill, BF16 weights, SDPA, cache positions, token order and native acceptance logic stay unchanged. Original methods and hooks are restored after each trial. Independent fresh caches are created by every native round.

The diagnostic calls the authenticated original generator method with only its new-token cap shortened from 2,048 to 16. Model/source/record fingerprints and effective configuration are saved in [result.json](result.json). Target/draft weights remain frozen. The saved-candidate control replaces the 15 draft sampler outputs solely to hold every verification prefix fixed. Ordinary proposals are a separate control.

| Fixture | Original generation | Default native verifier | Single-row-linear native verifier | First remaining traced difference at selected position |
| --- | --- | --- | --- | --- |
| `ultrachat_200k:train_sft:37680`, anchor 24 | 16/16 exact | 15/16; position 6 differs | 15/16; same position differs | Layer 2 output (third transformer layer) |
| `gsm8k:train:3821`, anchor 38 | 16/16 exact | 16/16 exact | 16/16 exact | First layer attention output projection |
| `ultrachat_200k:train_sft:164610`, anchor 19 | 16/16 exact | 16/16 exact | 16/16 exact | Layer 1 output |
| `ultrachat_200k:train_sft:61759`, anchor 22 | 16/16 exact | 16/16 exact | 16/16 exact | First layer attention output projection |

Every token-agreement column represents three repeats with the same verdict. Separate cached teacher-forced traces through the selected positions confirm the reference. On the new failure, original generation selects saved token 12946 from logits 36.0/36.0 for IDs 12946/16984; both default and modified native verification give 35.75/36.0 and select 16984.

On that new fixture, the selected row agrees through both of the first two transformer layer outputs under the intervention. Its layer 2 output then differs in 1,303 components, maximum 0.00390625. The final normalized hidden vector differs in 3,480 components, maximum 1.5, RMS 0.03517197. Only the first layer has detailed intra-layer trace sites; the layer-2 finding does not identify its exact internal operation. Other fixtures already differ at the first attention output projection after their first Q/K/V rows agree. These results reject the hypothesis that correcting linear row shape alone is sufficient. They motivate auditing all reductions, attention/cache layouts and reference execution history; they do not prove a defect in one attention kernel.

Ordinary DFlash produces the same proposal hash with and without this target-only intervention. Its first wrong proposed token relative to the saved response is position 7; both verifiers disagree at position 6, before that wrong input, and accept five proposals. Later logits conditioned on different candidate prefixes are not used as saved-label comparisons.

## Cost and resource accounting

Measurements below are medians of two instrumented, non-traced native rounds after a traced round on each fixture. They include synchronization and diagnostic hooks. They are sufficient to reject an assumption of a free correction, but are not an end-to-end serving or editor-speedup benchmark.

| Fixture suffix | Default block verification | Modified block verification | Default full native round | Modified full native round |
| --- | ---: | ---: | ---: | ---: |
| 37680 | 42.81 ms | 391.90 ms | 94.50 ms | 443.63 ms |
| GSM8K 3821 | 43.54 ms | 392.54 ms | 95.57 ms | 444.80 ms |
| 164610 | 44.92 ms | 392.41 ms | 96.98 ms | 445.55 ms |
| 61759 | 44.05 ms | 392.74 ms | 95.85 ms | 444.33 ms |

Job `8452037` ran on one RTX A5000 on `tron56`, four CPUs and 32 GiB host allocation. SLURM elapsed time was **2m17s**, process wall time 101.71s, synchronized target-forward time 20.3505s. Peak allocated GPU memory was 18,532,453,888 bytes (17.26 GiB), maximum host RSS 20,372,380 KiB. Runtime was Python 3.12.12, Torch 2.13.0+cu126, Transformers 5.15.0, seed 0, BF16 SDPA and TF32 off. The deterministic-algorithm flag was left at its original false setting to avoid confounding the one-variable intervention; no new claim of within-path nondeterminism follows from that setting.

The exact planned and actual target-forward count was **279**: 216 repeated source/default-native calls, 35 cached trace calls, 24 modified-native calls and four ordinary-control calls. The 300-call / 600-target-second limits were not hit. `complete` is true, `all_fixtures_exact` is false, `all_query_rows_exact` is true and no exception/error is present. SLURM `FAILED` / exit 1 is the intended negative correctness verdict, unlike the prior optional-probe budget exception. No second allocation, bank recollection or precision sweep ran.

Five CPU diagnostic tests pass, including intervention scope, vocabulary-head inclusion, output ordering, weight preservation, restoration after an invalid call, reporting and budget controls. The GPU experiment is the actual negative correction regression; the CPU tests are not substitutes for it.

## Why rounding to fewer digits is not a guarantee

Rounding moves the decision boundaries rather than eliminating them. For example, rounding these pairs to one decimal gives different argmax decisions even though their unrounded values are very close:

| Execution | Token A score | Token B score | Rounded pair | Choice with lowest-ID tie break |
| --- | ---: | ---: | --- | --- |
| First | 1.0499 | 1.0498 | (1.0, 1.0) | A |
| Second | 1.0499 | 1.0501 | (1.0, 1.1) | B |

The same boundary problem applies to a fixed number of significant digits. Coarser scores also create more ties and can change the target's decisions. A tolerance or empirical maximum error is not a proof of equality for every input. A certified bound could justify a fast path when the top-token margin exceeds twice the bound, with exact reference recomputation otherwise; no such end-to-end bound has been established here. No rounding/tolerance patch was applied.

## A concrete route to consistent target execution

For the requested guarantee, define one canonical target computation shared by generation and verification: pinned weights/tokenizer/template, positions/masks, cache semantics, logits processors/tie rule, arithmetic kernels, hardware/runtime and supported layouts. Within that defined scope, equivalent prefixes must have equivalent computations. A seed or deterministic-algorithm switch addresses repeatability, not equality between different arithmetic layouts. PyTorch explicitly documents differences between batched and sliced floating-point computations and limits cross-platform/version reproducibility. [Numerical accuracy](https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html), [reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html).

Two implementation routes exist:

1. Use the identical serial cached target procedure for both purposes, including canonical prefill and cache history. This is a correctness reference; recomputing all verification tokens serially gives up the block-parallel target evaluation central to speculative speedup. This session did not implement or validate a full-generation serial verifier.
2. Use batch-invariant kernels throughout both target paths, so supported batch/chunk layouts preserve the reduction arithmetic. This is the promising route for retaining useful parallel verification. It requires consistent matrix multiplication, normalization and attention, not only a one-row linear wrapper. Thinking Machines describes the required reduction and KV-layout invariants; its standalone operator library alone does not supply the entire attention solution. [Primary implementation explanation](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), [operator library](https://github.com/thinking-machines-lab/batch_invariant_ops).

Current vLLM documentation describes `VLLM_BATCH_INVARIANT=1`, lists Qwen3-8B as tested, and supports NVIDIA compute capability 8.0+. The feature is beta. Its speculative interface includes DFlash, and its trace-replay documentation explicitly conditions bitwise agreement between speculative and non-speculative logprobs on batch invariance. These are upstream capabilities, not a verified combination with this pinned checkpoint or our local editor integration. vLLM is absent from the current Nexus environment; Triton is installed. [Batch invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/), [speculative interface](https://docs.vllm.ai/en/latest/features/speculative_decoding/), [trace replay](https://docs.vllm.ai/en/latest/serving/online_serving/trace_replay/).

The next bounded test should therefore establish a compatible, pinned invariant target path and compare genuine greedy generation against genuine candidate verification on identical prefixes, then vary query length, cache chunking and batch companions. Check logits and tokens, including cache rollback and continuation after rejection. Forced trace-token replay alone cannot prove greedy output agreement. Preserve the old labels and report any difference from the new canonical outputs separately. Only after a consistent new path is demonstrated can a concrete regeneration proposal be evaluated; regenerating 50k responses under today's mismatching paths would not solve the problem.

There is no verified correction yet. The bank and editor training remain blocked.

## Artifacts

- [Result](result.json), SHA256 `f64712dd1a0600d88f780a23640d3f3e8abf1932be00a3ffc8e7dba6dc725dc2`.
- [Driver](../../../editor-experiment/replay_diagnostic/rowwise_session.py), SHA256 `cc3ecc04d78b6466c2e478af655534bcfd739bec42dfff977989fd700e2d7b13`.
- [Intervention](../../../editor-experiment/replay_diagnostic/rowwise.py), SHA256 `7c4f5cea3a6cfaf589a3806eb448712daf201b29746c6a1e2e029ed61d097fc1`.
- Reused [exact diagnostic](../../../editor-experiment/replay_diagnostic/exact_session.py), SHA256 `c0b60a2a474447a284b28d8f9efaa49174fa3c2a88a1dc5dbd558b4f18f656f0`.
- [Usage and call plan](../../../editor-experiment/replay_diagnostic/ROWWISE_REPLAY.md), [SLURM runner](../../../editor-experiment/replay_diagnostic/rowwise.sbatch), [CPU checks](../../../editor-experiment/replay_diagnostic/test_rowwise.py).

All three executed Python source hashes match local files and the remote files. Raw responses remain on Nexus. Remote results are under `/fs/classhomes/fpajak/speculative-decoding-project/research/editor-feature-bank/rowwise-replay-20261117/`, with log `rowwise-replay-8452037.log`. The user queue was empty after completion. Original workspace edits and both untrained token-only AR branches remain unchanged.
