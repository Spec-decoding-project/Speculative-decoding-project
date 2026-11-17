# Original greedy labels reproduce; block arithmetic changes a token

**The new mismatch is reproducible numerical execution sensitivity.** The actual original generator reproduces the first 16 saved tokens on all four fixed fixtures, three times each. Native block verification rejects one of those tokens on the new failing fixture, also three times. An isolated linear projection produces different outputs for a bitwise-identical input vector and unchanged weights when evaluated alone versus inside the 16-row block. This establishes shape-dependent floating-point differences, without claiming a particular vendor kernel defect or that this projection is the only contributor.

The diagnostic answers the user's question positively. It does **not** validate a correction or a training bank. Original labels remain unchanged, no mismatch was filtered, and no editor training or bank recollection ran. The [exact-token decision](../../adr/0001-preserve-exact-saved-target-tokens.md) remains binding.

## Matched-prefix reproduction

Job `8451880` invoked the authenticated original `QwenBackend.generate` method from `scripts/generate_qwen_responses.py`, with the original saved generation configuration. Only `max_new_tokens` changed from 2,048 to 16. This reproduces one full first block, not each complete response. The original generator SHA256 is `421e593f7f2ab56bfac09adc84f7d887455ebeb0955685ed8c22abf9b43f6ac0`.

Pinned target: Qwen3-8B `b968826d9c46dd6066d109eabc6255188de91218`. Pinned draft: Qwen3-8B-DFlash-b16 `9b41424b7109f9c5413454f481b09a82b85333f4`. Target/draft weights and source fingerprints were checked. Original prompt tokens, every generated prefix, cache lengths, and absolute positions were checked. Both models stayed frozen. Inference used BF16, SDPA, seed 0, no sampling, one beam, cache enabled and TF32 disabled; the effective generation configuration is in [result.json](result.json).

The native control calls the hash-verified released `spec_generate`. Its draft sampler alone is replaced with the 15 saved proposals; the anchor still comes from native prefill. This artificial candidate keeps the verification prefixes fixed and is neither editor output nor a repair-quality result. Position numbers below are zero-based within the 16-token block, including the anchor.

| Fixture | Anchor | Original generation, each of 3 repeats | Native saved-candidate verification, each of 3 repeats | Uncached replay, each of 3 repeats |
| --- | ---: | --- | --- | --- |
| `ultrachat_200k:train_sft:37680` | 24 | 16/16 exact | 15/16; position 6 differs | 15/16; position 6 differs |
| `gsm8k:train:3821` | 38 | 16/16 exact | 16/16 exact | 16/16 exact |
| `ultrachat_200k:train_sft:164610` | 19 | 16/16 exact | 16/16 exact | 16/16 exact |
| `ultrachat_200k:train_sft:61759` | 22 | 16/16 exact | 16/16 exact | 15/16; position 15 differs |

On the new failing fixture, position 6 predicts the seventh saved token at absolute position 30, from the identical context through absolute position 29:

| Execution | Saved token 12946 logit | Other token 16984 logit | Selected token |
| --- | ---: | ---: | ---: |
| Original cached greedy generation | 36.00 | 36.00 | 12946 |
| Independent cached teacher forcing | 36.00 | 36.00 | 12946 |
| Native cached block verification | 35.75 | 36.00 | 16984 |
| Uncached full-prefix replay | 36.00 | 36.25 | 16984 |

Every row above repeated three times with the same outcome. Original cached generation has a maximum tie between the two IDs and selects 12946. Block execution removes that tie. There is no random sampling involved. Independent cached teacher forcing on the new fixture covers saved positions 0–6; the later trace pass repeats that comparison once on every fixture through its selected position. The three original-generation repeats on the other fixtures also verify that all generated prefixes equal their saved prefixes; they are not three separate calls to the teacher-forcing helper.

Ordinary DFlash was also run once on the new fixture. Its first candidate error against the saved tokens is position 7, but native verification already disagrees at position 6 and accepts only five proposals. Thus the position-6 discrepancy also occurs with an ordinary candidate before its first wrong token. Later logits with different candidate prefixes are not evidence of saved-label disagreement and must not be interpreted that way.

## Where the computations first differ

At the corresponding input row, the embedding and first layer input normalization are bitwise identical. The first observed difference is the first layer query projection, before that layer's attention computation. Directly evaluating the captured query input with the same projection weights isolates this operation:

- Single-token input shape: `[1, 1, 4096]`.
- Block input shape: `[1, 16, 4096]`; its selected row is bitwise equal to the single-token input.
- New fixture: 1,310 of 4,096 output components differ; maximum absolute difference `0.001953125`, RMS difference `0.000106978463`.
- All three isolated repeats reproduce the same differences. Single-row output exactly matches the cached trace; the selected block output exactly matches the native trace.
- All four fixtures show this phenomenon, including those whose final token selections agree.

The new fixture's final normalized target hidden vector differs in 3,395 of 4,096 components, with maximum absolute difference 0.5 and RMS difference 0.01757325. These upstream differences rule out an explanation limited to vocabulary readout rounding. They do not quantify the causal contribution of each layer to the final token flip. No counterfactual replacement of only the first query projection was run.

| Isolated first-query projection on new fixture | Differing components | Maximum absolute difference |
| --- | ---: | ---: |
| Default BF16 | 1,310 | 0.001953125 |
| Disable reduced-precision BF16 reductions | 6 | 0.000244140625 |
| Compute projection in FP32, then cast to BF16 | 0 | 0 |

These last two interventions apply only to the isolated projection, not a full inference pass. The FP32 projection still differs in one component on the GSM8K control (maximum `0.0000019073486328125`), so it is not a general equality guarantee.

## Why a more precise vocabulary head is not a saved-token fix

Applying the same FP32 vocabulary matrix to each path's fixed final hidden vector yields:

| Hidden vector | Token 12946 | Token 16984 | 12946 minus 16984 |
| --- | ---: | ---: | ---: |
| Cached path | 35.90642548 | 36.09694672 | -0.19052124 |
| Native block path | 35.85815430 | 36.07027054 | -0.21211624 |

Both select 16984. Head-only FP32 therefore changes the cached path's original saved-token decision instead of restoring it. The existing saved answer depends on the original computed BF16 tie. Preserving that answer requires matching the relevant original computation; treating near-ties as equal or adopting a different precision policy does not establish the required contract.

For control 164610, the generic `alternate` field in some recorded summaries accidentally equals `expected` when `topk` orders a tie that way. Its signed margin is then uninformative. Use the recorded `top5`, `maximum_tie_ids` and actual token matches for that control. The two explicit alternatives on the failing fixtures are unaffected. This reporting limitation is preserved in the raw artifact rather than silently rewritten.

## Budget, incomplete probe, and artifacts

One RTX A5000 on `tron61`, four CPUs, 32 GiB host allocation; Python 3.12.12, Torch 2.13.0+cu126, Transformers 5.15.0. SLURM elapsed time was **3m50s**, including loading and the pause between stages. Peak allocated GPU memory was 21,045,345,280 bytes (19.60 GiB); maximum host RSS was 20,416,948 KiB. CUDA-synchronized top-level target-forward time was 20.8547 seconds. That timer excludes checkpoint loading, draft-only computation, isolated projection/head probes and the inter-stage pause; allocation elapsed time covers those costs. No per-probe timing claim is available.

The call-count plan contained an arithmetic error: baseline used **251** target forwards, and the four completed trace comparisons used **43**, reaching **294**. The optional whole-transformer FP32-linear intervention would need nine more calls, exceeding the 300-call limit. The hard pre-forward guard stopped it at exactly **300**, before call 301. SLURM status is `FAILED`, exit 1, from `Diagnostic forward budget exhausted`, not merely the intentional exit for a failed correctness baseline. No complete FP32 intervention result exists; do not interpret the six partial calls as a correction result. No budget extension or rerun occurred.

The baseline and all four trace comparisons were saved before this optional probe began and are complete in [result.json](result.json). Its SHA256 is `d468197e17b74c858c8df7a18eeeff7110310c4698291e740fba197bd97dccea`. Executed source hashes were checked against both remote and local files:

- [exact_session.py](../../../editor-experiment/replay_diagnostic/exact_session.py): `9d169721dc295b7fa8f9d7fefc3f63fc8f3d7b260ba189554e49fc2ca7bcbac3`.
- [exact_probes.py](../../../editor-experiment/replay_diagnostic/exact_probes.py), uploaded as `probes.py`: `63cb2a7acad233526e9f6bfe96485f22c951baa71f1354046eb9cb4fb3e31b75`.
- [Runner](../../../editor-experiment/replay_diagnostic/exact.sbatch), [usage](../../../editor-experiment/replay_diagnostic/EXACT_REPLAY.md), and [CPU tests](../../../editor-experiment/replay_diagnostic/test_exact.py). Two CPU tests passed for position/margin/tie reporting and the hard call/time caps; the GPU result is the numerical regression evidence.

Remote artifacts remain under `/fs/classhomes/fpajak/speculative-decoding-project/research/editor-feature-bank/exact-replay-20261117/`; allocation log is `exact-replay-8451880.log`. Raw response records remain on Nexus. There was no extra checkpoint copy and no live job remained after the allocation ended.

## Next bounded question

The source-data explanation is now tested: these four saved first blocks reproduce using the original greedy generator. The next correction hypothesis is to preserve original single-row linear arithmetic inside the native verification block, including the vocabulary head, while leaving prefill and weights unchanged. This directly tests the observed shape dependence and avoids assuming that higher precision preserves the saved BF16 answer. Attention and other batched operations could still cause disagreement, so success is not presumed.

Any such correction needs fresh independent caches, all four exact-token regressions, actual native execution and added-cost measurements before a bounded bank attempt. A sequential cached reference remains a diagnostic control, not an unmeasured speedup solution. Only a complete 128-repair/16-preservation bank with independent checks can unlock fitting. This session did not launch the next experiment.
