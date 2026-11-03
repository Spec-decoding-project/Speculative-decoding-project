# Public second-pass teacher: availability and compatibility

## Decision

A public DFlash/DTA checkpoint pair exists and is structurally compatible with the inspected D2SD loader. It is a candidate for a stronger teacher test, not a verified reproduction of the paper. No weights were fully downloaded, no model was executed, and no GPU money was spent.

The earlier statement that no usable released teacher had been found needs this qualification: we have now found accessible checkpoint files, but their training provenance, numerical compatibility and usefulness remain unverified. Do not call them author-trained or paper-matched weights.

This keeps the objective fixed: establish useful second-pass corrections, then learn a cheaper replacement. Finding another checkpoint is groundwork, not a research contribution or evidence of improvement.

## What is actually available

[DAPANTAO/D2-SD](https://huggingface.co/DAPANTAO/D2-SD/tree/d48de101eb52c71a03e810cdbcf7a7b6a0b7a6f3) contains a first drafter and a DTA second drafter. The pinned revision is `d48de101eb52c71a03e810cdbcf7a7b6a0b7a6f3`.

| File directory | Registry SHA256 for model.safetensors |
|---|---|
| qwen3-8b-perfectblend | 19819cdda20b72c348e2835e3fdc6d134687500a0944fc2b82299999bb970682 |
| qwen3-8b-dd-fused-prob-decay-perfectblend_16 | f18f7b030c6ad30cf01e601b674fbbe4ebdf88ee04a957d74e2380f28f0973c7 |

Each is 2,097,259,104 bytes, with 1,048,626,432 BF16 parameters in 58 tensors. Bounded range requests retrieved only the first 65,536 bytes of each file. Tensor dimensions and offsets match the remote lengths; the full payload hashes have NOT been checked. Identical headers mean identical architecture, not identical weights.

The first drafter also appears in [felixcheng97/D2-SD](https://huggingface.co/felixcheng97/D2-SD), with the same registry hash. Neither inspected model card establishes authorship, data, training dose, or a paper result. The [author's checkpoint-release reply](https://github.com/catnanami/D2-SD/issues/1#issuecomment-5058764178) predates these uploads and says approval was pending. It does not authenticate them. No author was contacted.

## Source and tensor contract

The [pinned author benchmark](https://github.com/catnanami/D2-SD/blob/4c4b4917e56bf004a87eeed1e40af7f9f51af986/benchmark.py) loads both drafter roles using `DFlashDraftModel`. Inspection of its module definitions predicts all 58 parameter names and shapes in both checkpoint headers exactly. Therefore the DTA config's different class name is not, by itself, a weight-layout incompatibility. This is a static check; runtime/API and numerical equivalence remain untested.

The generator chooses up to four uncertain boundaries using first-draft probabilities, preserves the prefix at each boundary, masks the suffix, and runs the second model. This provides a concrete source of conditional corrections to study. Its supplied decode timer resets after the first iteration, so use a separate complete-work measurement for any eventual runtime claim. Source: [pinned generator](https://github.com/catnanami/D2-SD/blob/4c4b4917e56bf004a87eeed1e40af7f9f51af986/generation/d3_generator.py).

Both checkpoints have five layers, width 4096, block size 16, and require five target feature layers totaling width 20,480. The pinned Qwen3-8B config matches these dimensions and token IDs. Our saved Qwen3-4B features have width 12,800 and cannot be substituted. Our 4B vocabulary projection cannot be reused either.

The pair alone needs 4.19 GB of storage; an 8B BF16 target adds roughly 16 GB before runtime memory. Local free space was about 4.50 GB, insufficient for the complete stack and recovery reserve. Do not download it locally or load the accompanying training_state.pt files. Inference needs only weights/configs and inspected code. Remote staging and its costs must be included in any future attempt.

## The next scientific question

**Does this separately trained conditional drafter supply useful repairs beyond running the original drafter again with the same revealed prefix?**

Mechanism: a model trained to complete a partially revealed draft might use those tokens better than a model trained mainly on masked blocks. That would create a useful conditional-completion function for a smaller editor to learn. The strongest counterargument is that its apparent gain comes from extra proposals or stronger standalone weights, or that it fails on our workload just as the recent teachers did.

Use the same target histories and prefix boundaries, selected from first-draft scores before checking target answers. Keep the horizon at 16 initially; do not mix the question with longer blocks. Compare both weight sets under ordinary masked input and revealed-prefix input. Also retain original first-pass proposals. Match proposal count and verification budget when comparing conditional reruns; report single-chain repairs/damage separately from best-of-proposal coverage. A selected wrong prefix limits both conditional models and must remain in the results.

This comparison can separate observed input effects within each model from differences between pretrained weights. Unknown training histories prevent attributing any between-model difference to one particular training algorithm. Revealed-prefix benefit within DTA alone could also reflect its expected input distribution; do not claim a general causal mechanism from that ablation.

Before compression, require useful corrections beyond the matched original-model rerun on independently selected responses, and assess how much remains relative to a strong compatible 8B refinement alternative. Choosing such a comparator is still outstanding. An advantage over the first draft alone is insufficient. Any free oracle union is an opportunity bound for the saved outputs, not a serving algorithm or speedup. A later student must preserve useful corrections while reducing complete cost, including projection and verification.

This is not a launch-ready protocol: full weight authentication, runtime compatibility, independent prompt selection, compatible strong comparator, measured staging requirements and a bounded cost estimate remain. There is no justification for another student fit yet. Only $1 remains unallocated under the current ledger; this assessment reserves none of it and assumes no additional authorization.

## Reproducibility and status

- `registry-responses.json`, `candidate-details.json`, `upload_commits.json`: source availability and provenance limits.
- `candidate-source-manifest.json`, `contract-sources.json`: pinned source URLs and hashes.
- `header-audit.json`, `*-header.json`: bounded remote safetensors inspection.
- `static-contract.json`: all expected tensor names/shapes and target-config checks.
- `analysis/audit_public_d2sd_contract.py`: local standard-library audit; no remote code execution.

No new training, generation, GPU allocation, external messages, or paid execution. Existing findings and accounting holds are preserved. The compression objective remains active and unmet.
