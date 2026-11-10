# Compact editor: overfit first, then test generalization

Proposed on 2026-11-10. No editor training or GPU feature collection was launched during this inspection.

## Decision

Use the existing Qwen3-8B responses. The data volume is sufficient for a small memorization test and a subsequent learning curve. Whether it is sufficient for useful generalization or a speedup remains an experimental question. Do not collect more target responses before measuring those questions.

The first deliverable is a small, repeatable experiment showing that the editor can repair real DFlash errors on its training examples while preserving the earlier correct tokens.

## What was actually verified on Nexus

Project: `/fs/classhomes/fpajak/speculative-decoding-project`.

A read-only scan ran in a SLURM CPU allocation from 15:10:22 to 15:10:55 UTC on 2026-11-10. The deployed `summarize_qwen_responses.py` matched the collection branch's script SHA256, `e2a09686780fe353dbe84481510aac0265a11f34336a12fc260449155d3cf03d`.

| Observation | Value |
|---|---:|
| Main collection: training responses | 50,000 |
| Main collection: validation responses | 1,024 |
| Main collection: generated tokens | 40,718,789 |
| Training generated tokens | 40,162,389 |
| Main collection: EOS completions | 48,683 |
| Main collection: length-capped responses | 2,341 |
| Main collection: input skips / missing work / partial tails | 0 / 0 / 0 |
| Complete main shards | 64 of 64 |
| Pilot responses | 384 |
| Unique prompt IDs across main and pilot | 51,024 |

The pilot overlaps the main collection; it adds no new prompt IDs. All record checksums and run/shard identities passed. This scan did not independently regenerate model outputs or verify each record against the original input Parquet files. No jobs appeared in the user's queue before the audit.

Responses preserve prompt IDs, source metadata, dataset split, prompt token IDs, response token IDs, and completion status. Generation used Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16, greedy decoding, thinking disabled, and a 2,048-token output cap.

Only `qwen3-8b` was present under `data/models/`. The inspected collection has no editor feature bank or matched DFlash model directory. `quota -s` returned no usable quota information; remaining user quota must be established before large downloads or caches.

## Match the editor to the data

The existing 3,024,384-parameter editor was configured for Qwen3-4B's hidden width of 2,560. The downloaded Qwen3-8B has hidden width 4,096. With the same two attention layers, rank 256, four attention heads, and block size 16, `CompactSecondPass` has **4,204,032 parameters**. This is the same design adapted to the target's interface, before any capacity experiment.

Use the existing Qwen3-8B target and a pinned revision of [`z-lab/Qwen3-8B-DFlash-b16`](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16). The official model card identifies Qwen3-8B as its target; its [configuration](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16/blob/main/config.json) specifies hidden width 4,096 and block size 16. Validate tokenizer, mask token, selected target layers, dtype, and checkpoint loading before collecting features.

Do not feed 8B features into a 4B editor checkpoint or use the 4B drafter with these responses. No new 4B response generation is needed for this proposed 8B experiment.

## Preparation: build a small feature bank

1. Select short, EOS-completed training responses from the pilot, preserving their exact saved token IDs. Use both math and chat prompts. Keep every context from one prompt in the same split. The initial test uses training examples only; it does not measure generalization.
2. Replay the target on the accepted prefix and capture only the committed-context features needed by the matched DFlash model. Check saved greedy labels against fresh target predictions on the selected examples. Investigate mismatches rather than silently changing the label definition.
3. Run the frozen DFlash first pass with the anchor and masked future slots. Save its final normalized block states, actual first proposed token and embedding, original proposal IDs, target label IDs, valid-position mask, anchor position, prompt ID, and model/schema fingerprints.
4. Sample enough candidates to obtain 128 distinct full blocks with a correct first proposal and at least one later error. These are reachable repairs: the editor cannot fix the first proposal, so a wrong first proposal would prevent any suffix repair from being accepted. Record the selection rate and excluded cases. This diagnostic selection must not become the evaluation distribution.
5. Add a separate small set of initially correct blocks to check preservation. Future target tokens are labels only; never put them into the editor inputs or DFlash masked positions.

The feature bank should support a teacher-free loss. The current `extract_record`, `materialize`, and full-training driver require a second-pass teacher even for their target-only arm; the full driver also hardcodes the historical 384/64-prompt bank sizes and uses a 4B-specific loader with missing pinned source assets. Implement a separate configurable 8B feature collector and overfit runner, reusing `CompactSecondPass`, rather than claiming those historical scripts can run unchanged.

## Experiment 1: can the small editor memorize corrections?

Freeze Qwen, DFlash, embeddings, and the vocabulary output head. Train only the compact editor from its identity initialization. Use direct cross-entropy against the saved target token IDs over all valid editable positions. Do not require a trained secondary teacher, KL loss, label smoothing, data augmentation, dropout, or weight decay for this diagnostic.

Starting optimizer: AdamW, learning rate `1e-3`, weight decay zero, FP32 editor parameters and loss. Use microbatch 1 initially, accumulating to an effective batch of up to 16. Reuse the frozen vocabulary head; calculate full-vocabulary loss in bounded chunks if needed. Recompute the folded output projection as editor parameters change; a stale detached projection would break training.

| Stage | Fixed training data | Proposed gate |
|---|---|---|
| A | 1 repairable block | All editable tokens correct; mean CE below 0.05 |
| B | 16 repairable blocks | All editable tokens correct across all 16 blocks; mean CE below 0.05 |
| C | 128 repairable blocks | At least 99% editable-token accuracy and at least 95% entirely correct edited blocks |
| Preservation | Separate initially correct training blocks, added to the fit | Report any newly introduced prefix errors; aim for none on this tiny fit |

These thresholds are proposed engineering gates, not established scientific cutoffs. Run nested data subsets from the same initial state for comparable curves. Evaluate every 25 updates, stop early after the gate holds for three evaluations, and cap each initial attempt at 2,000 updates. Profile the first 100 updates before setting a realistic remaining wall-time budget. Save the initialization, selected examples, optimizer state, curves, and decoded before/after examples.

Report training CE, editable-token accuracy, exact edited-block accuracy, first-error repair rate, earlier-prefix damage, original/edited accepted-prefix lengths, union coverage, gradient norms, and update count. Evaluate the edited branch directly: keeping the original as fallback can hide editor damage. A low loss alone does not establish useful repairs.

Finally reload the checkpoint, verify repeated predictions, and compare FP32 fitting with BF16 deployment. Confirm target verification and cache behavior on the selected blocks before interpreting their saved-label coverage as actual acceptance.

## Interpret failure before increasing complexity

If a stage fails, first check alignment and the training path: anchor/first-guess offsets, correct final states rather than old midpoint placeholders, future-label leakage, optimizer ownership, frozen versus trainable parameters, and gradient flow.

The editor's `up` matrix starts at zero. On the first backward pass, earlier coefficient layers can legitimately receive zero gradients while `up` learns; check those layers again after `up` has changed. This is not by itself a broken graph.

Retry the same examples with learning rates `3e-4` and `3e-3`, check the FP32 path, and inspect whether loss is still falling at the update cap. As a diagnostic control, optimize an independent hidden-state residual for each training slot through the frozen vocabulary head. That control is intentionally a memorizer and is not a deployable editor; it helps separate head/label/optimization failures from failures of the compact network.

If the controls fit but the compact network consistently cannot, test one capacity change at a time: rank 256 to 512, or two attention layers to four. Keep examples, optimizer search, and evaluation rules matched; report both equal-update results and convergence behavior. Measure added serving cost. If the small model memorizes but fails on unseen prompts, larger capacity is not yet the demonstrated remedy.

## Experiment 2: generalization, only after the fit test

Expand progressively to 384, 3,000, then 10,000 distinct training responses. Use the remaining 50,000-response pool if the learning curve supports further scaling. Start with a small number of diverse anchors per response; expand toward 32 only after measuring feature cost and repair coverage. Contexts from one response are correlated and are not equivalent to independent prompts.

Fix a prompt-disjoint development subset of the existing 1,024 validation prompts before tuning, and leave the remaining validation prompts untouched for the final comparison. Check duplicate prompt families across partitions. Evaluate unfiltered contexts, including wrong-first-guess cases, and report math/chat results separately. For length-capped responses, retain only fully observed training windows; never invent an EOS label. The initial overfit test avoids those cases entirely.

Compare frozen DFlash, DFlash plus the editor, and the strongest applicable score-based tree baseline. Evaluate both replacement and union semantics if both are candidates for deployment; include the corresponding verification cost. Introduce teacher-weighted or teacher-KL objectives only as later matched comparisons, after direct supervision works.

The decision metric for deployment is end-to-end wall-clock generation with the target output preserved, including the complete editor/readout/verification/cache cost. Better saved-context coverage does not establish a speedup.

## Resource plan and first deliverables

Use one available RTX A5000 allocation initially, with up to four CPUs and 32 GiB host memory. Feature extraction runs the frozen target and drafter under inference mode with batch one and short prefixes; measure actual peak memory before expanding context length. After feature capture, unload the target and drafter bodies and train from the cached states plus frozen output head. Cached first-token embeddings avoid retaining the whole input embedding matrix for this stage.

Do not materialize the entire planned feature set now. At 50,000 responses times 32 contexts, a single BF16 `[16,4096]` state array per context would consume about **195 GiB**, before metadata or other tensors. By contrast, 128 such state arrays use about 16 MiB. Start small, measure bytes/context and extraction time, and use bounded shards for later experiments. Full vocabulary logits should not become the default disk representation.

The next implementation should produce:

- A configurable feature collector for the pinned Qwen3-8B/DFlash pair, with prompt-level splits and provenance.
- A teacher-free overfit runner with configurable subset sizes and the checks above.
- A tiny immutable feature bank and per-stage checkpoints.
- A report of training curves, actual repaired and damaged examples, BF16 replay checks, resource use, and a recommendation to scale data, change capacity, or fix the pipeline.

Existing server responses remain the source data. The next proposed GPU work is the small feature-and-overfit experiment, not a full-corpus training run.
