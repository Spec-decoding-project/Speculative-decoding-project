# Autoregressive token editor: proposed replacement experiment

Prepared 2026-11-15 after the user's clarification. The two requested token-only variants are now implemented as experimental local models/trainers. They have only synthetic CPU validation; this is not a real-data training result or authorization to treat the previous replay gate as passed.

## Confirmed direction and open choices

The user wants autoregressive revision from tokens rather than the historical parallel hidden-state residual editor, and wants to investigate training on the existing 50,000 Qwen3-8B responses. Target and DFlash remain the frozen reference pair. The original workspace and historical evidence must be preserved.

The user selected both input contracts, to be implemented on separate branches. The draft-conditioned variant uses accepted context plus the entire DFlash token proposal, then sequentially generate a corrected suffix. The alternative already in `dflash/secondary.py` sees context plus anchor/first proposal and discards the original suffix; it is a suffix redrafter, not a whole-draft editor. Both are autoregressive and token-only. Every neural decoder still has its own internal hidden states.

For the first matched comparison, propose preserving the target anchor and the first DFlash proposal, as in the historical experiment. Revising the first proposal is a separate follow-up choice because it changes both reachable repairs and the verifier layout. Do not silently fix a wrong first proposal with a saved target label.

Initial implementation uses a configurable small decoder trained from scratch for a matched comparison; pretrained initialization remains a later experiment. Alternatives include a small decoder trained from scratch versus fine-tuning a compatible pretrained causal model. Do not inherit the old editor's 3M parameter claim. With the existing padded vocabulary of 151,936 and width 256, one token embedding table alone has 38,895,616 parameters; tying input/output embeddings shares this table rather than removing it. Factorization, copying or frozen shared embeddings are other choices with their own costs and must be reported explicitly. The existing pretrained secondary example is Qwen3-0.6B, not a trained compact editor.

## What the old experiments actually tested

`CompactSecondPass` has 3,024,384 trainable parameters at the 4B target's width 2,560, or 4,204,032 at the 8B width 4,096. It takes final normalized draft states `[B,16,H]` and the actual first-proposal embedding `[B,H]`. Two width-256 attention layers produce rank-256 residual coefficients for positions 2 through 15. The frozen target vocabulary head converts the revised states to token scores. All suffix revisions are produced in parallel. The shared base-model parameters are not included in the editor count.

The old controlled 4B training used 384 prompts x 32 contexts = 12,288 blocks, three epochs, 1,152 optimizer steps per arm, effective batch 32 (microbatches of four), AdamW at learning rate 1e-3 with weight decay 0.01, FP32 trainable parameters and BF16 autocast. Loss variants were target-token cross-entropy, extra weighting where a trained second-pass teacher was correct, and eligible teacher KL plus cross-entropy. Only the editor was updated. The teacher and base models were frozen for these fits. The teacher-free objective still had a teacher-dependent implementation path. Evaluation used 2,048 saved contexts from 64 previously exposed development prompts and one training seed.

The later retained-prefix objective changed the fixed-context union coverage gain from 3.0217% to 3.1700%: +0.1483 percentage points, paired prompt bootstrap interval [-0.5711, +0.8585]. Edited-prefix damage increased from 541 to 570 blocks. This did not establish an improvement or a generation speedup. The 8B collector has had zero editor updates.

References:
- `editor-experiment/compact_second_pass_prototype.py`
- `editor-experiment/compact_second_pass_full_training.py`
- `results/second-pass-profiling-2026-10-01/compact-training-completion/attempt2/RESULTS.md`
- `results/second-pass-profiling-2026-10-02/compact-retained-prefix/attempt1/RESULTS.md`
- `docs/research/feature-bank-2026-11-10/REPORT.md`

## Proposed example and training contract

A real training record needs a committed prefix P, an original DFlash candidate D, a corrected target continuation Y, prompt/split identity, anchor offset, and pinned source/model identities. The 50k saved target responses supply P and Y at sampled anchors; they do not by themselves supply D. Run frozen DFlash on those prefixes to produce proposals without exposing Y. Token-only collection avoids caching draft hidden features for the editor, but must retain correctness, provenance, and target-replay checks.

For the proposed whole-draft editor, present P and D as conditioning tokens in distinct segments, then a correction-output boundary and the preserved seed. At training time, feed earlier correct suffix tokens to predict the next one (teacher forcing); use causal attention, proper next-token shifting, and loss only on the corrected suffix. Segment/boundary encoding must be explicit and identical at inference; its vocabulary and parameter cost belong in the final configuration.

The full original D may be visible because it exists at inference. Later correct Y tokens must never be visible when predicting an earlier Y token. Teacher forcing is not the same as using a separate neural teacher model. An autoregressive decoder can compute all teacher-forced training positions in parallel using a causal mask; deployment generates sequentially from its own outputs.

For the preserved-first-proposal variant, use reachable examples for the initial memorization check. Measure the wrong-first-proposal frequency in the natural evaluation distribution instead of filtering those examples away to inflate results.

Simplified training loop:

```python
optimizer.zero_grad(set_to_none=True)
logits = editor(training_input_ids, attention_mask=attention_mask)
loss = suffix_cross_entropy(logits, shifted_correct_tokens, suffix_loss_mask)
loss.backward()   # Compute gradients; weights are still unchanged.
optimizer.step()  # Update the editor weights using those gradients.
```

These names illustrate the contract; this is not a runnable trainer. A production runner also needs padding-safe loss normalization, configuration/seed logging, finite-gradient checks, optimizer ownership audits, validation, and resumable checkpoints.

## Recommended experiment order

1. Settle the token input contract and initialization. Implement model, dataset construction, sequential decoding and a readable trainer. Test shape/shift/mask correctness with synthetic CPU fixtures, including perturbing future labels.
2. Reconcile the existing target replay mismatch before treating saved labels as faithful serving targets. Token-only input does not remove this question. Revisit the old hidden-state-bank requirement explicitly for this new architecture; do not claim that an unused feature bank is now required, or silently interpret the failed gate as success.
3. Run a tiny intentional memorization check first: 1, then 16, then 128 real blocks, with preservation controls. This is a training-pipeline test before committing substantial compute. Check both teacher-forced loss and free-running exact suffix generation, then reload and repeat. Proposed prior-stage caps (2,000 updates, evaluations every 25) are starting budgets to review for the chosen architecture, not already validated autoregressive thresholds.
4. Once the small fit works, increase distinct training responses toward 50k through bounded learning-curve points, for example 384, 3k, 10k and 50k. Start with few anchors per response. Measure both unique examples and token/update exposure; keep a fixed training recipe or report changes. Use a fixed prompt-disjoint development split and reserve untouched test prompts from the existing validation pool before tuning.
5. Evaluate actual sequential proposals on the natural held-out distribution. Report first-proposal failures, direct replacement damage, exact suffix matches, accepted-prefix changes, and union coverage separately. Include ordinary DFlash, the existing frozen autoregressive redrafter where feasible, and a matched editor trained without draft conditioning to test whether reading D helps.
6. Measure complete generation and round costs for replacement and union. Include the editor's context/draft prefill, sequential suffix steps, vocabulary projection, cache maintenance and target verification. Confirm target-only output agreement under the documented greedy execution contract. A model with fewer trainable parameters is not automatically faster.

If memorization fails, first verify labels, causal shifts/masks, gradients, optimizer membership, learning rate and train/evaluation precision. A positive memorizing control plus persistent compact-model failure can motivate a capacity comparison. Failure to overfit by itself does not establish insufficient capacity. If train performance is strong and held-out performance weak, study generalization and data coverage rather than diagnosing a broken training loop.

## What can be measured before training

An untrained model can establish interfaces, memory use and runtime cost on specified execution shapes. An oracle using known target labels can estimate repair opportunity on fixed contexts. A pretrained autoregressive model can supply an inference baseline without new training. None demonstrates that a new untrained editor learns useful corrections or repays its overhead.

For a matched fixed-workload approximation, speed ratio is `(new useful tokens / old useful tokens) / (new round time / old round time)`. A 5% token gain with 10% more time is `1.05 / 1.10`, approximately 0.955: slower. Full-generation measurement must follow because trajectories, cache behavior, EOS and context lengths change.

## Current status

Only tiny randomly initialized synthetic test models have been trained locally. No pretrained checkpoints were downloaded and no saved-response editor training has run. The current live replay diagnostic remains unresolved. Historical results and the original workspace were not modified. This proposal records the user's new direction without replacing the existing tracker plan while the old hidden-state-specific tracker gates are still pending revision for the new token-only architecture.
