# Sixteen real blocks, two autoregressive variants

This protocol is fixed before training, following the user's authorization to
run the next phase. ADR 0004 records the scope change. The experiment tests
whether the working one-block training path can memorize sixteen repairs and
whether its excluded-prompt behavior improves. It is not a speedup experiment.

## Data and controls

Re-audit the immutable `collection/` outputs of job 8453776 on CPU. Pin all five
file hashes, reconstruct the preceding one-block dataset, authenticate every
emitted target token against raw logit rows, and require matching full-vocabulary
hashes across all 768 greedy/native positions. Reconstruct genuine native blocks
from saved actual target inputs; verify prefix, positions, anchor, full labels,
native accepted/rejected counts and vocabulary limits. No target inference runs.

Remove only the preceding experiment's first-four-block retention cap. Keep all
other completeness/context checks. The four training prompts contain 45 usable
repairable blocks. For each prompt, sort repairs by absolute anchor and choose
indices `floor(k*(n-1)/3)` for `k=0,1,2,3`. This fixes sixteen distinct blocks
spread across the saved 64-token continuations. They overlap and come from four
prompts; report unique labeled positions as well as 224 token presentations.

Reuse the identical 32 development blocks from the eight excluded prompts for
comparison with the one-block fit. They include wrong-first proposals that the
editor cannot repair because it preserves the first proposal. Score the one
naturally correct validation block outside that set as a separate preservation
control. Zero correct training blocks exist in these captures. Do not fabricate
controls, move validation data into training, or present this as a paired
repair-plus-preservation training experiment. That broader question remains open.

## Exact training procedure

Both models have 40,750,336 parameters: width 256, two causal decoder layers,
four attention heads, a tied 151,936-token input embedding/output readout and
1,088 positions. They own all their parameters and consume token IDs only.
Context mode reads accepted context, anchor and first proposal. Draft mode also
reads the entire original DFlash proposal. Both generate fourteen revised tokens
one at a time. Target and DFlash are not loaded into this training process.

Reset each variant to seed zero and initial weight hash
`672080cdb5b25a32ae824198f3549e43979352bab0f621e75379dd9c46d03ab8`.
Each update zeroes gradients, visits all sixteen examples in fixed order and
accumulates microbatch-one cross-entropy gradients divided by 224 suffix tokens.
Teacher forcing supplies earlier correct suffix tokens while the causal mask
prevents later correct tokens from influencing earlier predictions. There is no
loss on the accepted context, draft-conditioning segment, anchor or first
proposal. `backward()` computes gradients; after all sixteen blocks, clip the
global gradient norm at one, then `optimizer.step()` updates editor weights.

Use AdamW at 0.001, FP32, zero weight decay, no dropout and no learning-rate
schedule. Every 25 updates, evaluate all sixteen training blocks using greedy
autoregressive generation with the editor's own earlier outputs. Require all
224 suffix tokens correct, all sixteen suffixes exact, and average teacher-forced
CE < 0.05 for three consecutive evaluations. Stop after passing, at 2,000 updates,
or at 600 training-loop seconds per variant (including periodic evaluation).
No retry, alternate seed, optimizer sweep or capacity increase is automatic.

Verify optimizer ownership, finite gradients, changed weights, causal future-
label exclusion on all training blocks, cached versus full-sequence decoding,
checkpoint weight hashes and identical metrics after checkpoint reload. Save
model, optimizer and RNG states. Only after fixing each checkpoint, evaluate
development/preservation and recreate the untrained baseline.

## Measurements and resource bounds

Report losses/curves, exact suffix/token accuracy, repairs/damage, original versus
edited versus union consecutive correct proposals (excluding anchor; maximum
15), source-level development summaries, and the single preservation observation.
Keep raw tokens in private artifacts. Use two warmups and ten CUDA-synchronized
latency samples per block including editor prefill and fourteen sequential
outputs. The target and drafter are unloaded; this measures isolated editor cost,
not full-system throughput or actual native edited acceptance.

One RTX A5000 on tron61, four CPUs, 32 GiB, 30 minutes, existing pinned Python/
Torch environment, no downloads. Verify no prior job is running, source/package
hashes and available storage. Write a unique job directory. Keep all previous
results and original response journals untouched. No automatic 128-block or 50k
training follows; report the results and the smallest justified next step first.
