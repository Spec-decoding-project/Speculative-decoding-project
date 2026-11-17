# Authorize one-block memorization before a complete bank

Update: the completed one-block phase is followed by the separately authorized
[sixteen-block phase in ADR 0004](0004-authorize-sixteen-block-memorization.md).
The original one-block protocol and evidence remain unchanged.

Accepted November 17, 2026, following the user's explicit request to train both
full-size autoregressive variants on the same single real repairable block and
evaluate excluded prompts before increasing the training set.

This supersedes ADR 0002's complete 128-repair/16-preservation-bank prerequisite
for this bounded milestone. It does not waive authentic native proposals, token
alignment, frozen target/drafter weights, prompt separation or data integrity.
Use the already validated invariant vLLM reference; do not expand the numerical
investigation. Original responses remain immutable.

Each 40,750,336-parameter variant starts from the same seed and sees one block.
Use FP32, AdamW 1e-3, no weight decay, microbatch one and gradient clipping at 1.
Evaluate every 25 updates, stopping after three consecutive exact 14-token suffix
evaluations with mean cross-entropy below 0.05, at 2,000 updates or 600 training
seconds. Require checkpoint reload and independent sequential-generation checks.

Eight prompts from the existing validation split, four per source, form a small
held-out probe. Select them before model training, exclude prompt/family overlap,
and use no evaluation result to choose weights or stop training. Collect at most
four complete native blocks per prompt from a 64-token reference window. Report
wrong-first-proposal cases and direct damage, not just eligible repairs.

This milestone tests real-data memorization and a small out-of-training probe.
Saved-context coverage and editor latency are not an end-to-end speedup claim.
No training on additional prompts, 50k regeneration or automatic sweep follows.
