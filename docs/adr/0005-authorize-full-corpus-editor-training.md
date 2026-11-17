# Train both token editors on the full training-prompt corpus

Accepted November 17, 2026, following the user's explicit request: perform a real
training run on all prompts and estimate when results will be available.

This supersedes the small-stage and no-50k restrictions in ADRs 0003/0004 for
this run. Process all 50,000 authenticated training prompts; preserve the 1,024
validation prompts outside optimization. The original response journals remain
immutable. Create separately versioned invariant-native continuations and actual
DFlash proposals needed for training. Do not repeat the earlier determinism matrix.

The first full run uses up to four complete native blocks per prompt from a
64-token canonical continuation. Include a naturally correct training proposal
when one exists. Train on reachable examples whose preserved first proposal is
correct; explicitly account for every prompt, incomplete window and wrong-first
case. No silent claim that all prompts contributed if some have no eligible block.
Evaluation keeps wrong-first cases. This covers the full prompt inventory, not
every token of every old saved response.

The target sees the complete prompt, up to the observed 3,241 tokens. Raise its
context capacity to 4,096, with the rest of the pinned invariant contract intact.
Validate 128 training prompts, including the longest cases and four prior fixed
fixtures, against independent greedy tokens/full-logit hashes before the full
collection. Every corpus token/block is checked against native target argmax and
acceptance evidence. Corpus-wide independent greedy duplication is not required
after this scoped validation; no universal numerical guarantee is inferred.

Both editors retain the existing 40,750,336-parameter architecture. Window only
the editor's accepted context to its most recent 1,024 tokens; retain the absolute
target anchor and window offset in provenance. No target prompt truncation or
parameter increase. Start both models from the same seed-zero initialization.

Train each for three complete shuffled epochs. Use FP32, microbatch one,
effective batch 32, AdamW peak learning rate 0.001, weight decay 0.01, gradient
clipping at one, 2% linear warmup and cosine decay to 10% of peak. Check finite
loss/gradients, save resumable model/optimizer/RNG/progress checkpoints, and score
development after each epoch. Choose the highest development mean correct prefix
(CE breaks ties), then evaluate the selected checkpoint once on final test.

Split the old validation inventory into 512 development and 512 final-test
prompts, balanced by source. All eight previously inspected prompts belong to
development. Preserve exact normalized family separation. Report repair, damage,
preservation, exact suffix/token accuracy, original/edited/union coverage and
isolated editor latency. Full-generation speedup remains a separate measurement.

Use one existing RTX A5000 at a time, four CPUs and 16 GiB host memory for the
initial resource profile and pipeline; increase the host reservation only if
measured necessity requires it. Initial full-pipeline wall limit is 24 hours,
with resumable chunks/checkpoints. No paid service, model download, capacity
sweep or extra hyperparameter run. Estimate elapsed time from the calibration
and actual collection progress, separately identifying scheduler uncertainty.
