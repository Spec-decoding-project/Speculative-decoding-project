# Authorize the sixteen-block repair fit with limited preservation evidence

Accepted November 17, 2026, after the user reviewed the completed one-block fit
and requested the next phase and its exact overfitting procedure.

Extend ADR 0003 to sixteen distinct repairable blocks per variant. Reuse audited
captures from job 8453776: four blocks spread across each of four training
prompts. Start both full-size models from the same initialization as before;
do not warm-start the one-block weights. Retain the same 32 development blocks
from eight excluded prompts, which have already been inspected and are not an
untouched final test.

The captured pool contains zero fully correct training blocks and one fully
correct validation block. Score that natural preservation control separately;
never move it into training or fabricate native controls. The earlier proposed
paired repair-plus-sixteen-preservation arm is deferred until genuine training
controls exist. This bounded run answers sixteen-block repair memorization and
provides only a single-block damage observation, not a general preservation
claim. Broader collection remains open and is not a prerequisite for this run.

Each update accumulates gradients over all sixteen examples, normalized over
224 editable tokens. Use unchanged FP32/AdamW 1e-3, no weight decay/dropout,
microbatch one, gradient clipping at one. Evaluate every 25 updates and require
all editable tokens exact plus mean CE below 0.05 at three consecutive checks.
Stop at 2,000 updates or 600 training-loop seconds per variant. Reload and audit
checkpoints, then evaluate excluded examples and isolated editor latency.

No new target inference, response regeneration, automatic 128-block fit, capacity
increase or large-data training is authorized by this milestone. Original
responses and previously executed code/results remain immutable.
