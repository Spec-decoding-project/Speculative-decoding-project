# Paired COPY-or-generate pilot

The user authorized continuing after the completed full-corpus result on November
20. This bounded experiment addresses the prefix damage in that result. It does
not repeat native collection or train another full-corpus model.

The original whole-draft editor and a COPY variant start from the same selected
epoch-three checkpoint, with identical shared weights and fresh optimizers. The
COPY variant adds a zero-initialized linear head: 257 parameters, for 40,750,593
total. Both have width 256, two causal layers, four heads, tied editor-owned
151,936-token embeddings/readout, and a 1,088-position budget. They read context,
the entire DFlash proposal, and the two preserved seed tokens. Fourteen editable
tokens are emitted sequentially. Neither loads the target or DFlash model.

COPY emits the original token at the current suffix position and feeds that actual
token into the next decoder step. It never reads reference labels at inference.
Training supervises COPY when the proposal token equals the reference; otherwise
it supervises the reference vocabulary token. Action cross-entropy is the training
objective. Separately reported induced-token cross-entropy adds the vocabulary
and COPY probability masses when both emit the same correct token. Inference
selects the highest-scoring action, not the highest marginal token probability.

## Fixed data and optimization

- 64 authentic native training blocks: 32 repairable and 32 naturally perfect;
  16 per source/kind group, all distinct prompts and exact families. Sources are
  UltraChat and GSM8K. No synthetic controls or held-out training rows.
- Selection fingerprint:
  `f286a92360b74187f78f93ff798caf3f9a5a9be2157d145126287b34c0485809`.
- Immutable bank SHA256:
  `c7f92fae5609137510a198297a31b5178c70c413f263f89f04560b98aa6bbb3d`.
- Warm checkpoint SHA256:
  `c79b8c59c1e2cf73a7adf7aa9738b1b273779d330fe32f8d2c4a15fd56b7e8d2`.
- Shared initial weights SHA256:
  `8babb2bfe3f4c11ef283312e22ccd2e20878438c96957302ff2c462748bbd382`.
- FP32, AdamW, LR 0.0001, weight decay 0.01, clip norm 1, seed 0,
  effective batch 64 through microbatch one. Fixed row order, identical arms.
- At most 2,000 updates per arm; evaluate training every 25. Stop after three
  consecutive evaluations with all 896 emitted suffix tokens correct and mean
  induced-token CE below 0.05. No development-based checkpoint selection.

The unchanged baseline package is vendored from the previous full-run sources;
the sole refactor exposes normalized decoder features before the tied readout.
Before any update, all 2,048 saved development proposals must replay exactly from
the original selected checkpoint in the pinned GPU environment. Loss agreement
tolerance is 1e-5. Tests cover COPY alignment, actual-token feedback, causal masks,
ownership, always-COPY failure, cached/full execution and checkpoint resume.

## Evaluation and limits

Reload each final checkpoint and require identical training losses and emitted
tokens. Score the 2,048 development blocks from 512 excluded prompts, with source,
natural preservation, and wrong-first breakdowns. Independently recount prefix
metrics and validate action-to-token traces. Report prompt-cluster bootstrap
95% intervals (10,000 draws, seed 0) for each arm versus DFlash and COPY versus
the matched control. These are exploratory development estimates, not untouched
test results or independent training-seed uncertainty.

Time 64 evenly spaced development blocks: two warmups and ten synchronized FP32
cached calls per block, median of block medians. Require cached/full token equality
and identical outputs across timing repetitions. No end-to-end speedup is inferred.
The old final test is not loaded. New confirmation prompts and complete-round
cost/fidelity measurement are required before any later generalization/speed claim.

## Runtime and preservation

One RTX A5000 on tron61, four CPUs, 5 GiB requested host memory (within the issue's
16 GiB cap), and two hours total including checks/evaluation. Each arm gets at most
3,150 optimization seconds; absolute cutoffs reserve time for both evaluations.
The persistent job start time prevents retries silently extending that budget.
Checkpoint and stop on a limit; no automatic sweep, capacity change, or extra run.

The completed draft best/latest checkpoints have independently hash-verified Mac
backups. The GPU preflight copies both to node scratch and verifies both before
atomically replacing their home paths with links. Original bytes remain unchanged.
This frees home quota for the new, regular-file pilot checkpoints. A physical,
fsynced three-checkpoint write-and-hash probe tests two final checkpoints plus one
atomic-save temporary. Original source snapshots and journals remain immutable.

The Slurm job uses a frozen source bundle, preserves package records and all
predictions, writes checkpoints every 25 updates, records its exit code, and runs
independently of SSH. Test-only pytest dependencies are bundled separately; the
pinned training environment is not modified. No prompt text/tokens are published.

## Decision

Memorization establishes mechanics only. A transfer candidate needs positive
development net prefix gain relative to DFlash, retained repairs and reduced
preservation damage versus the matched baseline. Passing fit with continued damage
calls for changing the objective/action policy, not automatically increasing model
size. Failed fit first requires checking labels, masking, gradients and optimization.
No further large fit or serving integration is automatically authorized by this pilot.
