# Full-corpus run is active; results are pending

Started **November 17, 2026 at 2:33:36 p.m. Eastern**, Slurm job **8454498**,
on one RTX A5000 on tron61, four CPUs and 16 GiB host memory. The job has a
24-hour wall limit and durable collection chunks/training checkpoints.

The current phase is native training-example collection. Both full-corpus
optimizer runs and final-test evaluation follow automatically after bank
validation. This document records a launch, not completed training results.

## Scope

- Process all **50,000 training prompts** and the **1,024 excluded validation
  prompts**. Preserve the original response journals. Generate separately
  versioned canonical target continuations and genuine DFlash proposals.
- Keep up to four complete reachable training blocks per prompt from a 64-token
  continuation, including a naturally correct proposal where available. Every
  prompt is accounted for; prompts without a usable full block are reported.
- Train both existing **40,750,336-parameter** variants from identical random
  weights for **three epochs each**. Context mode reads context/seed; draft mode
  additionally reads the full original proposal. The editor uses the most recent
  1,024 context tokens; the target receives the full prompt.
- FP32, effective batch 32, microbatch one, AdamW peak LR 0.001, weight decay
  0.01, gradient clipping at one, 2% warmup/cosine decay to LR 0.0001.
- Use 512 development prompts for checkpoint selection and 512 reserved final-
  test prompts afterward. Previously inspected prompts stay in development.
  Report repairs, damage, preservation, correct-prefix coverage and isolated
  latency. This does not measure full-generation speedup.

The [fixed protocol](PROTOCOL.md) and [ADR 0005](../../adr/0005-authorize-full-corpus-editor-training.md)
record the exact procedure and changes from the small memorization experiments.

## Completed checks and timing evidence

Preflight job **8454304** authenticated all 51,024 source records and 65 source
file hashes, with zero exact normalized training/validation family overlaps.
Prompt median/mean/max lengths are 93/171.22/3,241 tokens. Original response
length caps do not exclude prompts because new canonical continuation windows
are generated independently.

Calibration job **8454371** completed exit 0 in 7 minutes 50 seconds. The native
and independent greedy paths matched **all 8,083 generated tokens and full-logit
hashes across 128 training prompts**, including long contexts. All four prior
canonical fixtures retained their outputs. All 128 supplied usable training
blocks: 505 repairs and three naturally correct proposals, 508 blocks total.
Thirty local checks passed across the collection, model, training and resume
tests. A restored mid-epoch optimizer run matched uninterrupted weights.

Calibration projected 7.35 hours of native collection and approximately 55 minutes
of optimizer updates per variant for three epochs at the maximum retained size.
Its raw compute estimate was 9.18 hours, before I/O and evaluation.

The first three full-run chunks then processed 192 prompts and retained 763
blocks, with 66.46 seconds of generation and 68.16 seconds including chunk audit/
writing. Full collection omits calibration's full-logit hash transfers and uses
larger request chunks while retaining the four-active-request limit and all
per-record token/acceptance checks. Short-prompt collection averaged 0.3462
seconds/prompt versus 0.4951 in calibration.

Weighting that observed short-prompt rate by the full inventory, and retaining
the conservative calibration rates for longer prompts, projects **5.85 hours of
collection**. Add approximately **1.83 hours of training** for both variants,
bank validation, checkpoint I/O and evaluation. The updated planning range is
**8–10 hours total**, approximately **10:30 p.m. November 17 to 12:30 a.m.
November 18 Eastern**. This is an estimate, not a completion guarantee. Queue
delay is absent at launch; interruptions or materially different long-run rates
can move the finish time.

## Live artifacts and continuation

All raw examples, weights and results remain on Nexus:

```
/fs/classhomes/fpajak/speculative-decoding-project/research/full-corpus-run-20261117/
  status.json
  collection/progress.json
  collection/chunk-*.json.gz
  bank.sqlite
  bank.json
  training/{context,draft}/progress.json
  training/{context,draft}/{latest,best}.pt
  training/{context,draft}/development-summary-epoch-*.json
  training/{context,draft}/complete.json
  result.json
  REPORT.md
```

The last two files are written automatically after both selected checkpoints
finish final-test evaluation. A failure records its phase/return code and retains
completed chunks/checkpoints. Inspect the current job before any resubmission.
The source bundle is `research/full-corpus-fit-20261117-run`, frozen at commit
`c3b15d6e588cf4b5af078ee780b38530ab8ec15a`; all 28 uploaded files were checksum
verified. Calibration used `b2acf77`; the validated collector files are byte-
identical in the full run. Previous experiments and both AR branches are intact.

[Preflight/calibration/launch evidence](evidence/) and the
[active full-corpus task](https://github.com/Spec-decoding-project/Speculative-decoding-project/issues/20)
hold provenance and progress. Keep that task open until the full run and its
results are audited. No capacity change or extra hyperparameter run is queued.
