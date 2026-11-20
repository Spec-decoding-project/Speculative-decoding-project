# COPY-or-generate pilot: memorization passes, transfer fails

Both editors memorized all 64 real training blocks, including all 32 repairs and
32 naturally correct proposals. Reloaded checkpoints reproduce that result.
**The COPY editor still makes excluded development proposals worse overall.** It
protects correct tokens more often, but suppresses most useful repairs. This is
not a successful performance correction and does not justify a larger fit in the
same form without diagnosing the lost repairs.

The completed Slurm job **8488680** ran November 20 from **10:09:26 to 10:17:40
Eastern**, exit 0. Total elapsed time from the original pilot start was 10m32s,
within the two-hour cap. Nothing remains running or queued. No final-test rows
were loaded and no target/drafter training or new native collection occurred.

## What changed and how it trained

The matched control is the existing whole-draft autoregressive editor. The other
arm adds a 257-parameter COPY head, bringing its size to **40,750,593 parameters**.
It still reads tokens only: accepted context, the full DFlash proposal, and the
two preserved seed tokens. It emits fourteen suffix tokens one at a time. COPY
emits the original proposal token at that position; the next step receives that
actual token. Reference answers are used for supervision, never for inference.

Both arms began with identical shared weights from the completed full-corpus
draft checkpoint, with fresh optimizers. The 64 selected training blocks came
from distinct prompts and exact families, with 16 examples per source/kind group.
UltraChat and GSM8K each supplied 16 repairs and 16 natural preservation controls.
No development prompts entered optimization.

Each update accumulated gradients over those 64 examples, one example at a time:
FP32, AdamW LR 0.0001, weight decay 0.01, gradient clipping at 1. The baseline
predicts vocabulary tokens. COPY training instead rewards COPY whenever the
original token matches the reference, and otherwise rewards the reference token.
Earlier correct suffix tokens are inputs during training; evaluation uses the
editor's own generated tokens.

The stopping rule was fixed before launch: every 25 updates, generate all training
suffixes; require all 896 tokens correct and mean induced-token cross-entropy
below 0.05 at three consecutive evaluations. The maximum was 2,000 updates per
arm, subject to the shared time cap. Development results did not select checkpoints.

| Training result | Matched control | COPY editor |
|---|---:|---:|
| Updates at stopping | 100 | 250 |
| Three passing evaluations | 50 / 75 / 100 | 200 / 225 / 250 |
| Exact editable suffixes | 64 / 64 | 64 / 64 |
| Repair blocks corrected completely | 32 / 32 | 32 / 32 |
| Correct proposals damaged | 0 / 32 | 0 / 32 |
| Induced-token cross-entropy | 0.004466 | 0.004435 |
| Action cross-entropy | 0.004466 | 0.004462 |

Action loss and token loss differ for COPY because a vocabulary action and COPY
can emit the same correct token; the induced-token loss combines those probabilities.
The full [predeclared protocol](PROTOCOL.md) and [run configuration](evidence/run.json)
record the numerical, input and selection fingerprints.

## Excluded development result

This comparison uses **2,048 blocks from 512 development prompts**, not the
previously inspected final test. Correct prefix counts consecutive matching
proposals after the target anchor, with a maximum of 15. The original full-trained
warm checkpoint scored 3.4663 on this same development population before the pilot.

| Measurement | Original DFlash | Matched control | COPY editor |
|---|---:|---:|---:|
| Mean correct prefix | 3.5776 | 3.4736 | 3.4683 |
| Change from DFlash | — | −2.91% | −3.06% |
| Blocks improved | — | 267 | 38 |
| Blocks damaged | — | 273 | 94 |
| Perfect proposals damaged | — | 17 / 44 | 4 / 44 |
| Prefix positions gained | — | 819 | 67 |
| Prefix positions lost | — | 1,032 | 291 |
| Net prefix positions | — | −213 | −224 |
| Isolated editor latency | — | 19.57 ms | 19.85 ms |
| Offline best of original/edited prefix | 3.5776 | 3.9775 | 3.6104 |

Both sources worsen. COPY changes UltraChat from 2.4746 to 2.3301, and GSM8K from
4.6807 to 4.6064. The 320 wrong-first blocks stay at zero because the first proposal
is intentionally fixed. Among 1,684 repairable development blocks, COPY improves
38 prefixes but reproduces no complete fourteen-token repair suffix; the matched
control reproduces nine. COPY's higher total exact-suffix count mostly reflects
preserving already-perfect proposals.

Paired bootstrap intervals resample whole prompts, keeping each prompt's four
blocks together (10,000 draws, seed 0). The mean COPY change relative to DFlash is
**−0.1094 tokens**, with a descriptive 95% interval **[−0.1514, −0.0723]**.
The control interval is [−0.1948, −0.0127]. COPY versus control is −0.0054,
interval [−0.0962, 0.0850]; there is no demonstrated improvement between the arms.
These describe one exploratory development run, not fresh confirmation prompts or
variation across training seeds.

Latency uses 64 fixed development blocks, two warmups and ten synchronized cached
calls per block, with the target and drafter unloaded. No end-to-end generation
speedup was measured. The offline best-of-two prefix assumes an oracle choosing
the better original/edited branch and excludes additional verification cost.

## What the action traces establish

COPY chooses its COPY action at **23,097 / 28,672 editable positions (80.56%)**,
and copies all fourteen positions in 813 blocks. Of the control's 267 repairs:

- 23 remain repairs under COPY.
- **231 are lost because COPY repeats the original first wrong token.**
- Five are lost through an earlier damaging change.
- Eight are lost through an incorrect generated replacement at the first error.

COPY contributes 15 repairs the control did not make, giving 38 total. Thus 231
of the 244 lost control repairs (94.7%) have a directly observable COPY decision
at the original error. This explains the immediate behavior; it does **not** tell
us whether the COPY model's vocabulary head still had a useful alternative there.
Joint fine-tuning may have changed that generator too.

The trainability and data-integrity checks pass. The current blocker is transfer:
the editor learns the small bank but makes the wrong copy-versus-repair choices
on excluded prompts. More parameters are not indicated by failure to memorize;
there was no such failure. A larger COPY training bank could behave differently,
but this pilot does not establish that it would solve the problem.

The next smallest useful diagnostic is **frozen-model inference**, asking whether
COPY overrides good vocabulary predictions or whether joint fine-tuning damaged
those predictions. Inspect the vocabulary prediction and COPY score on the same
prefix at the 231 skipped repairs, and compare a COPY-disabled ablation with the
existing control. No optimizer updates are needed. That result distinguishes a
decision-policy problem from a generator-preservation problem before another fit.
This follow-up is proposed, not launched; serving remains blocked.

## Checks, recovery and artifacts

- All **36 local** editor/data/audit tests passed. All **23 editor tests** passed
  in the pinned GPU environment and in a self-contained local source bundle.
- Before training, the original checkpoint reproduced every one of its 2,048
  saved development proposals and the exact recorded loss (delta 0.0).
- COPY alignment, actual-token feedback, future-label masking, optimizer ownership,
  toy checkpoint resume, actual GPU reload and cached/full equality passed.
  The latter checked 64 fixed development blocks per arm; all timing repeats matched.
- An independent implementation recounted saved training/development predictions
  and checked action-to-token emission. A fresh Mac recount matched both GPU audit
  reports exactly. Update journals contain precisely steps 1–100 and 1–250.
- Both new checkpoints are backed up on the Mac and match their GPU SHA256 hashes.
  The original bank and warm checkpoint hashes remained unchanged.

The first preflight job, 8488678, stopped after 25 seconds before real-data training
or storage relocation. Two existing tests exposed missing helper files and a
Git-checkout assumption in the generic CLI. These packaging problems were reproduced
and corrected; job 8488680 used the remaining original budget. Model/data/optimizer
settings did not change. The frozen training source is **e2836e3f232f6736657b2974ebaa394cf5e96b82**
on `work/ar-copy-editor`; the first attempt at 84f48c3 remains preserved.

The run used one RTX A5000, four CPUs and 5 GiB requested host memory; Slurm peak
RSS was 4,030,840 KiB (3.84 GiB). Both completed draft checkpoints were copied and
hash-verified on tron61 scratch before their home paths became symlinks. Independent
Mac originals remain intact. A physical, fsynced 1,467,117,036-byte simultaneous
three-checkpoint probe passed before training.

Public summaries and fingerprints are in [evidence/complete.json](evidence/complete.json),
[the action-trace analysis](evidence/failure-analysis.json),
[baseline replay](evidence/baseline-replay.json), and
[backup verification](evidence/backup-verification.json). Raw tokens and checkpoints
remain private in the remote `research/copy-action-run-20261120/` directory and
the Mac backup `~/.local/share/wayfinder-artifacts/copy-action-run-20261120/`.
The analysis is reproducible with `copy_action_fit.analyze`; it does no inference.
