# Full-block editor training: completed comparison

The three training runs and their evaluations are complete. **None improves on
the earlier gated editor.** Early-position weighting is better than the matched
ordinary-training control, but does not beat the existing checkpoint. The more
aggressive preservation/repair objective never beats its starting calibration
score, so checkpoint selection retains the original generator. No new checkpoint
is recommended for native deployment or larger training from these results.

The GPU job completed in **24m07s** on one RTX A5000. Independent recounts on
Nexus CPU and the Mac agree, all selected checkpoints are backed up and verified,
and the user queue is empty. This is a completed negative comparison, not an
unfinished training attempt.

**What the editor reads.** The existing whole-draft editor already receives the
entire proposal. Its input is the latest 1,024 accepted context tokens, the
original 16-position block, and an output seed. Here the block consists of one
already accepted anchor plus 15 proposed tokens. The anchor and first proposed
token stay fixed; the editor generates the remaining 14 tokens one at a time.
Every generated suffix position can attend to the entire original block and to
the suffix tokens already emitted. It does not receive target or drafter hidden
states, correct future tokens, or a known error location at inference.

The generator remains a two-layer causal Transformer, width 256, four attention
heads, with its own tied token embedding/readout. It has **40,750,336 parameters**.
A separate **265-parameter linear COPY gate** chooses between the original token
at that position and the generator's candidate. The actual chosen token becomes
the input for the next autoregressive step. The gate is not the entire editor.

This experiment changes how that full-block-conditioned generator learns; it
does not add a previously absent block input or test a new attention architecture.

**The three approaches.** All start from the same earlier full-corpus generator
checkpoint and use identical training examples in identical order:

1. **Ordinary training (`uniform`):** ordinary cross-entropy gives every editable
   position equal weight. This controls for simply doing more training.
2. **Early-position emphasis (`early`):** weight suffix position j by exp(-j/7).
   The first editable position therefore receives more emphasis than later ones.
   This targets the fact that an early mismatch makes the later proposals unusable
   in a consecutive speculative prefix.
3. **Preserve and repair (`protect_repair`):** keep the early weights, multiply
   correct original-prefix positions by two, and the original first-error position
   by four. This explicitly teaches both restraint and repairing the first break.

The reference tokens determine those training weights only. The model never gets
the reference's first-error position when asked to edit a held-out block.

**Exactly how training ran.** Each arm uses 8,192 blocks from distinct
training-origin prompts/families: 4,096 UltraChat and 4,096 GSM8K. This includes
1,024 naturally perfect proposals, with the remaining 7,168 repairable. The 512
gate-calibration prompts/families are excluded from this fine-tuning selection.
These are continued fits of the existing full-corpus model, not fresh training
on all 50,000 prompts.

Each generator receives 768 optimizer updates: three passes over those 8,192
blocks, effective batch 32, microbatch one. During training, earlier correct
suffix tokens help predict later ones (teacher forcing). Weighted cross-entropy
measures prediction error; `backward()` computes gradients, and AdamW's
`step()` updates every generator parameter. Target and DFlash are not loaded or
changed. FP32, seed zero, peak learning rate 0.0001, 16-update warmup, cosine decay
to 0.00001, weight decay 0.01 and gradient clipping at one are fixed for all arms.

Every 128 updates, and at initialization, each model generates its own suffix on
the 512 calibration blocks. Selection uses correct-prefix length, breaking ties
with token accuracy. The unchanged starting checkpoint is eligible. After this
selection, the generator is frozen and a new COPY gate is trained on the same
1,536 fit blocks used by the earlier gate study; 512 calibration blocks select
that gate. Gate fitting uses the earlier reachable/decisive-prefix supervision
and weighted binary loss, AdamW at 0.001, batch 256, and fixed early stopping.

| Arm | Generator updates completed | Generator step selected | Gate updates completed / selected |
|---|---:|---:|---:|
| Ordinary | 768 | 640 | 800 / 780 |
| Early emphasis | 768 | 256 | 240 / 40 |
| Preserve and repair | 768 | **0, original generator** | 380 / 180 |

Training loss fell in all arms. Mean weighted loss over epochs one and three was
1.505→1.214, 1.345→1.060, and 1.370→1.001 respectively. These demonstrate
optimization of the training objectives; differently weighted losses should not
be compared directly across arms. They do not demonstrate held-out improvement.

**Held-out development results.** All checkpoint selections were fixed before
scoring the same 2,048 development blocks from 512 prompts: 1,024 blocks and 256
prompts per source. No development prompt enters gradient updates. This set has
been inspected in previous experiments, so these results are exploratory rather
than fresh confirmation. The old reserved test, inspected after the November 19
full-corpus run, was not evaluated again.

Correct prefix counts consecutive matching proposed tokens, from zero to 15,
excluding the anchor. Editable-token accuracy counts matches across all 14
editable positions, including positions after the first mismatch. Neither metric
is final-answer accuracy or measured output tokens committed by native DFlash.

| Candidate | Mean correct prefix ↑ | Editable-token accuracy ↑ | Blocks improved / damaged vs DFlash |
|---|---:|---:|---:|
| Unedited DFlash | 3.577637 | **32.4289%** | — |
| Original generator, no COPY gate | 3.466309 | 26.0568% | 335 / 360 |
| Earlier generator + earlier COPY gate | **3.699707** | 30.6571% | 269 / 187 |
| Ordinary, generator alone | 3.432129 | 25.7010% | 320 / 369 |
| Ordinary + newly fitted gate | 3.617676 | 28.8086% | 286 / 241 |
| Early emphasis, generator alone | 3.455078 | 25.8894% | 328 / 364 |
| Early emphasis + newly fitted gate | 3.672363 | 30.0502% | 281 / 223 |
| Preserve/repair selected original, no gate | 3.466309 | 26.0568% | 335 / 360 |
| Preserve/repair selected original + new gate | 3.699707 | 30.6571% | 269 / 187 |

The last row reproduces the incumbent's predictions; it is not a gain from new
generator training. The original gate replay also independently reproduces the
previously reported 3.699707 score and 30.6571% token accuracy.

| Candidate | UltraChat prefix | UltraChat token accuracy | GSM8K prefix | GSM8K token accuracy |
|---|---:|---:|---:|---:|
| DFlash | 2.474609 | 23.7514% | 4.680664 | 41.1063% |
| Earlier gated editor | **2.480469** | **21.2193%** | **4.918945** | **40.0949%** |
| Ordinary + gate | 2.390625 | 19.5382% | 4.844727 | 38.0790% |
| Early emphasis + gate | 2.439453 | 21.1844% | 4.905273 | 38.9160% |
| Preserve/repair selected original + gate | 2.480469 | 21.2193% | 4.918945 | 40.0949% |

Neither newly selected trained model improves on the incumbent on either source.
Both gated candidates have a positive overall prefix difference from DFlash,
driven by GSM8K, while all-position accuracy decreases. The predeclared criterion
required a prefix above 3.699707 **and** accuracy at least DFlash's 32.4289%.
No arm passes.

A 10,000-draw paired-prompt bootstrap, stratified by source, gives the following
95% intervals relative to the earlier gated editor:

| New gated arm | Mean prefix difference | 95% interval | Token-accuracy difference, percentage points | 95% interval, percentage points |
|---|---:|---:|---:|---:|
| Ordinary | −0.082031 | [−0.147949, −0.018555] | −1.8485 | [−2.3821, −1.3009] |
| Early emphasis | −0.027344 | [−0.083496, +0.031250] | −0.6069 | [−1.0672, −0.1604] |

The early-emphasis prefix difference remains uncertain; the point estimate is
lower, and there is no demonstrated improvement. These exploratory intervals
are not corrected for multiple comparisons and do not measure seed variability.

**Why this did not help enough.** The data show that optimization works but its
benefits do not transfer to the development distribution. Calibration prefix
improves from 9.1230 to 9.2461 for ordinary training and to 9.3262 for early
weighting, yet the selected models lose on development. Preservation/repair
weighting fails even this calibration comparison. The experiment does not
identify one proven cause.

There is a concrete distribution limitation: calibration is deliberately half
perfect proposals, fine-tuning is 12.5% perfect, and development has only 44
perfect blocks out of 2,048 (2.15%), plus 320 blocks with an incorrect first
proposal. Calibration prompts were also part of the historical base-model
training inventory, although excluded from the new generator and gate fits.
Thus calibration improvement is a weak proxy for broad unseen-prompt improvement.
Changing that sampling/selection policy is an untested follow-up, not an excuse
to relabel this comparison as successful.

The fixed-first-proposal design is another hard limit: those 320 wrong-first
blocks cannot gain a usable prefix regardless of how well the suffix is edited.
All 44 initially perfect proposals are counted in evaluation; the incumbent
damages nine, ordinary+gate damages thirteen, and early+gate damages eleven.
More emphasis on early tokens did not eliminate destructive edits.

**Cost and correctness checks.** On 16 fixed development blocks, five warm graph
replays each, the isolated gated-editor timing is 6.897 ms for ordinary, 6.892 ms
for early emphasis, 6.892 ms for selected-original preservation/repair, and
6.916 ms for the earlier editor. Input/output copies are included. Median graph
setup is 99.7, 99.8, 100.0 and 107.8 ms respectively. These small timing
differences do not establish a speed gain. All retain the same fourteen serial
generation steps. No new native DFlash throughput benchmark ran with these weights.

The causal-mask, finite-gradient, optimizer-ownership, cached/full inference,
checkpoint reload, gate-simulation and changed-input graph replay checks pass.
There are 18 passing checks in the pinned GPU environment and 22 targeted local
checks including the independent auditor and backup regression. The independent
auditor reconstructs predictions against the authenticated bank, verifies all
768 update records per arm and checkpoint selection, and computes the reported
statistics without importing PyTorch. Nexus and Mac audit results are identical.

**Decision.** Keep the earlier editor as the comparison baseline. These particular
fine-tuning objectives are not an improvement. This does not establish that a
whole-block editor cannot work, that training cannot improve, or that a larger
model is required. A next bounded study should first use a calibration sample
representative of natural blocks and separate from historical training, then
isolate a structural change such as making the first proposal editable or adding
an explicit proposal encoder/alignment mechanism. Those are proposals only;
no further experiment has been launched.

**Supporting files and reproducibility.**

- [Predeclared protocol](PROTOCOL.md), frozen with training source `dd31481`.
- [Independent recount and confidence intervals](evidence/independent-audit.json),
  [all development metrics](evidence/complete.json),
  [calibration histories](evidence/generator-calibration-history.json),
  [run/source/data pins](evidence/run.json).
- [Backup verification](evidence/backup-verification.json),
  [complete artifact hashes](evidence/export-manifest.json),
  [job states](evidence/final-job-status.txt), [checks](evidence/checks.json).
- Training: `editor-experiment/refinement_research/whole_draft_fit.py`;
  data selection: `whole_draft_data.py`; independent recount:
  `whole_draft_audit.py`; backup inventory: `whole_draft_export.py`.
- GPU job `8600245`: completed exit 0 in 24m07s on tron61, one A5000,
  four CPUs, peak host RSS 5,993,580 KiB. Source archive SHA256:
  `e11b3be145329793c4c1efe676c6056721187dda4a2bdfc3053c2b4456e15b5d`.
  All 59 recorded Python source hashes match the frozen commit.
- Private raw predictions, selected generator/gate pairs, source, logs and hashes:
  `~/.local/share/wayfinder-artifacts/whole-draft-8600245/` on the Mac.
  All 62 output/checkpoint files match the remote inventory (526,383,290 bytes).
  Raw token traces and checkpoint weights are not published in this repository.
- Remote results: `/fs/classhomes/fpajak/speculative-decoding-project/research/whole-draft-8600245/`.
  Original selected and resumable optimizer checkpoints remain in
  `/scratch0/fpajak-whole-draft-8600245/` on tron61. Scratch is not a durable backup.

The first CPU postprocessing job (`8600354`) passed its numerical audit, then
failed while staging redundant weight copies in Nexus home storage. A minimal
write probe confirmed `ENOSPC`. The six selected files were streamed directly
from node scratch in CPU job `8600808`, verified on the Mac, and only the three
new redundant home copies were removed. A small write then succeeded. The exporter
now inventories scratch by default rather than copying weights; regression and
CPU inventory job `8600874` pass. This storage recovery changed no trained weights
or evaluation data and repeated no training. Original workspace changes remain
intact. The investigation is tracked in
[Can full-block-conditioned editor fine-tuning improve both token accuracy and usable prefixes?](https://github.com/Spec-decoding-project/Speculative-decoding-project/issues/28).
