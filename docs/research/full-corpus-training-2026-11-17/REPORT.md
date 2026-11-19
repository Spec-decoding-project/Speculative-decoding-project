# Full-corpus training: completed result

Both token-only autoregressive editors completed three epochs, checkpoint reload,
reserved-test evaluation and an independent audit. **Neither improves DFlash when
its proposal is unconditionally replaced.** Reading the whole proposal helps
substantially, but damaging correct prefixes still outweighs the useful repairs.
No full-generation speedup has been demonstrated.

Recovery job 8474153 completed November 19 at **07:59:33 Eastern**, exit 0.
Independent audit 8474170 completed at **08:00:46 Eastern**, exit 0. No job remains
running or queued as of the 09:50 Eastern follow-up. See the
[storage recovery record](STATUS_2026_11_19.md) for the original interrupted run.

## What was trained and tested

Each editor has **40,750,336 parameters**, width 256, two causal decoder layers,
four attention heads and editor-owned tied token embeddings/readout. It receives
tokens, not target/drafter hidden states. Both preserve the anchor and first
proposal, then generate fourteen editable tokens autoregressively.

- Context mode reads accepted context and the two seed tokens.
- Whole-draft mode also reads the entire original DFlash proposal.

Both started from identical random initialization and trained only their own
weights: FP32, microbatch one/effective batch 32, AdamW, three shuffled epochs,
18,678 updates and 8,366,610 supervised suffix-token exposures per model.
This was not initialization from a pretrained small language model.
The fixed [protocol](PROTOCOL.md) gives the exact optimizer and numerical contract.

All 50,000 training prompts and 1,024 validation prompts were processed. The
bank contains **199,205 training blocks from 49,953 prompts**; 47 prompts had no
eligible complete block. Development has 2,048 blocks from 512 prompts. Reserved
test has 2,040 blocks from all 512 test prompts. This covers the full prompt
inventory, with up to four blocks per prompt, not every historical response token.
Original response journals remain unchanged. The training bank has 197,032
repairable and 2,173 naturally correct preservation blocks.

## Reserved-test comparison

Correct prefix means consecutive matching proposals after the target anchor,
with a maximum of 15. Both modes selected epoch three using development data.

| Measurement | Original DFlash | Context/seed editor | Whole-draft editor |
|---|---:|---:|---:|
| Mean correct prefix | 3.5422 | 2.0176 | 3.3446 |
| Relative change | — | −43.04% | −5.58% |
| Improved blocks | — | 275 | 281 |
| Damaged blocks | — | 955 | 375 |
| Unchanged prefix length | — | 810 | 1,384 |
| Initially perfect proposals damaged | — | 34 / 34 | 21 / 34 |
| Editable token accuracy | 32.68% | 12.76% | 25.60% |
| Exact generated editable suffixes | — | 4 / 2,040 | 26 / 2,040 |
| Isolated FP32 editor latency | — | 19.10 ms | 19.30 ms |
| Offline best of original/edited prefixes | 3.5422 | 3.9843 | 4.0206 |

Both sources worsen under replacement. Whole-draft mode changes UltraChat from
2.5502 to 2.3022 and GSM8K from 4.5264 to 4.3789. Of 512 test prompts, its mean
prefix improves on 160, worsens on 217 and stays unchanged on 135. The 324
wrong-first blocks remain at zero because that first proposal is preserved.

A paired prompt-cluster bootstrap, stratified by source, gives a descriptive
95% interval of **−0.309 to −0.086** tokens for the whole-draft replacement change
(point estimate −0.198). Context mode is −1.674 to −1.373 (point −1.525).
These are 10,000 resamples of this saved-context benchmark, not independent
training seeds or a speedup test. Blocks from one prompt are resampled together.

The best-of-two row assumes retaining the original branch and choosing whichever
has the longer valid prefix. It is useful evidence of potential complementary
repairs, but includes neither additional verification cost nor a working runtime
selection mechanism. The reported 19 ms is isolated editor latency with the
target and drafter unloaded. No end-to-end tokens-per-second conclusion follows.

## What the failure analysis established

The whole-draft editor gains **976 correct-prefix token positions** in its 281
improvements but loses **1,379** in its 375 regressions: a net loss of 403.
Of those losses, only 152 come from the 34 initially perfect proposals. Even
perfectly protecting that group would leave a net loss of 251. Protection must
also cover the valid prefixes of partially wrong proposals.

In 278 of the 375 damaged test blocks, the first change occurs within the first
three editable positions. That is 74.1% of the regressions. The development set
shows the same direction: 1,114 positions gained versus 1,342 lost. These counts
come from the saved tokens and do not assume a cause for the learned behavior.

Both models did learn. Development results across epochs were:

| Mode | Epoch 1 prefix / CE | Epoch 2 prefix / CE | Epoch 3 prefix / CE |
|---|---:|---:|---:|
| Context/seed | 1.7251 / 3.4586 | 1.9268 / 3.0795 | 2.0752 / 3.0171 |
| Whole draft | 3.2100 / 3.1115 | 3.3232 / 2.7888 | 3.4663 / 2.7712 |

The development DFlash baseline is 3.5776. The whole-draft final-100-update mean
training CE was 2.3160, compared with 10.2785 in the first 100 updates. These are
different shuffled batches, not a fixed training-set memorization evaluation.
This run did not apply the earlier one-/sixteen-block exact-overfit gate to the
full training set, so it does not establish whole-corpus memorization.

The existing objective predicts every suffix token using ordinary token
cross-entropy. It has no explicit COPY action and no special penalty for breaking
an early correct prefix. This makes a copy-or-edit objective a concrete hypothesis
to test. The results do not prove it will work. They also do not prove that simply
adding parameters or epochs is necessary: approximately 95.45% of this model's
parameters already reside in the 151,936-by-256 token embedding/readout table.

## Next smallest useful milestone

Focus on the whole-draft variant. Compare its unchanged objective with an explicit
**COPY-or-generate action**, preserving token-only inputs and autoregressive output.
COPY would emit the token at the current position of the original proposal; the
next step would condition on the token actually emitted. It must never consult
the reference answer during inference.

First establish that the revised path can learn both repair and preservation on
a fixed, small mixture of genuine training blocks, with the unchanged model as
a matched control. Use the existing bank and checkpoints. Check teacher-forcing
alignment, the COPY-to-token conversion, causal masking, checkpoint reload and
cached/full equivalence before interpreting learning. Keep the same model width
and depth; one explicit action does not require a larger language model.

Then assess repair gains and prefix damage on the existing development split,
and measure the added inference cost. Copying everything is the baseline, not
success. Passing training memorization alone is also insufficient. Advancement
requires useful excluded-prompt repairs after accounting for damage, followed by
a measured verification/latency budget before any speedup claim.

This is a proposed next experiment, **not an already trained correction**. No new
optimizer run or serving integration was launched during this completion audit.
The current final test has now been inspected; it may be retained as a transparently
reused benchmark, but a later confirmatory claim needs fresh held-out prompts.
No new data or credentials are needed from the user to prepare this bounded
experiment. Existing data and GPU access suffice.

The bounded comparison, checks and resource cap are recorded in
[Can a COPY-or-generate autoregressive editor preserve correct prefixes and retain repairs?](https://github.com/Spec-decoding-project/Speculative-decoding-project/issues/21).

## Verification and artifacts

- The preceding code test run passed **34 tests**, including interrupted atomic
  checkpoint saves and corrupted-result rejection. These unchanged unit tests
  were not rerun solely to produce this report.
- Both final checkpoints passed actual GPU reload/re-evaluation and cached/full
  generation agreement on 64 fixed test blocks. Timing repeats preserved outputs.
- Independent CPU audit recounted every development/test prediction, verified
  development-only checkpoint selection, bank/checkpoint hashes, SQLite integrity
  and zero exact normalized family overlap among train/development/test.
- All **65 original source-file hashes** are unchanged. The resumed first-epoch
  development report is byte-identical to the pre-interruption report.
- Final checkpoints are backed up on the Mac and hash-verified. A fresh local
  draft-mode recount is byte-identical to the completed remote audit.
- All 798 raw native-collection chunks are also backed up on the Mac and match
  the completed collection manifest.
- Both update journals combine into exactly steps 1–18,678 without duplicates or
  omissions. CE was reported by the model and is not reconstructed from argmax
  predictions by the independent auditor.

Frozen training source: **c3b15d6e588cf4b5af078ee780b38530ab8ec15a**.
Recovery/auditor source: **3406e63**. Bank SHA256:
`c7f92fae5609137510a198297a31b5178c70c413f263f89f04560b98aa6bbb3d`.

Aggregate evidence: [full result](evidence/final-2026-11-19/result.json),
[pipeline completion](evidence/final-2026-11-19/pipeline-complete.json),
[audit completion](evidence/final-2026-11-19/audit-complete.json),
[context recount](evidence/final-2026-11-19/context-audit.json),
[draft recount](evidence/final-2026-11-19/draft-audit.json),
[error accounting](evidence/final-2026-11-19/error-breakdown.json),
[bootstrap](evidence/final-2026-11-19/bootstrap.json),
[checkpoint backup hashes](evidence/final-2026-11-19/mac-backup-checksums.json).

Private outputs/checkpoints: Nexus `research/full-corpus-run-20261117/`; Mac
`/Users/franciszekpajak/.local/share/wayfinder-artifacts/full-corpus-run-20261119-complete/`.
The pre-recovery backup remains separate. Token-level corpora and checkpoints
are not committed to GitHub. The research outcome is negative for unconditional
replacement at this configuration, with measurable repair signal to investigate.
