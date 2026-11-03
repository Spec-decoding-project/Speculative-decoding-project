# What actually fails when the small editor tries to reproduce a second pass?

## Main finding

The evidence does **not** establish that useful second-pass corrections are impossible to compress. It now identifies something more concrete than a mismatch between loss and acceptance: **on sampled development failures, the learned correction usually moves the correct token further behind a wrong competitor, rather than merely moving it insufficiently toward winning.**

This is an observed computational failure of this frozen student. It is not yet an isolated explanation for why training produced that function. Capacity, representation, initialization, exposure and optimization must not be relabeled as proven causes.

Two new local audits used existing artifacts. No training, GPU execution, paid work, model downloads or new evaluation prompts were involved.

## 1. Many teacher repairs require few output changes

Authenticated all 2,048 development contexts from 64 responses and the five historical editors. Reproduced every original/edited prefix length and the earlier teacher-repair counts. Among the 353 blocks where the full teacher extends the original draft:

- **222/353 (62.9%)** contain exactly one wrong original token before the teacher's correct-prefix endpoint.
- Replacing just the original first wrong token with the known correct answer recovers **706/1,136 (62.1%)** of the teacher's additional matching positions.
- This does not supply an inference-time detector, a correct-token predictor or a computational shortcut. A single required token change can still require difficult context-dependent computation. The remaining 131 blocks require multiple changes to reproduce the teacher prefix.

Thus it would be unsupported to say the output changes are universally too extensive for a small editor. It would also be unsupported to infer that sparse output changes necessarily have a simple, cheap predictive rule.

## 2. Earlier damage is not the whole problem

For each frozen output, use target answers only to diagnose the consequences of two hypothetical repairs: restore earlier original-correct tokens, and correct the original blocking token. Do not rerun the neural model after patching. The teacher-probability student gives:

| Fixed-output condition | Teacher-added positions captured, out of 1,136 | Blocks reaching the complete teacher prefix, out of 353 |
|---|---:|---:|
| Observed student | 218 | 56 |
| Restore earlier correct prefix only | 253 | 67 |
| Correct the blocking token only | 545 | 156 |
| Restore earlier prefix and correct blocker | 678 | 203 |

Restoring earlier positions adds 35 captured positions; fixing the blocker adds 327. Across all five historical editors, those increments are respectively 35–59 and 278–351. This supports prioritizing the prediction at the blocking token over a story in which most useful corrections are already learned but merely hidden by earlier damage.

Even the combined answer-informed patch retains only 678 positions. Later mistakes still matter. None of these patches is deployable: they use the answer and error location. They isolate sequence-level requirements, not a neural cause or a new decoding algorithm.

## 3. New score-level evidence: wrong direction versus insufficient movement

Reconstructed the frozen teacher-probability editor on the same previously hash-selected 128 training contexts and 128 development contexts used by the earlier input-dependence audit. Kept weights, first-pass states and actual first-guess input fixed. Read both original and edited vocabulary scores in CPU FP32. For a correct token, define its margin as its score minus the highest score of any wrong token.

Among development contexts whose native recorded teacher extends the original prefix:

- There are **28** cases with an editable original first error.
- The editor chooses the correct token in **9** cases; it still chooses a wrong token in **19**. Local correctness alone does not establish a repaired whole prefix.
- In **17/19 failures**, the correct-token margin gets worse. These cases span **15 prompts**. Median margin change is **-1.20 logit units**, with changes ranging from -4.09 to -0.061. Their final margins are all negative.
- In **2/19 failures**, the margin improves but remains negative.
- In five of the 17 adverse cases, the correct token improves relative to the *old* wrong guess but falls further behind the strongest wrong competitor. Looking only at the old wrong token would therefore hide some failures.

On the selected training teacher-repair cases, the margin improves in 82/87 and the editor chooses the correct token in 51/87. On development teacher-repair cases those figures are 11/28 and 9/28. This is descriptive evidence of a change in learned behavior between the sampled groups. Different prompt difficulty and teacher-success selection remain confounders; these fractions do not identify dataset size or memorization as the unique cause.

### What can be ruled out about simply increasing correction strength

For fixed base scores `b` and fixed additive corrections `d`, the ideal real-valued gold margin at scale `a` is:

`m(a) = min_j [(b_gold - b_j) + a * (d_gold - d_j)]`, over wrong tokens `j`.

This is concave in `a`. If `m(1) < m(0)` and `m(1) < 0`, then for every `a > 1`:

`m(a) <= m(1) + (a - 1) * (m(1) - m(0)) < 0`.

Accordingly, making the same fixed correction stronger cannot fix those 17 negative-direction cases in this score model. This is a limited algebraic consequence, not an assertion about retraining, changing directions, shrinking the update, earlier-token preservation or native BF16 execution. The existing shrinkage audit separately found limited opportunity from reducing correction strength. Do not convert either result into a universal rejection of calibration.

### Numerical qualifications

The reconstructed edited tokens exactly reproduce all previously saved CPU input-audit outputs. They differ from original native BF16 outputs at 65/1,692 valid training token rows and 55/1,763 development rows. Union progress differs on six training contexts and two development contexts. Those discrepancies are preserved.

At the first-error position of the 28 development teacher-repair cases, CPU versus native **correct/incorrect classification agrees in every case**. The score margins themselves were not recorded from native CUDA, so score-direction claims remain CPU-reconstruction findings. The score audit took 199 seconds; no model parameters were updated.

## 4. What the earlier interventions do and do not explain

| Possible explanation | Existing evidence | Conclusion supported |
|---|---|---|
| The output representation cannot express the right token | Answer-informed coefficient witnesses corrected all 24 selected first errors using the frozen learned output projection, verified on GPU | Not a hard output-space impossibility on those cases; not a full capacity proof |
| The whole editor cannot fit these kinds of corrections | Focused training reproduced 64 chosen teacher repair prefixes while preserving 32 chosen correct blocks | Joint fitting of this selected set is possible; broader fitting and transfer remain open |
| The explicit guess input was never trained | All its weights changed; optimizer moments record gradient history; deployment casts verified | The omitted-parameter/disconnected-gradient explanation is ruled out for this implementation |
| The student already learned the intended explicit-feedback mechanism, but only transfer failed | Removing or shuffling the explicit guess gave no substantial useful advantage on sampled training or development data | The mechanism has not been demonstrated even during fitting; first-pass states may still carry implicit guess information |
| More overall teacher agreement is enough | Teacher agreement improved from 50.17% to 59.48%, without a reliable coverage improvement | Global imitation quality is not a sufficient advancement metric at this architecture and exposure |
| Prefix weighting alone solves it | The tested retained-prefix objective changed coverage from 3.02% to 3.17%, with uncertainty spanning no difference | That particular weighting package was insufficient; objective mismatch is not a complete diagnosis |
| Missing raw context is the established bottleneck | The tested context-access package did not improve reliably; fixing its optimization burst did not restore useful gains | Neither context absence nor that interface is established as the unique cause |
| All that remains is transfer | On a training sample the probability student captured only 447/1,094 teacher-added positions despite high total training coverage | Incomplete teacher capture already exists in fitting; other learned corrections inflate training coverage |
| Simply replace teacher-training prompts with different prompts | The matched old/new-data experiment showed no clear fresh-panel improvement | That equal-size replacement was insufficient; data scaling remains untested |

These statements refer to the cited concrete experiments, not all possible editors.

## 5. Why there is no honest single-cause answer yet

The student was not a scaled-down copy with only one variable changed. The full teacher uses five pretrained native DFlash layers, width 2,560, and direct context access. The original editor starts as two randomly initialized width-256 transformer layers, consumes saved first-pass states plus one guess embedding, and predicts a rank-256 additive vocabulary correction. Normalization, attention, nonlinearities, input interface, initialization and capacity differ together. The editor also relies on the frozen pretrained first drafter; the private-parameter ratio must not be presented as total-system compression.

The comparisons have investigated several remedies, but they do not isolate all those differences. Calling the failure simply "not enough data," "too few parameters," "bad loss," or "not enough information" would claim causation the evidence does not establish.

The most specific current diagnosis is: **this trained student often generates an adverse correction direction at the acceptance-critical decision on development inputs, and it does not faithfully recover all useful teacher decisions even on training inputs.** Selected answers are representable and fit-able. That combination leaves learning the correct input-to-correction mapping as the unresolved obstacle, with its architectural and training causes still unidentified.

This sharpens the research question rather than disproving it. Another fit should distinguish a stated cause while controlling the others; no architecture, data increase or loss switch is selected or promised by this audit. In particular, stronger scalar corrections and protecting earlier tokens alone do not address the dominant observed failures. The new facts do not establish novelty, economical compression or generation speedup.

## 6. Scope correction for the separate xPress experiment

The completed xPress experiment removed the cost of an already lightweight released refiner. Its approximately 0.4% timing difference does **not** measure the savings from replacing our full second-pass teacher or establish that D²SD-style full-pass compression is pointless. It remains a competitor-cost observation. Our earlier prototype is D²SD-inspired, not a complete reproduction of every D²SD configuration.

## Evidence and reproduction

- New sequence audit: `analysis/audit_compact_failure_decomposition.py`; `results.json`, `cases.json` here. All 69 input files authenticated against the historical audit; 2,048 contexts, 353 teacher-repair cases, five editors. Independent prefix-product checks reproduce every patched capture count.
- New frozen-score audit: `analysis/audit_compact_first_error_margins.py`; `margins/results.json`, `margins/rows.json` here. Uses existing pinned model/readout identities and preselected samples. No GPU is needed.
- Earlier source reports: `notes/second-pass-profiling-2026-10-02/{readout-witness,compact-rehearsal,compact-guess-dependence,compact-retained-prefix,context-access-training,context-smaller-update-parallel,teacher-unseen-comparison,pretrained-compression-assessment}`.

The scripts refuse to overwrite their output folders/files. All data are historical; no new held-out claim or paid execution occurred. Repeated contexts are not independent prompts. Source-level experiments and numerical limits remain preserved.
