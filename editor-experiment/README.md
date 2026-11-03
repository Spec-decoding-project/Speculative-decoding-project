# Second-pass editor: compressing a second drafting pass into a small network

Shrinav Loka and Frank. Code and results from October 1 to 6, 2026.
Results and numbers are in `../results/SUMMARY.md`. Per-run write-ups are in
`../results/second-pass-profiling-2026-10-0*/.../RESULTS.md`.

## Question

Speculative decoding lets a small drafter guess a block of tokens that the
large target model verifies in one pass. D2SD showed that a second, separately
trained drafter re-drafting from the first draft adds accepted tokens, but the
second pass costs almost as much as the first. We asked whether a much smaller
network, reusing what the first pass already computed, can reproduce enough of
the second pass's corrections to make generation faster.

## Setup

- Target: Qwen/Qwen3-4B, thinking off, greedy decoding.
- First drafter: z-lab/Qwen3-4B-DFlash-b16 (revision b74e3a3), frozen. One
  parallel pass proposes a block of 16 tokens.
- Teacher: a full copy of the drafter (537M parameters), trained to redraft
  the block with the first token guess visible. Frozen during editor training.
  Its draft is kept beside the original and the target verifies both.
- Editor: 3,024,384 parameters (`compact_second_pass_prototype.py`). Two
  attention layers of width 256 read the first pass's normalized hidden states
  (width 2560) and the first token's embedding, and emit a rank-256 residual
  added to the drafter's final hidden state at positions 2 to 15. The drafter's
  frozen vocabulary head produces the edited logits, so the correction can be
  folded into a 74 MiB matrix costing about 10% of a full projection. The
  up-projection starts at zero, so an untrained editor reproduces the original
  draft. The first guess and the original branch are never changed.
- Data: 384 training responses and 64 development responses, 32 draft
  contexts each, from GSM8K and UltraChat (12,288 train / 2,048 dev contexts).
- Training (`compact_second_pass_learning.py`): 3 epochs, effective batch 32,
  microbatch 4, 1,152 AdamW updates, lr 1e-3, weight decay 0.01, constant
  schedule, FP32 masters with BF16 autocast, one seed. Loss on the 14 editable
  positions only. Three arms with identical initialization and data order:
  1. cross-entropy on the target's correct token;
  2. the same, plus 0.5 extra weight on positions where the teacher was right;
  3. cross-entropy plus 0.5 x KL to the teacher's distribution.
- Metric: longest prefix matching the target's greedy output, with the
  original draft kept as fallback, averaged per prompt. Costs are complete
  round times on an A100 including verification of both proposals.

## Results (measured, dev set)

| Setup | Extra accepted positions | Extra round cost |
|---|---|---|
| Teacher (full second pass) | +10.2% | about +30% (derived) |
| Editor, correct-token loss | +3.0% | +5% to +6% |
| Editor, teacher-correct weighting | +3.7% | +5% to +6% |
| Editor, teacher KL | +3.5% | +5% to +6% |

- Break-even for the editor needed about +6%; a 5% net speedup needed +10.6%
  to +11.9%. No arm reached it.
- The editor recovered 218 of the teacher's 1,136 extra positions and 94 of
  its 353 repaired blocks.
- Follow-ups changed nothing: a smaller learning rate (+3.0%), a loss weighted
  toward preserving the correct prefix (+3.2%). Restricting supervision to the
  drafter's top four candidates made the editor worse than no editor (tree
  coverage 12.5% to 12.6% vs 14.2% untouched).
- The teacher itself does not transfer: it repairs 79 and damages 33 blocks on
  its training prompts, but repairs 35 and damages 41 on unseen prompts.
- Frozen DFlash fed its own completed draft for ten passes loses 49% of its
  correct positions on pass 2 and stays there
  (`../results/frozen-dflash-iterations-2026-10-06/analysis.json`).

## What we concluded

A drafter re-reading its own draft makes it worse. A trained second drafter
produces useful alternatives on some contexts, worth about 10% more accepted
positions when both drafts are verified, which is less than it costs. Our
editors recovered a third of that and lost to a free 24-node tree built from
the first drafter's own confidence scores. The likely causes, in order: the
teacher memorizes its 384 training prompts, so the editor imitates
prompt-specific repairs; the training dose (1,152 updates) was never scaled;
and the editor only sees drafter states while the teacher also reads the
target's committed features.

## Plan

1. Real generation speed on Qwen3-8B: plain DFlash, DFlash + full second
   pass, DFlash + best editor. 200 prompts each, wall-clock. This project has
   no end-to-end speed number yet.
2. Scale the editor's data 10x using the existing frozen teacher: run it on
   4,000 new prompts, keep only contexts where the target verified the
   teacher's redraft as better than the original, and retrain the same editor
   at 1x, 3x, 10x. If dev gain stays near 3.7% at 10x, dose is not the limit.
3. Only if 2 shows a scaling curve: train a D2SD-style variable-prefix second
   drafter on Qwen3-8B (none is released) and repeat.

Compute: UMIACS Nexus class cluster, RTX A5000 24 GB, free.

## Running

Not a one-command pipeline. Each experiment was launched by a cloud driver
that is not included; the modules here are the model, data, training, decode
and analysis code those drivers called. Entry points:

- `compact_second_pass_full_training.py` trains the three arms given saved
  feature banks produced by `compact_second_pass_features.py`.
- `compact_second_pass_decode.py` runs one complete verification round with
  the editor and measures cost against original-only and teacher rounds.
- `analysis/analyze_compact_full_training.py` reproduces the result tables
  from saved run outputs.

Requirements: torch, transformers, the two Hugging Face checkpoints above, and
a GPU with at least 24 GB.
