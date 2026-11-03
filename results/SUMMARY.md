# Speculative decoding research: what we have done so far

Written 2026-11-03. One page, plain words. Every number is from our own logs
unless marked (paper). Setting unless stated: target Qwen3-4B, drafter
z-lab/Qwen3-4B-DFlash-b16, GSM8K + UltraChat, 64 dev prompts x 32 contexts,
A100 on Modal. Total spend across all phases: about $100.

Detailed records: RESEARCH_STATE.md (full log), RESEARCH_HANDOFF.md
(reverse-chronological), notes/<date>/RESULTS.md (per experiment).

## The question

A small drafter guesses a block of tokens; the big model checks them in one
pass. Can the drafter's *second look* at its own draft make the draft better,
cheaply enough to speed up generation? Bar: 5% faster end to end.

## Phase 1, early September: train a better drafter

- Published post-trained drafter (Draft-OPD) scored 13.9% higher than the
  base drafter on our reward. Real, but not our method.
- Our own training on first-rejection points: no gain that transferred to
  new prompts (-0.5% to +0.2%, intervals cross zero). Closed.
- Verification-credit post-training pilot: primary recipe lost 6.7%. Closed.

## Phase 2, mid September: where the time goes

- Found the tile cliff: flash attention verifies rows in tiles (32 on H100
  FA3, 128 on the GEMM). A draft that crosses a tile boundary pays for a
  whole extra tile. Wrote SGLang and vLLM length caps and model-card fixes
  (experiments/cliff/). Then found NVIDIA had published the 128-row rule
  first. Engineering result, not novel.
- TreeFlash CUDA graphs: +25.1% on our setup. Engineering.
- Deferred exact verification: 1.30x end to end at 89K context. Below bar.
- Feeding drafted prefix tokens back into the drafter on Qwen3-8B hurt at the
  drafter's doubt slots (-22% and -35% of the gap). Target features for that
  prefix recovered 78-84% but need a target pass.
- Staging a block into 2-3 conditioned stages: at most 1.04-1.05x tokens per
  round; any number of stages at most 1.14x.

## Phase 3, Sept 22-29: conditional redrafting (the class project)

Second drafter pass reads [anchor, first token guess, masks].

- Untrained second pass alone: -1.42% vs the original draft.
- Keep both drafts, let the target pick (best-of-two): +5.13%.
- Trained full second drafter with feedback: +10.21% retained; matched
  no-feedback control +8.05%. Feedback effect +2.16 points [1.17, 3.18],
  concentrated at draft slots 2-4.
- Adapter + 2 private layers reusing draft states: +4.90% best-of-two,
  cost per accepted position 8.57% worse than plain drafting.
- Even a free second pass would cut time per token by at most 4.76%.
- Sept 24 audits closed every cheap substitute: candidate mixing (+0.2% on
  excluded prompts), cached suffixes, cross-round logit transport (-25%),
  longer drafts, cheaper tree readout. A 24-node tree built from the first
  drafter's own scores covered 565 vs 443 positions for free and beat all
  of them.

## Phase 4, Oct 1-6: compress the second pass into a small editor

3M-parameter editor reads the drafter's saved states, trained against a full
second-pass teacher. 384 training prompts, 1,152 updates per arm.

- Teacher (full second pass, with original kept as fallback): +10.2%.
- Students, three losses: +3.0% (correct-token), +3.2% (prefix-weighted),
  +3.7% (teacher KL). None beats a 5% round overhead. Candidate-focused
  distillation was *worse* than the untouched draft (-1.5 points).
- Tree coverage: original scores 14.2%, full teacher 16.9%, best student
  12.6%. Students lose.
- Why: the teacher learns its training prompts. Raw chain repairs/damages:
  training 79/33, dev 35/41. Reading the whole completed draft is worse:
  dev 25/94.
- Frozen DFlash fed its own completed draft for 10 passes (Oct 6): pass 2
  drops matching positions from 1,504 to 646 (-49%), stays there.
- Public D2SD weights on Qwen3-8B, one run, 8 prompts: original 140
  positions, +4 conditional branches 166 (+5.98%, interval [-0.6, +12.7]).
  The only 8B evidence we have.
- xPress-8B comparison: died downloading the checkpoint. Never measured.

## What the literature says (paper)

- D2SD on Qwen3-8B: naive resampling 5.75 -> 6.08 accepted tokens; DFlash
  reused as second drafter +14% tokens / +5% speed; separately trained
  variable-prefix drafter + 4 branches + joint verify: 5.31 -> 7.05 tokens
  (+33%), speed 4.16x -> 4.98x (+20%). A third level adds +8% tokens but
  cuts speed (per-cycle cost +13%).
- Branch-conditioned trees on DFlash are published: DARTree, DEdit, D2SD.
- D2SD's trained second drafter weights are not released.

## One-paragraph verdict

A drafter re-reading its own draft makes the draft worse. A separately
trained second drafter produces a useful *alternative* on some contexts,
worth about 10% more correct positions when the target verifies both, which
is less than the second pass costs. Every cheap replacement we trained
recovered a third of that and lost to a free score-based tree. The second
pass is worth having and too expensive to run; that gap is real and is the
open question, but our students have not closed it and our teacher does not
transfer.

## Open and runnable on the UMIACS A5000 (free, 24 GB)

1. End-to-end wall-clock on Qwen3-8B: DFlash vs DFlash + full second pass
   vs DFlash + best frozen editor. No real generation number exists yet.
2. Train a proper variable-prefix second drafter for Qwen3-8B (nobody
   released one). Multi-day job; the free GPU is the right tool.
3. Only if 1 shows headroom: collect 20k+ draft contexts and retrain the
   editor at a dose that could work.

## Rules we keep breaking and should not

- Fixed-context coverage is not a speedup. Nothing counts until measured on
  real generation.
- Compare against the strong free tree, not against the plain chain.
- Check prior art before calling anything new (DARTree, DEdit, D2SD, TAPS,
  NVIDIA 128-row rule).
- Label every number: measured, derived, estimate, paper.
