# Earlier branches preserve useful corrections, but the gain remains modest

## Decision
The early schedule is a validated building block on saved development contexts, not a demonstrated generation-speed improvement. It preserves nearly all of the comparable late method's coverage, but its proposal-cost saving is only 0.153ms. Do not expand this fixed schedule into a generation campaign or another training run on these results alone.

The next concrete question is whether scoring the whole vocabulary is necessary for these conditional corrections. A saved-output support check leaves that possibility open, but actual restricted execution may change the intermediate predictions. The author already supports candidate-restricted refinement; it must be applied to strong controls too, and is not a novel method by itself. Any next test must measure retained conditional value and comparable costs together, not just a faster isolated kernel. No next paid attempt is reserved.

## Completed full-panel result
All256contexts/64exposedresponses, all seven64-node conditions, two warmups and eight rotated paired repetitions per context. Same frozen released weights and authenticated first-pass states. No target/backbone transformer execution, training or generation.

| Condition | Total matching positions | Complete proposal ms, mean of256context medians |
|---|---:|---:|
| Shared64 corrected-score tree | 2350 | 2.050 |
| Published64 chain/tree union | 2310 | 2.056 |
| Late selection +3branch updates | 2434 | 3.083 |
| Late selection +6branch updates | 2475 | 3.603 |
| **Early selection, parallel completion** | **2430** | **2.931** |
| Same early inputs, serial completion | 2430 | 3.571 |
| Early selection, suffix initialized late | 2430 | 3.972 |

Counts include the root consistently. They are fixed-context coverage, not newly generated accepted tokens. Proposal timing includes scoring, host selection, copies, synchronization and tree packing; common drafting/readout and target verification are excluded. Graph setup:5.26seconds, not amortized into the table.

## What we learned
1. **Batching works for this particular conditional computation.** Parallel versus serial early completion gives identical trees, main chains and branches in all256contexts while saving0.640ms, about17.9% of proposal time. This is not17.9% generation speedup and does not contradict the earlier different batching experiment.
2. **Waiting for the final draft did not improve useful coverage for these selected branches.** Starting from the third rather than sixth iteration changes branch/tree contents in129contexts, but their reference-matching coverage is identical in every context after three conditional updates. This supports beginning these useful corrections earlier on this panel; it is not a general initialization theorem.
3. **Early selection loses little, but not nothing.** It selects the true first-error site in124contexts and its correct replacement in70; late selection gives126/71. The early method loses8positions and gains4 versus late3, net-4. This contrast combines fewer eligible sites with a numerical scoring-backend difference; do not attribute it solely to timing.
4. **The strong tree still limits competitive headroom.** Early parallel gains96positions and loses16 versus shared64, net80. It costs0.881ms more. A no-added-cost scenario averages3.41% savings[1.38,5.49]; the observed implementation does incur added cost.

The early fixed-prefix certificate holds on all256contexts.91whole blocks were already settled after three iterations; others had at least four settled positions. The unchanged main chain after3+3 exactly reproduces the native six-pass chain everywhere. Native wrapper checks at eight fixed contexts pass.

Early selection reuses the third pass's fused logits; the late selector explicitly recomputes unfused scores along the final chain. Top16 rankings differ somewhere inside the settled region in243contexts, mostly without changing selections. Holding eligibility fixed, this numerical scoring difference changes the two selected pairs in9contexts; restricting eligibility alone changes them in28; actual early/late selections differ in37. This does not violate the structural convergence statement, which assumes the same function/precision. Future reports must not claim all early score differences reflect less information.

## Economics, now measured on the same panel
All64responses contribute both coverage and timing, avoiding the preceding timing-subset mismatch. Equal-response bootstrap resamples prompts, not256independent contexts or individual timing repetitions.

With an **assumed40ms common round cost**, early parallel's modeled savings versus shared64 are **1.38%[-0.68,3.51]**. Math:1.45%[-0.41,3.51]; instructions:1.31%[-2.31,5.11]. Late6's corresponding estimate is1.18%[-1.39,3.97]. These are uncertain modeled gains, not measured total latency. The early method versus comparable late3 is0.29%[-0.14,0.71] in this scenario.

Thus the schedule makes an existing conditional operation cheaper without establishing the requested5% end-to-end improvement. The larger17.9% proposal-only batching number must not be used as the headline speedup.

## Local necessary-support check
In the preceding late6 screen,36improved contexts contained142new matching positions before subtracting displacement losses. Constraining suffix outputs to the original top16 would necessarily block7of those extra positions in four contexts, even while retaining the original baseline for free. The other135positions have necessary vocabulary support. Selected/clamped alternative tokens are exempt from the suffix restriction.

This is **not** a restricted-head replay or a guarantee of95% retention: intermediate choices can differ, and changes to main-chain refinement can change selection and tree scores. It only rules out immediate lack of candidate support as a dominant objection on those observed successful paths. Full output projection is a plausible cost to test; the released implementation already contains the idea. No new architecture/training/novelty claim follows.

## Verification and accounting
18localtests passed. AllGPU author3/author6/serial-main checks and repeated trees passed; fixed prefixes and clamping held. Weights unchanged. All256native main chains reproduce the previous evidence. Independent reconstruction/recount covered1792trees and all timing records, and exact analysis regeneration passed.

All nine artifacts/1.60MB recovered and verified. App `ap-s8bR9i5Q5SmEXM0pCU7Rto` stopped/zero tasks and owned volume removed; independently checked after closure. Worker169.43seconds; throughcleanup243.68seconds(4m04s). The provider call-graph snapshot continued displaying PENDING after worker evidence was complete; terminal function results and fresh app inspection resolved status, with no resubmission.

Full$1.50cap retained pendingbilling. **$15remains** in the current$30authorization. No active jobs, no retries, no new reservation. Research goal active/unmet.

Evidence: [analysis.json](analysis.json), [mechanism-check.json](mechanism-check.json), [candidate-vocabulary-support.json](candidate-vocabulary-support.json). Reproduce primary analysis with `analysis/analyze_xpress_early_branches.py --verify`; reproduce both posthoc checks with `analysis/diagnose_xpress_early_branches.py`, using `.venv-profile-launch`. Sources and protocol frozen under `frozen/`. Prior-art boundaries are in [PROTOCOL.md](PROTOCOL.md).
