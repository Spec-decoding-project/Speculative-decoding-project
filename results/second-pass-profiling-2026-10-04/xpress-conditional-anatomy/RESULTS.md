# Conditional continuation has useful support; selecting it cheaply remains unresolved

The complete 64-response panel provides stronger evidence than the earlier 16-context sample: after an oracle corrects the first fixed mistake, recomputing the continuation often helps more than copying the old suffix. This is a causal intervention on a fixed refiner, not an implemented decoding improvement.

## Completed comparison

All 256 preassigned contexts completed: four anchors from each of 32 GSM8K and 32 UltraChat development responses. These responses were already exposed during development. The released 4B xPress backbone/head stayed frozen. We recomputed its own backbone states from authenticated, prefix-only target features and reused the target's tied BF16 embedding/readout. No target transformer execution, training or new generation occurred.

Every proposal was saved before the diagnostic received future target labels. Eight preassigned explicit sequential checks confirmed that the one-pass, gold-conditioned calculation gives the correct greedy first-error length and the continuation length after one privileged replacement. Future-input checks passed. The backbone repeated exactly and matched the established wrapper. Fused/eager final chains differed in 15 contexts, all UltraChat; both controls are retained.

## What the intervention found

After six controlled refinement passes, 61 blocks already matched completely, 180 had a fixed first mistake, and 15 could still change at their first mistake. The primary intervention concerns the 180 fixed-error cases.

| Conditional continuation versus correcting one token and copying the suffix | GSM8K | UltraChat | Total |
| --- | ---: | ---: | ---: |
| More matching positions | 37 | 57 | **94** |
| Same matching length | 25 | 59 | **84** |
| Fewer matching positions | 0 | 2 | **2** |
| Net additional matching positions | +176 | +157 | **+333** |

The correct replacement ranked within the refiner's top two in 101/180 cases, top four in 143/180, and top sixteen in 169/180. The remaining eleven require a token outside that shortlist. These ranks are measured at the correct first-error location, which a real decoder does not know.

This establishes a specific conditional relationship: **for many of these fixed first-pass states, supplying a better earlier token changes later head predictions usefully.** It does not establish how to identify that token, how many parallel iterations are sufficient after it changes, or whether recomputation earns back its runtime.

The original published six-pass chain itself improved matching length in 63 contexts and damaged it in 13 compared with its own initial chain. Better individual-chain coverage alone is not the competitive benchmark.

## Opportunity beyond existing trees

The same-computation corrected-score 32-node tree covers 2,299 matching positions. Restricting privileged replacements to the top four, while retaining the whole tree for free:

| Oracle addition | Extra matching positions | Equal-response free-work modeled savings |
| --- | ---: | ---: |
| Replace one token; copy old suffix | 51 | 2.72% [1.74%, 3.79%] |
| Replace one token; conditionally continue to the next mistake | 351 | 14.21% [11.43%, 17.10%] |

Both domains show opportunity. The corresponding conditional figures are 11.26% for GSM8K and 17.16% for UltraChat. Against the published-spine control the overall figure is 14.17%, so the finding is not explained by selecting only the eager control.

**These are not speedups.** The calculation knows the first error and correct replacement, grants converged conditional continuation, retains fallback paths without charging a node budget, and assumes no extra computation. Actual generation and new-path verification were not run. Intervals resample 64 responses, keeping each response's four contexts together; they describe this exposed development panel.

## A cheap selector captures only part of the opportunity

After completing the primary comparison, we screened one fixed, label-free rule: choose two positions with the largest estimated first-failure mass along the refined chain's certified fixed prefix, then try the runner-up at each. This uses model probabilities as a ranking proxy; calibration is not established. No threshold or parameter was tuned to the outcomes.

- It selected the true first-error location in 126/180 cases, but supplied the correct alternative in only 71.
- Granting perfect conditional continuation for those selected cases adds 162 positions beyond the 32-node corrected-score tree: 6.57% free-work modeled savings [4.28%, 9.25%]. This does not include actual branch rollout or selection losses.
- Two complete branches plus the original 32-node tree can need as many as 62 nodes. Against a stronger **64-node corrected-score tree**, the same optimistic mechanism leaves 154 positions and 6.14% modeled savings [3.84%, 8.80%]. Even this comparison retains the entire control for free; a real fixed-budget union can lose its useful paths.
- Under the particularly optimistic model of identical fractional extra round cost on every response, only about **1.2% extra round cost** fits between that 6.14% opportunity and a 5% savings target. A full extra refinement stage is not automatically affordable.

Using the same two-site/runner-up-alternative concept from the initial draft instead of the completed refinement catches 43 correct replacements and leaves 5.01% free-work modeled savings. Selecting earlier could avoid serial work, but loses opportunity here. The two screens differ in available scores and the stability filter; they do not isolate a causal calibration improvement.

These screens only account for the identified stable-first-error mechanism. They are not measurements of actual finite-pass branches or a universal bound on every possible proposal algorithm.

## Next decision

The mechanism merits a **bounded branch-retention test**, not another trained editor. The concrete question is whether a small number of clamped conditional updates can recover the measured continuation benefit while fitting a tight computation budget.

Use two score-selected alternatives; preserve their known prefix, update only the remaining continuation, and retain the original proposal. First compare short refinement with a longer conditional reference on the same cached states, including failed branches and fixed-budget losses. Any cost comparison must include selection, full vocabulary work, packing and a comparably optimized 64-node control. More iterations or more alternatives are not presumed beneficial.

The cheap selection screen is marginal relative to 5%; smaller reliable gains still matter, but it does not justify a generation run yet. If short updates fail to retain useful support, or even their removable-cost limit cannot pay back, close this exact configuration. Do not silently tune the rule on these responses until it passes.

No novelty is claimed. Prefix-based redrafting exists in D²SD, causal refinement in xPress, and conditional tree expansion in other inspected work. A contribution must demonstrate a better correction-versus-computation tradeoff than those controls. This diagnostic supplies evidence for that question, not its answer.

## Verification, resources and reproduction

- 777 artifacts (38.54MB) recovered and hash-verified; 1,280 trees independently recounted and all eight sequential checks independently checked.
- GPU work completed in 196 seconds; total attempt through recovery and cleanup was 9m09s. Recovery of many small files accounted for much of the additional time; future runs should consolidate artifacts without weakening verification.
- App `ap-rKnyFvOIk79DuzhfcADwUa` is stopped with zero active tasks; owned temporary volume is absent. Independent post-closure checks confirm both.
- The full $1.50 cap remains conservatively held, not reported as settled billing. **$18 remains** in the newly authorized $30 allowance. No further run is reserved or launched.

Primary evidence: `analysis.json`, `attempt1/recovered/results.json`, and per-context proposal/diagnostic files. Secondary evidence: `first-failure-selector-screen.json`, `early-first-failure-selector-screen.json`, `matched64-oracle-screen.json`. Runner, inputs and source snapshots: `frozen/`.

Reproduce from the repository root with `.venv-profile-launch/Scripts/python.exe analysis/analyze_xpress_conditional_anatomy.py --verify` (recomputes and checks the saved result without overwriting it). Recheck the saved selector receipts with `analysis/screen_xpress_first_failure.py`. Scientific helpers and local tests are `experiments/xpress_conditional_anatomy.py` and `experiments/test_xpress_conditional_anatomy.py`.
