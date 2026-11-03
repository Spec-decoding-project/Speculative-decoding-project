# Conditional corrections survive; prototype cost still defeats them

All16 contexts and eight methods completed, with five alternating timed repetitions each. No training, target-model execution or new generation. Frozen recovered4B xPress weights and first-pass features;32 final nodes throughout. All data remain reused development evidence from eight responses.

| Method | Matching positions | Mean complete proposal time |
|---|---:|---:|
| Ordinary score tree |149|0.500ms|
| Published refined-chain union |157|1.979ms|
| Shared-spine corrected-score tree |169|2.596ms|
| Original seed/full history |170|9.725ms|
| Corrected seed/full history |175|10.231ms|
| Corrected seed/parent only, ungrouped |175|10.454ms|
| Corrected seed/parent only, grouped |175|9.024ms|
| Full-history depth-wise expansion |177|10.981ms|

Times include first readout, released six-pass refinement where used, shared scores, seeds, grouping, head calls, transfers and selection. They exclude the common drafter transformer, target verification and cache work. These are not complete rounds or generation speedups; they are reference Python builders with CUDA-graph head scoring, not optimal serving implementations.

## What the experiment resolves

The grouped candidate gains6 positions in5 contexts and loses none versus shared-spine. Its corrections therefore add proposal value on this panel. It nevertheless costs6.429ms more. In a labeled40ms common-work scenario, modeled time per matching position is7.17% worse (descriptive95%[-0.65% better,13.37% worse]). No generation test is justified by this measured implementation.

Grouping saves1.430ms versus the same parent-only method without grouping, about13.7% of that method's proposal time. All16 selected trees agree. However3 top16 rows change with GPU batch shape, and maximum compared log-score difference is0.13602. Mathematical duplicate-input equivalence did not yield bitwise CUDA score equivalence; preserve this qualification. Original-policy trees reproduce the previous GPU run on16/16 contexts.

Full history gives the same175-position coverage as parent-only, at1.207ms higher proposal cost. This panel supplies no coverage benefit for the extra history; it does not show the distributions or trees are identical or establish that history never helps.

Versus depth-wise expansion, the candidate gains4 and loses6 positions (one improved/two worsened contexts) while saving1.956ms proposal time. At40ms common work, modeled savings are only0.38%[-6.59,4.14]. Thus speed relative to the deliberately serial construction does not rescue the candidate against simpler controls.

## Follow-up diagnosis without another GPU run

The selector currently materializes and sorts every expanded path, even though it retains only32. A local host-only profile measured6.204ms for that eager selection and0.450ms for best-first selection on the same saved scores. The latter evaluated6,602 path scores instead of materializing30,737 paths. This machine's timings cannot be subtracted from cloud GPU proposal time or reported as a serving speedup.

Best-first enumeration preserves this particular policy: nonpositive edge log probabilities ensure every descendant scores no higher than its frontier ancestor. Expanding the largest frontier score, with the same ancestor-first tie rule, recovers the global top32. Newly proposed children still have no outgoing edges unless they were in the original seed. This is a known search optimization, not a novel selection objective.

The implementation matches all64 saved one-hop trees across four methods and300 independent exhaustive randomized cases, including zero-score ties and varying budgets. It was implemented only after the paid run. Its GPU-host integrated cost remains unmeasured, and its CPU timings do not establish that it is sufficient. Original paid-run sources are preserved with matching hashes under frozen/sources/.

## Decision and next action

Do not train another head or run generation yet. Keep the observed conditional corrections as evidence. Apply the identical best-first execution change to every applicable control before judging the algorithm's achievable cost. The next cost check must evaluate the full optimized proposal against the same shared-spine and depth-wise controls; a faster selector alone is insufficient. If the complete margin remains poor, close this configuration rather than extending the timing sequence into an open-ended kernel project.

No novelty or5% generation improvement is claimed. The high-level candidate is still conditional expansion of paths available from an initial draft using a released refiner; parent-conditioned trees and parallel causal scoring overlap prior work already reviewed. A favorable cost screen would still require fresh contexts, target fidelity and actual end-to-end measurements against strong equally optimized systems.

## Evidence and resources

All24 recovered artifacts (30,381,092bytes) authenticated. Independent128 coverage recounts and80 tree-policy reconstructions pass; frozen-weight version audit and stable timing outputs pass. Submission through cleanup83.58seconds; app ap-MuLJBgcZHdqvxsqIgzNh0X stopped, zero tasks, owned temporary volume absent, independently checked. No retry. Full1.50USD cap remains conservatively held, not a settled invoice;21USD remains in the new30USD allowance. No active paid work.

Evidence: attempt1/analysis.json, verification.json, recovery.json, closure-resource-check.json, all recovered comparison records; host-profile.json and lazy-selection-checks.json. Local CPU precursor: ../xpress-corrected-seed-replay/RESULTS.md. Goal active and unmet.
