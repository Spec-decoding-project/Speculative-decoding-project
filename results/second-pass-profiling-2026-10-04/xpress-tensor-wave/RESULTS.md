# Close the fixed one/two-wave shortcut

All 256 saved contexts from 64 exposed responses completed on one A100-SXM4-80GB. The shallow shortcut was both slower and less useful than the matched depth-wise width-eight control. Do not advance it to generation, train a selector for it, or repeat this same screen.

| Method, 64 nodes | Matching positions | Gained / lost positions versus shared | Mean of context-median proposal time |
|---|---:|---:|---:|
| Six-pass refined shared scores | 2,348 | — | 2.460 ms |
| Same seed, one completion wave | 2,462 | 115 / 1 | 2.860 ms |
| Same seed, two completion waves | 2,523 | 177 / 2 | 3.283 ms |
| Depth-wise conditional, width 8 | 2,677 | 338 / 9 | 2.560 ms |
| Depth-wise conditional, width 12 | 2,688 | 348 / 8 | 3.204 ms |

One/two waves damage zero fully correct baseline blocks. Both depth-wise variants damage one. These differences are reported rather than treating total coverage as the only outcome. Width twelve adds little progress relative to width eight and costs substantially more.

## What the result explains

The warm-start construction's refined shared tree already costs 2.46 ms. Two completion waves add another 0.82 ms. The complete depth-wise control, which does not need six seed-refinement passes, costs only 2.56 ms. Thus reducing the number of *additional* conditional waves did not reduce the whole proposal's cost. The shortcut also cannot follow newly discovered useful continuations as far. This is evidence against this implementation and fixed schedule, not a proof that all initial-draft reuse is futile.

Conditional scoring itself remains useful on these contexts. Width eight adds 329 net matching positions (14.01% aggregate) versus shared scores for 0.10 ms more proposal time. That is a positive control result. Conditional trees and depth-wise expansion already exist; this does not establish an original algorithm or defeat released DARTree.

## Cost interpretation

Timings include hidden/candidate preparation, seed refinement where required, GPU-resident selection and packed ancestor masks. All five methods use the same head/primitives and graph replay. Graph setup took 1.36 seconds, excluded from steady-state proposal timings and reported separately. They use base-top-64 candidate normalization and top-16 edges. This is not full-vocabulary policy equivalence. The width-eight/twelve controls do not pay unnecessary seed work.

Backbone execution, initial vocabulary projection, target verification, accepted-cache updates and end-to-end generation were not run. These are actual proposal timings, not whole-decoder timings. A fully optimized published implementation may also improve these primitives, and the shared tree's tensor beam construction is not assumed optimal against its CPU-heap alternative.

In the predeclared scenarios assigning an equal 5/10/20/40 ms common non-proposal cost to every condition, the two-wave candidate is respectively 16.56%/12.51%/9.80%/8.20% worse in modeled cost per matched position than width eight. Every corresponding descriptive prompt-bootstrap interval is adverse. It has no supported runtime case here.

Width eight versus shared has modeled savings of 11.37%/11.84%/12.15%/12.34% at those same common costs. At 40 ms, its interval is 9.39% to 15.44%. These figures assume fixed-context progress translates to rounds and equal excluded costs; they do not establish a generation speedup, equal verification time or preserved native output. Report them only as motivation to check the complete decoder.

## Verification and accounting

- 38 local tests and launch serialization passed before execution.
- Every eager audit matched the corresponding captured graph exactly; repeated graph outputs were stable.
- All 1,280 trees were reconstructed locally from the actual saved CUDA input paths and output scores. Independent ancestry, finite-score, mask, token and coverage checks passed. Exact analysis regeneration passed.
- No target labels entered proposal construction. No model weights changed, training ran, or fresh prompts were used.
- 265 artifacts, 12,508,718 bytes, were recovered with hashes verified. App `ap-aPTYcJDkHwcFjVGUvLeWIU` was independently confirmed stopped with zero tasks; owned temporary volume was removed.
- Worker execution was 193.52 seconds. Including queueing, staging, recovery and cleanup, the attempt took 488.53 seconds. Full $1.50 cap conservatively retained; actual settled billing is unknown. $10.50 remains in the existing $30 allowance, with no active paid job or new reservation.

## Next decision

Preserve the stronger conditional construction as a reference, not a novel success. Before further compression or another proposed shortcut, determine whether its benefit survives actual target verification and complete round costs against the released six-pass chain and a strong ordinary tree. Reuse existing target/cache verification machinery; include native numerical agreement and following-token cache checks. No new training is justified by this result.

This is a bridge toward eventual full generation, not a replacement success criterion. The original objective—useful end-to-end improvement with required correctness and a defensible contribution—remains unmet.

Evidence: `analysis.json`, `analysis-contract.json`, `attempt1/execution.json`, `attempt1/recovery.json`, `attempt1/recovered/`; `analysis/analyze_xpress_tensor_wave.py --verify`. Original protocol and sources remain frozen under `frozen/`.
