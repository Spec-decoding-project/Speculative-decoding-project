# Is the current full teacher a sufficient compression target?

## Finding

A perfect copy of the feedback teacher's current predictions would support a 5% modeled latency reduction only if uniform added round cost were at most **4.387%**. The compact implementation previously added **5.02-6.27%** on four different contexts with untrained weights. If those costs applied uniformly on the development panel, perfect teacher imitation would yield only **3.29-4.42%** modeled savings.

This identifies a target/cost mismatch to resolve before another imitation fit. It is not a measured speed comparison or a universal impossibility. The cost panel is very small, uses untrained proposals, and differs from the quality panel; trained, optimized overhead could be lower or higher. Smaller reproducible gains would still be useful, but would not meet the current 5% target.

## Existing predictions contain some complementary opportunity

On the same 2,048 saved contexts across 64 exposed prompts, retain the longest *complete predicted prefix* available among the named models. These are hindsight envelopes, not learned selection or feasible serving systems.

| Available predictions | Added prefix positions | Equal-prompt progress gain | Maximum uniform overhead for modeled 5% saving |
|---|---:|---:|---:|
| Feedback teacher | 1,136 | 10.21% | 4.39% |
| No-feedback teacher | 950 | 8.05% | 2.33% |
| Current probability-trained compact editor | 386 | 3.46% | -1.81% |
| Feedback teacher plus compact editor | 1,304 | 11.84% | 5.87% |
| Both teachers | 1,357 | 12.18% | 6.16% |
| Both teachers plus compact editor | 1,494 | 13.53% | 7.40% |

All include the unchanged original proposal as fallback. Negative allowable overhead means even free added computation fails the 5% point target under this model.

The compact editor adds168positions outside the feedback teacher, including137outside both teachers. Thus blindly copying one teacher would discard some useful student behavior. Adding those opportunities to the target description creates more modeled margin, but does not establish that a single compact model can learn them jointly. Running all these models is not being proposed: their computation would consume the assumed savings.

The feedback-teacher-plus-student envelope would model5.77%savings at5.02%uniform overhead but4.64%at6.27%; its allowable-overhead descriptive bootstrap interval is4.44-7.43%. All-three envelope models7.11%and6.00%at those costs, with allowable-overhead interval5.76-9.13%. These are dependent exposed-data diagnostics, not robust population margins or predicted speedups. In particular, success of a label-informed combination does not imply successful weight merging, distilled consolidation or selection before verification.

## Calculation and checks

For each prompt p, Pp is summed original prefix progress and Up summed available prefix progress. Under hypothetical uniform fractional added round cost c, the equal-prompt relative cost is (1+c)*mean(Pp/Up). A5%mean reduction requires c<=0.95/mean(Pp/Up)-1. It is not obtained by simply subtracting two percentage gains.

All source hashes match the earlier feedback-specific-capture audit. Context keys, original lengths, teacher anchors and capacities align. Full-teacher overhead4.387% exactly reproduces the historical sensitivity calculation. Combined extras independently match the earlier position partitions. A second aggregation of saved per-context rows reproduces all scenario thresholds. Bootstrap resamples entire prompts; it does not account for training-seed uncertainty or turn the exposed panel into a fresh test.

## Decision for this direction

The purpose remains a cheap useful second-pass replacement. Matching the current teacher more accurately is an intermediate metric, not sufficient success. A next proposal must explain at least one credible source of margin: lower complete overhead, corrections beyond this teacher's fixed predictions, or a stronger teacher target that can actually be compressed. Any additional training signal must preserve existing correct prefixes and successful student corrections; an oracle combination alone is not a method or a funding argument.

Do not launch another generic distillation loss merely because it gets closer to this teacher. Do not switch to model merging merely because complementary outputs exist. The new useful result is the quantified target/cost requirement, and the limited extra headroom available in already-known outputs. No trained method, novelty or speedup is established.

Sources: results.json, rows.json, verification.json; ../feedback-specific-capture/results.json; notes/conditional-full-reference-precision-2026-09-22/results.md; notes/second-pass-profiling-2026-10-01/compact-integrated-cost/attempt1/RESULTS.md. No new cloud execution or spending.
