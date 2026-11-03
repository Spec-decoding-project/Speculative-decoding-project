# The useful repair depends on coordinated score and feature changes

The post-hoc four-case intervention completed in9m39s on CPU. No backbone forward, training, target generation, GPU or paid execution occurred. It uses the saved first/second-pass states from the completed sixteen-context screen. The frozen refiner and all other inputs stay unchanged.

The head adds its correction to the backbone's vocabulary scores. A second backbone pass changes both those base scores and the hidden features the head reads. We independently substitute the two outputs, then perform the same six clamped refinement steps from the same draft.

| Context | Original scores/features | New scores only | New features only | Both new |
| --- | ---: | ---: | ---: | ---: |
| UltraChat12468/8: improved case | 6 | 6 | 5 | 14 |
| UltraChat12248/8: regressed case | 3 | 2 | 4 | 2 |
| GSM8K5091/8: unchanged complete control | 16 | 16 | 16 | 16 |
| UltraChat12677/24: unchanged incomplete control | 1 | 1 | 1 | 1 |

Values are matching-prefix positions including the anchor. All cases where the primary full/cheap prefix changed are included; controls were selected by a fixed hash from unchanged complete/incomplete cases. The incomplete control has a locked wrong first guess, so its unchanged prefix cannot demonstrate preservation of useful later corrections.

In the improved case, neither isolated substitution recovers the repair. Coordinated substitution extends the prefix by eight positions. In the regressed case, substituting the scores alone suffices to lose a position, while changing features alone gains one. Therefore a blanket rule of copying either output pathway is unsupported.

These are interventions on a fixed computational graph, not clean semantic interventions. Hybrid scores and features may be combinations the refiner was never trained to process; interactions and cancellation matter. The finding does not prove two physical student branches are necessary: one learned output function could approximate the combined result. It also does not identify which attention or MLP operations created the feature change, establish an information limit, or show transfer to new prompts.

Neutral replays reproduced both original full token vectors exactly in all four cases, independent prefix recount passed, and head weights remained unchanged. The source script hash matches the recorded plan. Evidence: results.json, per-context proposals, plan.json, verification.json.

This explains two observed outcomes without turning the narrow repair into a new training recommendation. The sixteen-context competitive result is unchanged: this untrained partial-feedback teacher has no established headroom for a student fit. Compression remains the research objective. No new spending; $3.50 remains.
