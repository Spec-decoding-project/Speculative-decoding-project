# Additive teacher-tree audit

The local audit completed on 2,048 saved contexts from 64 reused development responses. No training, generation, GPU allocation or paid execution occurred. Results are in `results.json`; the comparison was specified in `PROTOCOL.md`.

Keeping the ordinary tree intact and adding teacher branches protects its existing coverage. However, the added nodes also give an ordinary tree more opportunities. The relevant control therefore receives exactly the same number of nodes, chosen without reference labels.

| Candidate width | Modeled savings versus ordinary tree with equal nodes and width | Versus equal-node ordinary tree with width 16 |
|---|---:|---:|
| 4 | 2.62% | 1.72% |
| 8 | 3.38% | 3.21% |
| 16 | 3.82% | 3.82% |

These calculations assume equal complete-round time and use fixed-context progress. They exclude the extra cost of producing teacher predictions and constructing the union; they are not measured generation speedups.

At width 16 the descriptive prompt-bootstrap interval is 3.09–4.55%. The union averages 37.27 nodes. Against the equal-node ordinary control, it gains 716 matching positions and loses 148 across contexts. It never loses coverage against its contained 30-node tree, but that smaller control has fewer opportunities. The latter comparison gives 6.28% modeled savings, which must not be presented as the matched-budget result or confused with historical 6.28% findings from other experiments.

Adding only the teacher's greedy chain yields 1.10% modeled savings against the equal-node width-16 control, with worse results on general instructions. The fuller teacher tree is more useful in this analysis.

All 28,672 independent tree reconstruction checks passed, including checks against previously recorded endpoints and prefix closure. Candidate widths above 16, other proposal methods, actual runtime, fresh prompts and student learnability remain untested here. Reused-development intervals are descriptive, not fresh confirmation.

Decision: this establishes additional useful teacher proposals on these saved contexts, but does not justify another small-editor training run by itself. Even assuming the additional computation is free, the matched-control estimate remains below the 5% goal. Any next training proposal needs an explicit reason it can improve beyond merely copying this tested teacher policy, or a justified lower performance objective. This is not a general upper bound on redrafting.
