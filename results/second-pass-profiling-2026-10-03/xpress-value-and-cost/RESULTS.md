# Released 4B xPress: useful corrections, little removable cost in this setup

The approved attempt completed all sixteen fixed contexts from eight reused development responses. There was no training. The strongest supported decision is **do not train a smaller imitation of this refiner solely to obtain 5% lower latency in this execution setup**. The correction can improve proposals, but eliminating its computation saved only about 0.4% of round time. This does not close the broader question of learning better corrections.

## Design and scope

- Target: frozen Qwen3-4B, BF16, greedy; pinned revision in `runner-plan.json`.
- Reference: released `UIUC-SSAIL/Qwen3-4B-XPress-b16`, revision `fce84732637e1fb70bea3c70ca341131b8d49b61`; its own co-trained backbone and six-pass refiner.
- Four GSM8K and four UltraChat development responses, two frozen contexts each. These are exposed prompts, not fresh final evaluation.
- Proposal comparison: historical released DFlash, newly evaluated co-trained backbone without refinement, and that same backbone with refinement. Common prefix identities, committed anchors and recorded target continuations matched for all sixteen contexts.
- Competitive coverage control: retain the co-trained backbone's ordinary 30-node tree and add the refined chain; give ordinary controls the same resulting node budget (30–43). Labels do not choose the budget or proposals. These trees are reconstructed candidate sets, not executed tree decoders.
- Timing comparison: full refinement versus supplying its exact previously computed proposal. Both still execute the first drafter, vocabulary projection, target verification, acceptance and cache work. Twelve paired repetitions per context, alternating order, following two warmups: 384 timed rounds. The cached proposal is an optimistic nondeployable reference.
- Hardware: A100-SXM4-80GB. Target and refiner use CUDA graphs; the first drafter is not graph captured. Results apply to this execution mix. Model loading, graph setup and repetition resets are outside round timing; refiner graph setup was 16.94 ms.

## What refinement added

Coverage means consecutive positions matching the recorded target continuation, including the common committed anchor. It is not a count of extra generated tokens or measured speedup. Primary percentages give each response equal weight, combining its two contexts before taking the ratio.

| Comparison | Mean coverage change | Descriptive 95% prompt interval |
|---|---:|---:|
| Co-trained backbone chain vs historical released DFlash chain | +38.79% | +0.93% to +73.73% |
| Refined chain vs its own co-trained backbone chain | +23.13% | +6.25% to +44.38% |
| Refined union vs ordinary co-trained tree at the same node count | +4.14% | -2.98% to +12.59% |
| Refined union vs historical released DFlash tree at the same node count | +38.02% | +9.45% to +65.99% |

Refinement improved five contexts, worsened none and tied eleven relative to the co-trained single chain. Total matching positions went from 135 to 154. The equal-response percentage above is not the ratio of these pooled totals.

Against the stronger same-checkpoint, same-node-count tree, the refined union improved three contexts, worsened two and tied eleven: 157 matching positions versus 152. The uncertainty does not establish a reliable incremental advantage over this ordinary tree. The much larger comparison against historical DFlash must not be attributed solely to refinement: the released xPress backbone was trained differently too. This does not show that co-training alone caused the backbone difference.

## What cheaper refinement could save

| Measurement | Observed reduction from supplying the exact proposal | Descriptive 95% prompt interval |
|---|---:|---:|
| Complete-round wall time | 0.406% | -0.020% to +0.961% |
| CUDA-event elapsed time | 0.400% | -0.025% to +0.936% |

Mean context-median wall times were 38.760 ms with refinement and 38.603 ms with its exact proposal supplied. Paired proposals, verification outputs, decisions, next features and retained caches matched exactly. Both paths executed the first drafter once. Thus this intervention measures the runtime consequence of removing the refiner within the specified harness, without requiring a student to reproduce its behavior.

Even this optimistic intervention falls well below the frozen 5% criterion. Its small positive timing estimate overlaps zero; do not describe 0.406% as a proven speedup. Under the illustrative unchanged-other-cost model, a free replacement would still need about 4.84% additional progress **over xPress itself** to reach 5% lower time per position. This is a conditional calculation, not a universal bound, and cannot be combined with reconstructed-tree coverage to claim a competitive generation speedup.

Separate instrumented profiling recorded mean inclusive CUDA regions of approximately 0.634 ms for refinement, 9.522 ms for the first network pass, 14.515 ms for target verification and 4.206 ms for target cache/features. These instrumented intervals are not additive substitutes for uninstrumented paired timing. Optimizing the rest of the execution path or changing workload/hardware can change the removable fraction.

## Interpretation and next decision

1. There is direct evidence that the released refinement computation changes proposals usefully on some of these contexts. We did not identify a general linguistic cause of those corrections or prove that a smaller student can learn them.
2. The same-checkpoint ordinary-tree comparison substantially reduces the apparent benefit. More alternatives must remain a control for a proposed correction method.
3. Faithful cheaper imitation of this already inexpensive refiner has insufficient measured cost headroom here. Another compression fit is not justified by this result.
4. Keep the research question focused on corrections that add useful progress beyond strong alternatives. A further training proposal needs a concrete behavior with competitive value or a different measured cost advantage; neither a new head size nor another loss is selected here. No automatic follow-up is authorized.

## Validation, limitations and resources

All 89 recovered artifacts (16,469,355 bytes) match their remote SHA-256 manifests. Forty-eight chain lengths were independently recounted from tokens, masks and labels. All sixteen first-pass parity checks, paired cache/state comparisons, finite checks and the final unchanged-model-version audit passed. The planned panel and all 384 timing records are complete.

One native-reference numerical disagreement is preserved: `gsm8k-5091`, anchor index 24. Both paired paths accepted the same sixteen positions, but the bonus token differed from the freshly reconstructed causal reference (393 versus 10471). The paired timing intervention remains exact; this is not evidence of native-output-preserving generation. Do not drop this context or claim full decoder fidelity.

The bootstrap intervals describe only eight exposed responses and do not establish generalization. No full responses were generated; no executed ordinary-tree timing, fresh DFlash serving measurement, student training, quality-equivalence claim or new algorithm result is supplied.

Execution through recovery and cleanup took 9.04 minutes. GPU shutdown was verified with zero active tasks; the owned temporary volume is absent. The full approved **$1.50** remains conservatively charged as an accounting allowance pending billing, not as a measured invoice. Unallocated campaign balance is $0.0050666768450267075; no reservations or active paid jobs remain. No retries occurred.

## Evidence and reproduction

- Frozen design: `runner-plan.json`, `local-bundle.json`, `authorization.json`.
- Execution and shutdown: `attempt1/execution.json`.
- Raw evidence and hashes: `attempt1/recovered/`, `attempt1/recovery.json`.
- Complete analysis: `attempt1/value-and-cost-analysis.json`.
- Post-recovery validation: `attempt1/verification.json`.
- Analyzer: `analysis/analyze_xpress_value_and_cost.py` from the repository root. Its CLI refuses to overwrite the existing analysis; reproduce into a separate evidence copy if needed. No GPU is required for reanalysis.

Historical experiments and the original prepared bundle remain preserved.
