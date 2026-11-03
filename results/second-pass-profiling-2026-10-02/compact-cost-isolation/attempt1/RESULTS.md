# Trained editor cost isolation: correction computation is a material cost

Completed in 4.19 minutes through recovery and cleanup. All four contexts and240timed rounds completed; no training or retries. One saved probability-trained compact checkpoint on an A100-SXM4-80GB, same original first pass and same graphed target verification in all conditions.

## Result

Live editor and supplied-proposal control produced identical proposals, full verifier winner vectors, accepted/bonus tokens and all36layers of compacted cache tensors on each context. Supplying the corrections from device memory bypasses only their computation; first-pass drafting remains. The supplied-proposal control is nondeployable.

| Context | Original round | Trained editor union | Same proposals supplied | Original / union progress | Median paired correction-production cost |
|---|---:|---:|---:|---:|---:|
| Math anchor8 | 34.752ms | 37.389ms | 34.655ms | 3 / 3 | 2.470ms |
| Math anchor24 | 35.075ms | 37.731ms | 35.405ms | 16 / 16 | 2.438ms |
| Chat anchor8 | 35.625ms | 38.974ms | 36.563ms | 3 / 3 | 2.487ms |
| Chat anchor24 | 36.836ms | 39.764ms | 36.963ms | 7 / 8 | 2.485ms |

Round columns are medians; the final column is the median of paired repetition differences, so it need not equal the difference between the displayed medians. The live editor is more expensive than replay in79of80paired repetitions. Twenty repetitions per context describe timing stability, not80independent workloads.

Live added round overhead is7.57-9.40%. With the same revised proposals supplied, observed overhead is-0.28%to2.63%. The slightly negative value is a timing observation, not proof that extra verification has negative intrinsic cost. The difference includes embedding lookup, editor computation, extra vocabulary scoring, selection and differences in replay bookkeeping. It does not isolate the neural trunk or a particular kernel.

One of four contexts gains one accepted position. Its live cost per position falls5.55%, while the other three worsen7.57-9.40%. Grouping the two contexts per prompt, math cost per position worsens7.58%and chat improves1.22%. Two exposed prompts do not support a general performance claim. No complete response was generated.

## What this changes

The previous5-6%untrained overhead was not an immutable downstream verification floor. Here, producing corrections consumes roughly2.45ms, while using them without producing them is closer to the original round cost. Optimizing the correction-production path is therefore technically relevant; shrinking the model or implementing a vocabulary shortlist is not yet the uniquely justified optimization.

The immediate distinction worth measuring is launch/intermediate-tensor overhead versus device computation. A captured frozen editor path compared with eager execution, with identical proposals and equally optimized shared work, would test an established implementation technique. This is not a claim of novelty and does not establish that graph replay preserves predictions or saves enough time here. No follow-up has been launched.

Cost improvement alone still cannot rescue the current full-panel3.46%progress result for the5%goal: its hypothetical free-computation saving remains3.25%. Better learned proposals remain necessary. This diagnostic clarifies the cost allowance a better editor must fit within; it is not a reason to replace research on learning with kernel work.

## Checks and scope

All first-pass capture checks, finite/committed causal decisions, next-cache checks, EOS guards, replay parity and repeat consistency passed. All36layers matched between the live and replayed compact union in each context. Loaded and final compact checkpoint tensors match exactly; target/drafter parameter versions remain unchanged. The recovered evidence includes no failed causal checks. This does not establish whole-generation/native trajectory fidelity.

All22artifacts,732574bytes, were recovered and independently hash-verified. Same-context repeated execution occurs on A100SXM4, whereas the prior untrained diagnostic used A100PCIe. Do not interpret cross-run timing differences as a model effect. Fixture construction, initial cache clones and proposal precomputation are outside round timing equally where applicable. Fold preparation is separately retained. Shared draft work remains eager and target verification graphed; no production-serving superiority claim.

## Accounting

App ap-d0P2YI1aa1mbAqssomiAaQ is stopped with zero active tasks; the owned temporary volume is verified absent. Retain the full$1.25attempt allowance, not a settled bill. Unallocated campaign allowance is$0.98506668. No new budget authorization, retry or follow-up spending occurred.

Evidence: analysis.json, recovery.json, execution.json, recovered/summary.json and paired replay-parity artifacts. Pricing reference for the bound: https://modal.com/pricing. Earlier protocol and preflight remain preserved.
