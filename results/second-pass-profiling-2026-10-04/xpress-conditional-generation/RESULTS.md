# Full generation: promising work reduction, failed fidelity and incomplete panel

## Decision

Do not claim an output-preserving generation speedup or pay merely to finish the two missing prompts. The reference construction generated stable outputs and reduced verifier work, but the frozen success screen cannot pass: several outputs differ from native greedy generation, and only10/12 assigned prompts completed. The previous fixed-context13.32% cost-per-position finding remains intact and limited to that scope. No new training, tolerance change, retry or follow-up allocation.

The next necessary question is a validation/serving prerequisite: can the conditional construction and its controls share a target execution path that preserves the required greedy reference across query shapes, without erasing the benefit? Use existing batch-invariant implementation research before another numerical-debugging run. Do not assume every discrepancy is numerical or implement an expensive new target engine by default. The saved outputs establish disagreement, not its operator-level cause.

## Completed observations

Six GSM8K and four UltraChat prompts completed every assigned condition and both timing repeats,60 measured generations total. The last two UltraChat prompts were not started before the fixed recovery-reserve guard. No full12-prompt average, bootstrap interval, setup-amortized12-request result or success claim is reported.

Every table row below is included; differences in output content/length make these descriptive observations, not equivalent-work speedups. Seconds are medians of two repeated whole requests with models loaded. Final GPU output materialization is included; conversion of the final token tensor into a host JSON list and artifact I/O are outside the request timer.

| Prompt | Released chain: seconds / tokens / calls | Shared CPU heap: seconds / tokens / calls | Conditional tree: seconds / tokens / calls |
|---|---:|---:|---:|
| gsm8k-798 |0.640 /284 /21|0.683 /284 /21|0.609 /284 /18|
| gsm8k-126 |0.346 /114 /11|0.333 /116 /10|0.341 /116 /9|
| gsm8k-790 |0.715 /222 /23|0.638 /222 /20|0.575 /222 /17|
| gsm8k-1897 |0.416 /148 /13|0.375 /148 /11|0.345 /148 /10|
| gsm8k-1164 |0.815 /292 /27|0.759 /289 /24|0.685 /283 /21|
| gsm8k-2351 |0.451 /147 /14|0.433 /144 /13|0.426 /147 /12|
| ultrachat-12506 |4.685 /512 /164|3.851 /512 /128|3.428 /512 /110|
| ultrachat-12253 |4.129 /478 /145|3.560 /509 /118|3.685 /512 /118|
| ultrachat-12591 |4.443 /512 /157|3.793 /512 /126|3.471 /512 /111|
| ultrachat-12573 |5.234 /512 /184|4.425 /512 /147|3.848 /512 /123|

On these completed requests, conditional used549 verifier calls, shared618, released759. Output totals were3248,3248,3221tokens respectively; equal total token counts do not imply equal continuations. Conditional hit the512token cap on four prompts; both other conditions on three. Conditional request time was lower than shared on eight completed prompts and higher on two. Native-output agreement was4/10 conditional,4/10 shared and5/10 released. Both repetitions were stable for every completed condition.

Native-reference runs consumed103.03seconds; measured trials116.28seconds; policy initialization/warmup5.50seconds; model loading71.34seconds. Other elapsed work includes the two published-generator checks, imports, evidence serialization/commits and framework overhead; those components were not separately timed. The original estimate did not leave enough room for this total overhead. Do not infer that the model computations alone caused the cutoff.

## What the wrapper check actually found

On the first GSM8K and UltraChat prompts, the optimized released-chain wrapper produced exactly the same final tokens as the pinned published generator. Raw acceptance-vector equality failed in both, for a fully explained terminal reason:

- GSM8K: all21 shared rounds match; the published generator then ran one more round to consume an already-known EOS. Our wrapper returned that known token directly.
- UltraChat: all164 shared rounds match; our wrapper returned the already-known512th token, while the published generator ran another round and truncated its output to512.

The original raw acceptance-check failures are preserved. The posthoc [terminal accounting audit](terminal-accounting-audit.json) establishes agreement of every nonterminal round and the final output in these two checks. It does not prove full general compatibility, change the frozen success result, or explain the other native disagreements. The terminal fast path applies to all three tested conditions, so it is not unique to the conditional method.

The published generator itself differs from native output on `ultrachat-12506`, matching the custom chain output. This rules out the custom wrapper as the sole source of that particular disagreement. The earlier fixed-round study directly observed a score-tie sensitivity. Those facts support investigating numerical consistency, but do not establish a cause for every present divergence. This run did not record complete first-divergence score/cache tensors; do not claim it did.

## Scope and limitations

Fourteen CPU tests used real small target/drafter models across consecutive rounds, request resets, token caps, EOS and cache/feature continuity. Source/code freeze preceded selection of12inputs excluded from212 named historical input/plan manifests. No oversized manifest was skipped. This clearance is not exhaustive semantic deduplication or proof of absence from pretraining.

Weights stayed frozen by construction, but the worker's final whole-panel version check was not reached after the deadline exception; do not mark it as having passed. All60 saved records independently pass output-prefix, length/stopping, acceptance-count, finite-positive time and stable-repeat checks. Exact analysis regeneration passed. Whole-model operator correctness on all differing trajectories is not established by those checks.

Target/proposal graphs were used for all policies; the first drafter remained uncaptured. Setup/warmup cost was4.404seconds released,0.603seconds shared and0.490seconds conditional, measured in that order with shared runtime/kernel caches. These are order-dependent observations, not a robust cold-start comparison. The missing full panel prevents the planned12-request setup-charged comparison.

No quality-equivalence conclusion follows from final-token disagreement. No agreeing-only performance aggregate is substituted. Conditional/depth-wise tree construction remains established prior art; adapting the released xPress head is not yet a demonstrated original algorithmic contribution.

## Budget and resources

The original controller49589 exited normally after reporting the guarded failure. App `ap-Qh4woEtQyti81MDoLLEL8G` was independently confirmed stopped with zero active tasks. Its owned volume is absent. Nine artifacts,387358bytes, were recovered and verified. Staging through cleanup took599.30seconds. No retry or extension.

The full **$3 cap** is retained conservatively pending settlement, leaving **$5.50** of the existing allowance. The research goal remains active and unmet.

## Evidence

- [Protocol](PROTOCOL.md), [frozen input/plan](runner-plan.json), [prompt clearance](prompt-clearance.json), [local tests](local-checks.json), [source hashes](local-bundle.json).
- [Analysis](analysis.json), [timed records](attempt1/recovered/trials.json), [native/published references](attempt1/recovered/references.json), [setup](attempt1/recovered/setup.json), [terminal error](attempt1/recovered/terminal.json).
- [Execution/cleanup](attempt1/execution.json), [recovery hashes](attempt1/recovery.json), [independent shutdown](attempt1/independent-shutdown.json).
- Reproduce analysis only: `.venv-profile-launch/Scripts/python.exe -X utf8 analysis/analyze_xpress_conditional_generation.py --verify`.
