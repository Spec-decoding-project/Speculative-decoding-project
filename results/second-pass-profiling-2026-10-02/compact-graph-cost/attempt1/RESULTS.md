# Capturing the frozen compact editor reduces its execution overhead

Completed 2 October 2026 local time. No training occurred. This is an implementation cost result, not a learned-method or full-generation speedup.

## Matched results

Four historical contexts from two prompts, 20 paired repetitions per context and four conditions: 320 timed rounds. All conditions share the same eager first drafting pass and captured target verification. The supplied-proposal condition is a nondeployable cost control. The original and revised proposals are verified together in the editor conditions.

| Context | Original round ms | Eager editor union ms | Captured editor union ms | Supplied proposal union ms | Original / union progress |
|---|---:|---:|---:|---:|---:|
| GSM8K 1330, anchor 8 | 32.764 | 35.238 | 33.203 | 32.975 | 3 / 3 |
| GSM8K 1330, anchor 24 | 33.743 | 36.530 | 34.165 | 33.942 | 16 / 16 |
| UltraChat 12035, anchor 8 | 33.640 | 36.729 | 34.219 | 34.031 | 3 / 3 |
| UltraChat 12035, anchor 24 | 35.058 | 37.361 | 35.061 | 35.031 | 7 / 8 |

Displayed round times are per-condition medians. Median paired correction-production costs, relative to supplied proposals, fell from 2.222–2.821 ms to 0.117–0.280 ms. Paired medians need not equal differences between the displayed medians. Added complete-round time versus original fell from 6.57–9.18% to 0.007–1.722%. Near-zero overhead on one context is a timing observation, not zero intrinsic verification cost.

The graph includes embedding lookup, the existing 3,024,384-parameter editor, folded vocabulary correction and token selection. Copies of each current first-pass state, scores and first guess into persistent buffers are counted. One graph is reused across the four contexts; no target answers or saved revised proposals enter it.

## Correctness and setup

Captured and eager paths matched exact first-pass states/scores, revised scores, token proposals, verifier logits, outcomes and all 36 layers of compacted key/value cache. Repeated outputs were stable. The editor checkpoint remained identical. These checks cover the tested contexts and runtime, not every input or deployment shape.

Graph warmup/capture took 22.517 ms. Folding the frozen vocabulary readout took 289.727 ms and is required by both measured editor paths. Graph capture alone amortizes against the eager editor after about 10–11 rounds here. That calculation does not waive readout setup when comparing with original DFlash. Whole-request measurements must charge both appropriately.

## Research implication

The earlier roughly 8–9% overhead was not an intrinsic cost floor. Better learned corrections now have a more plausible route to paying for execution. Current learned coverage is unchanged, and remains weak. Four cost contexts cannot be combined with the separate 16-response teacher-opportunity pilot as if measured jointly. That pilot allowed only about 1.383% uniform round overhead for a modeled 5% saving under perfect teacher capture; the present 0–1.7% measurements leave that question open. A student can also learn corrections outside the teacher's saved outputs, so this is not a universal bound.

CUDA graphs are established engineering, not the proposed research contribution. No population interval, fresh generalization, or complete-generation speedup is established. The next consequential uncertainty is learning useful transferable corrections, with data collection, retention and matched controls specified before launch.

## Recovery and accounting

Execution through cleanup took 199.856 seconds. All 27 artifacts (979,141 bytes) were authenticated and recovered. App `ap-jIIicAXwrtwrseHZ6txbaZ` is stopped with zero tasks; the owned temporary volume was deleted. Provider-reported compute is $0.07830684, not settled billing. The full $2.50 allowance remains retained.

Evidence: `analysis.json`, `recovery.json`, `execution.json`, `provider-closure.json`, and `recovered/`. Analyzer: `analysis/analyze_compact_graph_cost.py`. Historical source modules and previous findings are preserved.
