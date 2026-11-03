# Conditional proposals survive actual target verification and full round costs

## Decision

The frozen xPress conditional construction is a useful reference for further research. On this fixed development panel it improves actual accepted progress enough to pay for its additional round work. This supports an actual-generation comparison; it does not justify claiming a novel algorithm, an end-to-end speedup, or starting another compression fit.

The next unresolved question is whether the benefit persists when each method follows its own consecutive decoding rounds, including request setup, changing draft/target caches, stopping and all runtime overhead. Use the released chain and the faster shared-score CPU-heap tree as controls; keep ordinary-tree evidence. Do not substitute the slower shared GPU implementation as the headline control.

## Complete result

All 32 contexts from 16 exposed development responses completed. The sample was fixed by response-ID hashes, eight responses per domain, two predetermined anchors each; outcomes did not select it. Six alternating-order measured repetitions followed two warmups. These are not fresh held-out prompts.

| Construction | Actual accepted positions | Mean complete-round time | Pooled ms per accepted position |
|---|---:|---:|---:|
| Released six-pass chain | 223 | 34.489 ms | 4.949 |
| Ordinary unrefined tree | 228 | 36.526 ms | 5.126 |
| Shared scores, CPU heap | 248 | 36.974 ms | 4.771 |
| Shared scores, GPU tensor builder | 248 | 37.521 ms | 4.841 |
| Conditional depth-wise width-eight tree | 303 | 37.585 ms | 3.969 |

“Accepted positions” includes the known root processed this round; the pending bonus token is the next round's root and is not counted twice. Results use actual target decisions, not the old saved continuation as a label oracle.

The primary paired description computes total median round time divided by total accepted positions within each response, then averages relative savings equally over responses. Its values differ from ratios of pooled totals above.

| Conditional versus | Equal-response round-cost saving | Descriptive 95% interval | GSM8K / UltraChat |
|---|---:|---:|---:|
| Released chain | 19.73% | 11.44–28.62% | 17.48% / 21.97% |
| Ordinary tree | 16.36% | 6.29–26.25% | 25.96% / 6.75% |
| Shared CPU heap | **13.32%** | **5.43–21.99%** | **17.64% / 9.00%** |
| Shared GPU tensor builder | 14.58% | 6.77–23.13% | 19.04% / 10.12% |

Intervals use 10,000 response-cluster bootstrap resamples stratified by domain. They describe this small exposed panel, not population generalization or independent evidence from every token/timing repetition. Against the shared CPU heap, conditional construction reduced cost per accepted position on 9/16 responses; gains on some responses outweigh small added costs where coverage ties.

## Useful changes and damage

Against the strongest shared-score tree, conditional construction gained **56 positions in 11 contexts**, lost **one position in one context**, and tied in 20. No full-horizon shared prefix was damaged. The net gain is 55/248 = 22.18% in pooled progress; the equal-response gain is 24.03% [9.33%, 42.32%].

All 32 conditional accepted sequences agreed with the ordinary/shared controls over their common emitted prefix. The extra progress is therefore not an artifact of following different continuations within these comparisons. Shared CPU and GPU implementations produced identical accepted sequences and bonus tokens across the whole panel, while the CPU heap was about 0.547 ms cheaper per round.

The mechanism supported here is practical: reuse the same first-draft features and base probabilities, but score a branch's continuation using that branch's token history instead of reusing one chain's conditional scores everywhere. Target weights, released draft/head weights and tree query budget are unchanged between shared and conditional trees; no training occurred. This is evidence for the complete construction. It does not isolate a pure single-variable effect: search width, number of score evaluations and initialization schedule also differ. Earlier clamping/history studies supply more controlled mechanistic evidence; this test supplies actual verification and cost evidence.

## Correctness and numerical limitation

Eight CPU tests used a small real Qwen3 target before launch. On GPU, every context passed native first-draft parity. All 160 measured method/context paths passed visibility, positions, finite-result and all-36-layer cache-compaction checks. Repeated rounds were stable. An independent analyzer reconstructed every accepted path directly from saved paths and actual verifier choices; exact analysis regeneration passed.

The conditional, ordinary and both shared constructions passed all 32 recorded comparisons of accepted-path winners against freshly evaluated causal prefixes, including the next token after cache compaction.

The released chain had **one numerical disagreement**, at `ultrachat-12644`, anchor 440. A verifier tie (22.375 versus 22.375) differed from the fresh causal evaluation (22.375 versus 22.250). Its emitted prefix differs from the conditional path there. The comparison remains in every table; no agreeing-only result is substituted. Thus the chain comparison cannot be described as an output-preserving speedup. Passing the conditional checks on these contexts is not a guarantee about future full generation.

Saved top-two rankings sometimes order tied candidates differently from `argmax` (112 node observations). The analyzer uses the separately saved actual `argmax` posterior for acceptance, verifies any top-two disagreement is a tie, and does not reinterpret ranking order as the decoder's decision.

## Timing scope and remaining limits

Measured rounds include first-draft computation and vocabulary projection, proposal creation and transfers, mask/position preparation, target verification, acceptance, target-feature extraction and accepted-cache compaction. Target and proposal work use CUDA graphs. The first drafter is equally uncaptured in all arms. Prefix cache reset, initial prefill, graph setup, loading, auditing and artifact saving are outside repeated round timing and cannot be claimed free in a full serving workload.

This is a complete reference-round comparison, not a fully optimized production benchmark. No full response was generated, no request-level latency was measured, and no output-quality study was performed. The conditional width-eight method is not an official reproduction of DARTree with its own trained head. Conditional/depth-wise construction, candidate restriction and graph replay are established techniques; novelty remains unestablished. See the pinned DARTree/SpecExec prior-art evidence in `../xpress-exact-tree-local/sources/`.

## Execution and budget

NVIDIA A100-SXM4-80GB, PyTorch 2.9.1+cu128. Worker completed in 276.46 seconds; staging through recovery/cleanup took 478.13 seconds. Peak allocated CUDA memory was 10.96 GB across the comparison, not a per-method memory benchmark. All 73 artifacts (3,347,191 bytes) were recovered and hash-verified. App `ap-uvKiy8KzDor2mx5bVoZR4Y` was independently confirmed stopped with zero active tasks; its owned volume is absent. No retry or training.

The full **$2 cap** remains conservatively charged pending settlement. **$8.50 remains** from the existing $30 authorization. No new spending is allocated. The research goal remains active and unmet.

## Evidence

- [Frozen protocol](PROTOCOL.md), [plan](runner-plan.json), [bundle hashes](local-bundle.json), [local checks](local-checks.json).
- [Analysis](analysis.json), [analysis contract](analysis-contract.json), [cross-method prefix check](accepted-prefix-crosscheck.json).
- [Worker result](attempt1/recovered/results.json), [runtime](attempt1/recovered/runtime.json), [execution and cleanup](attempt1/execution.json), [recovery hashes](attempt1/recovery.json), [independent shutdown](attempt1/independent-shutdown.json).
- Reproduce: `.venv-profile-launch/Scripts/python.exe -X utf8 analysis/analyze_xpress_conditional_round.py --verify` from the project root. This reads saved evidence only and launches no GPU work.
