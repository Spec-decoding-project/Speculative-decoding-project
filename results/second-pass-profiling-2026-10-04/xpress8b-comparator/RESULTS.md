# xPress8B comparison: staging cutoff, no scientific measurement

## Outcome

The one authorized attempt stopped while downloading the pinned xPress checkpoint. The CPU staging deadline raised `TimeoutError` in the download loop. The cached target checks and first-drafter download/hash verification had completed; xPress authentication had not. No GPU call was submitted, no context was evaluated and no training occurred.

This supplies no evidence for or against the second-pass teacher's advantage over xPress. The completed D2SD findings remain unchanged, and the goal of learning a cheaper second-pass replacement remains active and unmet.

## What failed operationally

The 5–10 minute estimate was too optimistic. The attempt lasted 884.56 seconds through recovery and cleanup (14m45s). The fixed staging guard stopped it as designed. No retry or deadline extension occurred.

The recovered record identifies the transfer phase, but does not record bytes received or interval throughput. Therefore it does not establish why this transfer was slow or support an accurate revised download estimate. This was not a GPU-capacity failure, model incompatibility finding, or negative learning result.

Before another launch, improve the transfer diagnostics and establish checkpoint staging readiness separately from the scientific comparison. Preserve the frozen scientific comparison; do not change the research hypothesis because a download timed out. Do not request another identical attempt without addressing staging. No new attempt or spending is authorized.

## Evidence and cost

All three recovered files (486 bytes) passed an independent hash check. All 56 frozen source hashes also passed. Modal independently reports the app stopped with zero active tasks. The exact owned temporary volume is absent; the shared target cache is preserved.

Provider-reported compute is $0.03806402 (about four cents), not a settled invoice; storage may be separate and metering may lag. The approved $1 ceiling remains held conservatively, not reported as actual spend. Goal authorization is $41, allocated $41 including unsettled allowances, with $0 unallocated. No reservation or active paid job remains.

App: `ap-pzNTDhrPLfqTioG322eAgg`. Evidence: `attempt1/execution.json`, `attempt1/recovered/staging-failure.json`, `attempt1/recovered/authenticated-first.json`, `attempt1/recovery.json`, `attempt1/metering.json`, and `closure-audit.json`. The original frozen plan and sources remain under `frozen/`.
