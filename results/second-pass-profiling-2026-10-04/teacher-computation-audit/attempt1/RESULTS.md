# Context-routing diagnostic: stopped at instrumentation fidelity gate

## Outcome

One approved attempt ran and stopped as specified. Six of 256 contexts completed; the seventh failed the native-versus-instrumented token agreement gate. None of the fourteen contexts where conditioning uniquely repairs the draft had been reached. The planned question about useful feedback-dependent repairs remains unanswered. No training, generation or retry occurred.

This is a measurement-compatibility failure, not evidence against compression or feedback-dependent contextual computation. The native controls reproduced their saved historical proposals before the failure. Replacing native SDPA with explicit attention to expose/intervene on scores changed a control prediction even before an intervention. The local small-model checks did not establish released-weight compatibility; that limitation was included in the protocol, and the real-weight gate caught it.

## What changed in the failing control

Failure: `gsm8k-1386`, anchor index 20, 459 committed context rows. The conditioned teacher disagreed at future draft position 10:

| Score | Native SDPA | Instrumented explicit attention |
|---|---:|---:|
| Token 3070 | 21.375 | 21.375 |
| Token 1414 | 21.250 | 21.375 |
| Selected token | 3070 | 1414 |

The instrumented scores contain an exact top tie; normal argmax selects the other token. Both conditioned and unconditioned score vectors show maximum absolute differences of 0.25, although unconditioned output tokens agree. Same-trace neutral replay remains exactly equal to the instrumented conditioned scores. This locates the observed mismatch between native and instrumented computation, rather than the neutral replacement itself. It does not isolate the responsible floating-point operations or establish harmlessness.

The changed token comes after the first mismatch with the recorded target. Both native and instrumented conditioned matching prefixes remain seven root-inclusive positions. Thus this particular numerical disagreement did not change the prefix metric. The frozen gate required complete proposal-token agreement and was stricter than prefix agreement; it was not relaxed after the result. A future revision would require a justified validation design, not pretending this attempt passed.

## Preserved partial observations

All six completed contexts were from two GSM8K responses; none was from UltraChat. One was a teacher repair already available without feedback and remained repaired under each intervention. Five were cases where conditioning did not extend the original.

In one of those latter cases (`gsm8k-1386`, anchor 12), the normal conditioned prefix was six positions versus the original/unconditioned ten. Freezing context selection restored ten; freezing context mass retained six; replacing both gave seven. This is a single recorded example of changing a harmful computation. It is not evidence about the useful feedback-dependent repairs or the full panel, and does not imply a deployable selection rule.

Among completed records, route replacement changed total context attention mass by at most 4.17e-7 before the probability cast. After BF16 casting, the maximum change was 0.00415, about 0.415 percentage points. The mathematical control therefore works to floating-point precision before casting, but the actual low-precision operation is not a perfectly isolated mass-preserving intervention. This limit must remain visible in any future causal interpretation.

The independent analyzer correctly refuses a full-panel summary. Individual token-based prefix counts were independently recounted. The primary result is incomplete, not a negative mechanism verdict.

## Execution, evidence and cost

- GPU: A100 80GB PCIe; Torch 2.9.1+cu128. Earlier source run used A100 SXM4; native saved proposals nevertheless reproduced on the evaluated contexts. No cross-hardware latency comparison was attempted.
- Initial deployed teacher digest matched the pinned checkpoint. Final whole-model weight audit was not reached, and must not be claimed complete.
- All ten recovered artifacts, 55,447,338 bytes, matched remote and local SHA-256 values. They include the failing inputs, complete available failing-probe outputs and paired attention traces. Frozen code hashes remain unchanged.
- App `ap-k92lN5MrKvpHjasJxIE6h0` is stopped with zero active tasks. Owned temporary volume deletion is verified. No shared archive was removed.
- Elapsed through recovery and cleanup: **3.68 minutes**.
- Provider-reported app compute at the first query: **$0.05375276**. It can lag and is not a settled all-in invoice. Conservatively retaining the approved **$1.50** allowance is budget accounting, not a claim that $1.50 was spent. No automatic refund or follow-up authorization is inferred. Campaign unallocated balance remains $0.0050666768450267075.

## Next decision

Do not launch another paid attempt by default or train an editor from these observations. The saved failure supports local work on an instrumentation method/validation rule that can distinguish the intended computational change from backend numerical differences. The existing input/score snapshot permits that design work without another collection run. No remedy has been validated yet.

Evidence: `execution.json`, `recovery.json`, `recovered/failure-snapshot.pt`, `failure-score-audit.json`, `verification.json`, `metering.json`, and the six records in `recovered/predictions.json`. Historical findings and the originally frozen protocol remain preserved.
