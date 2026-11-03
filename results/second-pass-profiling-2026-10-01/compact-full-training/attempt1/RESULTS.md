# Controlled editor attempt interrupted before training

A local DNS failure interrupted observation of the running Modal call. The launcher treated that observation failure as an execution failure and stopped the GPU. The remote worker consequently recorded KeyboardInterrupt during teacher feature extraction. This is an operational failure, not evidence against the learning hypothesis.

All 292 recovered files passed size and SHA256 checks. They contain complete reusable features for 285 prompts: 7904 training and 1216 development anchors, 9120 total. No optimizer update or development evaluation ran. GPU shutdown with zero tasks and deletion of the owned temporary volume are verified.

The observation repair now polls the same submitted call through connection errors and ordinary observation timeouts, while retaining an independent deadline. Actual remote function timeouts remain terminal. Local fault-injection checks passed; a live outage has not validated the repaired path. The original launcher and failure evidence are preserved.

The next substantive step remains the unchanged three-arm learning comparison, reusing these features. No learning benefit, speedup or novelty is established by this attempt.

Budget reconciliation retains $1.50 rather than the original $4 maximum, leaving $6.1351 unallocated. The conservative component bound is $1.3433 including contingency; this is not provider metering or a settled invoice. [Published rates](https://modal.com/pricing) and [calculation](cost-reconciliation.json).
