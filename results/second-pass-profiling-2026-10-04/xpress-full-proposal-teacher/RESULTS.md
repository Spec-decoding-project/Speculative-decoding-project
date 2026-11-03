# Stronger teacher attempt: stopped during calibration

The matched learning comparison remains untested. Neither planned384-update fit, development evaluation, student fit, or target generation ran. Calibration copies did take optimizer steps and were discarded; they are not trained scientific results.

## What completed

- Preflight authenticated all899 inputs and37 source files. The released checkpoint and native source imported on the reused image.
- A100-SXM4-80GB, PyTorch2.9.1+cu128, Transformers5.10.2. The released BF16 refiner's exact final scores, tokens and input gradients matched literal six-step autograd at training context ultrachat-12450/31.
- Both temporary backbone modes reached optimizer calibration without a recorded gradient/optimizer failure. Their timings and exact update counts were not persisted before the later exception, so do not invent a completed runtime forecast.
- The first-draft equality guard failed while timing the strong control. The two full scientific fits had not begun.

## Preparation defect and limits of the diagnosis

Collection and training explicitly selected SDPA's math backend. The strong-control timing call at launched worker line163 did not surround its first-pass recomputation with the same context. The shared initial_draft function also did not enforce the backend itself. The runtime file's "deterministic SDPA math" entry therefore described an intended policy that was not enforced on every call.

A new local witness test reproduced the policy gap: FlashAttention remained enabled at the unguarded model entry point. This establishes a concrete preparation error. It does not establish which CUDA kernel was selected in the failed call or prove that this alone caused the changed tokens. The failed attempt did not retain the differing score vectors, original calibration hidden states, selected failing context, or cumulative timing arrays. That evidence gap is also a preparation failure. No tolerance is relaxed and the original parity failure remains a failure.

Local repair now enforces SDPA math inside both initial_draft and second_pass, independently of caller context. The caller's context is restored afterward. The witness test failed before this repair and passed after it; the21-test model/panel/report/recovery suite passed in35.34seconds. This validates backend selection policy, not released-weight CUDA replay or learning benefit.

The repaired worker additionally persists calibration states/scores/tokens and intermediate timing arrays, and captures actual mismatch outputs. The recovery helper now handles Modal's FileNotFoundError translation for absent manifests. Normal manifest recovery worked in this attempt; this latter issue did not cause its scientific stop. These are unlaunched repairs. All37 launched source files, the original plan and bundle are preserved under attempt1/frozen.

## Scope of the research evidence

The new local opportunity calculation remains valid separately: in234 of256 historical contexts the first guess was correct. A privileged perfect suffix under the fixed proposal policy covers3,801 positions versus2,696 for conditional64. It supports structural room for improvement, not learnability or compression. Its equal-response percentage is different from a pooled-token percentage; see opportunity.json.

This attempt adds a successful CUDA head-gradient check and a failed execution-compatibility check. It supplies no evidence for or against useful learned full-proposal corrections. It does not prove the second-pass hypothesis, disprove small editors, or justify another architecture/loss change.

The planned comparison specifically tests completed-draft feedback versus first-guess feedback at matched training exposure. It is not a comprehensive comparison with ordinary pooled single-pass continued training. Even a later positive result needs stronger optimized cost comparisons, a learned compact replacement, and fresh actual generation validation before a method or speedup claim.

## Resources and next action

Controller99968 exited successfully after recovering the remote failure. App ap-TmmQL20DsVnY79SjM5fNM1 is stopped with zero tasks; owned volume vo-2oe7YsUDix7qpQsOFZxVLJ is deleted. Six evidence files totaling1,960bytes were recovered and hash-verified. Total local elapsed287.99seconds, including staging/recovery/cleanup; provider app lifecycle254.16seconds.

Provider-reported app compute is0.08814409USD, not a settled invoice and not a complete storage bill. The full3.50USD cap remains held under the existing conservative accounting. No active reservation/job remains; no funds were reclaimed. The existing30USD goal allowance has0USD unallocated. No retry or further spending occurred.

Keep the research objective active and preserve the comparison. The local policy defect is repaired, but a further released-weight CUDA attempt would need available explicitly authorized funding. Do not claim the paired teacher comparison is launch-ready or quietly shorten its dose to fit a different budget. No new scientific recipe is justified by this operational failure.

Evidence: attempt1/execution.json, recovery.json, metering.json, recovered/terminal.json, recovered/gradient-check.json, attempt1/analysis.json, opportunity.json and repair-checks.json.
