# Accelerated download succeeded; model placement failed

The full pinned xPress checkpoint downloaded, reconstructed and passed SHA256 verification in 21.98 seconds using the accelerated client. This establishes that the revised transfer route worked on Modal in this attempt.

Staging then failed at `os.link`: the mounted storage rejected the hard link. This was a preparation error. The local file-system test did not establish cloud support for that operation. No GPU call was submitted, no context was evaluated, and no research comparison or training result exists.

The attempt finished recovery and cleanup in 79.74 seconds. All six recovered files (1,947 bytes) passed independent hash checks. The app is stopped with zero tasks, and the owned temporary volume is independently confirmed absent. Shared caches are preserved. Reported compute is $0.0026564, not settled billing; storage/build charges may be separate.

The narrow local repair uses the verified model at its existing path, adds only its small pinned configuration file, and passes a path under the declared volume mount to the GPU worker. It does not link, move or duplicate the weight file. Tests with link operations forbidden pass, including nested checkpoint paths, unchanged weights, exact configuration and path containment. Full released-model loading through that corrected path remains unverified on the GPU.

The scientific worker, prompts, references, models and analysis remain unchanged. This is an operational fix, not evidence for a speculative-decoding improvement. Both previous attempts and their frozen sources are preserved. No retry has launched.

The current goal authorization is $42. An accounting proposal retains $0.25 for each of the two closed, verified CPU-only staging attempts, covering the full staging/shutdown CPU envelope plus $0.20 each for other costs. This would release $1.50 from their prior $2 combined holds. It is an estimate with reserves, not a settled invoice, and has not been applied. User approval is required before applying it and launching another one-attempt, $1-capped comparison. Total authorization would stay $42.

Evidence: `attempt1/execution.json`, `attempt1/recovery.json`, `attempt1/metering.json`, `attempt1/recovered/transfer-second.json`, `attempt1/recovered/staging-failure.json`, `path-repair/local-checks.json`, and `accounting-proposal.json`.
