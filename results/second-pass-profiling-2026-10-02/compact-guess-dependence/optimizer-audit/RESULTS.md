# Explicit-input optimizer audit

## Finding

The explicit first-guess input was neither omitted from the optimizer nor persistently disconnected from the loss. This closes a concrete implementation explanation for the weak input-intervention effect, without identifying the learning failure's cause.

All three original objectives used AdamW over `editor.parameters()` in one ordered group. The input projection has 655,360 weights. Every weight changed from initialization, and every saved Adam second-moment entry is nonzero. The checkpoint records 1,152 optimizer steps for every parameter. Nonzero second moments demonstrate gradient history; weight decay alone cannot explain them. This does not prove nonzero gradients on every step.

| Objective | Initial input-projection norm | Final norm | Weight-change norm | Nonzero second-moment entries |
|---|---:|---:|---:|---:|
| Correct tokens | 9.2373 | 30.4318 | 29.0265 | 655,360 |
| Teacher-correct emphasis | 9.2373 | 29.6840 | 28.2506 | 655,360 |
| Teacher probabilities | 9.2373 | 29.6981 | 28.2530 | 655,360 |

Norms describe parameter movement, not useful influence or suitable update size. The final FP32 model casts exactly to the saved deployed BF16 checkpoint for every parameter in all three arms. The optimizer checkpoints, initial checkpoint and deployment checkpoints were verified against the recorded recovery hashes. Parameter IDs were mapped using the inspected ordered single-group construction and checked tensor shapes. No training was replayed.

The inspected update code refreshes the differentiable folded vocabulary projection within each microbatch, backpropagates the loss before calling the optimizer, and does not detach the editor coefficients. Only the supplied frozen features and teacher probabilities are detached. There is no stale detached editor readout in this path.

## Relation to prior findings

The earlier initialization assessment already showed the zero output projection blocks coefficient gradients only on the first update; that finding is preserved, not presented as new. This audit adds parameter-specific optimizer history for the explicit-token projection in the original three-arm run.

Combined with the training/development input interventions, we can now say: this input was trained and affects outputs, but it has no demonstrated substantial useful prefix contribution in the sampled probability-trained editor. We cannot say the input is inherently useless, the model converged, or the architecture cannot learn a useful conditional rule.

## Research decision

Do not rerun the fit to fix a nonexistent omitted-parameter bug, or diagnose the weak input effect as a persistent zero-gradient initialization failure. Do not treat a requirement to mimic teacher input dependence as the ultimate goal: a cheaper editor that uses first-pass states directly would also satisfy the practical objective if it generalizes and wins on latency. The unresolved outcome is useful learned corrections at affordable cost, with matched controls to distinguish any claimed conditioning mechanism.

Evidence: results.json and analysis/audit_compact_input_optimizer.py. No GPU work, no new weights, no cloud cost. The current input-intervention evidence remains CPU FP32 and uses exposed data.
