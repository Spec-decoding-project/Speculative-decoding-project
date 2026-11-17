# Version the batch-invariant canonical target

Update: [ADR 0003](0003-authorize-one-block-memorization.md) supersedes the
complete-bank prerequisite for the explicitly authorized one-block fit and
held-out evaluation. The invariant reference and immutable originals remain.

Accepted November 17, 2026, after the user reviewed the invariant-kernel result
and selected “Adopt invariant vLLM and the small versioned pilot (Recommended)”.

For new versioned experiments, use the pinned batch-invariant vLLM target after
supported-configuration validation. This supersedes ADR 0001's requirement that
new labels equal the old generator's tokens. It preserves ADR 0001's requirement
that original response files remain immutable and that mismatches cannot be
filtered or hidden.

The observed distinction is concrete: the invariant target consistently selects
`Employee` (16984) where the original generator selects `employee` (12946). Both
are reproducible in their respective computations. Adopting a new numerical
target definition does not declare the original generator incorrect.

The authorized next step is a separate four-prompt response pilot, after repeated
validation with chunked prefill enabled in every engine. Both genuine greedy
generation and DFlash verification must agree exactly on emitted token IDs and
corresponding full-vocabulary logit hashes. Preserve the original prompt/source
IDs, splits, model/tokenizer hashes, old record hashes and all new runtime/data
provenance. A mismatch or capped pilot response prevents a passing pilot.

This decision does not authorize 50k regeneration, editor training or a feature-
bank pass. A complete, independently audited 128-repair/16-preservation bank
under the canonical contract must still precede fitting. The guarantee is scoped
to the defined model/runtime/hardware/layout contract; finite tests cannot prove
all future inputs or arbitrary hardware/version changes.
