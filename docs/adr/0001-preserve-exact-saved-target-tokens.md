# Preserve exact saved target tokens

Accepted November 16, 2026: the user rejected creating replacement labels from the native verifier. The original saved greedy continuations remain the reference token sequences for both autoregressive editor experiments; collection and verification must reproduce those tokens exactly before a training bank can pass.

The next work is to diagnose and reconcile source-generation, cached replay and block-verification behavior. Greedy decoding selects the maximum of the computed logits; its absence of sampling does not itself guarantee equality across floating-point execution shapes. That numerical limitation does not relax the user's exact-token requirement. Preserve original files and failure fixtures; do not relabel, accept near-ties, filter mismatches or reinterpret the data as approximate distillation targets.

Changing precision or execution settings is a diagnostic probe until exact saved-token agreement is demonstrated under a documented serving path, with any added cost recorded. If agreement cannot be achieved within the bounded investigation, report that limitation and keep training blocked. The rejected four-fixture label-generation proposal must not be run.
