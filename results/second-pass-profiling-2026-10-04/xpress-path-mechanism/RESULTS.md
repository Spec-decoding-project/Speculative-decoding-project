# Known-path refinement: local mechanism and support check

**Decision:** do not run another teacher or editor fit. A released refiner can evaluate concrete draft histories without another backbone pass. The next useful test is whether a single batch of such evaluations supplies useful new children, rather than merely reranking an unchanged set. This is an unvalidated candidate execution method, not an established contribution or speedup. No paid work occurred; the new allowance remains $24.50.

## What changed our decision

The completed teacher-support experiment showed a feedback benefit but no improvement over the initial teacher. This check therefore asks about using an already inexpensive learned refiner, not compressing that teacher again. The earlier xPress cost-removal study also excludes a 5% improvement from making its existing head free on that setup. Any new method must improve proposals, not just shrink the same computation.

The candidate is **one batch of conditional expansion from already proposed paths**:

1. The first draft supplies a bounded tree of possible continuations. An existing refined chain can be included, with its cost charged.
2. Evaluate the released causal refiner at every existing parent, conditioning each evaluation on that parent's actual token history. These histories are already known, so different depths can be evaluated together.
3. Add predicted children, select a fixed-size prefix-closed verification tree, and let the unchanged target verify it.

This is not unlimited autoregressive expansion in one pass. A newly added child's children require another conditional evaluation unless that path was already present. The hypothesis is that broad, one-step conditional repair earns more progress per unit time than serial expansion; it may fail because the initial tree lacks the needed parents, scores misrank useful paths, or extra scoring consumes the benefit.

## Algebra and executed checks

For fixed first-pass hidden features, xPress's projected input at slot j is a function of those features and the token preceding j. Its latent at slot k is

`m[k,c] = sum(j <= k) L[c,k,j] * projected_input[j,c]`

followed by a positionwise MLP and vocabulary readout. There is no recurrence through a previous output latent. If a path's token identities are supplied, its next-position latent can therefore be computed directly. This follows from the pinned author's implementation, not from assuming a small model can learn a new mapping.

The reference implementation projects each unique (slot, preceding-token) pair once, gathers the appropriate ancestor rows, applies the requested mixer row and evaluates only the requested MLP/readout row. It is CPU reference code, not an optimized GPU kernel. Shape validation and causal-mixer validation deliberately remain in this reference; do not time it as a serving implementation.

- FP64 synthetic comparison against the author's full-block function: maximum absolute error 2.22e-16. Future-input isolation passed. Permuting independent paths differed by at most 2.22e-16, within the declared 1e-12 mathematical-check tolerance. The first exact permutation assertion failed on this rounding difference; this is not bitwise backend equivalence.
- Existing **8B** released head, weights promoted to FP32 on CPU: 192 queries from all 16 saved traces. Full-block and requested-row evaluations chose identical tokens in all 192. Floating-point score differences remain in the raw report; no BF16/CUDA parity is established.
- Changing an earlier guessed token while keeping the immediate predecessor fixed changed the winning prediction in 2 of 64 prescribed comparisons. This confirms that the released function can use longer history on these inputs. Without target labels for those interventions, it says nothing about whether either prediction improved.
- No training, target generation, correctness-of-generation test or runtime comparison ran.

Evidence: [local-check.json](local-check.json). Code: `experiments/xpress_path_refinement.py`, `analysis/check_xpress_path_refinement.py`.

## Does the proposed support contain enough opportunity?

Separately, the authenticated **4B** value-and-cost panel has eight reused development responses and 16 contexts. Do not combine its model results with the 8B numerical checks as one experiment.

We reconstructed an ordinary 128-node tree from each co-trained backbone's saved top-16 probabilities, then included the existing six-pass refined chain. Verification retains the historical matched budget, 30–43 nodes. Target answers are used only to compute these optimistic ceilings; the entire correct path fits within that budget if an oracle knew it.

| Candidate set / reference | Total matching positions |
|---|---:|
| Ordinary tree at the historical verification budget | 152 |
| Existing refined union at that same budget | 157 |
| Better of these two per context, answer-informed | 160 |
| Perfect pruning of the prebuilt larger candidate pool | 165 |
| Perfect pruning plus a perfect new child at every existing parent | 174 |

Pure reranking of this fixed pool offers 7.97% mean equal-response, equal-round-cost modeled savings over the existing refined union, with a descriptive interval of 2.87–13.71%. Even perfect selection leaves only 3.23% uniform extra round-cost allowance to reach 5% savings. Relative to an oracle choosing the stronger existing policy per context, that allowance falls to 0.10%; that latter comparator is not a deployable baseline.

Allowing one perfect new child expands the ceiling to 15.49% modeled savings over the refined union, leaving a 12.41% uniform round-cost allowance under perfect selection. This is **not** a forecast: it assumes both correct child predictions and perfect allocation, and its interval describes only eight exposed responses. No evidence yet shows that xPress supplies those children or scores them appropriately.

The one-hop ceiling follows directly: a prefix-closed original tree matching k positions can match at most k+1 after adding one generation of children, capped by the valid window. Newly added nodes are not expanded again. This ceiling is specific to this supplied candidate pool, not all redrafting algorithms.

All source opportunity files matched their recovery hashes and reference identities. Matching-prefix counts were also checked through a literal reference-prefix intersection after verifying tree prefix closure. Evidence: [fixed-path-ceiling.json](fixed-path-ceiling.json) and the subsequent, explicitly extended [fixed-path-and-one-hop-ceiling.json](fixed-path-and-one-hop-ceiling.json). Historical results are unchanged.

## Prior work and novelty boundary

- [xPress](https://arxiv.org/html/2608.02438v1) supplies the causal mixer and existing trained weights. Its candidate-restricted readout is already implemented. Neither is ours.
- [PCTree](https://arxiv.org/html/2608.02123v1) already applies a released parent-conditioned head to alternate branches without training.
- [DominoTree](https://arxiv.org/html/2607.08642v1) and [DARTree](https://arxiv.org/html/2608.13524v1) already carry full-prefix recurrent correction through branches. DARTree batches expansion by depth and defers pruning. Full history and batched branch expansion are not novel claims.
- [JetSpec](https://arxiv.org/html/2606.18394v3) explicitly describes parallel tree-causal scoring across depths using a trained head. Thus even across-depth parallelism is not a broad novelty claim. The locally pinned `path_conditional_refresh.py` is a planned depth-wise implementation, not evidence that its entire published approach is absent or that our variant is novel.
- [DRelay](https://arxiv.org/html/2610.01439v1), published October 1, combines global candidate context and causal selection with joint training. It further limits any generic claim about repairing an initial draft using draft information. We have not evaluated it or established comparable performance.

At most, the narrower candidate is an execution/quality tradeoff: use an existing small, nonrecurrent refiner to conditionally expand a prebuilt proposal set in one batch, without additional backbone execution or training. Whether this has a distinct, useful advantage over existing methods remains open. Do not present a head substitution as an established new algorithm.

## Next decision, before any training

The smallest relevant GPU measurement would reuse the released **4B checkpoint and its own features**, collect full conditional scores for every proposed parent, and compare one-batch expansion with shared-score and immediate-parent-input controls at identical final node budgets. No target answer may choose paths, pool size or runtime inputs. Count repairs and lost correct prefixes and time the complete additional path preparation, scoring, child selection and tree construction. Full-vocabulary normalization and any shortlist restriction must be explicit.

The stronger comparisons also need the same refiner applied through depth-wise conditional expansion and its published six-pass chain/tree alternative, with comparable execution optimization. Shared scores alone are insufficient. Any favorable small screen must subsequently survive fresh prompts, actual target verification, complete generation timing and fidelity checks.

Proceed to a generation test only if measured proposal gains and complete costs leave credible room for the 5% goal against those controls. If actual conditional children fail to improve the pool or the cost exceeds the observed benefit, close this configuration. Do not fall back automatically to a new loss, training sweep or increasingly large candidate pool.

No GPU bundle is ready or reserved yet. The next action is a bounded value-and-cost diagnostic design using this existing refiner, not a training proposal. Goal remains active and unmet.
