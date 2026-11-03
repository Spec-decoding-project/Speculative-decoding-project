# First-token freedom, conditional teacher scores, and the cost of compression

All work in this audit was local and cost $0. The research objective remains unmet. This extends an already-known control—the ordinary tree that can change the first guess—to the newly collected wider candidate pools. It does not claim discovery of a new decoding technique.

## The practical comparison must allow first-token alternatives

Keeping the first guess fixed is useful for isolating changes to later predictions. It is not a sufficient competitive baseline. The first guess is wrong in331of2,048saved contexts. The fixed-first teacher and student cannot recover a longer matching prefix there.

At four candidates and30nodes, ordinary first-token alternatives gain384positions on those wrong-first contexts, while losing76positions elsewhere as the same node budget is redistributed. This reproduces the historical free-first result; at16candidates the corresponding changes are+402and−105.

| Candidate pool | Ordinary tree with first-token alternatives | Teacher-scored tree that still fixes first token |
|---|---:|---:|
| Four | 18.60% coverage above original single draft | 16.90% |
| Eight | 19.19% | 18.32% |
| Sixteen | 19.28% | 18.96% |

These are saved-context coverage figures, not speedups. At16candidates the teacher-minus-ordinary difference is−0.326percentage points with descriptive95%interval[−2.110,+1.423]. This does not establish either superiority or equivalence. The earlier positive matched-fixed-first comparisons remain valid as mechanism controls; they did not establish competitive superiority over this stronger ordinary method.

## Preserve both behaviors without inconsistent conditioning

We tested a direct combination using the same saved scores and node budget:

1. Allow ordinary alternatives to the first token.
2. On the branch whose first token equals the teacher's conditioning token, use teacher scores for later positions.
3. On other first-token branches, retain original scores. Do not apply a teacher conditioned on one token as if it had seen another.

This rule needs no labels to construct a tree, but it still requires the full teacher to provide its scores. Positions remain factorized after the first token; these are not full sequential conditional probabilities.

| Candidate pool | Combined-policy coverage above original single draft | Modeled savings versus equally wide ordinary free-first tree, before teacher cost |
|---|---:|---:|
| Four | 21.59% | 2.38% [1.66,3.11] |
| Eight | 22.91% | 2.94% [2.05,3.84] |
| Sixteen | 23.56% | 3.34% [2.34,4.33] |

At16candidates, this gains878positions and loses386across contexts relative to the ordinary tree. Both domains have positive modeled savings: math4.12%, instructions2.56%. Full-normalized teacher scores are similar, with3.31%modeled savings. This supports complementary roles for ordinary first-token branching and teacher-conditioned later scores. It does not establish a5%generation improvement, novelty, or cheap student learnability.

## This output rule does not rescue existing students

We applied the same matching-branch rule to all three frozen top-four students. Previously selected training-only strengths were preserved; no development tuning or fitting occurred.

| Frozen student | Full-strength modeled savings versus ordinary free-first tree | Previously training-selected strength |
|---|---:|---:|
| Full-vocabulary teacher imitation | −1.19% | −0.57% |
| Candidate-only teacher imitation | −1.36% | −0.46% |
| Candidate correct-token training | −3.04% | +0.067% [−0.095,+0.225] |

Negative means worse, even before paying for the student. The last result is tiny, uncertain, and has opposite domain signs. No new serving route or another fit is justified for these checkpoints.

## Could lower round cost nevertheless rescue a perfect chain editor?

The old16-context timing panel measured approximately31.53msfor the compact-editor round and33.21msfor the ordinary free-first tree. Those timings and the64-response saved coverage are different measurements: saved/live predictions differed, so combining them gives **a hypothetical scenario, not measured performance**.

If an editor perfectly copied the current teacher's original-plus-one-chain progress and cost31.53ms, that scenario is2.45%slower than the ordinary free-first tree. To reach5%lower time per token at that teacher progress, its round would have to cost no more than29.24msunder the measured33.21msreference. Even the original chain's measured30.95msis above that number, though it is not a proven hardware lower bound.

Alternatively, at the31.53mscost assumption, teacher-chain progress would need to increase uniformly by about7.85%over its current value—equivalent in this sensitivity calculation to18.85%coverage above the original draft rather than10.21%. That is a requirement illustration, not a prediction that training can realize it.

Do not assign the simple two-chain editor's timing to the combined teacher-scored **dynamic tree**: it requires different selection and packing work. Equally optimized ordinary trees could also reduce their cost.

## Research decision

There are two unresolved obstacles, not merely an inefficient implementation:

- The strongest measured uses of current teacher scores still offer only about3.3%modeled savings before extra cost under this fixed-budget setup.
- Existing small editors do not preserve even that useful ranking advantage.

Keep the broad cheap-second-pass objective active, but do not launch another imitation fit from these frozen teacher results. A next learning proposal needs a stronger useful target or a demonstrated cost advantage against the unrestricted ordinary reference. First-token-aware tree construction is a necessary comparison, not a novel contribution. More support-oracle headroom alone cannot establish learnability.

## Verification and evidence

Free-first reference:6,144independent tree reconstructions and2,048historical endpoint checks. Matching-branch teacher:300exhaustive short-tree synthetic cases,18,432independent reconstructions and6,144original-reference endpoints. Frozen students:12,288independent reconstructions. Input hashes and all favorable/unfavorable per-response results are preserved. Bootstrap intervals are descriptive after development reuse. No GPU, new training, new response, live-generation test, or paid execution occurred.

Evidence: [free-first-reference.json](free-first-reference.json), [branch-consistent-teacher.json](branch-consistent-teacher.json), [branch-consistent-students.json](branch-consistent-students.json), [results.json](results.json) (cost scenarios).
