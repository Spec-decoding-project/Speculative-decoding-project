# Candidate-focused teacher imitation did not beat original draft scores

All three matched fits and planned evaluations completed. The fixed epoch-three
learning screen failed. No generation comparison, larger model, longer fit or
follow-up spending is justified by this result alone. The broad question of
cheaply capturing useful second-pass work remains open.

## Primary development result

64 reused development responses, 2,048 saved contexts; one training seed.
Same original top-four candidates, original probability mass on that set,
fixed x1, original chain retained, 30 total tree nodes. Percentages below are
equal-prompt matching-prefix coverage gains over the original single draft,
including the known root. They are **not measured generation speedups**.

| Scores used to build the tree | Extra saved-context coverage |
|---|---:|
| Original drafter; no editor | 14.158% |
| Full second-pass teacher | 16.903% |
| Small editor: full-vocabulary teacher imitation | 12.513% |
| Small editor: candidate-focused teacher imitation | 12.644% |
| Small editor: candidate correct-token training | 10.622% |

Candidate-focused imitation versus:

- Original-score tree: **-1.514 percentage points**, paired 95% interval
  [-2.006, -1.011]; adverse point direction in both domains.
- Full-vocabulary imitation: +0.130 points [-0.439, +0.697]; no reliable
  advantage, with opposite domain directions.
- Correct-token training: +2.022 points [+1.388, +2.673]; positive in both
  domains. This relative win does not overcome the original-score reference.

Candidate-focused scores gained 142 matching positions in 111 contexts but lost
344 in 223 contexts relative to original scores; 1,714 contexts tied. Under the
uniform-round-cost model, this progress would require rounds to become about
1.36% cheaper just to break even. An extra editor has no demonstrated route to
that saving. This calculation is not a measured slowdown.

## Learning and what failed

All arms: 3,024,384 parameters; identical initial weights, 384 training responses,
same scheduled anchors and order, three epochs / 1,152 updates. FP32 masters,
BF16 deployed checkpoints, AdamW .001 and .01 weight decay. Full forward
computation is shared; candidate-only inference projection was not implemented.

The original drafter's conditional mismatch to the teacher on development was
0.2423. After training it was 0.2709 for full imitation, 0.2934 for candidate
imitation, and 0.4004 for correct-token training. Lower is closer to the teacher.
Both imitation arms reduced this mismatch on the fixed training probe instead:
0.2498 initially, falling to 0.2007 and 0.2136 respectively. Thus the targeted
objective did not improve development imitation in this fit.

On the 64-response training probe, original-score tree coverage was +15.335%;
full imitation +14.821%, candidate imitation +14.053%, and correct-token training
+20.816%. This separates two observations:

1. The correct-token editor fitted useful training corrections but transferred
   poorly. It is not true that this architecture cannot learn any useful edits.
2. The imitation arms reduced average training mismatch without beating the
   original-score tree even on that training probe. Closer probabilities alone
   were insufficient for the useful-prefix objective at this exposure.

This does not identify an information limit, capacity limit, optimizer cause,
convergence point or universal failure of distillation. Objective changes also
change update magnitudes. One seed and exposed development data cannot establish
broad generalization. No hyperparameter sweep or architecture choice was tested.

The saved epoch-one/two/three tree gains were 13.214/13.193/12.513% for full
imitation, 13.102/13.014/12.644% for candidate imitation, and
11.625/11.262/10.622% for correct-token training. None beat original scores.
Earlier checkpoint selection is therefore not hiding a win in these saved epochs;
unsaved checkpoints and different training recipes remain untested.

## Hindsight teacher-opportunity audit

The projected teacher improved 293 development contexts and worsened 199,
gaining 624 matching positions and losing 289 versus original-score trees.
Candidate imitation retained only 96 of those 624 extra positions (15.4%),
while losing 269 positions in contexts where the teacher offered no benefit.
Full imitation retained 99; correct-token training retained 72.

Candidate imitation raised the first missing correct path's score in 135 of
the 293 teacher-beneficial contexts. In 65 of those cases it still gained no
prefix coverage: raising a correct path need not make it win against competing
paths. This is a post-hoc, reference-dependent description, not a deployable
selector or causal explanation. It identifies both insufficient useful capture
and damage, rather than reducing the failure to overhead alone.

## Evidence checks and numerical limitation

- All 2,048 native original/teacher development readouts matched saved argmax
  predictions; training order matched the authenticated historical exposure.
- All three arms completed 1,152 updates; final BF16 weights match their saved
  FP32 recovery checkpoints; complete optimizer steps and moments are retained.
- Full-imitation final parameters exactly reproduced the historical matched
  teacher-only fit. The new objective comparison is not a failed control replay.
- All 47 recovered artifacts were authenticated. Training/development IDs and
  exact input-token sequences do not overlap; semantic similarity is not excluded.
- Independent frontier enumeration reproduced 10,240 primary tree results using
  the frozen Python 3.11 floating-point operation contract.

The first independent verifier used an algebraically equivalent probability
normalization and recomputed entire path sums. It failed an exact recount because
rounding changed nearly tied node choices. The failing verifier and a concrete
case are preserved, not erased. On all scorers, the alternative calculation
changed nine context/scorer outcomes. Coverage becomes 14.164% original,
16.897% teacher, 12.513% full imitation, 12.654% candidate imitation and 10.622%
correct-token training. Point ordering and the negative practical conclusion
are unchanged. No tolerance was introduced into the primary selector, and no
GPU score or trained checkpoint was changed to repair the local verifier.

## Execution, cost and next decision

Completed in **14m22s through recovery and cleanup** on A100-80GB PCIe.
App `ap-dQtT8avUgP0PnqWeUsCu7O` is stopped with zero active tasks. The exact owned
scratch volume `vo-XgUQwSsjgnICroBND7QkUx` is absent; the retained research archive
is untouched. Recovered 47 artifacts / 370,776,938 bytes.

The full **$3** cap remains a conservative unsettled allowance, not a billed
expense. **$2.53506668** remains unallocated, with no active reservation or job.

Close this candidate-focused objective at the tested dose. Do not launch generation
or simply repeat/extend it. Before a different fit, the research needs a concrete
reason it will capture useful teacher decisions while preserving the original
scorer's strengths; merely reducing average imitation loss has now failed that
test. The original goal remains active and unmet. This is neither a publishable
method claim nor evidence that no compact editor can work.

Evidence: [protocol](../PROTOCOL.md), [analysis](analysis.json),
[independent verification](verification.json), [learning curves](learning-curve.json),
[teacher opportunity audit](teacher-opportunity-audit.json), [execution](execution.json),
[recovery receipt](recovery.json), [frozen inputs](../frozen/manifest.json).
