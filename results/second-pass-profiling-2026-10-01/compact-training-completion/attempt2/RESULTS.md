# Compact second pass training results

The small editor learned useful corrections, but teacher probabilities did not demonstrate an advantage over the matched training controls. This recipe has not established sufficient progress for the intended five percent latency improvement. No generation-speed claim follows from this study.

## Fixed final comparison

All three arms used the same frozen drafter, small editor architecture, initialization, ordered training data, optimizer and precision. Each received1152updates across three epochs on384training prompts. The primary checkpoint was epoch three, fixed before execution. Each checkpoint was evaluated on the same2048saved contexts from64previously exposed development prompts. One training seed was used.

Coverage counts the known root plus consecutive matching future tokens, retaining the original proposal as fallback. Damage below concerns the edited proposal; it does not subtract from retained-union coverage.

| Training supervision | Equal prompt coverage gain | Extra positions | Blocks repaired | Edited prefixes damaged |
|---|---:|---:|---:|---:|
| Correct tokens | 3.02% | 347 | 161 | 541 |
| Extra emphasis on teacher correct rows | 3.70% | 419 | 192 | 513 |
| Teacher probability feedback | 3.46% | 386 | 176 | 503 |

Teacher probability feedback exceeded ordinary training by0.439percentage points, with paired prompt bootstrap95percent interval[-0.120,1.031]. Against the weighting control it was0.237points lower, interval[-0.940,0.424]. Both intervals span zero. This fails to demonstrate superiority; it does not prove equivalence or absence of useful teacher information. The teacher-probability point estimate trails the weighting control in both domains.

## Which teacher corrections were recovered

The full teacher produced1136extra covered positions and repaired353blocks. Teacher probability training recovered218of those same extra positions, about19.2percent, and improved94of those353blocks. It missed259teacher-repaired blocks while improving82other blocks where the teacher had not improved coverage. Its386total extra positions therefore must not be described as386recovered teacher positions. The weighting-only control recovered240teacher extra positions and improved106teacher-repaired blocks.

The teacher baseline is10.20799percent on the same root-inclusive definition. A previous8.3969percent conversion was wrong: historical detail rows exclude the root, but historical summaries already include it. The [denominator audit](denominator-audit.json) reconciles all64prompts with freshly reconstructed features. Editor gains and paired contrasts are unchanged by this correction.

## Cost and learning interpretation

Under a conditional fixed-context conversion that assumes unchanged original round time and a completely free editor, modeled mean savings are2.87percent,3.50percent and3.25percent for the three arms. This is not a bound on future generation trajectories, but supplies no economic case for the five percent target. Real added editor and verification cost would reduce those modeled savings. Earlier untrained integrated timings cannot establish trained serving performance.

Training CE declined in all arms, and final coverage exceeded the first epoch in each. The run does not establish convergence or a capacity limit. It establishes what this dose and recipe achieved. Teacher KL here changes confidence-dependent smoothing and alternative-token probabilities together; even a positive contrast would not have isolated token ranking alone.

A post hoc training-only audit found that7.76percent of teacher-eligible rows are at the original first rejection with a correct frozen first guess;52.36percent already match the original token. These overlapping descriptive categories do not establish a harmful objective or justify removing preservation supervision. [Allocation analysis](training-auxiliary-allocation.json).

## Decision and execution

### What the probability comparison actually changes

A direct check of the implemented loss clarifies its mechanism. On a teacher-correct row, let p be the student's probability distribution, q the teacher's, y the recorded correct token and j a competing token. With auxiliary weight lambda=0.5, the weighting control has loss 1.5 CE(y,p), while the probability arm has CE(y,p)+0.5 KL(q||p). At identical student logits, the weighting control's direct gradient-descent pressure to increase the margin z_y-z_j exceeds the probability arm's by 0.5(1-q_y+q_j). This is nonnegative. The probability arm transfers alternative-token probabilities while reducing the hard-label margin pressure relative to that control.

This is a logit-space identity, not a claim about the actual neural parameter update: Adam, shared parameters, other examples and diverging trajectories matter. It does not prove why this experiment failed or that stronger margins would generalize better. The implementation identity was verified through autograd on 128 synthetic eligible rows, maximum error 4.44e-16: [loss-margin-audit.json](loss-margin-audit.json). This is a mathematical check, not another empirical learning result.

The distinction is established prior art: [Decoupled Knowledge Distillation, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html) separates target-class and non-target-class components and studies their coupling in vision tasks. It does not establish a speculative-decoding improvement or justify adopting its method here without evidence.

Consequently this run tests the complete proposed supervision package, not the isolated usefulness of knowing alternative-token probabilities. The package did not demonstrate superiority. A further causal test would need to separate confidence/smoothing from the identity of alternatives; it should not be presented as a novel distillation idea or launched merely because this algebra supplies another possible explanation. Existing checkpoints alone do not supply that counterfactual.

### Post hoc anatomy of missed teacher repairs

The saved-token audit partitions all 353 teacher-repaired development blocks. For the probability-trained editor at epoch three, 94 improve, 155 retain the original first wrong token, 83 substitute another wrong token at that position, and 21 correct that position but damage an earlier prefix. Thus 238 of the 259 missed blocks still have the original first-error position wrong. Earlier-prefix damage occurs in 81 blocks overall and overlaps those categories; it is not an additional disjoint count. Preserving earlier predictions alone cannot recover most missing teacher repairs.

The editor matches 531 of the teacher's 1,136 extra positions individually, but only 218 belong to its usable consecutive prefix. This gap demonstrates why individual token matches cannot substitute for prefix coverage. It does not show that a gating rule can recover the remaining matches: the correct choices are known only afterward from recorded labels.

Across the three saved epochs, the probability-trained editor improves 163 of the 353 teacher-repaired blocks at least once, versus 94 at the fixed final epoch. An impossible oracle that picks the best epoch separately for every block gives 6.63% total union progress gain. That is evidence of changing corrections across checkpoints, not a deployable selection method, a speedup, or grounds for choosing a new primary checkpoint. All three objectives show similar behavior (correct-token only: 157 teacher blocks helped at least once; weighting control: 167).

This audit narrows the next question to learning and retaining the first useful correction while preserving its prefix. It does not establish why that learning fails, or justify a particular new loss, model size, or training extension. No new GPU work occurred. Reproducible counts and examples: [teacher-miss-anatomy.json](teacher-miss-anatomy.json), generated by `analysis/compact_teacher_miss_anatomy.py`.

Do not launch a larger generation benchmark, automatic training extension, or new architecture from these results. Preserve the useful corrections and the failure to demonstrate added value from teacher probabilities. The next explanation to investigate is why this recipe misses most teacher-repaired blocks; recovered examples permit further local comparison. Neither hidden-state information loss nor insufficient model capacity has been demonstrated.

All3456updates and18432checkpoint evaluations completed. All479artifacts were recovered with hash/size verification; the app stopped with zero tasks and its temporary volume was deleted. The two operational failures remain preserved separately. The controller was corrected to handle the installed SDK's actual polling timeout.

The continuation finished in26.32minutes through recovery and cleanup. The conservative allowance is$1.75, below the$3.50cap, with$4.1351remaining after reconciliation. This is not a settled invoice. [Accounting](cost-reconciliation.json) uses [published rates](https://modal.com/pricing). No follow-up GPU work was launched.

[Full analysis](analysis.json) includes all fixed epochs, domain results, paired intervals, exact repair overlap and training losses. [Execution](execution.json) and [recovery manifest](recovery.json) preserve operational evidence.
