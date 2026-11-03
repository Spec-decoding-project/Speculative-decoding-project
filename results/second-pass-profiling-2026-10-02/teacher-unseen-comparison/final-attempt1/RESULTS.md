# Fresh comparison of training data and teacher guidance

The small editor made useful additional proposals on 64 fresh responses, but neither changing to data unseen during teacher training nor learning the teacher's probabilities produced a clear improvement over the simpler training controls. This completed comparison does not justify scaling this recipe or claiming a generation speedup.

## What was tested

Six versions of the same 3,024,384-parameter editor received 1,152 updates each. Three used the original 384 training responses; three used 384 different responses outside the full teacher's training set. Within each set, the losses were ordinary correct-token training, additional weight on teacher-correct tokens, and guidance from teacher probabilities at those same positions. Initialization, data order within each set, optimizer, precision and training exposure were matched. All six epoch-three checkpoints were frozen before fresh outcomes were collected; no checkpoint was selected using those outcomes.

The fresh panel contained 32 GSM8K and 32 UltraChat responses, with 2,048 saved draft contexts. The original and edited drafts were both retained. Coverage counts the longer consecutive match with the recorded target continuation, including the shared known root. It is an opportunity for a verifier to use, not measured generation latency.

## Results

Additional coverage over the original DFlash proposal, with each response weighted equally:

| Training method | Original training data | New training data |
|---|---:|---:|
| Correct tokens only | 2.75% | 2.79% |
| Extra weight on teacher-correct tokens | 2.89% | 2.91% |
| Teacher probabilities | 2.83% | 2.77% |

The new-data teacher-probability editor's 95% prompt-bootstrap interval is 2.29–3.28%. It repaired 172 prefixes and damaged 436 edited prefixes, including 54 that were originally fully correct. Retaining the original proposal protects its coverage on these fixed contexts; the extra editing and verification work still costs time.

The planned paired comparisons are small and uncertain:

| Comparison | Difference in percentage points | 95% interval |
|---|---:|---:|
| Change in teacher-probability advantage over weighted labels when changing data | −0.075 | −0.776 to +0.631 |
| New-data versus old-data teacher probabilities | −0.056 | −0.659 to +0.538 |
| New-data teacher probabilities versus weighted labels | −0.139 | −0.664 to +0.376 |
| New-data teacher probabilities versus ordinary training | −0.013 | −0.458 to +0.409 |

The new-data teacher-probability editor adds 3.36% coverage on math and 2.19% on instructions. Its difference from the weighting control is positive on math and negative on instructions. These findings do not establish a benefit hidden by the aggregate.

## What this says about the research question

The full second-pass teacher adds 10.92% coverage on these same fresh contexts, with a 95% interval of 9.52–12.39%. It repairs 412 prefixes. Its useful corrections are therefore still present outside its own training data.

The compact editors' additional coverage is about 25–27% of the teacher's additional coverage. This compares coverage totals; it does not prove that the students reproduced that fraction of the exact same corrections. A student can make corrections the teacher does not make.

The data change made the teacher's supervision more representative on the measurements that motivated it, but did not improve the fresh result materially. This weakens that proposed remedy for this editor and training dose. It does not prove that data never matters, that the editor lacks the necessary information, or that small editors cannot work.

Training cross entropy fell in every arm. That shows optimization changed predictions, not that useful prefix corrections converged. Teacher-correct guidance applied to about 67.1% of valid training positions on the original data and 37.7% on the new data. This was a matched update-budget comparison, not an equal guidance-dose comparison. Combined gradient norms were logged; isolated auxiliary-gradient contributions were not measured.

## Limits and next decision

There is one training seed. The 10,000 paired, domain-stratified bootstrap resamples measure prompt uncertainty conditional on these fitted weights, not seed robustness or simultaneous confidence across comparisons. Twenty-four responses reached the 512-token cap. All six fixed checkpoints were reported, so the slightly highest observed arm is not a newly validated selected winner.

No speculative generation or timing ran on this fresh panel. The earlier four-context graph timing result remains separate. Coverage around 3% offers little headroom toward a 5% latency reduction under a simple progress-per-round model, even before charging extra work; this is an economic warning, not a measured speed result or universal bound.

Preserve this completed negative comparison. Do not scale the recipe, extend training automatically, or start generation testing merely to pursue the unchanged 5% goal. The broader research question remains active. A further training experiment needs a concrete explanation beyond reusing this data change or assuming more teacher supervision must help.

## Evidence and resources

The independent audit checked saved labels against native continuation tokens, recounted all six editors' predictions, and verified checkpoint identities and unchanged weights. All four runtime compatibility controls passed exactly. These checks do not establish deployed speculative-decoding correctness on every trajectory.

The evaluation finished through recovery in 19.83 minutes. The app is stopped with zero tasks and its scratch volume was deleted. The retained research archive holds the full tensors; 529 metadata and prediction files, totaling 13,407,865 bytes, were recovered and verified locally. The $4 allowance remains conservatively retained; it is not a settled invoice. The campaign has approximately $16.54 unallocated after allowances, with no active job reservation.

Sources: [primary analysis](analysis.json), [teacher comparison](teacher-opportunity-analysis.json), [execution and shutdown](execution.json), [independent audit](archive-metadata/final-audit.json), [recovery hashes](archive-metadata-recovery.json), [training exposure](../fit-attempt1/training-exposure-analysis.json), and [frozen protocol](../final-frozen/protocol.md). Reproduce the analyses with `analysis/analyze_teacher_data_final.py` and `analysis/analyze_teacher_final_opportunity.py` from the repository root.
