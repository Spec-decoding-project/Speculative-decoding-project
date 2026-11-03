# Where compact editors miss full-second-pass repairs

This is a local, post hoc audit of five fixed final checkpoints on the same64 development prompts and2,048 saved contexts. No fitting, checkpoint selection or GPU run occurred. Input hashes and token-label progress were checked.

| Editor | Teacher-repaired blocks also improved | Teacher extra positions captured consecutively |
|---|---:|---:|
| original | 82 /353 | 184 /1,136 |
| context | 86 /353 | 172 /1,136 |
| smaller_update | 98 /353 | 191 /1,136 |
| weighted_ce | 106 /353 | 240 /1,136 |
| teacher_kl | 94 /353 | 218 /1,136 |

The full second pass repaired353 blocks. Across the five editors,181 were repaired by at least one and172 by none. Of those172,143 have the original first-error position wrong in all five predictions. In29, at least one editor corrects that position but breaks an earlier prefix. These are disjoint common-miss categories, not additional independent samples.

Smaller updates gained44 teacher-repaired cases but lost32 compared with the previous context checkpoint. These counts concern that subset; correction length and other blocks explain why total retained coverage did not improve. Stabilization shifts learned corrections without demonstrating broader useful coverage.

An impossible selector choosing the best of all five editors separately for each block yields 7.951% additional retained progress, versus the teacher's 10.208%. Under the declared fixed-context model, that oracle permits only 2.392% common round overhead for5% savings. This is not a deployable selector, an economic measurement or an upper bound on what a new trained model could learn. It does not justify training a gate or merging the checkpoints.

## Research implication

The immediate missing evidence is whether correct-token scores improve but remain below a wrong alternative, or whether the editor fails to move them in the right direction. Winner-only records cannot distinguish these cases. A future frozen-checkpoint measurement would compare correct-token minus best-wrong-token margins at the same first-error positions, including successful repairs and unchanged correct-prefix controls. It must include score changes and margins in comparable conventions, not compare arbitrary logit offsets.

Margin movement alone would not prove that scaling, calibration, more training or a new loss can recover useful prefixes. Before funding such a measurement, specify which intervention each possible outcome would justify and inspect existing score artifacts. No new training, paid readout or selection method is queued by this audit.

Independent review confirmed the category indexing and143/29 split. The dominant shared obstacle is persistent first-error prediction failure; its cause remains unresolved. The findings do not establish insufficient information, insufficient capacity, irreducible errors or failure of small editors in general.

Evidence: [full counts](results.json), [matched records](matched-records.json).
