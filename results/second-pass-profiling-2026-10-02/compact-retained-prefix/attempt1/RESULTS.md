# Compact retained-prefix training comparison

Both arms completed 1,152 updates. The unchanged control reproduced all three historical development evaluations exactly.

| Final training objective | Retained progress gain | Repaired blocks | Damaged edited prefixes |
|---|---:|---:|---:|
| uniform_ce | 3.0217% | 161 | 541 |
| retained_prefix | 3.1700% | 184 | 570 |

Candidate minus uniform CE: +0.1483 percentage points; paired prompt bootstrap interval [-0.5711, +0.8585]. Domain contrasts: {'gsm8k': -0.2294986137605669, 'ultrachat': 0.5261102962207407}.

Exploratory screen: not passed.

Full-second-pass repair overlap: {"uniform_ce": {"teacher_repaired_blocks": 353, "teacher_extra_positions": 1136, "teacher_blocks_repaired_by_editor": 82, "teacher_extra_positions_captured": 184}, "retained_prefix": {"teacher_repaired_blocks": 353, "teacher_extra_positions": 1136, "teacher_blocks_repaired_by_editor": 95, "teacher_extra_positions_captured": 176}}.

Modeled latency changes under assumed round overheads (positive is slower): {"0": -3.016332278371159, "2": -1.0766589239385782, "5": 1.8328511077102982, "6.3": 3.0936387880914573}. These are sensitivity calculations, not timing measurements.

The original draft is retained as a fallback. These figures measure fixed-context union coverage on 64 reused development prompts, not generated acceptance or latency. Edited-prefix damage is reported separately and does not subtract union coverage; generating and verifying the alternative still costs time. The objective is established prior art. This comparison changes the full weighting package, not one isolated causal ingredient. One seed and repeatedly used development prompts do not establish generalization.

Recovered and authenticated 32 artifacts (124,420,901 bytes). Shutdown and temporary-volume deletion are verified. Elapsed through cleanup: 14.58 minutes. Retained allowance: $1.55, not an invoice. Remaining after allowances: $2.24. No controller resubmission.

[Execution](execution.json) · [Closure](closure.json) · [Protocol](../PROTOCOL.md)

## Decision

Close this exact objective package without a generation run or automatic objective sweep. The small positive point estimate is uncertain and differs in direction across domains. Even the zero-overhead cost scenario falls short of the 5% goal. More repaired teacher cases did not yield more recovered teacher positions. This does not establish equivalence or a general limit on prefix-aware training, capacity, information or the broader cheap-editor direction.

Independent review recounted final tokens and teacher overlap and verified exact historical control predictions. [Review](review.json).
