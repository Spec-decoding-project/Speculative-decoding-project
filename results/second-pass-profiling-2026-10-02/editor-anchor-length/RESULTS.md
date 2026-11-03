# Does context length hide a useful editor regime?

The frozen editors have a stronger descriptive signal at recorded anchor positions 20–128. This does not yet justify deployment, a gating fit, or new training. No GPU work occurred.

All 2,048 development contexts were joined by prompt and anchor ordinal to the full teacher records. Fixed powers-of-two anchor groups were chosen before calculating these grouped outcomes. They partition the data exactly; all-panel gains reproduce earlier results. Source SHA-256 hashes are in results.json. These are repeatedly exposed development examples, not new confirmation.

## Shortest anchor group

This group contains 404 contexts from 56 prompts. Progress counts the longer matching prefix available from the original proposal and edited proposal, including the known root. It is not a measured generation result.

| Frozen model | Equal-prompt progress gain | Largest total round overhead permitting modeled 5% savings |
|---|---:|---:|
| Correct-token training | 4.37% | None, even at zero overhead |
| Teacher-correctness-weighted training | 7.93% | 1.73% |
| Teacher-probability training | 8.16% | 1.10% |
| Retained-prefix training | 5.22% | None, even at zero overhead |
| Full second-pass teacher | 17.39% | 7.89% |

The two promising compact rows have unadjusted prompt-bootstrap overhead intervals of [-0.31%, 4.19%] and [-1.49%, 4.20%]. These intervals are descriptive, not corrected for looking across models and groups. The median prompt progress gains are 4.46% and 1.14%, respectively. Average progress gain and average reciprocal cost differ; the larger progress percentage does not imply a better modeled latency result.

The full teacher's point estimate allows more cost in this group than on the whole panel (7.89% versus 4.39%). That shows where useful correction opportunity is concentrated in these records. It does not show that shortening context causes the improvement or that compact training on this subset would capture it.

## Decision and limitations

The historical compact implementation added approximately 5–6.3% round time on a different four-context timing panel. That cannot be substituted for subgroup runtime. Nonetheless, the present estimates supply no positive evidence that simply restricting the existing editor to early anchors reaches the 5% goal. Every other compact model/length-group point estimate also fails that goal even with zero added cost.

Do not select a best group and report it as fresh success. Prompt membership, task mix, progress denominators and difficulty vary across anchor groups. This audit cannot distinguish those explanations. The recorded anchors span only 20–702; nothing here establishes behavior at genuinely long contexts. Nor does subgroup arithmetic establish full-response savings from a gate, because unedited rounds and changing trajectories would still count.

The next useful modeling question remains recovery of stronger corrections at the same cost. Before funding an early-context specialization, its rationale would need to distinguish greater available teacher opportunity from easier student imitation, with a training-only selection rule and untouched evaluation. No such successor experiment is selected by this audit.

## Avoided repeated work

The saved strength assessment still lacks learned full score corrections needed to diagnose over- or underscaling. No scalar sweep was funded. The proposed teacher-checkpoint stability check already existed in `compact-training-fit-audit/attempt1/teacher-suitability.md`; it finds that 535 missed positions were stable across teacher epochs. It was not repeated or counted as a new finding.

The source audit also confirmed that the current CE-plus-KL objective's mixture interpretation was already documented. The general connection between distillation and soft-label regularization is established research, not a new explanation discovered here: [Yuan et al., CVPR 2020](https://arxiv.org/abs/1909.11723). Its vision results do not diagnose our editor's failure.

Reproduction: `analysis/audit_editor_anchor_length.py`. No model weights, historical outcomes, budget entries, or closed attempts were changed. Research remains active.
