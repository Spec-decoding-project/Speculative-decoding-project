# Some teacher signal transfers; rescaling and gating are insufficient

Completed locally for $0 using saved scores from the three matched fits. No
training, GPU allocation, fresh evaluation prompts or new checkpoint. The
previous negative full-strength result is unchanged. The research goal is active.

## What this changes in our explanation

It is too strong to describe the students as learning no transferable teacher
signal. On all 64 reused development responses together, infinitesimally moving
from original scores toward each student's scores reduces teacher-imitation loss.
Both domains have the same favorable initial direction. At full strength all
three instead increase that loss. Excessive correction strength is therefore
part of the observed failure for this probability objective.

This does not mean the transferred signal yields useful accepted prefixes.
Training-selected attenuation leaves the two imitation students below original
score selection. The attenuated correct-token student is only +0.105 percentage
points in coverage [-0.062, +0.275], with opposite domain directions. Its modeled
additional-round-cost allowance is only 0.097%, before an editor is charged.
This is not a demonstrated latency benefit.

## Controlled direction test

At each fixed original candidate set, preserve its mass m. With fixed residual
r = log S_C - log Q_C, define p(a)=softmax(log Q_C+a*r) and
F(a)=mean[m KL(T_C || p(a))]. Then

- F'(a)=mean[m sum_k (p_k(a)-T_k) r_k].
- F''(a)=mean[m Var_p(r)] >= 0.

The convexity statement concerns this fixed-direction loss, not tree coverage.
Analytic derivatives passed 100 synthetic finite-difference comparisons.

| Student | Development derivative at zero | Paired prompt 95% interval | Strength chosen on training only |
|---|---:|---:|---:|
| Full-vocabulary imitation | -0.0540 | [-0.0649, -0.0447] | 0.713 |
| Candidate-focused imitation | -0.0391 | [-0.0471, -0.0318] | 0.630 |
| Correct-token training | -0.0351 | [-0.0462, -0.0253] | 0.121 |

Scalar choices minimize teacher KL on the fixed 64-response training probe,
not development. Bisection uses convexity; there is no development-selected
deployment scalar. The original no-editor KL is 0.2423. At the selected strengths
the development losses are 0.2455, 0.2539 and 0.2411 respectively. Training's
preferred strength still does not reliably transfer to the useful-prefix metric.

| Scoring method | Extra coverage over original single draft | Difference from original-score tree |
|---|---:|---:|
| Original-score tree | 14.158% | — |
| Training-scaled full imitation | 13.423% | -0.735 pp [-1.181, -0.294] |
| Training-scaled candidate imitation | 13.647% | -0.511 pp [-0.910, -0.099] |
| Training-scaled correct-token editor | 14.264% | +0.105 pp [-0.062, +0.275] |

All five fixed diagnostic strengths (0,.125,.25,.5,1) are in `results.json`.
None has a reliable positive development coverage contrast. Do not promote a
best-in-hindsight grid point. These exposed-development intervals are descriptive,
not untouched-test confirmation. Native BF16 scaled-output fidelity is untested.

## Does the direction address the decision that matters?

A post-hoc audit classifies all 2,048 contexts under the restricted four-candidate
output and fixed first guess:

| Original-score tree situation | Contexts |
|---|---:|
| Correct path could be extended using available candidates | 710 |
| Next correct token absent from the four candidates | 779 |
| First fixed guess is wrong | 331 |
| Original chain already covers the valid continuation | 183 |
| An alternative tree path already covers it | 45 |

The latter groups either cannot gain under this restriction or require preserving
existing coverage. This is not an inference-time classifier: target references
identify the categories.

For the 293 contexts where teacher scores improve the tree, candidate imitation
initially improves the correct missing path's margin against the weakest optional
retained node in 161 cases and worsens it in 132. Full imitation gives 171/121
with one tie; correct-token training gives 169/124. The direction is mixed even
at these useful decisions. Positive average imitation alignment is not enough.
All 2,130 frontier-margin derivatives passed finite-difference checks. A favorable
margin slope cannot guarantee eventual inclusion when other node ranks also change.
This is output-change analysis, not causal attribution of training gradients.

## Could a perfect gate rescue these frozen outputs?

At each context, let an oracle use target references to choose the longest
matching tree. Charge zero editor, selection and additional round cost; keep
the same 30-node ceiling for the chosen tree. This is deliberately optimistic.

| Oracle choices | Modeled savings versus original-score tree |
|---|---:|
| Original or full-strength candidate-imitation editor | 1.063% [0.840, 1.301] |
| Original or any of the three full-strength editors | 2.087% [1.709, 2.506] |
| Original or any editor at the fixed grid/training-selected strengths | 2.429% [2.044, 2.847] |

Thus a gate selecting among exactly these frozen possibilities cannot reach 5%
on this panel even under zero-overhead assumptions. This is not a bound on all
continuous scales, new candidates, new weights, different workloads, or real
generation trajectories. It does not justify a learned gating project.

## Keep the teacher's opportunity scoped correctly

The full second pass provides several possible outputs; no single restricted
output represents all its usefulness:

- Perfect switching between original-score tree and the original/full-teacher
  chain gives 4.004% zero-added-cost modeled savings [3.269, 4.763].
- Perfect switching between original and teacher-scored top-four trees gives
  4.436% [3.838, 5.036].
- Allowing both teacher output forms gives 5.907% [5.103, 6.720].

The last figure is an unimplementable reference-informed selector, not an
achievable student result. Even perfect realization of its observed mean would
allow only about 0.963% uniform added round cost to retain 5% modeled savings.
All necessary computation is excluded from these optimistic scenarios.

For the teacher chain, 195 contexts extend beyond the original-score tree,
adding 574 positions. Of those, 210 positions lie at or after the first correct
token outside the original top four. Thus restricting training to four candidates
does discard some useful full-teacher work. Conversely, 364 positions remain
representable within that candidate support; missing candidates do not explain
all failed imitation. These are hindsight opportunity counts, not learnability.

## Research decision and prior art

Close rescaling or gating these frozen scores as a route to the 5% target. Do not
turn the tiny attenuated CE point estimate into a paid generation proposal. Keep
the broader cheap-second-pass question, with a more precise requirement: a future
student must create substantially more useful continuation choices while preserving
the original scorer's strengths. Average probability imitation, lower runtime,
or a gate around these predictions alone is insufficient in the measured panel.

Do not relabel a generic path-aware loss as a new contribution. TAPS already
combines local target-preference KL with a reach-probability loss and selects
prefix-closed trees. That is relevant prior art, not evidence that the same
objective will solve our second-pass compression problem. [TAPS, Sections
4.2–4.4](https://arxiv.org/html/2606.00487v1).

The earlier D²SD review and variable-prefix study remain relevant and closed;
neither a new uncertain-cut selector nor a variable-prefix replay is automatically
justified. [D²SD reread and existing controlled results](../../second-pass-profiling-2026-10-02/d2sd-reread.md).

No new architecture, data scale, or paid fit is selected by this audit. Before
one is selected, its reason for producing new useful corrections must exceed the
small existing-output opportunity quantified here. Research is active and unmet.

## Evidence and accounting

Independent aggregation checked all 2,048 oracle contexts, teacher-source alignment,
and the original training comparison's positive-position counts. There were 12,288
endpoint context evaluations; exact endpoint assertions checked 384 aggregated
prompt totals. The earlier near-tie numerical sensitivity remains a limitation.

No additional paid execution; $2.53506668 remains unallocated, zero active jobs.
Evidence: [direction results](results.json), [frontier alignment](frontier-alignment.json),
[gating bounds](gating-ceiling.json), [teacher-output opportunity](teacher-output-opportunity.json),
[teacher-chain anatomy](full-teacher-switch-bound.json), [verification](verification.json),
[pre-calculation protocol](PROTOCOL.md).
