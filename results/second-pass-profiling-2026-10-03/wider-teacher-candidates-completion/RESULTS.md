# Wider candidates help teacher ranking, but do not supply sufficient performance margin

The frozen comparison is complete across all 64 development responses / 2,048 contexts. The successful completion collected only the 664 missing records; 1,384 earlier records remain unchanged. The first preparation failure and second attempt's timeout remain recorded. No new model was trained and no generation timing was measured.

Follow-up qualification: these matched fixed-first-token comparisons isolate teacher scoring, but do not establish an advantage over an ordinary tree allowed to change the first guess. The [stronger-reference and conditioning audit](../teacher-chain-cost-margin/RESULTS.md) supplies that comparison and preserves all findings below.

## Question and controls

Does limiting the teacher to the original drafter's top four tokens discard useful corrections? Compare nested pools of four, eight and sixteen original candidates, at the same 30-node tree budget. Keep the original chain and first token fixed. Within each pool compare original scores with teacher preferences, preserving the original total probability mass on that pool. Also retain full-normalized teacher scores as the predeclared secondary analysis.

Candidate inclusion never uses target labels. Historical top-four IDs remain unchanged. Extra IDs come from native top16 scores, ordered by score then ID within that returned set; boundary ties follow the backend's topk choices. No claim of universal tie-independent coverage is made.

## Completed findings

| Candidate pool | Original-score tree: coverage gain over single original draft | Teacher-scored tree: same reference | Modeled time saving of teacher versus equally wide original tree, **before teacher cost** |
|---|---:|---:|---:|
| Four | 14.16% | 16.90% | 2.24% (prior comparison) |
| Eight — primary | 14.83% | 18.32% | **2.81% [1.84, 3.77]** |
| Sixteen — secondary | 15.02% | 18.96% | **3.14% [2.06, 4.20]** |

The percentages in the middle columns use the single original draft as denominator; the right column compares the two trees at the same pool size. They are not interchangeable. Coverage includes the known root and is calculated on saved target continuations. The right column assumes identical round time, averages response-level cost ratios and excludes all teacher computation. It is not actual generation latency or a universal achievable bound.

The primary teacher-versus-original difference is +3.50 percentage points of original-draft coverage, with descriptive prompt-bootstrap interval [2.33,4.69]. Modeled savings are positive in both domains: math3.64%, instructions1.97%. Sixteen-candidate modeled savings are math4.17%, instructions2.11%. Full-normalized teacher scores give similar outcomes:2.83% at eight candidates and3.19% at sixteen, against their matching original-score trees. All conditions are in the analysis; no development winner is promoted as a confirmed method.

## What this changes

The top-four limit excluded some useful teacher choices. Wider ordinary candidate lists improve little, while teacher rescoring makes better use of the added candidates. Thus candidate availability and ranking are separate bottlenecks here.

Nevertheless, even assigning the teacher's computation zero cost leaves this measured score policy below the5% goal. A student copying these score policies perfectly would reproduce these choices, not automatically yield a5% speedup under the equal-round-cost model. A student could learn different/better choices, and real round costs can differ; those are new hypotheses, not established consequences of imitation.

Candidate-support oracles show much larger unrealizable room: perfect reference-informed choices within top8 could give50.29% coverage over the original single draft, versus18.32% for the actual teacher-scored tree. This oracle uses the answers. It does not show that the first-pass features make the required decisions cheaply learnable, and must not justify a fresh student training run by itself.

Decision: **do not launch a wider-candidate student fit merely to copy the current teacher.** The next learning proposal needs a reason to obtain better useful decisions than these frozen teacher scores, or a demonstrated change in complete round cost. The broader cheap-second-pass research question remains active. Candidate widening and rescoring are established techniques; this diagnostic makes no novelty claim.

## Verification and limits

- Exact native BF16 original and teacher top-four scores and winners reproduced for every context; first-context repeats passed in each successful collection allocation.
- 18,432 independent tree reconstructions and2,048 historical endpoint checks passed. All781 trees certified unchanged under original-score widening remained unchanged; the457 individually excluded first-correct tokens were not rescued by unchanged-score widening.
- The missing-context complement is exact, all2048 keys appear once, common input files are identical across allocations, and recovered files were rehashed. Source attempts are linked in [consolidated/provenance.json](consolidated/provenance.json).
- Same reused development panel; no untouched final validation. Bootstrap intervals describe this analysis after substantial development reuse.
- Fixed first guess, mandatory original chain, 30-node budget, saved-context score trees. No actual tree verification, changed generation trajectories, target-output preservation, timing, student learnability, or production benefit established.

## Execution and cost

Completion app `ap-p2gM0h8dfAY2EgQJQsOP7B` finished in1m41s through cleanup. Five files /12,557,680bytes recovered; zero active tasks; owned volume `vo-VaNQWuqOVZQ44hbI5Sk0Sp` absent. Its$0.50cap is retained as an unsettled allowance. Including the failed and partial attempts, $2.50 is conservatively held, **not a billed-cost claim**. Campaign unallocated balance is$0.03506668; no paid jobs/reservations remain. The retained training archive was never mounted or modified.

The previous rank-mask error and runtime underestimate were preparation failures. Actual gather/mask tests now cover all saved contexts. Completion reduced growing-file rewrites, but these runs do not isolate the runtime effects of logging from hardware and context differences.

Evidence: [consolidated/analysis.json](consolidated/analysis.json), [fit-attempt1/execution.json](fit-attempt1/execution.json), [fit-attempt1/recovery.json](fit-attempt1/recovery.json), [PROTOCOL.md](PROTOCOL.md).
