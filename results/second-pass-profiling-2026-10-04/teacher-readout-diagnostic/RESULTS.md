# Frozen refinement helps the failed teacher; removing it reveals no strong compression target

## Completed intervention

Both retained teacher checkpoints were evaluated on all256saved contexts from64development responses. No training, fresh prompts or target generation occurred. For each unchanged backbone state, compare its direct vocabulary prediction with its existing six-step frozen-refinement prediction. Preserve the original first guess and original six-step seed. All512final token vectors reproduced the prior experiment exactly. Four preassigned complete-score checks matched the historical refiner helper exactly. All weights remained unchanged.

## Result

| Trained input | Direct raw prefix total | With refinement | Contexts helped / hurt by refinement | Positions gained / lost |
|---|---:|---:|---:|---:|
| First guess |1849|2000|56 / 6|163 / 12|
| Completed draft |1500|1657|54 / 15|195 / 38|

Counts include the known anchor consistently. Removing refinement reduces raw-prefix coverage by7.22%averaged over responses for first-guess conditioning (descriptive95%interval -9.74%to-4.69%), and9.08%for completed-draft conditioning (-13.50%to-4.76%). These are fixed-context output comparisons, not generation speedups.

The completed-draft backbone's direct predictions repair22original prefixes and damage114, including42fullycorrect originals. After refinement this becomes25repairs and94damages. First-guess direct predictions give29repairs/71damages, versus35/41afterrefinement. The original fallback remains available; these raw-chain damages do not imply loss of protected original proposals or final-output errors.

## Competitive opportunity

The stronger released conditional control covers2700positions. Free hindsight union with the completed-draft direct chain reaches2705: only5newpositions in3contexts, +0.43%averaged by response (descriptive interval0.00%to1.12%). The corresponding first-guess direct chain adds2positions in2contexts, +0.10%averaged by response. These are upper bounds only for extracting these particular greedy chains on this panel; they do not limit alternative score-based proposals or newly learned models.

Even the optimistic union is insufficient to motivate a5%progress route based simply on removing this head. Keeping only the original and direct completed-draft chains is24.97%belowtheexistingconditionalcontrol; node counts and costs differ, so that comparison establishes neither a latency slowdown nor an equally priced method comparison.

Intermediate completed-draft totals across direct and refinementsteps1–6 are1500,1622,1644,1648,1652,1657,1657. First-guess totals are1849,1943,1977,1994,1997,1999,2000. These recorded trajectories help describe correction propagation; no intermediate checkpoint or iteration count was selected as a new method.

## What this changes

The simple hypothesis that the frozen head is the main cause of missing useful corrections is not supported: it rescues more correct progress than it removes. Useful greedy output is already insufficient at the backbone's direct readout. This does not prove a hidden-state information limit, that joint head/backbone training cannot help, or that a different output function cannot extract better predictions. The backbone was trained through this head; removing it is an inference intervention, not a test of retraining it.

Do not fund a student to imitate the direct readout, remove the head as a remedy, or assume unfreezing the head will solve the deficit. Preserve the failed teacher recipe and the broader compression objective.

The remaining actionable distinction is whether these full teachers learned useful repairs on their actual training examples but failed to transfer, or failed to learn useful prefix repair even there. Current training logs report loss, not accepted-prefix utility, so they cannot answer. Any further diagnostic should evaluate retained checkpoints on a fixed, appropriate sample of their authenticated training inputs and compare against equally exposed development cases, preserving differences in difficulty/eligibility rather than calling them causal evidence. This differs from the earlier compact-student rehearsal; it concerns the newly trained full teachers and requires no new fit. No such run is prepared or launched by this report.

## Limits and evidence

One seed, repeatedly exposed development panel, recorded target continuations, no native generation or end-to-end runtime. Labels affect metrics only. Independent direct prefix recount and all652recoveredfilehashes passed. Both source checkpoints remain unchanged. Source inputs,39sourcefiles, plan and protocol are frozen underlaunch-snapshot; source hashes and per-prompt comparisons are inattempt1/analysis.json. Recorded hidden tensors permit future readout analysis without redoing the transformer.

GPU calculation took57.42seconds; app lifecycle120.91seconds. Full staging, execution, recovery and cleanup took419.96seconds (7minutes).652artifacts/62604145bytes recovered and verified. Appap-VUJft5mPSinJeOLIAEx7uf stopped0tasks, ownedvolumevo-1jEIZTXRqo38K2VZMcyqwW absent. No retry occurred.

Reported compute is$0.0652386, not a settled invoice. Retain the$1.25all-inhold; $2.50remainsunallocatedwithinexisting$40authorization. No previousholdwasreclaimed and nofollow-upspendingoccurred. Researchgoalremainsunmet.
