# Teacher-unseen collection pilot: complete

2 October 2026. All16preassigned new-training responses and512contexts collected. No editor training, retries, final-test generation or model selection occurred. Two saved-context compatibility checks reproduced states, teacher states, labels and winners exactly before collection. Final teacher identity and numerical-buffer checks passed.

## Useful corrections exist, with limited measured opportunity

| Panel | Reachable original errors | Teacher prefix repairs | Extra covered positions | Equal-response coverage gain |
|---|---:|---:|---:|---:|
| Eight GSM8K responses |203|51|176|9.328%|
| Eight UltraChat responses |185|22|36|4.538%|
| All sixteen |388|73|212|6.933%|

The teacher repairs18.81%of reachable errors on these new-training contexts. It also damages71alternative prefixes. Retaining the original prevents those damages from reducing union coverage, but neither generating nor verifying the second proposal is free. Five responses hit the512token cap. There is no measured generation speedup or new compact-editor result.

The teacher-unseen result is descriptively much closer to the historical development repair rate23.01%than the teacher-training rate96.19%. These are different prompt cohorts; this does not prove memorization or causally explain the editor's earlier transfer failure. New-data teaching is feasible, not demonstrated superior.

## Economic qualification before a larger fit

If a hypothetical student reproduces ALL these teacher-added positions, the fixed-context equal-prompt model gives6.296%saving at zero extra computation. For5%modeled saving, uniform complete-round overhead can be no more than1.383%. At2%overhead it gives4.422%saving; at7.57–9.40%it gives0.798–2.512%higher cost per progress. Those latter overheads are earlier small-panel measurements, not measured costs on these16responses.

This calculation prices these specific teacher outputs; it is not a ceiling on a newly trained model, which could learn corrections absent from the teacher. It nonetheless prevents describing simple faithful imitation at the old measured overhead as an established path to5%lower latency. Sixteen training prompts are not a population estimate precise enough to kill the wider direction.

The planned data-regime comparison remains scientifically informative, but passing this collection pilot does NOT automatically promote six fits. Before committing, distinguish a transfer-mechanism study from an end-to-end improvement claim and account for the limited economical margin. No new objective or architecture is selected from this small panel.

## Measured cost and storage

Native collection and original bank construction took311.07seconds; conditional feature extraction took21.23seconds. Extrapolating the domain means to448responses gives145.2minutes for collection alone, versus the earlier165minute estimate. Setup, evaluation, I/O, training and recovery are separate, and per-response variability remains.

The attempt finished through recovery and cleanup in605.60seconds (10.09minutes). Fifty-six artifacts totaling285,843,109bytes were recovered and checked against remote hashes. They include raw banks as well as compact features. Retaining the same evidence per response across448responses would project about8.00GB before reserve/checkpoints; the earlier2.23GBestimate covered compact features only. Full-study storage therefore needs an explicit evidence-retention plan, not just the feature-bank estimate.

Provider-reported app compute is$0.35582853, not a settled invoice. The complete$2.50allowance remains conservatively retained; campaign unallocated funds are$45.03506668. GPU shutdown and zero active tasks were independently confirmed; the owned temporary volume is deleted. No follow-up launched.

## Verification and scope

`analysis/analyze_teacher_unseen_collection.py` authenticated every recovered file, joined raw token predictions to native labels/metadata, verified original inputs and x1, and recounted progress. An independent metadata-based loop reproduced512contexts,388reachable errors,73repairs and212extra positions. Full per-response records remain in analysis.json and rows.json; the conditional economic arithmetic is in economic-scope.json. Execution and provider-closure.json preserve lifecycle/accounting evidence.

All prompts remain new-training data. The64candidate final prompts have not been generated or used in fitting. Goal remains active; compatibility success and teacher opportunity are not training success.
