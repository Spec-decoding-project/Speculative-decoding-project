# Can the no-feedback teacher replace the original drafter?

## Finding

The historical8.051%no-feedback improvement was retained-union coverage, not standalone drafter performance. The original branch supplies a substantial fallback. Recounting all2048development contexts gives:

| Model | Original plus revised coverage gain | Revised branch alone, still using original first guess |
|---|---:|---:|
| Feedback teacher |10.208%|4.304%|
| No-feedback teacher |8.051%|2.600%|

All percentages are equal-prompt accepted-progress gains, root inclusive, on64exposed prompts. They are not latency savings. Revised-only no-feedback gains5.388%on GSM8K and loses0.188%on UltraChat. Its revised branch repairs299contexts, adding950positions, but is shorter than the original in237contexts, losing654positions. Feedback repairs353contexts/adds1136positions and loses704positions in249contexts. Original total progress is11005positions. Raw pooled ratios differ from equal-prompt ratios.

## What can and cannot run without the first drafter

FullConditionalReference.forward in no_feedback mode consumes masked token embeddings, raw committed target features, positions and attention mask. It does not insert the guessed token or use the original drafter's computed hidden states. Its underlying mathematical forward therefore does not need that original forward. The wrapper still checks state ownership; extracting independent inputs would require a verified implementation.

However, evaluation in conditional_full_reference_precision_worker.py only records private predictions at positions2onward. Position1is copied from the original draft. Training also uses cut=1and supervises the suffix. The private model's own first-future-token prediction was neither evaluated nor saved here. Thus removing the original forward is not the same as preserving the evaluated proposal. We do not have a measured replacement drafter.

## Recorded-suffix envelope

For no-feedback, fixing its first predicted future token to the known target answer while retaining the recorded suffix gives11.937%equal-prompt progress gain. This is a nondeployable upper envelope for these suffixes, not an achievable outcome or speedup. It permits repair of blocks where q0's first guess was wrong. The actual private first prediction could cause additional early rejections, and its relationship to suffix correctness is unknown. The original-first2.600%result is not a lower bound on standalone performance.

For feedback the analogous output-only oracle is12.077%, but changing that output does not recondition the teacher. It must not be read as what the teacher would predict after receiving a corrected first token.

## Decision

Do not substitute8.051%as the benefit of ordinary standalone fine-tuning or as an already-established cheap alternative. A future second-pass contribution should compare against a properly trained and evaluated one-pass drafter under matched resources. The saved no-feedback control is informative about conditioning but incomplete as that serving baseline.

The next required observation for repurposing this checkpoint would be its raw first-token predictions and complete revised prefixes on the same fixed contexts, with no q0 fallback. The existing suffix records cannot supply it. First-token-specific training would be a changed recipe, not a recovery of missing evaluation. No model execution, training, GPU job or spending was launched in this audit.

## Verification

Source evaluation files were authenticated against stored hashes. Every original and alternative accepted length was independently recomputed from saved token sequences and historical labels.4096arm-context rows align by prompt/anchor. Separate aggregate checks verify gains minus fallback losses equal revised-only progress and uniqueness holds. See results.json, rows.json, verification.json. Reproduce with analysis/audit_single_pass_reference.py.
