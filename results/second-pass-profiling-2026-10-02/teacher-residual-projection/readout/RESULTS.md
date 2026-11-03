# Projected teacher states: token corrections and remaining gap

All 128 selected contexts from 64 exposed development prompts completed locally in 228.8 seconds. The matching target vocabulary readout was verified by file SHA-256, target revision and source-shard identity. A separate recount of every saved token sequence reproduces all metrics. Additional cloud spending: $0.

## Results

Each projected proposal receives the full teacher's state for that example. This is an oracle reconstruction diagnostic, not a learned cheap editor.

| CPU FP32 proposal | Extra matching-prefix positions with original retained | Repairs | Damaged edited prefixes | Captured full-teacher extra positions |
|---|---:|---:|---:|---:|
| Original suffix read out again | 9 | 3 | 1 | 3 |
| Full teacher | 71 | 29 | 11 | 71 |
| Teacher projected onto existing learned basis | 31 | 15 | 5 | 21 |
| Teacher projected onto training-fitted PCA basis | 46 | 21 | 10 | 39 |
| Teacher projected onto random basis | 31 | 13 | 3 | 21 |
| One fixed training-mean correction | 10 | 5 | 7 | 6 |

The PCA projection preserves 39/71 = 54.9% of the full teacher's additional prefix positions. Its 46 total additional positions include seven outside the teacher's own matching-prefix extension. The existing basis preserves 21/71 = 29.6%. These counts are not fractions of corrected individual token guesses anywhere in a block; they count consecutive matching-prefix extension.

The PCA proposal gives 9.09% equal-prompt union progress above the recorded original, versus 5.80% for the learned-basis projection and 13.04% for the full teacher. The paired PCA-minus-learned-basis contrast is +3.29 percentage points, with a descriptive prompt-bootstrap interval of [-0.97, +8.95]. This small reused sample does not establish a reliable population advantage between bases. The first comparison is also not against the actual trained editor: both projections are supplied teacher-informed coefficients.

## Numerical qualification

All six conditions use identical CPU FP32, vocabulary-chunked readout arithmetic. This differs from the original BF16 GPU scoring. Against saved winning tokens, original-suffix readout differs at 39/1,763 valid positions across 35 contexts; teacher readout differs at 25/1,763 positions across 19 contexts. Native recorded teacher opportunity on this sample was 70 positions, versus 71 under this CPU readout. The nine extra positions from merely rereading the original suffix illustrate why these results must not be labeled native-decoder improvements.

All contexts remain in the analysis. No agreeing-only subgroup is promoted. Original recorded proposals and their first guess are retained, and each edited alternative is scored against the same recorded labels. This isolates a fixed-input readout question; it does not verify a generated trajectory or target output equivalence.

## Economic interpretation and next decision

Using the same illustrative equal-prompt reciprocal-progress cost model, PCA's oracle proposals permit only 1.12% additional total round time for a 5% latency saving. At 5% added round time, modeled savings are 1.36%. These are scenarios, not measured execution costs; historical editor overhead was measured on another panel. Perfectly predicting these projected teacher states would therefore not, on this sample and cost model, make the current implementation a convincing route to the 5% goal.

Do not launch a PCA-coefficient training run from the hidden-state result alone. The projection does preserve some useful corrections, but Euclidean reconstruction fidelity is a weak substitute for the token-and-cost objective. A newly trained model could produce different, better tokens than this particular projection; this is not a capacity bound on rank256, a bound on all representations, or a proof that hidden-state supervision cannot help.

Any further compression diagnostic should first test preservation of the teacher's relevant token decisions, rather than choosing a basis by reconstruction energy and assuming the decision benefit follows. That is a criterion for the next research choice, not authorization for an automatic metric sweep or another paid run. Research remains active.

Files: `results.json`, `verification.json`, `rows.json`, and `cost-scenario.json`. Reproduce with `analysis/audit_projected_teacher_tokens.py` and `analysis/verify_projected_teacher_tokens.py`. No model weights or historical experimental outcomes were changed.
