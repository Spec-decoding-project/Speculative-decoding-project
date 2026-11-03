# Wider-candidate comparison: recovered subset valid, primary panel incomplete

The masked-label repair worked. The fixed evidence reserve stopped collection after 1,384 recovered contexts: all 32 GSM8K responses, 11 complete UltraChat responses, and eight contexts from a twelfth. The remaining 664 contexts were not recovered. There is no completed 64-response estimate.

All recovered original and teacher top-four score vectors passed exact native parity. The analysis independently rebuilt 12,456 trees and matched all 1,384 historical original/teacher endpoints. All 622 recovered contexts covered by the prior whole-tree exclusion certificate remained unchanged when widening original candidates, as predicted.

For the completed responses only, teacher preferences with original candidate-set mass added the following matched positions over **equally wide original-score trees**, at the same 30-node budget:

| Candidate pool | 32 math responses | 11 instruction responses |
|---|---:|---:|
| Four | +250 | +25 |
| Eight | +318 | +30 |
| Sixteen | +366 | +18 |

These are raw fixed-context progress differences, not speedups. The instruction subset is incomplete and ordered by prompt ID; it cannot represent the planned instruction panel. Both damage and repairs are included in these net counts. The final partial response is retained in context records but excluded from this table.

This supports investigating the distinction between candidate availability and ranking, but does not establish a useful small editor, sufficient runtime margin, or the primary full-panel comparison. No training is justified by this partial result alone.

Runtime estimate failed: collection plus frequent growing-file saves exceeded the planned time. Stage timings were not recorded, so their individual shares are not established. The next completion preserves numerical work, runs only the exact missing keys, and saves per prompt to reduce repeated serialization/writes. Original failure, partial data, and timeout remain preserved.

App `ap-5STRqEnVNi8nyshwQwGSkv` is stopped with zero tasks; owned volume `vo-9iUnFKVIBLdJJvRPWOLEK1` is absent. Four files totaling25,974,428bytes were recovered and rehashed. Cleanup finished approximately9m49s after launch. The full$1cap remains an unsettled allowance, not actual billed usage. The separate completion is capped at$0.50 within existing funds.

Evidence: [analysis.json](analysis.json), [execution.json](execution.json), [recovery.json](recovery.json). The analysis explicitly labels the primary comparison incomplete and omits partial-panel headline means and intervals.
