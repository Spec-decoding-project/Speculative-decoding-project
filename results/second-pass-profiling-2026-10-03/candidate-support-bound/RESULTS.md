# Wider original candidate lists do not automatically repair missing paths

All 2,048 saved development contexts were analyzed for $0. This is a bound on the unchanged original-score selector, not a learned method or generation speedup.

The original tree includes its mandatory 16-node chain and 14 additional nodes. A token omitted from the sorted top-four candidates cannot score above the fourth candidate. Multiplying that bound by its parent's path probability bounds every omitted child. If it falls strictly below the lowest selected optional path score, the child cannot enter the current 30-node tree. Descendants cannot have a higher probability than their parent. Adding more candidates elsewhere cannot lower the selection cutoff.

| Observation | Contexts |
|---|---:|
| Entire tree certified unchanged under candidate widening | 781 / 2,048 |
| Next correct continuation absent from top four | 779 |
| Of those, widening alone still cannot admit it at this budget | 457 |
| Teacher chain helps where that next token is absent | 47 |
| Of those 47, original scores still exclude the token under widening | 30 |

The remaining cases are unresolved, not demonstrated benefits from widening. Different scores, a larger checking budget, or a different output construction are outside this certificate. In particular, teacher rescoring could make a previously excluded candidate useful.

Validation: all historical original-tree progress values reproduced. Exhaustive full-vocabulary enumeration checked 828 generated small examples (172 others could not fill the chosen limited-tree budget). An additional exact-decimal audit checked 56,501 outward-rounded interval bounds, candidate score ordering, and original argmax membership. It reuses tree construction; the exhaustive enumerator is the independent selection check. No acceptance tolerance was changed.

Decision: do not present a wider shortlist as an established repair for the editor. First distinguish candidate availability from ranking changes. The broad cheap-second-pass question remains open; no training or paid diagnostic is authorized by this result alone.

Evidence: [results.json](results.json), [verification.json](verification.json). Reproduce with Python 3.11: `analysis/audit_candidate_support_bound.py`, then `analysis/verify_candidate_support_bound.py`.
