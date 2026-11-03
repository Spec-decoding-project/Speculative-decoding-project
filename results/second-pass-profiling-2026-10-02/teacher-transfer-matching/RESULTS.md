# Does observable repair difficulty explain the teacher transfer gap?

2 October 2026. Local saved-record analysis; no fitting, new prompt exposure or cloud execution. The preceding confidence-only reply was no progress. This turn reauthenticated all 448 feature banks and produced a new matched comparison.

## Finding

Among cases where x1 is correct and the original draft has another error, the full fixed-x1 teacher repairs 8,587/8,927 training contexts (96.19%), versus 353/1,534 development contexts (23.01%). These are useful prefix repairs, not arbitrary token accuracy.

To check whether the gap merely reflects easier observed positions, training contexts were reweighted to the development distribution within exact domain, original first-error position and valid continuation-length groups. A second specification also groups absolute context length into <128, 128–255, 256–511 and >=512 tokens. No teacher outcome enters group selection.

| Descriptive comparison | Training repair rate | Development repair rate | Development reachable contexts covered |
|---|---:|---:|---:|
| Original reachable contexts | 96.19% | 23.01% | 1,534 |
| Matched domain, error position, available length | 95.77% | 22.99% | 1,531 (99.80%) |
| Also matched context-length bucket | 95.58% | 23.03% | 1,520 (99.09%) |

In the last comparison, mean extra positions per reachable context are 6.026 versus 0.744; mean fraction of remaining possible positions recovered is 61.43% versus 8.79%. The gap therefore persists after adjusting these observed features. These context-weighted conditional metrics are NOT the equal-prompt coverage percentages used in the primary experiments.

Semantic difficulty, specific tokens, prompt distribution and teacher training exposure remain confounded. This is not proof of memorization or a causal estimate of teacher-data reuse. No inferential confidence interval treats these correlated contexts as independent. The development prompts have been repeatedly inspected.

## Input contract checked

`extract_record` obtains x1 from the reconstructed original scores and checks that its embedding equals the restored visible token. The full teacher is called with that same guessed token. `materialize` checks original and teacher winners and provides that same x1 embedding to the editor. The teacher additionally accesses committed target features, while the compact editor accesses final first-pass states. There is no demonstrated gold-x1 versus guessed-x1 mismatch in this compact feature path. Compression of the richer teacher input remains intentional, not a resolved information-sufficiency claim.

## Next comparison made more specific

Keep the fixed-x1 conditional task and 3.024M editor unchanged. Study whether the source of teacher-labeled training examples changes useful transfer:

1. Original teacher-training responses versus new responses excluded from teacher training and all development/final sets.
2. In each regime, use ordinary correct-token CE, teacher-correctness-weighted CE, and teacher-probability KL alongside CE. The weighted-label control matters: KL versus plain CE alone mixes probability supervision with extra weight on teacher-correct positions. Reuse the existing implemented coefficients and normalization.
3. Initialize all six small editors identically from a fresh initialization; do not continue the old-data-trained editor on the new-data arm and call that an equal-exposure comparison. Match distinct response counts, domain mixture, anchor policy, updates, optimizer and per-response presentations. One seed is exploratory; not a robust-method claim.
4. Keep the existing development set for continuity and predeclare fresh held-out responses for final assessment before fitting. Never promote old development responses into training. Measure teacher opportunity and correctness on the new training bank before funding all fits. Include preservation and unsuccessful teacher cases, not just successful repairs.
5. Probability-specific evidence is KL versus weighted CE within each data regime. Improvement in every loss on new data is a data effect. Better teacher imitation without more held-out prefix progress is not success. Changing datasets cannot by itself isolate memorization. This is holdout-data distillation, not full cross-fitting.

The rationale has primary-literature support: [Dao et al., Knowledge Distillation as Semiparametric Inference](https://arxiv.org/abs/2104.09732) studies teacher estimation effects and cross-fitting on tabular/image tasks. It does not establish a cure for this drafter.

## Readiness and remaining work

This supports a specific test of the transfer explanation, not confidence in a positive training outcome. It is NOT launch-ready. Dataset exclusion, response count, native collection cost, available disk, teacher opportunity and exposure sizing still require concrete checks. Do not adopt a tiny fixed training dose merely to obtain a cheap negative. Collection can dominate the small fits; the earlier cached-fit timings do not price fresh response generation.

No automatic variable-prefix retraining, objective sweep or architecture expansion follows. Prior closed recipes and their results remain unchanged. Useful deployment must ultimately beat equally optimized alternatives with complete runtime accounting.

## Reproduction

`analysis/audit_compact_correction_exposure.py` reauthenticated the underlying banks and regenerated all 14,336 rows. `analysis/audit_teacher_transfer_matching.py` checks identities, coverage and two equivalent implementations of the standardization. `results.json` records all strata, source/code hashes and limitations. No new cloud cost. Goal remains active.
