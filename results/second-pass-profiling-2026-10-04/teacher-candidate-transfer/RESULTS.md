# Available candidates do not explain the teacher's transfer failure

This is a $0 analysis of the completed training-utility diagnostic. It changes the next decision: do not train a student on these checkpoints or fund a confidence filter for their saved predictions. Useful second-pass compression still requires a stronger source of transferable corrections.

## Question, mechanism and counterargument

Are development repairs missing because the original draft cannot even propose the right token, or because the teacher fails to choose it reliably? If candidate availability differs substantially between training and development, that could explain some of the gap. Conversely, similar availability does not prove equivalent context difficulty, a particular internal mechanism, or that a small classifier can make the right choice.

The saved candidate scores come from the released readout **rescored on the completed original draft**. They are not target-model likelihoods, not the teacher's scores, and not necessarily the scores that produced the previous iteration's original token. This explains why some originally wrong tokens have a correct alternative ranked first after rescoring.

## Correct alternatives are usually already present

Among editable first errors, excluding locked first-guess errors and originally complete blocks:

| Released rank of correct token | Training opportunities | Development opportunities | First-guess teacher repairs, training | Repairs, development |
|---|---:|---:|---:|---:|
| 1 | 13 | 14 | 6 | 4 |
| 2–4 | 118 | 124 | 65 | 30 |
| 5–16 | 22 | 23 | 6 | 1 |
| Outside saved 16 | 12 | 10 | 2 | 0 |

The correct token is in the top four in **131/165 training cases (79.4%)** and **138/171 development cases (80.7%)**. The first-guess teacher repairs **34/138** top-four development opportunities. Within ranks 2–4, its repair rate falls from 55.1% on training to 24.2% on development. Before training those same counts were 16/118 and 31/124. Thus training learned many choices on familiar contexts while not improving that large development stratum.

The whole-draft teacher repairs 41/118 versus 20/124 rank-2–4 cases. Fixed rank-group standardization would predict approximately 82.7 first-guess repairs or 53.7 whole-draft repairs if training rates carried over to the development rank mix; actual counts are 35 and 25. This is descriptive adjustment for one feature, not a causal or deployable prediction.

All 41 first-guess development regressions and all 94 whole-draft regressions displace a correct token ranked first by the released scores. Being ranked first does not establish certainty. No conclusion that hidden information is unavailable follows from this audit.

## Could the released score preference filter harmful changes?

For each helpful or harmful prefix edit, compare the released log-probability of the new token with the original token at their first divergence. Both histories agree before that point. Among pairs present in the saved top16, higher support for the proposed change discriminates helpful from harmful edits with development AUROC **0.727** for first guess (prompt-bootstrap 95% interval 0.624–0.819) and **0.786** for whole draft (0.706–0.859).

That supports a limited observation: harmful changes tend to fight a stronger existing preference. It does not validate a filter. Helpful and harmful changes overlap, neutral-prefix changes are excluded from this conditional analysis, six whole-draft harmful changes are outside the saved shortlist, and no threshold was tuned. The same reused development panel cannot validate a new learned policy.

More importantly, filtering damage would not establish enough new value. The strong comparator already covers the entire corrected prefix in **33/35 first-guess repairs** and **22/25 whole-draft repairs** on development. Avoiding a bad edit does not automatically add useful progress beyond that comparator.

## Stronger opportunity bound: perfect tokenwise mixing

As an exploratory bound, give an impossible oracle the correct answer at every position. Let it choose the correct token from either the original or the teacher output, even when each whole chain contains errors. Then add this mixture to the strong comparator for free.

| Saved teacher | Extra development positions beyond strong control | Equal-response coverage gain, with zero extra cost |
|---|---:|---:|
| First guess | 17, across two contexts | 0.64% |
| Whole draft | 12, across three contexts | 0.59% |

Training counterparts are 107 positions / 5.51% and 65 positions / 2.99%. The first-guess development bound's descriptive interval is 0–1.68%; whole draft 0–1.42%. These do not predict runtime savings.

This extends, rather than contradicts, the earlier bound on whole-chain selection. It closes the objection that substantial existing teacher value might merely be hidden behind a few damaged earlier positions. It applies only to the two saved top-token sequences on these contexts. It does **not** bound alternative logits, repeated redrafting, newly trained models, or a student that learns different predictions.

## Decision and next research requirement

Candidate absence is not the main observed limitation. The teachers fail to transfer useful choices among already plausible alternatives, while frequently changing existing correct choices. Confidence information has some predictive value, but even perfect preservation and selection among their saved tokens leaves little incremental development coverage. A confidence gate or another student imitation fit is therefore not justified for these checkpoints.

Keep the second-pass compression objective. Before another fit, the next teacher task must produce materially useful corrections on untrained responses relative to a strong comparator, or show a credible route to equivalent useful progress at substantially lower cost. Do not use improvements over a disrupted teacher input or a weak single chain as evidence of that requirement. These results do not establish whether changing data, optimization, model capacity or conditioning will achieve it.

A switch to an author-trained second-pass checkpoint could separate our limited teacher-training recipe from the redrafting hypothesis, but its availability must be established first. On this check, [D²SD's author repository](https://github.com/catnanami/D2-SD#models) still says pretrained checkpoints will accompany a future camera-ready release and directs users to their own checkpoints. That README is not evidence of a currently usable downloaded teacher, nor proof that none exists elsewhere. Do not launch a reproduction on an assumed asset or repeat the closed variable-prefix experiment without a new distinguishing premise.

## Verification and resources

137 source files authenticated against the completed recovery manifest. Candidate indexing/censoring checks, independent grouped counts, known-answer AUROC cases, and enumeration of all mixtures in a small counterexample pass. The complete mixture-bound analysis reproduces exactly. Sampling uncertainty uses prompts, not individual token edits; other counts are descriptive.

No GPU run, training, file deletion or spending. The campaign remains at **$1.00 unallocated**, with no active job. The previous goal turn was progress; this turn adds evidence that rules out a candidate-availability explanation and a substantial saved-token filtering repair. The overall research goal remains active and unmet.

Evidence: [candidate audit](analysis.json), [score-preference analysis](margins.json), [token-mixture bound](token-mixture-oracle.json), [verification](verification.json). Reproduction code: `analysis/analyze_teacher_candidate_transfer.py`, `analysis/analyze_teacher_edit_margins.py`, `analysis/teacher_token_mixture_bound.py`.
