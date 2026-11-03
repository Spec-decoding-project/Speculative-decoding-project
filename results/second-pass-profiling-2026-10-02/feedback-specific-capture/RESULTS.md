# Which teacher improvements did the compact editor recover?

The existing large-model comparison already established +10.208% retained progress with first-guess feedback versus +8.051% without it, a +2.157 percentage-point difference under matched training. The new analysis joins those outcomes to all four final compact editors on the same 2,048 contexts and64development prompts. It does not rerun or reclassify the original experiment.

## Teacher position partition

Of1,136positions added by the feedback teacher,729are also added by the no-feedback teacher and407are not. The no-feedback teacher adds another221positions outside the feedback teacher's extension. Thus the pooled net difference is407-221=186positions. These pooled counts and the equal-prompt2.157point contrast use different weighting.

The groups are **model-specific**, not proof of which individual tokens causally require feedback. The teachers underwent separate matched training, so both parameters and their input processing differ at evaluation. The no-feedback model also benefits from continued training and a second computation; this is not a pure decomposition of universal language mechanisms.

## Compact editor capture

| Compact training | Shared positions captured, out of729 | Feedback-model-only, out of407 | No-feedback-model-only, out of221 | Additional positions outside both teachers |
|---|---:|---:|---:|---:|
| Correct labels |145|39|26|137|
| Teacher-correctness-weighted labels |191|49|25|154|
| Teacher probabilities |178|40|31|137|
| Retained-prefix objective |142|34|29|158|

The probability-trained editor captures24.4%of shared teacher positions but9.8%of feedback-model-only positions. The weighted-label control captures more of both groups. Different group difficulty, extension length and task composition prevent interpreting these rates as a causal explanation. In particular, they do not prove the small model ignores its first-token input.

The probability-trained editor also adds137positions outside both teacher extensions—35.5%of its386total added positions. This is direct evidence that teacher retention alone is an incomplete success metric. It does not imply that those improvements can simply be combined with the teacher's missing corrections without interference.

## Research implication

Do not describe the current compact result as demonstrated compression of the distinctive feedback benefit. Its gains remain real fixed-context observations, but the mechanism is not isolated. Conversely, do not judge every future trained model against a teacher-derived projection as a universal ceiling: the existing students already produce some improvements outside both teachers' prefixes.

A focused remaining question is whether the trained editor's useful changes depend on its explicit first-guess input. A frozen-model input ablation can measure that computational dependence while holding first-pass states and weights fixed. It cannot replace a matched training-without-feedback comparison or establish information absence: the guessed token is already a deterministic function of the first-pass scores, and removing an input can create distribution shift. A correct-input advantage over a wrong or missing input alone also does not establish a speed advantage over an equally cheap model.

This identifies a mechanism audit, not a recommendation to add another architecture, rerun training automatically, or fund another objective sweep. No paid work occurred. All source hashes are recorded, and an independent explicit set-intersection recount reproduces every teacher/student count. Research remains active.

Evidence: `results.json`, `verification.json`; reproduction: `analysis/audit_feedback_specific_capture.py`. The full-teacher results remain in `notes/conditional-full-reference-precision-2026-09-22/results.md`.
