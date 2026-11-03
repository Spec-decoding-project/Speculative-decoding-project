# The conditional-response diagnostic does not directly explain the original editor's failed repairs

The earlier score-response mismatch is real, but the deployment distinction is decisive. All 68 highlighted useful teacher switches occur where the original first guess is wrong and the supplied alternative is correct. The original deployment keeps its original first guess in both proposals. It therefore stops before reaching any of these 68 suffix corrections. These cases do not directly diagnose its missed useful repairs.

This is a qualification of the learning recommendation, not a reversal of the measured score differences or a claim that input-response training could never help other cases. It narrows when that evidence matters: an editor actually serving alternative first guesses would need to respond to those guesses. The original fixed-first-guess policy does not require the same counterfactual behavior.

## Recount

All 2,048 contexts in the previous diagnostic were joined to the authenticated saved full-teacher predictions. No model ran and no data or checkpoint changed.

| First-guess relationship to the recorded target continuation | Contexts |
|---|---:|
| Original guess correct | 1,717 |
| Original wrong, second candidate correct | 147 |
| Neither candidate correct | 184 |

All 68 highlighted teacher switches are in the second group. Of these, 60 correct the token immediately following the changed guess, seven correct the next position after that, and one corrects a position four tokens later. This describes selected useful switches; it is not the distribution of all teacher corrections or proof of a general linguistic mechanism.

For a readable example, the recorded continuation begins “of customer service.” Under the original first guess “a,” the teacher proposes “of a customer.” When supplied “customer” instead, it moves “service” into the following slot. The score margin for “service” over the old wrong token changes by +16.5; the student's corresponding margin changes by only about +0.062. This demonstrates a specific conditional mismatch. Yet neither proposal can help the original fixed-first-guess policy while it retains “a.” The remaining teacher continuation is also imperfect; this is not a wholly repaired block.

Four examples were selected by a deterministic hash of context identity, rather than by reading their text first. All 68 decoded cases are retained in `cases.json`. Token fragments, including partial words, must not be presented as complete linguistic units.

## Consequence for the next learning decision

Do not train a paired-input student merely because it fails this diagnostic. First specify a competitive deployment policy that needs those input changes, or demonstrate a learning benefit on the unchanged original-input policy with matched controls. A weak response to an unused input is insufficient causal evidence for why ordinary-input student repairs failed.

The stronger-backbone screen currently running keeps each context's own first guess fixed across the full-pass and cheap-head routes. It evaluates corrections that those actual proposals could use. Its result remains necessary before choosing this released model as a new compression teacher.

The compact student's correct-token loss covered all valid suffix positions, including contexts with a wrong original first guess. Do not confuse that with the historical full teacher's training eligibility. The relevant input-support limitation is that the student saw only the original explicit guess for a given saved state.

## Evidence and limits

`results.json`, `cases.json` and `verification.json` preserve the counts, examples and source hashes. All 2,048 teacher token arrays match the authenticated historical recovery artifact. All 68 original-wrong/alternative-correct relationships and correction positions pass the independent join. Tokenizer is the local Qwen3 tokenizer used for the original collection.

No new estimate of generation speed, capacity limits, generalization or novelty follows. No training, GPU use or spending occurred. Existing $3.50 authorization remains unchanged. The second-pass compression goal remains active.
