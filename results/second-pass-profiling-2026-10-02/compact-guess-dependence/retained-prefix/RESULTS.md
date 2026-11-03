# Does the retained-prefix editor use its explicit first guess?

## Result

A small useful contribution is present in this sampled reconstruction, but it is not a substantial repair mechanism yet. The real guess adds two retained positions over either removing or permuting the explicit guess. These are two contexts from different prompts; the other 126 contexts tie in union progress.

| Explicit input | Extra retained positions over original | Edited prefixes improved / damaged | Equal-prompt union progress gain |
|---|---:|---:|---:|
| Actual first guess | 33 | 18 / 33 | 4.9138% |
| Zero embedding | 31 | 17 / 32 | 4.8060% |
| Other-prompt guess | 31 | 17 / 32 | 4.8060% |

Actual-input advantage is 0.1078 percentage points. The descriptive prompt bootstrap interval is [0, 0.2694] points, not confirmatory inference on fresh data. Union means retaining the original branch as a fallback; damage counts describe the edited branch and are not subtracted directly from union coverage. This experiment does not price that fallback.

## Design and checks

Frozen retained-prefix epoch-three checkpoint; 128 previously selected contexts from 64 exposed development prompts. No fitting, GPU work, or prompt reselection. First-pass representations and the first guess in the proposed output stay fixed. Only the explicit embedding input changes. Donor permutation stays within domain and changes the token identity in 125 contexts.

The actual guess changes 35 valid tokens versus zero and 49 versus the shuffled input. Useful progress changes are far rarer than token changes. An independent recount checks full saved tokens, accepted prefixes, donor permutations, unchanged output prefixes, and aggregate statistics. It does not independently validate the neural forward calculation.

The useful cases are gsm8k-3863 anchor 4 (progress 14 versus 13) and ultrachat-12609 anchor 21 (16 versus 15). Both actual-input token sequences match their saved native evaluation exactly. The zero/shuffled counterfactuals themselves have not been evaluated natively on GPU.

Across the complete sample, CPU FP32 reconstruction differs from native BF16 on 31 of 1,763 valid editable tokens, with two additional union positions in one other context. These discrepancies remain a limitation; the experiment is not a native serving result. Runtime was 125.39 seconds; additional paid execution $0.

## What this says about the explanation

The ordinary probability-trained editor had 33 extra positions with each input condition on the same sampled development contexts. The retained-prefix objective excludes examples whose immutable first guess is already wrong, but also changes positional weighting, smoothing, and normalization. Therefore the new result cannot attribute better input dependence to exclusion alone.

The prior full-panel retained-prefix comparison remains controlling for overall performance: 3.1700% progress gain versus ordinary CE's 3.0217%, a 0.1483-point difference with an interval spanning zero. The selected 128-context input audit must not replace that 2,048-context comparison. Cross-checkpoint differences are not a clean mediation experiment.

This narrows the conclusion: useful explicit feedback is not wholly absent, but the tested training package has not turned it into substantial additional progress. Wrong-first-guess examples cannot be treated as an established explanation or an already-demonstrated fix. Information about the guess also remains implicit in the saved states, and interventions disrupt the normal relationship between inputs. Nothing here establishes that another training procedure cannot work.

## Decision

Do not repeat this retained-prefix fit, launch native generation solely for this two-position effect, or claim a masking-based contribution. The remaining method-level challenge is learning substantially more economical corrections, not proving that the network reacts to an input. A new training proposal must identify an observable failure it changes and a comparison that isolates that change; renaming existing loss terms does not meet that standard.

Artifacts: results.json, rows.json, donors.json, verification.json. Original checkpoint and full-panel outcomes remain preserved.
