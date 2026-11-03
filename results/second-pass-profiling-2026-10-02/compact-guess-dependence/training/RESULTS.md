# Explicit feedback: training-sample follow-up

## Result

The editor's explicit first-guess input provides no clear useful advantage even on sampled training contexts. This does not support describing the problem solely as failure to transfer an already-learned explicit-feedback rule.

Same frozen checkpoint, 128 contexts from 64 hash-selected training prompts, two anchors each; fixed first-pass states and output first guess. Donors are a within-domain permutation, all 128 guesses changed. No fitting or cloud execution.

| Input | Added union-prefix positions | Repairs | Damaged edited prefixes | Equal-prompt gain |
|---|---:|---:|---:|---:|
| Actual | 117 | 44 | 16 | 21.720% |
| Zero | 117 | 44 | 16 | 21.785% |
| Shuffled | 116 | 43 | 16 | 21.589% |

Actual-minus-zero contrast -0.065 percentage points, descriptive prompt-bootstrap interval [-0.781,+0.586]. Actual-minus-shuffled +0.130 points, interval [-0.586,+0.781]. Zero changes 29 valid editable tokens; shuffled changes 55. Actual beats zero on one context and loses on one; actual beats shuffled on two and loses on one. All other contexts tie in union progress.

The earlier development sample yielded 33 added positions in all three conditions. These two audits therefore show weak useful explicit-input dependence on both sampled splits. They do not establish trained-no-feedback equivalence. First-pass states may carry implicit guess information; input interventions change the relationship between states and visible tokens.

The large training progress gain is not a generation speedup: the original branch is retained without charging its verification cost. It is also not comparable to a fresh held-out result. The complete historical training-panel result remains unchanged.

## Numerical and verification scope

CPU FP32 execution differs from saved native BF16 predictions at 65/1692 valid editable tokens, changing union progress in six contexts and reducing its total by four positions. Results concern the CPU intervention. Independent saved-token recount, donor permutation, per-domain token-frequency preservation and unchanged output prefix checks pass. Sample identities, feature bytes and checkpoint were hash-checked. Execution took 115.16 seconds, with $0 cloud spend.

## Decision

Do not default to a larger dataset as the sole remedy for transferring an explicit-feedback mechanism that this model has not demonstrated even on training. Also do not force sensitivity to arbitrary wrong first tokens: that fixed output token would already cause rejection. The research remains about economical correction, not reproducing an internal mechanism for its own sake.

The companion [mechanism analysis](../MECHANISM.md) explains why ordinary output distillation does not identify reliance on explicit feedback, checks related primary literature, and separates that logical limitation from an unproven claim about our optimization failure. No new loss, architecture or GPU run is selected by this result.

Evidence: results.json, rows.json, donors.json, verification.json. Both this training sample and the earlier development sample use the same conditional editor; no new training occurred.
