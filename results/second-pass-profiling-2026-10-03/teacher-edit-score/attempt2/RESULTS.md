# The full second pass supplies useful scores, but the economical margin is narrow

## Completed comparison

One A100-80GB PCIe attempt completed all 2,048 saved contexts from 64 reused development responses. It reconstructed native BF16 scores from saved second-pass states; it performed no transformer generation, training or checkpoint updates. All reconstructed teacher argmax tokens matched both historical references. Independent analysis reproduced every selected edit and coverage count.

The primary comparison changes only which model supplies probabilities to the same one-substitution rule. Both choose from the original drafter's top four tokens, preserve its first guess and retain the original branch. The selector never receives target labels, validity masks or observed rejection positions. Labels are used afterward to measure matching prefixes.

| Score source for choosing one edit | Extra matching-prefix coverage over the original draft |
| --- | ---: |
| Original drafter | 7.048% |
| Full second-pass teacher | 8.725% |

Paired difference: **+1.677 percentage points**, descriptive 95% interval **[+0.723, +2.643]**. Math improves from 7.773% to 9.673%; chat from 6.323% to 7.777%. Teacher scoring improves 220 contexts and worsens 116 relative to the original-score rule; 1,712 tie. It gains 426 matching positions and loses 258, net +168. It repairs 421 original-prefix failures versus 317 for original scores.

These are fixed-context coverage measurements, including the known root and retaining both proposals. They are not generated-token savings, a measured speedup or evidence a cheap student can reproduce the decisions. Positive paired development evidence passes the protocol's score-quality condition; the protocol's economic condition remains unresolved.

## Stronger ordinary-tree control

A conventional original-score tree still exceeds the single-edit policies at identical actual node budgets: 13.142% versus 7.048% for the original-score edit, and 12.886% versus 8.725% for the teacher-score edit. Equal node counts do not imply equal construction or verification time.

An explicitly exploratory follow-up substitutes teacher scores into the existing 30-node tree construction, leaving candidate tokens, original-chain preservation, first-token restriction and budget unchanged:

| Tree scoring | Extra matching-prefix coverage |
| --- | ---: |
| Original scores | 14.158% |
| Second-pass scores | 17.075% |

Paired difference: **+2.917 percentage points**, descriptive interval **[+2.041, +3.794]**. Math: 13.254% to 17.037%; chat: 15.062% to 17.113%. Teacher scoring improves 302 contexts, worsens 191, and ties 1,555; net +358 matching positions. Independent frontier enumeration reproduces the prior original-score trees exactly on all 2,048 contexts. Eighty exhaustive small-tree checks also pass with unsorted teacher candidate scores.

This is a controlled change to candidate selection on the saved panel, not an isolation of x1 conditioning. Teacher fine-tuning, model capacity and second-pass computation remain bundled. Nor are these teacher scores path-conditioned on every possible branch: they remain positional scores from the recorded second-pass input.

## Which part of the edit decision helps?

A post-hoc factorial check holds either the chosen position or the replacement-token ranking fixed:

| Position selected by | Replacement selected by | Coverage gain |
| --- | --- | ---: |
| Original scores | Original scores | 7.048% |
| Teacher scores | Original scores | 7.503% |
| Original scores | Teacher scores | 7.884% |
| Teacher scores | Teacher scores | 8.725% |

Replacing token ranking alone adds +0.836pp [0.208, 1.483]. Changing position alone adds +0.455pp [-0.213, 1.146], with a negative chat point contrast. The interaction is +0.385pp [0.088, 0.707]. This supports replacement choice as part of the observed benefit; it does not identify a learnable feature or prove a universal decomposition. These hybrids hold the original policy's position fixed and do not re-optimize the complete policy under mixed scores.

## Cost and interpretation

Under an equal-response cost-per-progress model, preserving all observed teacher-policy benefit permits only **1.495%** additional round cost over the original-score single-edit rule before breaking even. For tree scoring the corresponding margin is **2.437%** [1.687, 3.191]. A perfect zero-cost imitation of this observed tree-score improvement would give about 2.38% lower modeled cost per matching position versus the ordinary tree, below the campaign's 5% ambition. These are conditional arithmetic, not measured latency or a ceiling on other possible learning outcomes. Imperfect transfer narrows the margin.

The strongest supported conclusion is therefore: **the second pass provides some useful ranking information within an unchanged candidate set.** It has not shown that another compact network can recover that information economically or that this constitutes an original method.

## Decision and next useful check

Do not start another training run yet. We already have frozen matched mixture and teacher-only students. Their historical evaluation saved argmax branches; improved probability imitation did not improve that branch metric. This new result makes it useful to check those existing score distributions with the same one-edit and 30-node tree rules before inventing or fitting another editor. That is inference-only and directly asks whether any second-pass ranking benefit was already learned but not exposed by the previous output rule.

Preserve both complete matched checkpoints and all 64 responses; no checkpoint, temperature, threshold or mixture search. Require historical native prediction parity before interpreting scores. Compare against original scores and the now-measured teacher, and retain the original negative argmax findings. A benefit must also survive complete added cost and comparison with an equally optimized ordinary tree. A small scorer trained from scratch would require a distinct rationale and evidence beyond generic learned selection.

## Relation to existing work

[OPT-Tree](https://arxiv.org/abs/2406.17276) already optimizes expected acceptance through tree construction. [TAPS: Target-Aware Prefix Tree Selection](https://arxiv.org/html/2606.00487v1), sections 4.2–4.4, already trains a lightweight scorer on target verification feedback and uses it for prefix-tree selection. Therefore learned candidate scores, prefix-aware selection and distillation are not themselves novel. Our remaining question concerns compressing useful second-pass computation and demonstrating a cost advantage; changing the training teacher alone would not establish that advantage. This TAPS is distinct from the task-specialist paper with the same acronym at arXiv:2603.27027.

## Execution and evidence

The separately recorded attempt followed a preserved pre-GPU upload failure. It completed and cleaned up in **3m25s**, including 131 seconds of worker analysis. Five artifacts totaling 6,008,915 bytes were recovered and hash-verified. App `ap-RZswzHmnER4yamm9mS0jm9` is stopped with zero active tasks; exact owned volume `vo-gAEFXUT01uEUFDFHhKc7ca` is absent. The full $1 authorization remains a conservative unsettled allowance, not an invoice; $6.78506668 is unallocated and there are no active job reservations.

Evidence: [primary analysis](analysis.json), [matched-node controls](control-analysis.json), [tree-score follow-up](tree-score-analysis.json), [decision decomposition](decision-decomposition.json), [verification](verification.json), [execution](execution.json), [recovery hashes](recovery.json). All intervals are descriptive because this development panel has been used repeatedly, and the tree/decomposition studies were post-hoc. No new training or generation run has been launched.
