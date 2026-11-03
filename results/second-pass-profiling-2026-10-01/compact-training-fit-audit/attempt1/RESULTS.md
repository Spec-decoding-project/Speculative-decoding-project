# Frozen editor training-fit audit

The compact editors learn useful in-sample corrections, but recover only a small fraction of the full teacher's extra positions even on training data. Weak development performance cannot be explained solely as successful teacher imitation that then fails to transfer.

All three fixed epoch-three checkpoints were evaluated without updates. Their predictions on all 2,048 saved development contexts reproduced exactly before proceeding to 12,288 training contexts. Total: 43,008 model-context evaluations. No new prompts, checkpoint selection, generation or training.

| Training method | Training progress gain | Development progress gain | Teacher extra positions recovered, training | Teacher extra positions recovered, development |
|---|---:|---:|---:|---:|
| Correct tokens | 15.28% | 3.02% | 15.8% | 16.2% |
| Teacher-correct weighting | 17.75% | 3.70% | 18.1% | 21.1% |
| Teacher probabilities | 17.08% | 3.46% | 17.4% | 19.2% |

Progress is equal-prompt, root-inclusive retained-union coverage compared with the original draft, not latency. Teacher-position recovery is the overlap of additional accepted positions, not the ratio of aggregate gains. These are different denominators and answer different questions.

## Why the teacher matters to interpretation

The full teacher itself adds 101.50% progress on training contexts but 10.21% on development contexts. The training gain uses all 384 prompts; the earlier September 22 report's 90.95% figure used a training probe and is not the same panel. This substantial teacher train/development gap existed before compression. Different prompts and teacher opportunity prevent a causal attribution of all student differences to transfer alone.

The probability-trained editor repairs 4,043 training blocks, including 4,004 of the teacher's 8,587 repaired blocks. It recovers 9,399 of the teacher's 54,132 extra positions (17.4%). On development it repairs 176 blocks, including 94 of the teacher's 353, and recovers 218 of 1,136 positions (19.2%). Repairing a block at all does not mean retaining its full corrected suffix. The extra-emphasis control remains numerically strongest on both splits; this audit adds no new independent evidence of superiority.

## Decision

### Where the missing teacher positions go

The probability-trained editor misses 44,733 teacher-extra positions on training data. Of these, 23,519 (52.6%) belong to blocks where it still gets the original first error wrong; 17,900 (40.0%) are later positions lost after an initial useful repair; and 3,314 (7.4%) are blocked by damage to an earlier prefix despite correcting the original first error. The categories are disjoint accounting, not separate causal mechanisms.

On development, it misses 918 teacher-extra positions: 735 (80.1%) remain blocked at the original first error, 123 (13.4%) follow an initially useful but incomplete repair, and 60 (6.5%) are blocked by earlier-prefix damage. It reproduces the teacher's full improved prefix in only 56 of 353 teacher-repaired development blocks, versus 564 of 8,587 training blocks. Teacher repair lengths differ substantially between splits.

This distinguishes two concrete deficits: useful first-error decisions on development examples, and sustained corrected continuations even on training examples. A preservation-only change does not address most missing progress. The measurements do not establish that reweighting, a larger editor, more training, or changed inputs would solve either deficit. [Complete decomposition](miss-decomposition.json), reproduced by `analysis/compact_fit_miss_decomposition.py`.

Do not claim that distillation already compresses the full second pass well, or that the only remaining issue is unseen prompts. There is a large in-sample teacher-correction gap too. This does not prove an information or capacity limit: training exposure, optimization, representation restrictions and objective alignment remain possible explanations. Frozen evaluation cannot isolate them.

No automatic extension, new architecture or generation benchmark follows from this audit. The concrete next analysis is to separate failure to fix the first rejected token from failure to preserve the remaining teacher-correct suffix on training data, using these recovered predictions. Any further fit should vary one justified cause rather than treating a train/development gap as a diagnosis.

## Execution and evidence

The job initially queued, then completed within its bounds. All 14 artifacts (41,384,641 bytes) were recovered and authenticated; the app has zero active tasks and the temporary volume is deleted. No retries occurred.

Elapsed through recovery and cleanup: 10.86 minutes. Conservative retained allowance: $1.25, below the $1.50 cap; billing is not settled. Remaining campaign allowance: $2.8851.

[Analysis](analysis.json) · [Execution](execution.json) · [Recovery](recovery.json) · [Accounting](cost-reconciliation.json) · [Teacher split opportunity](../teacher-split-opportunity.json)
