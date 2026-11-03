# Useful learning exists, but the current full teachers transfer it poorly

The first-guess teacher learned useful prefix repairs on the training sample. Its improvement did not carry over to the saved development panel. The completed-draft teacher recovered much of the damage caused by its new input, but remains a weak compression target even on training examples. These are inference-only measurements of already trained checkpoints, not a new training attempt or a generation speedup.

## What completed

All 256 preassigned training contexts from 64 responses completed under four conditions: before/after training, with first-guess-only/completed-draft input. The same strong released conditional comparator was measured on each context. All 256 original first drafts reproduced; all 32 preassigned historical development vectors reproduced. Frozen-weight audits passed. The two saved checkpoints are unchanged.

The saved schedule confirms that 229 sampled contexts were eligible and actually presented during training, with 260 total presentations. All 256 remain reported, including fully correct drafts and the 27 untrained wrong-first-guess contexts. First guesses were never replaced with correct labels. Independent recount and all 214 recovered file hashes pass.

## Repairs versus damage

Each panel has 256 contexts. A repair means a longer correct prefix than the original draft; damage means a shorter one. These are raw edited chains. Keeping the original proposal prevents those raw regressions from automatically becoming losses in accepted proposal coverage.

| Teacher and panel | Repairs before training | Damage before | Repairs after training | Damage after | Fully correct blocks damaged after |
|---|---:|---:|---:|---:|---:|
| First guess, training | 26 | 27 | 79 | 33 | 12 |
| First guess, development | 41 | 23 | 35 | 41 | 13 |
| Completed draft, training | 13 | 149 | 51 | 60 | 31 |
| Completed draft, development | 19 | 144 | 25 | 94 | 39 |

For eligible examples, first-guess training increases equal-response raw prefix length by **17.34%** versus its untrained version (descriptive 95% interval 9.66–25.57%). On eligible development examples it decreases it by **4.11%** (−7.47 to −0.82%). These are paired checkpoint effects within each panel, not a randomized causal comparison between training and development.

Completed-draft training increases prefix length by 71.45% on eligible training inputs and 32.00% on eligible development inputs. Those large percentages are relative to its badly disrupted untrained completed-draft condition. They are **not** improvements over the original drafter or strong comparator. Its raw training chain still contains only 1,968 matching prefix positions versus 2,055 for the original chain; development is 1,657 versus 2,049.

## Value beyond the strong released method

Giving each teacher's saved chain to the strong comparator for free, and keeping all useful alternatives with hindsight, adds:

| Teacher | Training: added positions / response-average gain | Development: added positions / response-average gain |
|---|---:|---:|
| First guess | 102 / 5.25% | 14 / 0.46% |
| Completed draft | 64 / 2.90% | 12 / 0.59% |

These are all-context summaries. Restricting to eligible inputs gives 5.78% and 3.58% on training, versus 0.46% and 0.59% on development. No additional time or node budget is charged. These are therefore optimistic opportunities for these particular saved greedy chains, not speedups or general capacity limits.

The original-plus-edited pair alone remains below the stronger comparator even on training: −7.01% for first guess and −13.61% for completed draft across all contexts. This compares coverage only; candidate count and runtime differ. It does not establish a latency ranking.

## Alternative explanations checked

Original draft accuracy is similar across the sample and development: mean prefix 8.027 versus 8.004, and 64 versus 62 fully correct blocks. This does not establish equivalent difficulty: strong conditional coverage is 2,531 versus 2,700 positions.

The first-guess learning/transfer discrepancy occurs in both domains. Training adds 80 math prefix positions and 96 instruction positions versus the untrained teacher; development loses 92 and 38 respectively. It also appears within coarse original-error groups: early-error training cases gain 134 positions, whereas early-error development cases lose 30. Training gains are not confined to repeated examples: the 204 examples presented once gain 121 positions. These are descriptive strata, not causal controls or evidence for a specific data-scaling law.

The completed-draft teacher spends much of its improvement restoring already-correct drafts: 320 of its 743 net training positions come from originally complete blocks. Its remaining damage and weaker development repairs mean the evidence supports both an input-adaptation problem and poor transfer; it does not identify the responsible internal operation.

## Decision for second-pass compression

Do not train a smaller student to imitate either checkpoint on this evidence. This diagnostic rejects the blanket claim that the full teacher cannot learn useful corrections, but also rejects treating a lower training loss as sufficient teacher quality. The first-guess teacher exposes useful training-specific corrections; the completed-draft teacher has substantial disruption to overcome. Neither supplies enough demonstrated additional development value to justify compression now.

The next mechanism question is what distinguishes the corrections that remain useful across prompts from those that only improve trained examples. Start with the retained tokens, candidate scores, training exposure and context metadata; establish a concrete signal before choosing a change to data or supervision. Do not infer that more epochs, a different head, unfreezing the refiner, or simply feeding the whole draft is the remedy. A matched single-pass training comparison would still be needed before attributing transferable gains to draft feedback itself.

One training seed, reused development data, 64 sampled training responses, coarse difficulty checks, no convergence test, and no randomized exposure intervention limit attribution. Compression remains an active, unmet objective.

## Execution and evidence

GPU calculation: 91.76 seconds. Including staging, recovery and cleanup: **4 minutes 16 seconds**. App `ap-Q2SfGqQBjjUoLSANoSJ577` stopped with zero active tasks; owned volume `vo-rh5f8I9uNt8F0gllqVZBVH` is absent. Recovered 214 files, 35,317,016 bytes. No optimizer updates, retries or target generation.

Provider-reported compute: **$0.1011624**, pending invoice. Retain the full **$1.50** accounting hold; **$1.00** remains unallocated within the existing $40 goal authorization. No historical holds were released.

Evidence: [frozen protocol](PROTOCOL.md), [primary analysis](attempt1/analysis.json), [independent verification](independent-verification.json), [sample balance](sampling-balance.json), [descriptive strata](descriptive-strata.json), [execution and recovery](attempt1/execution.json), [metering](attempt1/metering.json). Source and input hashes, checkpoints, fixed sample and exposure records are in the preserved launch snapshot.
