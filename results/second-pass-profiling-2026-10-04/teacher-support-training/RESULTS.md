# Broader teacher training did not improve the starting model

All three matched fits and all planned evaluations completed. The frozen advancement screen failed under both proposal policies. Supplying token feedback remains useful in this setup, but adding formerly excluded examples did not establish an improvement over the starting teacher or equal-exposure old-support training. No new generation speedup was measured.

## Question and comparison

The old teacher's suffix loss excluded 2,274 of 12,288 training contexts because the original first guess was wrong. That was appropriate when the output had to retain that guess. Allowing a second first-token candidate made those contexts relevant, but did not prove that their exclusion caused the teacher's remaining errors.

Three arms began from the same saved feedback-trained checkpoint. Each completed 384 optimizer updates over the same response order with exactly matched per-update supervised-position weights:

- Old-support feedback: replace excluded examples with eligible examples from the same response, matching supervision length.
- All-support feedback: include every original training context and reveal its correct first token during training.
- All-support no feedback: use the same broader examples and labels, with that input masked.

Evaluation never supplies a correct token by looking at the answer. It uses the original drafter's first two candidates, 2,048 saved contexts from 64 reused development responses, and the fixed final checkpoint. The target, original drafter, embeddings and output projection remain frozen. Initial feedback and masked-input predictions were also evaluated before training. This is one continued-training dose and one seed, not a convergence or from-scratch comparison.

## Competitive results

The primary policy keeps the original chain and allocates exactly 30 nodes using branch-conditioned scores over the original candidate support. Its comparison is an ordinary 30-node tree from the same original drafter.

The percentages below are **modeled time-per-progress savings if complete round costs were equal**. They are computed from matching-prefix coverage with equal response weighting. They are not measured latency, and they do not charge the added teacher computation.

| Condition | Modeled savings versus ordinary tree | Descriptive 95% prompt interval |
|---|---:|---:|
| Starting teacher, feedback present | 4.107% | 3.124% to 5.106% |
| Same starting weights, feedback masked | 1.893% | 1.058% to 2.767% |
| Extra old-support training, feedback | 3.556% | 2.571% to 4.555% |
| Broader training, feedback | 3.837% | 2.822% to 4.837% |
| Broader training, no feedback | 1.641% | 0.920% to 2.365% |

The broader-feedback arm versus:

- **Starting teacher:** -0.300% modeled savings, interval [-0.806%, +0.198%]; 232 matching positions gained and 268 lost.
- **Old-support continuation:** +0.269%, interval [-0.277%, +0.794%]; 255 positions gained and 220 lost. This does not establish a reliable training-support advantage.
- **Same broader data without feedback:** +2.226%, interval [+1.519%, +2.937%]; 508 positions gained and 234 lost. Feedback helps at this matched exposure. The feedback arm evaluates two conditional distributions; the masked arm reuses one, so their runtime costs differ.

The predeclared secondary policy also permits new full-vocabulary tokens by adding both teacher chains and matching the node budget across all conditions. Broader feedback gives -0.609% modeled savings versus ordinary expansion, interval [-1.628%, +0.422%]. Math is positive and chat negative. Its node budget differs from the previous two-condition study because this experiment matches all five conditions; preserve both studies separately.

Neither policy reaches the frozen 10% free-cost advancement screen. Under the primary equal-round-cost model, the broader-feedback point estimate would not reach the eventual 5% latency goal even before charging extra teacher work. This is a result about these policies and contexts, not an upper bound on redrafting in general.

## What happened in the targeted cases

In 147 contexts, the original first guess is wrong and its second candidate is correct. These reference-selected cases are an explanatory subset, not a serving selector.

- Starting teacher: 293 consecutive correct suffix positions after the visible first token.
- Broader-feedback teacher: also 293; 27 positions gained and 27 lost, across 15 improved and 14 worsened chains.
- Both still fail immediately after that visible token in 55 cases and complete all valid positions in six.
- At the fixed tree budget, this subset falls from 515 to 499 matching positions: 10 gained, 26 lost.

The added examples therefore did not provide the hoped-for net improvement on the very subset motivating this test. Compared with the matched no-feedback arm, however, broader feedback adds 113 suffix positions on this subset. Feedback is useful; this particular data-support change is not an established remedy.

These observations do not identify a unique cause of the learning limitation. They do not prove insufficient capacity, missing information, convergence, or that more data can never help. They do rule out presenting this completed recipe as a successful fix.

## Validation and operational deviations

All three checkpoint file hashes and actual weight-state digests pass. Actual per-update supervision weights match exactly, all arms complete 384 updates, initial native prediction checks pass, and evaluation leaves trained weights unchanged. The frozen target/original-drafter version audit passes. Evaluation journals match complete files, with no duplicated evaluation rows.

An independent dictionary-frontier implementation reproduces 10,240 primary and 12,288 secondary endpoints. The first checker run exposed a Python-version difference: Python 3.12 changed floating-point `sum`. Explicit sequential accumulation reproduces the frozen Python 3.11 calculation. The unchanged Python 3.13 default changes three of 10,240 primary endpoints; the candidate's modeled estimate becomes 3.844% instead of 3.837% and still fails the screen. Frozen analysis and thresholds were not rewritten. See [Python's documented change](https://docs.python.org/3/library/functions.html#sum), `attempt1/summation-diagnosis.json`, and `attempt1/python-summation-sensitivity.json`.

A pre-training interruption during bank hashing was followed by progress in the same submitted GPU call, consistent with Modal's automatic container restart. The exact provider interruption reason was not separately confirmed. `retries=0` did not enforce the promised no-provider-restart behavior; this deviation is retained. No second call was submitted, no training preceded that interruption, and the original absolute deadline remained in force. [Modal execution semantics](https://modal.com/docs/guide/functions)

The restart also left an old inventory file. The final inventory hashed that old file before overwriting itself. Ordinary recovery correctly rejected the resulting self-entry. Recovery-only reconciliation required that entry to match the separately preserved interruption inventory; all other file sizes and hashes remained unchanged. Both records and the initial recovery failure remain saved. This was not a rerun or a change to scientific evidence.

## Resources and decision

The three fits took 537.34, 558.94 and 561.98 seconds. Submission through recovery and cleanup took 56.11 minutes; compute was stopped before the large checkpoint downloads finished. All 30 artifacts, 3,471,587,969 bytes, are authenticated locally. The app is stopped with zero tasks and its owned temporary volume is absent.

Retain the full **$4 accounting allowance**, not a claimed invoice. **$24.50 remains unallocated from the newly authorized $30.** No follow-up paid work was launched.

Close this exact continued-training recipe. Preserve all checkpoints and the useful feedback effect. Do not automatically extend training, repeat a data/loss/size sweep, or fund a cost-only optimization of this policy. The broader goal remains active and unmet.

The next local investigation should examine whether an already inexpensive learned refiner can supply useful branch-specific corrections without another full backbone pass. This is not yet a new method or a funded experiment: parent-conditioned trees already exist, and previous xPress memoization and cost-removal experiments remain closed. Source-level differentiation, a concrete computational mechanism, and evidence of additional competitive value are required before another launch.

## Reproduction

Use `.venv-profile-launch` (Python 3.11) for `analysis/analyze_teacher_support.py`; use `.venv-depth` for `analysis/verify_teacher_support.py`, whose accumulation order is explicit. Then run `analysis/explain_teacher_support.py`. All scientific runtime sources remain bound to the original `frozen/local-bundle.json`.

Core evidence: `attempt1/analysis.json`, `verification.json`, `explanation.json`, `execution.json`, `recovery.json`, `provider-interruption.json`, `manifest-self-reference-recovery.json`, and `recovered/`. The original failed checker and recovery records are preserved alongside the successful checks.
