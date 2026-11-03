# Separating teacher prediction from draft-feedback computation

## Finding

The earlier diagnosis is valid but combines two different tasks: compressing the full teacher's alternative predictions, and compressing the changes specifically caused by showing it the first draft token. A new join of existing observations shows that student failures occur in both. We cannot explain the gap solely as failure to use token feedback.

No training, model forward pass, GPU execution, downloads or paid work occurred. This is a post hoc analysis of 256 saved contexts from 64 exposed development responses, not a new evaluation panel.

## New matched-case analysis

Use the frozen feedback-trained teacher twice: first with the future masked, then with original DFlash's first guess visible. Copy that same first guess into both output proposals. Keep teacher weights, original fallback, target reference, positions and vocabulary-readout shape fixed. Join the proposals to the existing probability-distilled compact editor at the identical prompt and anchor.

The conditioned teacher extends the original draft in 42 of these contexts:

| Teacher behavior | Contexts | Teacher-added matching positions | Positions captured by student | Contexts where student fails to extend original |
|---|---:|---:|---:|---:|
| Without feedback, teacher already reaches the conditioned teacher's endpoint | 27 | 83 | 29 | 19 |
| Only the conditioned teacher extends the original | 14 | 35 | 3 | 11 |
| Both extend, conditioning extends further | 1 | 4 | 0 | 1 |
| Total | 42 | 122 | 32 | 31 |

Thus 19 of the 31 student failures to extend the original occur where the teacher can already supply its entire correct prefix without feedback. This does not prove those predictions use identical internal mechanisms in the two teacher runs. It establishes that explicit first-token feedback is not necessary to obtain those saved correct prefixes from this teacher.

Across all 256 contexts, conditioning improves retained coverage in 15, hurts in nine and ties in 232. It adds 38 matching positions and loses 12, net 26. The two teacher conditions add 122 versus 96 positions over the original, with 84 overlapping positions. These are pooled descriptive counts, not independent samples or population effect sizes.

The previously completed equal-prompt analysis on this subset found +7.72% coverage for the unconditioned alternative and +9.02% for the conditioned alternative. The paired difference was +1.30 percentage points, interval [-0.97,+3.80]. These figures do not replace the separate full-panel +10.21% teacher result. Conditioning helps particular contexts, but its positive panel-average effect in this comparison is uncertain.

The unconditioned input mask differs from the feedback teacher's training mask. The comparison estimates this specified within-checkpoint input intervention, not what equally trained unconditional versus conditional systems would achieve. Keeping both proposals costs computation and verification. No generation speedup is established.

## What computation actually differs

The executed teacher implementation is `FullConditionalReference` with the precision-preserving wrapper. It copies and trains the entire five-layer DFlash model, including its private context projection. It receives the committed target features and token/mask inputs, optionally replacing one mask with the original first-token embedding. It does not receive the original draft's intermediate states, full predicted block or future target features.

The compact editor is different: it projects all final normalized original-draft states from width 2,560 to 256, adds a broadcast first-token embedding and learned positions, processes them with two small transformer layers, and adds a rank-256 correction to the original vocabulary scores. Its original version cannot directly revisit the committed context. It relies on what the first pass encoded and on what the learned input projection retains. This is an architectural restriction, not evidence that needed information is actually absent.

Consequently, the historical distillation target was broader than “copy a cheap consistency check on a finished draft.” It includes predictions made by a separately trained, full-width contextual model. Small output changes do not imply a small prediction function, but neither do these architectural differences prove that compression requires a large model.

## A concrete feedback path, derived from the pinned implementation

The native attention computes queries from current draft states, and keys/values from both committed context and draft states. Each layer reuses projected committed context but computes fresh queries from the evolving block.

At fixed weights, positions and context, replacing only x1 has the following dependency structure:

1. At the first layer, queries at the later masked positions are unchanged; x1's key and value change. Scores against committed context at those later positions are unchanged before softmax, although changing the draft keys can change their total normalization and attention mass.
2. Attention propagates x1's effect to later positions. Their subsequent MLP calculations can change.
3. From the next layer onward, their queries can change. This allows different relative attention to committed context, alongside changed interaction within the block and changed nonlinear processing.

This establishes how feedback *can* affect a correction. It does not establish which path caused any observed correct token, whether it is necessary, or whether a compact approximation can reproduce it. The ordinary first pass already performs contextual processing; the possible added operation is processing under a changed token input.

Source inspection used the actual hash-pinned `zlab_4b_b16__dflash.py` loaded by `conditional_redraft_loading.py`, not merely a related DFlash-family implementation.

## What earlier negative experiments leave unresolved

- Output witnesses and focused rehearsal establish selected answers are expressible and selected repair sets fit-able. They do not establish that the editor inputs support a simple, transferable correction rule.
- The frozen editor often decreases the correct token's margin in sampled development failures. This identifies its wrong computation, not why training learned it.
- Adding raw context through a newly trained compact interface did not improve reliably, including after the instability intervention. That does not isolate the value of the teacher's already-trained contextual computation.
- Improving global teacher imitation did not reliably improve useful prefixes. This does not prove that better training is irrelevant; it shows overall agreement is inadequate as the sole target metric.
- The teacher and editor differ simultaneously in context access, width, depth, initialization, normalization, nonlinearities and readout restriction. Their comparison cannot assign the failure to one of these differences.

## Next discriminating question

For the small subset where feedback genuinely changes a failure into a repair, does the repair require changed access to committed context, or does draft-local processing suffice within the trained teacher?

A possible diagnostic would hold teacher weights fixed and preserve its normal draft-side computation, while replacing context-attention scores at later layers with those recorded in the unconditioned run. Keep draft scores current and normalize the combined attention normally. This removes changed query-to-context affinities while retaining draft-dependent attention mass and the teacher's trained transformations. It is a precisely defined hybrid computation, not a deployable editor or a pure semantic intervention.

Such a probe needs unmodified-run reproduction, neutral replacement controls, both repair and damage cases, successful-path controls, and separate reporting of shared repairs versus feedback-dependent repairs. Loss of repairs would support dependence on that path in the fixed teacher; survival would only show this particular path change was unnecessary. Neither result proves a cheap student exists or uniquely explains past training failures. Joint removal and individual-layer interventions can differ because pathways compensate or interact.

The saved output records cannot answer this operation-level question; they lack the required paired attention traces. No implementation, spending request or GPU launch is prepared by this report. The new evidence supports separating teacher behaviors before selecting a compression method, rather than repeating a small-head fit.

## Reproduction and verification

`analysis/audit_teacher_conditioning_overlap.py` produces `conditioning-overlap.json` and `conditioning-overlap-cases.json`, refuses output overwrites, and records SHA-256 identities for its three source files. It joins all 256 cases uniquely, checks original tokens and historical teacher lengths, reconstructs student and teacher prefixes from token identities, independently checks prefix arithmetic, and matches teacher repair counts/capture against the earlier full-panel audit. Context observations are grouped within 64 prompts; no new confidence interval is inferred from token counts.

Sources: the October 3 same-weight redrafting report and saved predictions; October 1 probability-student evaluation; October 3 compact-failure decomposition; `conditional_full_reference.py`, its precision wrapper, `compact_second_pass_prototype.py`, and the pinned native DFlash source.
