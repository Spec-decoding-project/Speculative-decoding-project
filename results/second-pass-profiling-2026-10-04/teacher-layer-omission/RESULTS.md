# Local teacher layer-removal diagnostic

Directly removing one of the teacher's five layers failed on all three retained examples where the full teacher repaired the draft. Every four-layer variant lost all eight teacher-added prefix positions. This is a concrete negative for direct, untrained layer omission on these four snapshots; it is not evidence that trained compression is impossible.

The supplemental no-feedback control also lost all four of its added positions under every omission. Thus the intervention disrupts ordinary prediction too. It does not isolate a special redrafting operation or prove that five layers are a minimum required depth.

## Why this check followed the context intervention

The prior 256-context intervention found that most of fourteen feedback-only repairs survived restored within-context preferences. That did not remove any whole trained layer or show its remaining computation was redundant. The new check actually bypasses one full layer, retaining all other trained teacher weights and operations, and supplies no full-pass teacher output to the shortened execution.

Source review also confirmed two constraints before testing: the teacher's weights differ from the original drafter, so their cached states are not interchangeable; and our September22 pretrained-derived reuse models already learned corrections but had adverse complete cost per accepted position. Neither simple reuse nor pretrained initialization can be represented as a new untested fix.

## Inputs and controls

Use all four existing context-diagnostic snapshots, selected before these omission outcomes. They cover three MATH prompts: one feedback-only repair, one shared repair extended by feedback, one repair already available without feedback, and one nonrepair. No independent/fresh/randomly representative sample is claimed.

Run the authenticated 537,427,200-parameter BF16 teacher with its native SDPA implementation and native rotary buffers, sharing the existing target embedding/readout. Conditioned input sees original x1; unconditioned input keeps it masked. Both output candidates preserve the original x1. The unchanged original draft remains a fallback. No new target generation, training or cloud execution.

The shortened forward replaces exactly one native layer by the identity residual path. Its attention, MLP and normalization calls do not execute. Each omission removes 100,930,816 parameters from active teacher-layer computation, leaving436,496,384 private parameters; this is not a latency or total-serving-memory saving. Other layers and the shared vocabulary readout remain. Persistent cache-layout conversion and actual GPU runtime were not tested.

Three unit tests passed, including operation counters proving the skipped layer does no attention/MLP work, exact restoration on success/error, frozen weights, first-token handling and rejection of unsupported cached execution. Full native outputs repeat exactly after all omissions. An independent token recount verifies every row.

## Conditioned teacher results

Prefix lengths include the common root. The last five columns each omit only the named layer:

| Context | Original | Full teacher | Omit 1 | Omit 2 | Omit 3 | Omit 4 | Omit 5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k-2541/20 | 6 | 7 | 2 | 5 | 5 | 4 | 6 |
| gsm8k-1330/12 | 11 | 14 | 2 | 6 | 8 | 9 | 7 |
| gsm8k-6953/28 | 7 | 11 | 2 | 2 | 6 | 4 | 4 |
| gsm8k-1330/4 | 9 | 8 | 2 | 5 | 4 | 5 | 3 |

The original fallback sums to33 positions; original plus full conditioned teacher covers41, while every omitted-layer variant plus original covers33. Standalone conditioned teacher prefixes sum to40 because it damages the nonrepair example; fallback accounting preserves that distinction. No omitted variant adds useful coverage even at hypothetical zero added cost on these examples.

## Supplemental no-feedback control

This control was specified after seeing the conditioned failures and uses the same four inputs/all five omissions. Full unconditioned teacher prefixes are6,14,8,9 respectively, preserving37 positions with the original fallback. Every omission again returns original-plus-alternative coverage to33. Some conditioned-versus-unconditioned differences survive in the damaged models, but none yields additional coverage over the original. Sensitivity to the explicit guess is therefore not sufficient to establish useful correction.

CPU native prefixes match the prior GPU prefixes for all eight input-condition comparisons. Complete conditioned token sequences differ from saved GPU sequences by one token in two contexts, both outside the first mismatch; unconditioned sequences match. Shortened-model predictions have not been evaluated on CUDA. These CPU-specific observations are not GPU compatibility or serving-speed evidence.

## Decision and limits

Do not pay for a larger untrained layer-removal sweep or call one of these shortened models ready for recovery training. The local examples supply no positive shortcut to advance. Removing an entire trained stage changes inputs expected by subsequent stages; the test cannot tell whether recovery training, width compression or a different approximation would preserve the function. Those remain separate hypotheses, not demonstrated solutions.

The useful result is narrower: we have not found a spare whole layer that can simply be omitted in this teacher, and the observed damage is not confined to feedback processing. We should preserve this result instead of claiming the small editor failed because a specific teacher layer contains an indispensable redrafting operation. Further design needs an explicit approximation and an achievable runtime budget; no new architecture or training recipe is selected by this check.

No GPU diagnostic bundle was promoted after this local negative. Research remains active; the cheaper reliable replacement remains unestablished.

## Time, invocation record and evidence

Conditioned real-weight check:115.80seconds. Successful no-feedback check:113.51seconds. One earlier supplemental local invocation exited with code1 and an empty output log; its cause is unknown and recorded in supplemental-invocation-note.json. The subsequent unbuffered invocation completed. No paid execution, new downloads or remote resources; campaign funds unchanged.

Evidence: local-results.json, unconditioned-local-results.json, verification.json, local-tests.xml, source-manifest.json, PROTOCOL.md and both execution scripts. Prior research artifacts/checkpoints remain unchanged. This report does not establish novelty, universal depth requirements, broader generalization, or actual speedup.
