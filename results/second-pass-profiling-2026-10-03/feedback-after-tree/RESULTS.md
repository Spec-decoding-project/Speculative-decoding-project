# Explicit first-token feedback after an ordinary-tree control

The local comparison completed on 256 saved contexts from 64 reused development responses. No new model execution, training or paid work occurred. The primary contrast holds the trained feedback checkpoint's weights, source context, output first token and verification-node allowance fixed. Only the saved neural input condition changes: either the first guess is visible to the model or it is copied into its output afterward.

Each condition retains the ordinary 30-node tree, adds its proposed chain and fills the remaining places with ordinary candidates, up to exactly 45 nodes. The independent reference is an ordinary 45-node tree. All selection uses predictions, not reference answers.

## Primary result

Showing the first guess to the trained feedback model adds **0.478% matching-prefix coverage** relative to the same weights without that input. The descriptive paired 95% interval is **-0.331% to +1.354%**. There are 19 gained matching positions and seven lost; 11 contexts improve and five worsen.

Under equal complete-round time, the same contrast corresponds to **0.358% modeled savings**, interval **-0.432% to +1.181%**. These are different transformations of per-response progress, not two estimates of measured runtime.

Against ordinary 45-node expansion, the conditioned proposal gives 1.337% more coverage [-0.428%, +3.327%], or 0.798% modeled equal-round-cost savings [-0.790%, +2.542%]. Neither establishes a competitive generation advantage. The point direction is favorable on math and unfavorable on general instructions.

## Secondary controls

| Checkpoint and extra proposal | Modeled equal-round-cost savings vs ordinary 45-node tree |
|---|---:|
| Released q0, conditioned | -0.457% [-1.956%, +1.214%] |
| Trained feedback, unconditioned | +0.394% [-1.184%, +2.097%] |
| Trained feedback, conditioned | +0.798% [-0.790%, +2.542%] |
| Trained no-feedback, unconditioned | -0.014% [-1.593%, +1.673%] |
| Trained no-feedback, conditioned | +0.541% [-1.156%, +2.265%] |

The no-feedback checkpoint's within-weight conditioning contrast is +0.608% coverage [+0.037%, +1.262%]. This small secondary effect is preserved; it does not establish that adding its proposal beats the ordinary tree. Released q0 without conditioning reproduces the ordinary tree exactly, as required.

## Meaning and limits

The earlier positive single-chain comparison remains valid. It did not establish that explicit first-token feedback adds substantial value beyond inexpensive alternatives already available from the original model. This new comparison supplies that missing control for these saved single-proposal interventions.

The result does not identify all causes of the separately observed teacher probability-tree benefit. It also does not show that richer feedback, a better-trained teacher, or a different student cannot help. The feedback checkpoint's masked first pass and the no-feedback checkpoint's conditioned pass differ from their respective training modes. This is an input intervention at fixed weights, not a causal estimate of retraining with and without feedback. Intervals on reused development prompts are descriptive; four contexts per response limit precision.

Equal nodes do not mean equal runtime. Teacher computation and selection costs are unmeasured here, and recorded reference matches do not establish live generation fidelity.

## Decision

Do not justify another small-editor fit by saying that a full second pass works and therefore its first-token-feedback benefit should be compressible. The available evidence supports useful alternative proposals, but only a small, uncertain incremental effect from that explicit input after the ordinary-tree control. Alongside the additive teacher-tree audit, this means we need a stronger competitive teacher task or an independently justified cost advantage before another training recipe is ready. More imitation data or a larger editor is not a demonstrated remedy.

This preserves the broader research question rather than replacing it with a baseline implementation. No successful cheap-redrafting method or new speedup has been established.

## Verification and sources

All 5,632 construction/count checks passed. Original/teacher records and candidate-bank hashes were authenticated; all 768 checkpoint/context records join exactly on original tokens, labels and masks. Heap and frontier enumeration agree at every ordinary budget from 30 through 45. The selected proposal unions are prefix closed, keep the original tree, have exactly 45 nodes, and agree with an independent prefix recount. Complete metrics, per-response totals, rows and source hashes are in `results.json`; reproduction is `analysis/audit_feedback_after_tree.py` from the repository root.

The prior same-weight report is [here](../same-weight-redrafting/attempt1/RESULTS.md); the distinct all-context teacher-tree audit is [here](../additive-teacher-tree/RESULTS.md). A targeted source check found no download response in the fetched [D²SD checkpoint issue](https://github.com/catnanami/D2-SD/issues/1). This is not proof that compatible weights cannot exist elsewhere, and no substitute checkpoint was obtained.
