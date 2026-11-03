# Compact editor: context-access comparison

Adding raw context access did not pass the defined improvement screen. Do not advance this tested recipe to a generation-speed comparison on these results.

All three models completed 1,152 updates across three epochs. All 64 previously exposed development prompts and all 384 training prompts were evaluated; the original CE control reproduced its earlier development predictions at all three epochs. The primary checkpoint remained epoch three. Independent token-label recount reproduced the primary analysis exactly.

| Model | Training progress gain | Development progress gain | Dev repairs / damage |
|---|---:|---:|---:|
| Original editor | 15.28% | 3.02% | 161 / 541 |
| Relevant committed context | 12.36% | 3.12% | 168 / 518 |
| Unrelated committed context | 8.48% | 2.89% | 169 / 636 |

Progress is equal-prompt, root-inclusive retained-union coverage over the original draft. Keeping both branches protects the original prefix in this metric; it does not make editing or extra verification free. Repairs and damage describe the edited branch. None of these percentages is a generation speedup.

## Fixed primary contrasts

- Relevant context minus original editor: +0.099 percentage points; 95% prompt interval [-0.575, +0.763].
- Relevant context minus unrelated committed context: +0.234 percentage points; 95% prompt interval [-0.459, +0.878].

Intervals use the prespecified 20,000 within-domain prompt-bootstrap samples, seed 1902. Blocks within a prompt are not independent samples. The screen required positive intervals against both controls and positive aligned-versus-original point differences in both domains. One seed leaves initialization and training-seed uncertainty unmeasured.

## What was tested

The new editor keeps the old two-layer trunk and adds one rank-256 context lookup before it. It has 9,709,056 parameters versus 3,024,384 originally. Both context models started identically; shared original parameters matched across all arms. Data, CE, labels, masks, minibatch order and optimizer exposure were held fixed. The extra raw-feature normalization, rotary attention and parameters are a package, not a pure measure of information availability.

The unrelated-context control uses a fixed different prompt within the same split and domain. Its committed rows are resampled to the recipient length; repeated or skipped rows are a disclosed corruption artifact. Superiority over this control alone would not establish a useful gain over the original editor. No future target rows entered the context inputs. No new continuations were generated.

## Interpretation limits

The saved training logs reveal a qualification: relevant-context loss reached 72.58, with 37 updates above 10; unrelated-context loss reached 14.85, with 5 above 10. Original-editor loss never exceeded 4.34. The threshold is post hoc and descriptive. The same-minibatch audit confirms these spikes were absent from the original arm. Finite-value checks passed, but that does not establish stable optimization. Both context interfaces used the original learning rate without clipping; neither choice is established as the cause. No per-step gradient or activation traces were retained, so the origin remains unresolved. This weakens interpreting this recipe's negative outcome as evidence against the usefulness of context itself. See [optimization audit](optimization-audit.json).

This result does not prove that the original states contain all useful information, that context never helps, or that a compact editor cannot work. It evaluates this interface, initialization and three-epoch dose. Training-versus-development differences are descriptive and confounded by prompt difficulty; they do not uniquely diagnose memorization. The earlier 24 output-space witnesses prove individual feasible corrections, not that this network can learn to choose them jointly.

The fixed-first-guess bound separately shows repair opportunity in 74.9% of these development blocks. That rules out blaming every missed opportunity on the immutable first guess, but does not show those repairs are learnable. Previously observed full-teacher gains and prior failed compact objectives remain preserved.

No automatic training extension, architecture sweep or further paid launch follows this result. A further experiment needs a specific distinguishing question supported by the combined findings.

## Follow-up: batch chronology and relevant stability research

The post hoc audit reconstructs the exact 12,288-example training order. Relevant-context loss exceeds 10 at 37 updates, all between epoch-three steps 87 and 124. Across that 38-update window its mean loss is 22.78, versus 2.83 for the original editor and 2.88 for the unrelated-context editor on the same minibatches. Average committed-context length is 250 tokens, compared with 248 before and 246 afterward; task proportions and distinct-prompt counts are similar. This is not a single repeated bad batch. These summaries cannot exclude an input-specific trigger or establish which operation became unstable.

Both context versions have larger changes to their key/value projection weights between epochs two and three than between one and two. The relevant-context key-weight L2 norm grows from 35.69 to 75.12; value-weight norm from 16.55 to 33.06. These are endpoint Frobenius norms, not attention logits or spectral norms, and cannot establish attention entropy collapse or its timing. See [batch and checkpoint audit](spike-batch-audit.json).

Two established mechanisms merit distinguishing: excessively sharp attention, and growth in residual/output activations. Zhai et al. associate attention-entropy collapse with unstable training and study spectral reparameterization; this suggests measuring attention entropy, not inferring it from our weight norms. [ICML 2023 paper](https://proceedings.mlr.press/v202/zhai23a.html). Wortsman et al. reproduce attention-logit growth and output-logit divergence in smaller models, but explicitly study slow divergence rather than loss spikes. Their results therefore guide measurements without diagnosing this episode. [Primary paper, Sections 1 and 3](https://arxiv.org/html/2309.14322v2).

The source normalizes inputs before learned context projections, but does not normalize the resulting queries/keys. Input normalization therefore does not bound the learned attention logits independently of projection weights. Nevertheless, the current evidence does not select QK normalization, clipping, a lower learning rate, or spectral reparameterization as a proven repair. Those would be established training techniques, not our speculative-decoding contribution.

Importantly, the first two checkpoints also lacked a clear context benefit before the large spike: original/context progress was 2.79/2.53% and 2.71/2.77%. Fixing instability cannot be promised to close the teacher–editor gap. The result still rejects advancing this exact recipe to generation timing.

A faithful causal diagnostic would replay the relevant-context arm from its original initialization, preserving data, weights, loss, precision, order and optimizer. Original loss-trace agreement must precede interpretation. Record per-layer gradient/update norms, attention-logit ranges and entropy, context residual norms, and output-correction magnitudes through the observed window; retain full optimizer snapshots before the event. Distinguish where changes precede the loss rise before choosing one intervention. Epoch-two BF16 deployment weights with a fresh optimizer are not an exact continuation. The first 896 updates took 310 seconds historically; this excludes staging, target loading, compatibility checks, instrumentation, recovery and cleanup, so it is not an execution-time quote or spending authorization. No new run is prepared or launched by this audit.

## Execution and accounting

Recovery verified 35 artifacts totaling 457,807,297 bytes. The app stopped with zero tasks and owned temporary storage was deleted. Elapsed through cleanup: 46.93 minutes. Retained allowance: $3.00 under the $4 cap, not a settled invoice. Remaining unallocated campaign allowance: $9.04. No retries.

[Primary analysis](analysis.json) · [Independent recount](independent-recount-analysis.json) · [Execution](execution.json) · [Accounting](cost-reconciliation.json) · [Protocol](../PROTOCOL.md)
