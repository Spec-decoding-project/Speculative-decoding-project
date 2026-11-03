# Public D2SD screen: conditional repairs exist; competitive value remains uncertain

## Outcome

All eight preselected development prompts and all32 contexts completed. The public DTA checkpoint supplied some corrections beyond matched ordinary rerunning. The incremental advantage is small and uncertain on this panel; it does not yet justify student training or establish a practical speedup.

| Proposals available | Total matching positions | Equal-prompt gain over original |
|---|---:|---:|
| Original first draft |140|reference|
| Original +4 conditional reruns of the first model |158|16.07%|
| Original +4 conditional runs of the second model |166|22.58%|
| Original + second model with no guessed-token feedback |153|14.86%|

Each32-context total includes its known target root. We select the longest matching proposal using reference answers: these are fixed-context coverage numbers, not measured emitted-token savings or latency reductions. Percentages weight each prompt equally and therefore differ from ratios of the pooled totals.

Against the matched ordinary conditional reruns, second-model coverage improved5.98% on average, with a descriptive prompt-bootstrap95% interval of−0.55% to12.71%. Five prompts improved, one worsened, two tied. Math averaged+0.42%; chat+11.53%. At the context level:8 better,3 worse,21 tied. The effect is neither precisely estimated nor uniform across domains.

## What draft feedback contributed

Some apparent benefit does not require feedback: retaining the original and the second model's root-only proposal gives153positions. The second model alone gives135, versus140 for the first model, so this is complementary output rather than evidence of a stronger standalone model.

Nevertheless, **four contexts contain ten additional positions beyond both ordinary conditional rerunning and the second model without guessed-token feedback**. They occur in three prompts; seven of the ten positions come from one chat response. This is concrete but concentrated evidence that conditioning this model can make useful choices. It does not establish general transfer or a scalable learning rule.

Examples, selected after inspecting outcomes and therefore explanatory only:

- Revealing the first guess `'s` changed `Let's try` to `Let's break`, matching one additional target token before another error.
- Preserving four guesses through `Dog video**:` and its line break enabled `This video` instead of `The video`, recovering two positions.
- Preserving eight guesses through `"Hill Field"** is likely` enabled `a poetic or`, recovering three positions.
- Preserving four guesses through `a direct translation,` enabled the beginning of `**"Campo de`, recovering four positions.

These are changes under a controlled input intervention: the model weights and target history stay fixed while some draft guesses replace masks. They locate useful conditional behavior. They do not isolate individual responsible words, attention operations, latent representations, or the computation a small editor would need. Later suffixes remain often wrong. See `decoded-examples.json`; its tokenizer matches the pinned8B tokenizer SHA256 exactly.

The successful retained prefixes contain1,4,8 and4 guesses. Our historical fixed-x1 editor therefore did not receive the same explicit inputs as three of these successful proposals. That observation alone does not justify repeating the completed variable-prefix training study or asserting that longer prefix input fixes student transfer. It identifies what this specific teacher was given.

## Repairs, damage and cost

Across128 conditional proposal chains per model, ordinary rerunning repaired7 original first-error prefixes and damaged23; DTA repaired17 and damaged8. Ordinary reruns damaged the one originally fully correct context in all four chains; DTA damaged none of its four chains. Chain counts are correlated, not128 independent examples. Keeping the original protects coverage from these regressions; a deployed system still pays to produce and verify alternatives.

Mean measured uncached embedding+draft+vocabulary forward time was9.229ms for ordinary conditional reruns and9.231ms for DTA. The first-pass mean was10.730ms. These single-forward diagnostic timings omit deployed cache savings, verification, selection and complete round costs. Four serial conditional proposals cost roughly37ms of this measured work, which demonstrates why compression matters; it does not quantify a deployable speedup or economically achievable student target. Native reference collection took50.87seconds.

## Decision

The screen answers an availability question and provides concrete conditional repairs to study. It does not establish sufficiently broad headroom against strong8B refinement systems. Do not train a student merely because22.58% exceeds the old teacher's percentage: the target, checkpoint, prompts and proposal budget differ.

Preserve the three distinctions:

1. Ordinary conditional rerunning itself supplied substantial coverage on this panel.
2. Different second-model weights supplied complementary proposals even without draft feedback.
3. Second-model feedback added some further useful positions, concentrated in a few contexts.

The next learning decision needs evidence that the desired correction set survives comparison with a strong compatible refinement alternative and leaves enough room for a cheaper editor after its full costs. The saved cases permit local analysis of which prefix decisions changed. No new fit, architecture choice or additional paid run is approved by these results. Public checkpoint training provenance remains unknown; this is not a paper reproduction or a novel method.

## Execution and verification

Single app `ap-JiPlQKdpkTwiezOQyltXeY`; GPU worker completed in107.91seconds. Whole attempt, recovery and cleanup:275.44seconds, about4m35s. Both public model hashes and all five cached target shard hashes passed before allocation.32 repeated first proposals matched. Small-model author/uploaded-code and buffer-loading checks passed locally; the actual released-weight run completed without nonfinite outputs.

All47 recovered files,305718bytes, independently hash-verified; prefix metrics independently recounted. App stopped with0tasks and owned temporary volume independently confirmed absent. Shared target cache preserved. Provider-reported compute is$0.08949038, not a settled invoice. Retain the full$1 allowance including staging, storage, failures and recovery; current$40 goal authorization has$0 unallocated. No retry, student training, or subsequent paid job.

Evidence: `analysis.json`, `descriptive-cost-and-counts.json`, `decoded-examples.json`, `closure-audit.json`, `attempt1/{execution,recovery,metering}.json`, and `attempt1/recovered/`. Frozen sources/protocol in `frozen/`. Compression goal remains active and unmet.
