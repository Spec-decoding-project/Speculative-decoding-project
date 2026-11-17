# Native cached verifier passes fixed fixtures, but the bank fails on a new prompt

Subsequent user decision: preserve exact agreement with the original saved greedy tokens. The separate-label proposal below was rejected and must not be run; see the [accepted decision](../../adr/0001-preserve-exact-saved-target-tokens.md). The experiment results below are unchanged, and training remains blocked pending exact replay diagnosis.

The unchanged released DFlash cached first-round verifier accepts the saved candidate on the original failing fixture and both controls. The old uncached replay still fails. A new, explicitly named collection contract therefore checks the native cached execution path without changing model precision or any saved labels. The bounded bank outcome is recorded below; a three-fixture success alone does not unlock training.

## Actual native verifier and matched-prefix controls

Diagnostic job `8432866` ran the hash-verified released `DFlashDraftModel.spec_generate` with the original frozen BF16 target/draft, SDPA and disabled TF32. `max_new_tokens=1` executes exactly one complete draft/verification round before truncating the returned sequence; hooks captured its actual target calls and cache/position metadata. Ordinary DFlash proposals were measured separately.

For the oracle control, only the draft sampler's 15 proposed tokens were replaced by the saved correct proposal. The anchor still came from native target prefill. This holds every relevant token prefix fixed while exercising the real verification call. It is an explicitly artificial diagnostic candidate, not an editor result, training example selection criterion, or performance claim.

| Fixture | Original cached generation | Uncached replay | Native verifier with saved candidate | Ordinary DFlash proposals accepted |
| --- | --- | --- | --- | ---: |
| `gsm8k:train:3821`, anchor 38 | 16/16, three repeats | 16/16, three repeats | All 15 proposals, three repeats | 4 |
| `ultrachat_200k:train_sft:164610`, anchor 19 | 16/16, three repeats | 16/16, three repeats | All 15 proposals, three repeats | 4 |
| `ultrachat_200k:train_sft:61759`, anchor 22 | 16/16, three repeats | Position 15 gets 3070 instead of 279, three repeats | All 15 proposals, three repeats | 2 |

The anchor is counted in the 16 saved positions but not in the 15 proposals. Cached teacher forcing on the same saved prefixes matched all 16 positions, with or without an explicit all-ones mask. Input hashes, cache lengths, explicit positions and first candidate errors are in [diagnostic.json](diagnostic.json). Native verification logits after an ordinary candidate's first wrong token are on a different context and are not interpreted as saved-context label comparisons.

## Numerical localization

At the original failing position, the input token embedding and first input normalization are exactly equal across cached single-token and native block execution. The first observed difference is in the first layer's query projection: 1,277 dimensions differ, maximum absolute difference 0.001953125. Its input row and weights are unchanged; the matrix's row count/execution shape differs. Later target representations also differ. This establishes an early shape-sensitive projection difference, not a particular vendor kernel bug or a proof that this is the only source of numerical variation.

| Single change from default | Old uncached mismatch | Native verifier on three saved candidates |
| --- | --- | --- |
| None | Still fails | All pass |
| Force SDPA math backend | Disappears on these fixtures | Fails position 5 on control 164610 and position 15 on fixture 61759 |
| Disable reduced-precision BF16 GEMM reductions | Still fails | All pass |
| Compute transformer linear projections in FP32, returning BF16 activations | Disappears on these fixtures | All pass |

Disabling reduced-precision BF16 reductions reduces the first query-projection discrepancy to four dimensions, maximum 0.0000004768371582, without fixing the original uncached token mismatch. FP32 transformer projections remove the observed first-layer Q/K/V differences, but differences still appear by the attention output projection. These probes are evidence of numerical execution sensitivity. The FP32 probe casts one linear operation's inputs and weights transiently; it is not a full FP32 transformer checkpoint and its deployment overhead was not benchmarked. No probed precision/backend setting was adopted for collection.

## Explicit collection-contract change

The original collector's `uncached-first-round-v1` equality requirement remains available as the default for reproducing the failure. The new opt-in `native-cached-first-round-v1` contract uses the released first-round layout:

1. Prefill the exact saved prefix with explicit positions and an empty target cache; require the resulting anchor to equal the saved anchor.
2. Verify the complete saved 16-token candidate using that prefix cache and positions anchor through anchor+15. The first 15 verification logits must predict saved positions 1 through 15 exactly. The final bonus logit is outside this saved block.
3. Extract DFlash features from the prefix only and construct the ordinary proposal with a fresh draft cache. Future correct labels are used only for target verification, never as proposal inputs.
4. On the original three fixtures, compare collected target features, final normalized draft states and actual proposal IDs bitwise against a genuine released `spec_generate` round. Require those regressions, exact native label replay, all existing identity/no-future-label/readout controls, and the full bank count before success.

This change was explained before the collection attempt. It does not make the original uncached check pass, relabel a response, tolerate mismatches or silently exclude failing candidates. It defines a **fresh first round at each saved prefix**, not equivalence to every long-running cache history or union-verification layout. Any native disagreement still stops collection. Native/legacy contract identifiers are saved in both audit and manifest, checked independently, and preserved when exporting a token bank for either autoregressive editor.

Eight feature/contract CPU tests passed. After integrating the exporter/reader change, all 19 feature and autoregressive tests passed, including rejection of an unknown token-bank replay contract. The original three-fixture red signal remains preserved by the diagnostic; no claim of universal numerical invariance is made.

## Resources and artifacts

Job `8432866` used one RTX A5000 on `tron60`, four CPUs and 32 GiB host allocation, with Python 3.12.12, Torch 2.13.0+cu126 and Transformers 5.15.0. Elapsed time was 4m12s including the baseline/probe pause. The baseline took 3.2430 seconds; native probes including draft verification/loading took 33.3735 seconds, for 36.6165 seconds total within the 10-minute cap. Peak allocated GPU memory was 18,725,072,384 bytes (17.44 GiB), and maximum host RSS was 20,467,556 KiB. Exit status 1 deliberately preserves the old failed baseline after saving the complete results.

The original target/draft fingerprints and generator/source metadata are in [diagnostic.json](diagnostic.json); the executed native probe hash was checked against [native_probes.py](../../../editor-experiment/replay_diagnostic/native_probes.py). Raw response records remain on Nexus. No new checkpoint copy/download or editor training occurred.

The single bounded native-bank attempt is job `8432927`, output `data/editor_feature_banks/native-bank-20261116/`. It retains the 128-repair/16-preservation requirement, 384 overlapping pilot IDs, EOS-only selection, source balancing, maximum 32 anchors per eligible response, context cap 1,024 and 2,700 capture seconds within a one-hour allocation. The [runner](../../../editor-experiment/replay_diagnostic/recollect.sbatch) starts an independent checker only if collection exits successfully. A completed partial attempt or an empty checker output is not success.

## Bank outcome

The single permitted full-bank attempt failed at candidate seven after saving six repairable blocks and no preservation blocks. Its original three fixed-fixture regressions all passed: target features, normalized draft states and actual proposal IDs exactly matched the released native first round, and saved labels passed native verification. Existing no-future-label, deterministic capture, identity and direct/folded controls also passed.

The new failing fixture is `ultrachat_200k:train_sft:37680`, absolute anchor 24, block position 6. Saved token 12946 disagrees with native cached verifier token 16984; the other 15 saved positions and the separate anchor match. This new discrepancy was not subjected to the earlier three-fixture numerical probe matrix, so its specific numerical mechanism is not established. It shows that changing to native cached replay is insufficient to validate the existing saved labels across the bounded sample.

[Bank audit](bank.audit.json) preserves the new fixture, original labels, chosen contract, source/model hashes, all three passing regressions and the failure. The partial manifest contains six repairable blocks whose stored tensor hashes were checked after the run. There is no complete manifest and the independent full-bank checker was not run because collection failed. Partial tensor integrity is not a passed bank gate.

Job `8432927` ended with exit 1 in 2m43s on `tron54`, maximum host RSS 20,949,448 KiB. The exception occurs before end-of-run GPU peak/capture totals are assigned, so those measurements are unavailable for this bank attempt. The user queue was empty afterward. No source response was rewritten, mismatch ignored, corpus recollected or editor trained.

Outcome: **inconclusive for bank feasibility; training remains blocked**. We now have evidence that the existing cached-greedy responses cannot simply be assumed to be exact native-block labels. Repeating precision switches or skipping difficult records is not an established solution. The next decision is whether to create a separate native-verifier label set, retain saved responses as distillation targets with explicitly different evaluation semantics, or keep pursuing exact agreement with the original saved-token contract. A [concrete proposed next experiment](NEXT_EXPERIMENT.md) is provided for that decision; it has not been authorized or run.
