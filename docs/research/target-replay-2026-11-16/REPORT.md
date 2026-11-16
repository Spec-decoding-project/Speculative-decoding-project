# Target replay mismatch reproduced and localized; bank gate remains failed

The original failure is repeatable. The saved responses agree with the original cached greedy generation path, but the uncached multi-position replay chooses a different token on the same saved prefix. The difference exists in the target backbone's final representations; changing only vocabulary readout precision or row count does not reconcile it. This narrows the diagnosis without establishing a correction or a valid replacement verification contract. No full-bank recollection or real-data editor training ran.

## Reproduction and controls

SLURM job `8432640` used one RTX A5000 on `tron49`, four CPUs and a 32 GiB host allocation. Python 3.12.12, Torch 2.13.0+cu126, Transformers 5.15.0, BF16 target weights, SDPA, seed 0 and disabled TF32 match the original target settings. Every pinned target file hash was checked, the original response-generator hash matched its run manifest, all pilot record checksums were verified, and the selected source rows matched both successful bank controls. Tokenizer output and rendered-prompt hashes matched on all three fixtures.

The diagnostic loads only the frozen target and the three fixed fixtures, omitting DFlash, editor code, bank selection and training. Each baseline performs the original separate anchor check and the original uncached 16-row continuation replay three times. This is a reduced reproduction of the actual failing target call, not an injected failure or a synthetic numerical example.

| Fixed fixture | Anchor | Uncached baseline, each of three repeats | Original cached generation, each of three repeats |
| --- | ---: | --- | --- |
| `gsm8k:train:3821` | 38 | 16/16 match | 16/16 match |
| `ultrachat_200k:train_sft:164610` | 19 | 16/16 match | 16/16 match |
| `ultrachat_200k:train_sft:61759` | 22 | 15/16 match; position 15 expects 279, gets 3070 | 16/16 match |

All separate anchor checks passed. All fixtures are their responses' first block, so generating the first 16 tokens from the original prompt tests the complete history of interest. The generation comparison uses the original collector's explicit greedy/cache/EOS/BOS/pad settings, with its output cap reduced from 2,048 to 16; no earlier output history is omitted. A separate cached teacher-forced path consumes the identical saved prefix at every step, avoiding changed-context confounding, and also matches 16/16 on every fixture.

The original equality check remains red: `BASELINE_PASS=False`, with three identical position-15 discrepancies. The program intentionally returns exit status 1 after saving all probes. SLURM therefore reports `FAILED` with exit code `1:0`; this is the preserved correctness verdict, not an allocation crash.

## Discriminating probes

Before running probes, the ranked hypotheses were: vocabulary-head rounding/row-shape effects; upstream cached-versus-uncached execution differences; and replay input/position/generation mismatch. The probes hold saved token inputs fixed and separate readout-only changes from upstream changes.

For the failing position, signed margin below is logit(279) minus logit(3070):

| Internal representation | Readout | Logit 279 | Logit 3070 | Signed margin | Chosen token |
| --- | --- | ---: | ---: | ---: | ---: |
| Uncached multi-position | BF16 | 37.5 | 37.75 | -0.25 | 3070 |
| Same uncached representation | FP32 | 37.5371323 | 37.6547394 | -0.1176071 | 3070 |
| Cached, identical saved prefix | BF16 | 37.5 | 37.5 | 0 | 279 |
| Same cached representation | FP32 | 37.4652939 | 37.3890991 | +0.0761948 | 279 |

The FP32 probes cast the BF16 final representation and stored head weights to FP32 for projection. They are **not** a full FP32 transformer run. Their opposing margins demonstrate that head rounding alone is insufficient to explain or correct this fixture.

Changing BF16 readout from 16 rows to one row at a time caused zero argmax changes on all three fixtures. The analogous FP32 change also caused zero argmax changes. Running the whole model with `logits_to_keep=1` instead of 16 preserved its final representations exactly and still chose 3070 at the failing position. An explicit all-ones attention mask likewise preserved the uncached representations exactly.

The cached and uncached final-position representations differ in 3,662 of 4,096 dimensions on the failing fixture, with maximum absolute difference 0.625. They also differ on both successful controls, whose greedy outputs stay stable. This localizes a relevant difference upstream of vocabulary readout. It does not isolate which attention/projection/reduction operation produces it, establish a full FP32 reference, or rule out every possible cache implementation issue. Execution-shape numerical variation is consistent with the observations; it is an inference, not a demonstrated kernel-level root cause. [PyTorch documents why mathematically equivalent batched and sliced computations need not be bitwise identical](https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html).

FP32 readout is not a generally safe replacement for the original BF16 contract: on the otherwise successful `ultrachat_200k:train_sft:164610` control, it changes position 5 from saved token 374 to 646 when applied to the uncached representation. No precision change was adopted for collection.

## Artifacts, budget and limits

- [Complete machine-readable result](result.json): hashes, pinned target files, runtime/configuration, input hashes, repeated verdicts, cache positions, logit margins and readout probes. Raw response records remain on Nexus.
- [Diagnostic source and bounded runner](../../../editor-experiment/replay_diagnostic/README.md). Local source hashes were verified against the executed session and probe hashes.
- Remote artifacts: `research/editor-feature-bank/replay-20261116/`; log: `target-replay-8432640.log` under the existing Nexus project.

The job finished in 4m40s including setup and the pause between baseline and hypothesis probes, within the single 30-minute allocation cap. Baseline calls took 7.6884 seconds and probes 10.3711 seconds: 18.0595 seconds total, below the 10-minute forward budget. Peak allocated GPU memory was 18,960,980,480 bytes (17.66 GiB); maximum host RSS was 18,309,524 KiB. The user's queue was empty afterward. Project disk use was 27,772,416 KiB before the run; the quota command returned no usable per-account quota. No checkpoint was copied or downloaded.

The original mismatch is not fixed. No native DFlash cached verification path, layer-level numerical localization, corrected collector, or full 128-repair/16-preservation bank was validated in this allocation. No relabeling, tolerance relaxation, mismatch filtering, or corpus-wide cache was introduced. The old bank's two partial blocks remain ineligible for training.

## Outcome and next question

Outcome: **diagnostic progress, inconclusive for the required faithful bank**. Source cached generation is reproducible; the proposed uncached equality gate is still false on the fixed fixture. This says nothing about editor capacity or whether either autoregressive variant can learn useful corrections.

The next bounded diagnostic must compare the actual native cached block verifier with the original cached greedy path on these same fixtures, then localize the first upstream discrepancy sufficiently to justify a correction or an explicit revised target execution contract. Preserve both original outputs and include any added verification cost. Do not silently substitute cached-label agreement for native verification agreement. Only after that contract is justified can one original-budget full-bank attempt and independent integrity check unlock real-data fitting.
