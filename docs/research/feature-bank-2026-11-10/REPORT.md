# Initial feature-collection gate: replay mismatch, training blocked

The bounded experiment did **not** produce a validated 128-repair/16-preservation bank. A one-block smoke test passed, but the larger attempt stopped at its third candidate because fresh target replay disagreed with one saved target token. This is inconclusive about feature-collection feasibility and editor capacity; no editor training occurred.

## Verified results

The new teacher-free collector uses the released, hash-verified Qwen3-8B DFlash forward implementation. Target weights/config/tokenizer files and draft weights/source passed their pinned fingerprints. Both checkpoints loaded without missing, unexpected or mismatched parameters.

| Check | Observation |
|---|---|
| Main response journals | 50,000 train, 1,024 validation; checksums verified |
| Pilot | All 384 IDs overlap training; shared prompt/response tokens and finish status agree |
| Exact normalized message-family train/validation overlap | 0; semantic near-duplicates not established |
| Initial exclusions | 12 output-capped pilot responses, 5 over the 1,024-context-token cap |
| Bounded candidate schedule | 7,340 full windows; at most 32 per eligible pilot response |
| Frozen target / drafter parameters | 8,190,735,360 / 1,048,626,432; zero trainable |
| Frozen vocabulary head | 622,329,856 parameters; zero trainable |
| Editor trainable parameters | 4,204,032; no optimization performed |
| Identity editor at native 15-row readout shape | Bitwise unchanged logits on the control fixture |
| Final draft normalization capture / repeated native forward | Bitwise equal on the control fixture |
| Future-label perturbation | Constructed prefix/masked draft inputs unchanged |
| Nonzero FP32 direct/folded readout | Zero argmax changes; maximum absolute difference 0.00003052 (smoke), 0.00002539 (bank control) |

The direct/folded check used `atol=0.002, rtol=0.0002`. It does not establish trained BF16 deployment equivalence. Native checks cover an uncached first-round forward, not full generation/cache behavior.

## Runs and failing fixture

Both GPU jobs used one RTX A5000, four CPUs, 32 GiB host memory, BF16 frozen models, Transformers 5.15.0 and Torch 2.13.0+cu126.

- Smoke job `8344696`: completed in 3m15s including setup; captured one repairable block. Capture/control time was 9.98s. Peak allocated GPU memory was 18,682,468,352 bytes (17.40 GiB). The tensor file was 140,880 bytes. SLURM reported maximum host RSS of 20,899,064 KiB.
- Bank job `8344771`: failed the correctness gate after 1m24s including setup; saved two partial repairable blocks before candidate three failed. SLURM reported maximum host RSS of 20,926,376 KiB. No complete bank manifest exists, and the partial bank is not approved for training.

Failing prompt ID: `ultrachat_200k:train_sft:61759`; absolute anchor: `22`. At block position `15` (zero-based), the saved token is `279`, but fresh uncached batched target replay chose `3070`. The other 15 positions matched. The separate anchor check passed. Tokens were not relabeled, and the mismatching candidate was not silently excluded.

The cause has **not** been established. Cache/row-shape numerical differences, replay alignment/configuration, and source-generation differences must be distinguished on this exact fixture. Passing file hashes and checksums rule out those particular integrity failures; they do not prove that two inference execution paths produce identical argmax decisions.

## Artifacts and reproduction

- [Smoke audit](smoke.audit.json), collector code at `0f2c11c`.
- [Failed bank audit](bank.audit.json), collector code at `288da22`.
- [Collector and usage](../../../editor-experiment/feature_bank/README.md).
- [Pinned model metadata](../../../editor-experiment/feature_bank/model_pin.json).
- [Saved-bank checker](../../../editor-experiment/feature_bank/inspect_bank.py).

Source hashes are embedded in the audits. The smoke manifest SHA256 is `d46409bf89d762439c4e1cf54354abf3328e5ba43fa612406ee103155373a28e`. The failed attempt preserves its fixture and traceback. Raw tensors, candidate records and logs remain on Nexus under `/fs/classhomes/fpajak/speculative-decoding-project/`:

```text
data/editor_feature_banks/smoke-20261110/
data/editor_feature_banks/bank-20261110/
editor-features-8344696.log
editor-features-8344771.log
research/editor-feature-bank/
```

The smoke command used `--repairs 1 --preservation 0 --seconds 300`. The bank command used the collector defaults: 128 repairs, 16 preservation blocks, 1,024 context tokens, 32 anchors per response and at most 2,700 capture seconds. Neither approached the candidate or time limit before the correctness outcome.

Six CPU integrity/selection tests passed. A regression test first reproduced and then verified the fix for grouping real source metadata by dataset instead of including per-row provenance. The bank run includes that fix; the smoke run predates it. The independent saved-bank checker was implemented, and its on-server validation was launched, but the SSH session was interrupted before its result was retrieved. Do not count that additional check as passed; retrieve `research/editor-feature-bank/bank-integrity-result.json` or rerun it next session.

Before downloading the 2.1 GB draft, allocated home usage was measured as 26,272,333,824 bytes. The [UMIACS class-account documentation](https://wiki.umiacs.umd.edu/umiacs/index.php/ClassAccounts) states a 30 GB home quota; the quota command itself returned no usable per-account limit. A single checkpoint copy was downloaded and its SHA256 verified, with no duplicate model cache. No full-corpus features were cached.

## Next decision

Diagnose the saved-token replay mismatch on the fixed failing fixture, using the two successful bank candidates as controls. Keep the one-block training ticket blocked until the replay contract is justified and the full bounded bank passes collection and independent integrity checks. Do not increase editor capacity or start larger training based on this result.
