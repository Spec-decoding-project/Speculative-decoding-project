# Independent legacy target replay check

Completed in this conversation on November 17, 2026, before the separately
versioned invariant-vLLM pilot. Job **8452102** reproduced stable disagreement
between the old target execution paths. The later [canonical pilot](../vllm-canonical-pilot-2026-11-17/REPORT.md)
and [ADR 0002](../../adr/0002-version-the-invariant-target.md) record subsequent
progress; this diagnostic does not invalidate that newer numerical contract.

## What ran

One RTX A5000 on tron56, frozen BF16 Qwen3-8B, SDPA, Torch 2.13.0+cu126,
Transformers 5.15.0, Python 3.12.12 and TF32 disabled. The original collector,
model/tokenizer fingerprints and response checksums were verified. No draft or
editor was loaded. The original generator's cap alone changed from 2,048 to 16.

Three repeats per fixture/path compared the first sixteen saved tokens. Every
replay used the saved candidate tokens, so comparisons after a disagreement
still concern identical saved prefixes. The cached block test reproduced the
native first-round target call layout without invoking DFlash in this experiment.

| Fixture | Original generation | Cached block16 | Uncached replay |
|---|---|---|---|
| GSM8K 3821, anchor 38 | 16/16 each repeat | 16/16 | 16/16 |
| UltraChat 37680, anchor 24 | 16/16 each repeat | 15/16; position 6: 12946 -> 16984 | Same mismatch |
| UltraChat 61759, anchor 22 | 16/16 each repeat | 16/16 | 15/16; position 15: 279 -> 3070 |

Within every fixture/path, all three complete logit tensors had identical hashes.
Original generated tokens and the prefixes consumed at each step matched the
saved data. This establishes repeatable disagreement across execution paths,
not observed random variation between identical calls.

## New discriminating checks

Holding prompt prefill, tokens, weights and precision fixed, cached replay used
chunks of 1, 2, 4 and 8 tokens once each per fixture. One-token replay matched
original generation's full logits bit for bit on all three. Chunks 2/4/8 all
flipped the same position on 37680; the other two fixtures' tokens remained exact.
The baseline block16 had already repeated three times.

Strict `torch.use_deterministic_algorithms(True)` completed two block16 repeats
per fixture and left the entire logits unchanged from the default block path.
No unsupported-operation error occurred. The setting did not reconcile the paths.

| Selected position | Original / chunk1 logits | Block16 logits | Outcome |
|---|---|---|---|
| 37680: tokens 12946 / 16984 | 36.00 / 36.00 | 35.75 / 36.00 | Exact tie becomes a different winner |
| 61759: tokens 279 / 3070 | 37.50 / 37.50 | 37.25 / 37.25 | Tie persists; uncached 37.50 / 37.75 flips it |

Exact equality to the maximum confirmed the ties. Argmax selects the lowest
maximal token ID. BF16 spacing in this range is 0.25; this is a representation
fact, not a certified bound on network error.

The earlier [exact diagnostic](../exact-replay-2026-11-17/REPORT.md) isolated
different query-projection outputs for a bitwise-identical input row and weights
at different matrix row counts. The [rowwise intervention](../rowwise-replay-2026-11-17/REPORT.md)
made early projections agree but did not fix the final token. Their raw artifact
hashes and recorded comparisons were checked in this conversation; those GPU
traces were not rerun as part of this independent job.

## Interpretation and limits

Shape-dependent finite-precision arithmetic is the supported explanation. The
precise contribution of each operation to the final token flip remains unisolated.
Source generation reproduced these first blocks; the experiment supplies no
evidence of incorrect original collection or insufficient editor capacity.
Three selected prompts do not estimate corpus-wide mismatch frequency, validate
full-response equivalence, or establish a serving-speed improvement.

285 of 300 allowed target forwards completed; synchronized target-forward time
was 15.960 seconds, process wall time 104.714 seconds, and SLURM elapsed time
1m47s. Peak allocated GPU memory was 16,430,469,120 bytes. Exit 1 deliberately
preserved the failed equality assertion after all probes completed. The raw
result has `complete=true` and no error/traceback.

## Durable evidence and reproduction scope

- [summary.json](summary.json): public derived observations, repeat hashes,
  margins, chunk probes and provenance; full response-token arrays are omitted.
- [diagnose.py](diagnose.py) and [executed-run.sbatch](executed-run.sbatch): exact
  executed sources. The runner contains the original remote locations, so it is
  a historical artifact rather than a reusable launch command.
- [Primary-source research](PRIMARY_SOURCES.md): pinned PyTorch/Transformers
  references and similar reported behavior, with applicability limits.
- Remote artifacts: `/fs/classhomes/fpajak/speculative-decoding-project/research/target-shape-review-20261117-DVjVRt/`;
  complete private result: `output/result.json`; allocation log:
  `/fs/classhomes/fpajak/speculative-decoding-project/target-shape-review-8452102.log`.

The script uses the existing original collector, target checkpoint and pilot
journals plus `research/editor-feature-bank/feature_bank/` helpers. It writes a
new output directory, records the baseline, then waits up to 180 seconds for
`probes.ready` before running the probes. The archived source is unchanged;
future execution requires an explicit bounded run with new paths and review of
the current contract. Archiving did not rerun GPU work.

Raw result SHA256:
`6dc1b7f0782efb7d531144a35ab2c10fc40757b6e66e5af7491cebb4d820dc29`.
Executed diagnostic SHA256:
`f5844147d391a3b7782c33c2b6e0875f7fa336ac68bad501bb5534a201a4622f`.
