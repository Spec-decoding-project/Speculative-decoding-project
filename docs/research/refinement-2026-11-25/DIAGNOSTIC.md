# Frozen COPY diagnostic: both decision and generator failed

The COPY model sometimes overrules a correct vocabulary prediction, but its vocabulary predictions also deteriorated during the previous joint fine-tuning. **Freeze the original full-trained generator for the next learning comparison.** Changing the COPY threshold alone leaves the damaged generator in place.

Job 8579035 completed November 25 in 4m01s, exit 0, using one A5000/four CPUs/5 GiB. Source commit `54cef7d1f0b36d7fd83592fa14d3d013c977b981`. No optimizer updates, target inference or final-test rows were involved.

At the same actual token prefixes for the 231 repairs lost by copying the first error:

| Observation | Cases |
|---|---:|
| Correct vocabulary alternative overridden by COPY | 86 |
| Vocabulary alternative itself incorrect | 145 |
| Full-trained starting generator correct | 211 |
| Those correct starting predictions lost after COPY fine-tuning | 131 |

The original full-trained generator, matched control and COPY model each reproduced all 2,048 saved development outputs exactly. Prefix/action alignment checks passed. Disabling COPY without changing any weight gives mean correct prefix **1.0796**, compared with **3.4683** with COPY enabled and **3.5776** for DFlash. It repairs 43 blocks and damages 1,254; all 44 originally perfect proposals are damaged. Net prefix positions: 69 gained versus 5,185 lost. The paired-by-prompt change relative to DFlash is −2.4980 tokens, descriptive 95% interval [−2.6621, −2.3379].

An independent Mac recount reproduced both raw-output audits and the 86/145/131 diagnostic counts. All model/input fingerprints stayed unchanged; cached and full execution agreed on 64 ablation blocks. These are exploratory development results, not new confirmation prompts or generation speedup measurements. Looking at known error positions is a diagnostic, not an inference policy.

The next comparison freezes the original 40,750,336-parameter generator and trains only small COPY-versus-generator gates on 1,536 training prompts, selecting on 512 separate training-origin calibration prompts. The rationale and predeclared bounds are in [HISTORY.md](../../../HISTORY.md).

Aggregate results: [complete.json](evidence/diagnostic/complete.json). Independent recount and private artifact hashes: [mac-audit.json](evidence/diagnostic/mac-audit.json). Raw tokens/checkpoints remain private under remote `research/refinement-diagnostic-20261125/` and the Mac `~/.local/share/wayfinder-artifacts/refinement-diagnostic-20261125/` backup.
