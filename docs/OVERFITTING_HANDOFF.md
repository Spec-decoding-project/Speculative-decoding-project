# Overfitting archive: start here

Snapshot recorded November 17, 2026. This branch preserves completed work and
instructions for inspecting it. Creating this archive launched no experiment.

## Read the current state first

1. [Sixteen-block report](research/sixteen-block-fit-2026-11-17/REPORT.md): both
   token-only editors memorized all sixteen blocks after 125 updates. Development
   predictions still damaged many original correct prefixes.
2. [One-block report](research/one-block-fit-2026-11-17/REPORT.md): both memorized
   one real block after 100 updates, with poor excluded-prompt transfer.
3. [Canonical-target report](research/vllm-canonical-pilot-2026-11-17/REPORT.md):
   versioned invariant vLLM greedy/DFlash paths matched all 3,898 emitted tokens
   and corresponding logit hashes on four complete pilot responses.
4. [Independent replay diagnostic from this conversation](research/independent-target-replay-2026-11-17/REPORT.md):
   the old Transformers paths repeat exactly within each shape but disagree
   across shapes. This is historical diagnostic evidence, not the latest target
   contract or a reason to discard the later invariant pilot.

The current decisions are [ADR 0002](adr/0002-version-the-invariant-target.md),
[ADR 0003](adr/0003-authorize-one-block-memorization.md) and
[ADR 0004](adr/0004-authorize-sixteen-block-memorization.md). They explicitly
supersede the earlier old-token/full-bank prerequisites for the documented
versioned pilot and bounded one/sixteen-block fits. Original response files
remain immutable; the new numerical target has separately versioned outputs.

### Model families are different experiments

| Model | Inputs / parameters | Evidence at this snapshot |
|---|---|---|
| Historical Qwen3-4B compact editor | Draft hidden states; 3.02M trainable parameters | Completed older teacher/distillation experiments under `results/` |
| Proposed Qwen3-8B compact editor | Draft hidden states; 4.20M parameters | Initial feature-bank/replay work; the new AR fit results do not establish a fit of this editor |
| Current context AR editor | Committed tokens, anchor, first proposal; 40,750,336 parameters | One- and sixteen-block memorization passed |
| Current whole-draft AR editor | Same inputs plus original proposed suffix; same parameter count | Same milestones passed |

Both AR variants own their embeddings/readout and generate fourteen editable
tokens sequentially. Qwen and DFlash supplied verified examples earlier; neither
was loaded or optimized during the sixteen-block fit. The two AR modes started
from the same random initialization, independently of the one-block checkpoints.

## Exact archive provenance

- Branch base: `fecf1bb8d2b9ef299bcbdd66262975705fb6afb4` from
  `work/editor-feature-bank`, including the completed sixteen-block report.
- Executed sixteen-block driver: `2bcdc06e1b7c688be918d1336ffad46951b1d434`.
- `editor-experiment/autoregressive_editor/` was copied unchanged from
  `456513249eb4f8bd9d1a118c5eb7c8909a8c30c2` (`work/ar-draft-editor`), except
  for the README's archive-status pointer. Its eight executed Python files match
  the [recorded source manifest](research/sixteen-block-fit-2026-11-17/evidence/source-manifest.json).
  This package lived on a separate branch and was assembled into the remote
  source bundle; including it here makes the archived training imports complete.
- The original chat worktree, `qwen-draft-model-changes-2`, remained on
  `learning-2` at `8491b95`. The other worktrees were not switched or edited to
  create this branch. Uncommitted plans from those worktrees are not implied to
  be part of this snapshot.
- [GLOSSARY.md](../GLOSSARY.md) preserves the vocabulary read from the feature-bank
  worktree. Results and decisions are authoritative in their dated records.

## Completed run inventory

| Job | Work | Process outcome | Scientific outcome |
|---|---|---|---|
| 8452102 | Independent legacy target-shape diagnostic | All probes completed; deliberate exit 1 | Exact equality assertion failed; shape-dependent disagreement confirmed |
| 8453328 | Invariant target pilot | Exit 0 | Four complete responses passed the declared equality checks |
| 8453776 | One-block AR fits | Exit 0 | Memorization passed; excluded-prompt transfer poor |
| 8454166 | Sixteen-block AR fits | Exit 0, 90 seconds | Memorization passed; broad development damage remains |

The sixteen training blocks came from four prompts, not sixteen independent
prompts. Their 224 editable-token presentations cover 215 distinct positions.
Development used 32 blocks from eight already-inspected prompts. Context/draft
editors repaired 0/1 blocks and damaged 20/19 respectively. The single separate
preservation control was damaged by both. This is neither an untouched final
test nor a native edited-generation speedup result.

## Mental model of remote execution

```text
Mac worktree -> uploaded source snapshot -> sbatch on Nexus submission host
    -> SLURM allocation on tron61 -> Python process -> logs/checkpoints/results
    -> agent reads artifacts and writes a report
```

Git branches organize source on the Mac. The GPU job uses an uploaded package
and its recorded hashes. Local edits do not automatically update that package.
SSH provides a command connection; SLURM owns a submitted batch job independently
of that connection. The `/fs/classhomes/` project is shared storage, while the
Python environment below is on node-local scratch and must be checked on tron61.

Remote project: `/fs/classhomes/fpajak/speculative-decoding-project`.
Submission endpoint: `fpajak@nexusclass.umiacs.umd.edu` (observed host node 01).
Use the normal UMD VPN and SSH/Duo flow. Existing SSH control sockets are temporary
session state, not durable access instructions or credentials.

### Inspect without changing a run

After SSH login:

```sh
cd /fs/classhomes/fpajak/speculative-decoding-project
squeue -u fpajak
sacct -j 8454166 --format=JobID,JobName,State,ExitCode,Elapsed
scontrol show job 8454166
tail -n 20 research/sixteen-block-fit-8454166/train.log
cat research/sixteen-block-fit-8454166/exit-status.txt
cat research/sixteen-block-fit-8454166/training/result.json
```

Use `scontrol` while SLURM still retains the job; use accounting and artifacts
afterward. `tail -f` follows live logs, and Ctrl-C stops only that log viewer.
An empty queue means no current job, not a successful experiment. Exit zero means
the process completed; inspect metrics, integrity checks and the report to decide
what it establishes. Exit one may be an intentionally preserved failed assertion
or an actual exception; the report/result distinguishes them.

At 2026-11-17 18:54:33 UTC, job 8454166 was `COMPLETED`, exit `0:0`, and the queue
was empty. This is a dated observation, not a continuing monitor.

### Where the artifacts are

Paths below are relative to the remote project unless absolute:

| Item | Location |
|---|---|
| Sixteen-block uploaded source | `research/sixteen-block-fit-20261117/` |
| Uploaded file hashes | `research/sixteen-block-fit-20261117/source-manifest.json` |
| Input captures | `research/one-block-fit-8453776/collection/` |
| Selected sixteen-block dataset | `research/sixteen-block-fit-8454166/dataset.json` |
| Main SLURM log | `ar-sixteen-block-8454166.log` |
| Actual training log | `research/sixteen-block-fit-8454166/train.log` |
| Training outputs | `research/sixteen-block-fit-8454166/training/{context,draft}/` |
| Checkpoints | The two output directories' `checkpoint.pt` files, with model/optimizer/RNG state |
| Isolated runtime | `/scratch0/fpajak-vllm034-Ry21jsII/venv/bin/python` on tron61 |
| New target pilot | `data/qwen_responses_vllm_invariant_pilot_20261117/` |
| Original responses | `data/qwen_responses_pilot/` and `data/qwen_responses_50k/` |

The launcher redirects Python stdout/stderr into `train.log`; an empty main SLURM
log therefore does not show inactivity. Check the saved `packages.txt`, `run.json`,
source hashes and dataset hash before reusing a runtime or checkpoint. Recorded
package versions differ between the old Transformers diagnostic and later vLLM
experiments; they are not interchangeable environments.

Large checkpoints, response arrays and private input captures stay on Nexus.
Temporary Mac paths in historical reports are conveniences, not durable backups.
The tracked reports, aggregate JSON and source manifests provide the Git record.

## Continuing from this snapshot

Re-check the queue, current branch/revision and later ADRs before planning work.
The inherited protocols describe completed bounded runs; the archive itself does
not queue jobs or authorize another fit. Historical `.sbatch` files name existing
source/output locations and need a deliberate new run plan before reuse.

The latest report proposes measuring the yield of more distinct training prompts
and genuine preservation examples, then comparing repair-only with mixed fits
while holding architecture fixed. That follow-up was not run at this snapshot.
The demonstrated gap is generalization/preservation, not an established parameter
capacity limit. Keep full-generation utility and complete verification cost as
separate future measurements.

The tutoring examination in this conversation remains pending at its first greedy
block-tracing exercise; no answer has been graded. These experiment records are
not evidence of the user's independently demonstrated implementation skills.
