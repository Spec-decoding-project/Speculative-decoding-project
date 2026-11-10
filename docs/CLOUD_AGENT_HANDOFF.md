# Agent handoff: Nexus access and Qwen data collection

Updated **2026-11-10** for moving to another workspace. Give this file to the next
agent. It contains operational instructions, not passwords or phone access.

## Start with this request

> Continue Franciszek's Qwen3/DFlash data collection. Read this handoff and the
> linked guides before changing anything. First reconnect to Nexus using VPN,
> interactive SSH, and a Duo Push that Franciszek approves on his phone. Check the
> existing jobs, count actual saved responses, inspect storage usage, and report
> any failures. Preserve existing data and resume only work that is actually
> incomplete. Then continue the external-storage plan after resolving its missing
> destination/authentication inputs. Do not infer current completion from old logs.

## Get the code in the new workspace

Repository: https://github.com/Spec-decoding-project/Speculative-decoding-project

Collection branch: **`Frankstuff/gsm8k-ultrachat-data-collection`**.
This handoff, collection scripts, pinned configurations, tests, and CodeTour are
published together on that branch. A fresh clone can select it directly:

```bash
git clone --branch Frankstuff/gsm8k-ultrachat-data-collection \
  https://github.com/Spec-decoding-project/Speculative-decoding-project.git
cd Speculative-decoding-project
```

The collection branch starts from `cb44a766fa40ff56fdf7ea5d1898a54e9ac1fb1f`.
`origin/main` has since gained separate work, observed at `e1c9fc8` on November 10.
Use the collection branch to reproduce this handoff; integrating it into newer
main requires reviewing those changes. Do not replace main with this older base.
The exact published collection commit is available in the accompanying handoff
message and through `git rev-parse HEAD` on this branch.

Large files under `data/`, environments, `.context/`, private logs, and SSH sockets
are gitignored and are **not** transferred by cloning. Source data can be rebuilt
from the pinned configuration; the generated results are on Nexus. Keep the old
workspace until any wanted local artifacts have been copied or verified on Nexus.

## Access: computer VPN, SSH, then approval on the phone

The agent does not need to control the phone. It selects **Duo Push** during SSH
login; Franciszek opens/approves the resulting Duo Mobile notification himself.
There is no phone-control credential or reusable Duo approval in this handoff.

The computer running SSH must be connected to the UMD GlobalProtect VPN, portal
`access.umd.edu`. A cloud runner does not inherit the Mac's VPN or open Terminal
tab. Check network access before debugging passwords. A network timeout occurs
before authentication; no MTU change was needed in the earlier successful login.
[UMIACS class access](https://wiki.umiacs.umd.edu/umiacs/index.php/ClassAccounts).

Account: `fpajak`. Normal endpoint: `nexusclass.umiacs.umd.edu`.
The last successful host was `nexusclass01.umiacs.umd.edu`; the alias sometimes
reached node 00 and timed out during the SSH banner. An agent may try the explicit
node 01 endpoint if the alias stalls, while retaining host-key verification.

From a VPN-connected computer, use a PTY-capable terminal tool:

```bash
ssh -tt -o ConnectTimeout=15 -o ConnectionAttempts=1 \
  fpajak@nexusclass.umiacs.umd.edu
```

1. Verify a new host fingerprint using the [official SSH instructions](https://wiki.umiacs.umd.edu/umiacs/index.php/SSH)
   and [UMIACS host-key reference](https://intranet.umiacs.umd.edu/hostkeys). Do not
   disable host-key checking to get past a mismatch.
2. Obtain the directory passphrase through supported secret input, or let the user
   type it interactively. Enter it only at the SSH password prompt, where input
   echo is disabled. Do not put it in Git, command arguments, transcripts, or logs.
3. Read the Duo menu and choose **Duo Push**; it was option `1` previously. The
   user requested the mobile notification, not a phone call.
4. Immediately tell the user that the notification was sent and wait for approval.
   If it expires, check that the user is ready before sending another. Do not
   repeatedly trigger pushes or attempt to automate approval on the phone.
5. Confirm a remote shell with `hostname`, `id -un`, and `pwd`.

`BatchMode=yes` cannot perform the initial password/Duo login. A fresh cloud
machine may require VPN setup and interactive authentication before any of this
works. If that is unavailable, report the network/authentication blocker and
continue code or public-data work independently.

### Reuse the approved connection for commands and copies

Optionally start the initial login as a control master. Run this on the client
instead of the simple SSH command above:

```bash
NEXUS_HOST=nexusclass.umiacs.umd.edu
NEXUS_SSH_DIR="$(mktemp -d /tmp/nexus-agent.XXXXXX)"
chmod 700 "$NEXUS_SSH_DIR"
NEXUS_SSH_CTL="$NEXUS_SSH_DIR/control"
printf '%s\n' "$NEXUS_SSH_CTL"
ssh -tt -M -S "$NEXUS_SSH_CTL" -o ControlPersist=no \
  -o ConnectTimeout=15 -o ConnectionAttempts=1 "fpajak@$NEXUS_HOST"
```

Leave the authenticated master PTY running. Record its host and absolute socket
path in runtime state. Variables do not automatically persist across separate
tool processes: set `NEXUS_HOST` and `NEXUS_SSH_CTL` to those same values in each
subsequent client process before running the examples below.

```bash
ssh -S "$NEXUS_SSH_CTL" -o BatchMode=yes "fpajak@$NEXUS_HOST" 'hostname'
```

A previous successful session used node 01 with
`HostKeyAlias=nexusclass.umiacs.umd.edu` and `StrictHostKeyChecking=yes` against an
already trusted alias entry. That is not a substitute for validating host keys
on a new machine. Keep the host choice consistent for the session and copies.

Close only the agent's own master when remote work finishes:

```bash
ssh -S "$NEXUS_SSH_CTL" -O exit "fpajak@$NEXUS_HOST"
```

## Known data, jobs, and freshness

**The last successful remote inspection was November 3.** The November 5 login
attempt reached Duo but expired without approval. No fresh inspection was made
for this November 10 handoff. Current job states and response counts are unknown.

| Item | Previously verified state |
| --- | --- |
| Remote project | `/fs/classhomes/fpajak/speculative-decoding-project` |
| Source pool | 215,337 eligible unique training prompts from GSM8K and UltraChat-200k |
| Original pilot | 384 prompts, 192 per source, seed 42 |
| Larger export | 50,000 training + 1,024 validation prompts; all pilot IDs are in training |
| Source verification job | `8143715`, completed successfully |
| Qwen setup job | `8146461`, completed successfully |
| Four-response sample job | `8146570`, completed successfully; all four ended with EOS |
| Remaining pilot job | `8146716`, last observed running; target 384 including the four samples |
| Larger array job | `8146717`, 64 shards, three concurrent GPU workers; target 51,024 |
| Last observed response counts | Pilot 13, larger array 11 on November 3; stale snapshots, not current totals |
| DFlash contexts/features | None collected by this pipeline; feature collector still to implement |

Do not add pilot counts to the larger run as if all prompt IDs were distinct.
Do not confuse source-manifest planned counts with generated examples. The local
pilot copy contains only the four downloaded sample responses.

Under the remote project root:

```text
data/prompt_corpus/             original raw sources, manifest, and 384-prompt JSONL
data/editor_prompts_50k/        verified Parquet input export
data/qwen_responses_pilot/      run.json and original-pilot response journals
data/qwen_responses_50k/        run.json and 64 response journals/sidecars
data/models/qwen3-8b/           full BF16 model snapshot and pinned_revision.json
.venv-qwen/bin/python           Python 3.12 inference/data environment
scripts/                       deployed source/export/generation utilities
configs/                       source and generation pins
qwen-responses-8146716.log      pilot log
qwen-responses-8146717_*.log     array logs
```

The remote directory is **not a full Git checkout**. Its utility deployment does
not include the new progress inspector, tests, or tours as of the last transfer.
Clone full code elsewhere if needed; preserve the existing run directories and
environment. The old Mac workspace is:
`/Users/franciszekpajak/conductor/workspaces/speculative-decoding-project/ndjamena`.
Its private `.context/` includes sample responses and old logs but is not needed
to understand this handoff or reconstruct the source datasets.

## First remote task: inspect, do not launch duplicates

On the submission node:

```bash
cd /fs/classhomes/fpajak/speculative-decoding-project
squeue -u fpajak
sacct -S 2026-11-03 -j 8146716,8146717 \
  --format=JobID,State,ExitCode,Elapsed,NodeList
show_qos
show_partition_qos
```

An empty `squeue` means no currently queued/running jobs, not proof of successful
completion. Inspect accounting, logs, and journals. If older accounting is no
longer available, report that limitation and use output verification as evidence.

Copy **only the new progress inspector** from the client checkout; this does not
change the generator that existing runs pin:

```bash
scp -o ControlPath="$NEXUS_SSH_CTL" -o BatchMode=yes \
  scripts/summarize_qwen_responses.py \
  "fpajak@$NEXUS_HOST:/fs/classhomes/fpajak/speculative-decoding-project/scripts/"
```

Then run the scan in a CPU allocation from the remote project directory:

```bash
srun --partition=class --account=class --qos=default \
  --cpus-per-task=1 --mem=2G --time=00:10:00 \
  .venv-qwen/bin/python scripts/summarize_qwen_responses.py \
  data/qwen_responses_pilot data/qwen_responses_50k > qwen-progress.json
```

The report separates EOS answers, length-capped answers, oversized-input skips,
remaining work, tokens, bytes, and unique prompt IDs. It reads bounded snapshots
of active journals, checks saved-record hashes and shard identity, and does not
alter incomplete final lines. It does not independently validate the original
input dataset or model outputs. Full `--verify-only` checks belong after a writer
stops; see [Qwen response collection](QWEN_RESPONSES.md).

Measure home/project usage and the actual user quota before further downloads or
feature generation. The filesystem's global free space is not the user's quota.
Report disk use, failed jobs, and incomplete shards before deciding what to resume.

## Preserve the experiment and resume contract

Target: `Qwen/Qwen3-8B`, revision
`b968826d9c46dd6066d109eabc6255188de91218`, BF16 CUDA/SDPA, thinking disabled,
greedy (`do_sample=False`), 2,048 output-token cap, 6,144 input-token cap.
Overlong inputs are explicitly skipped. Length-capped responses remain marked.
The tested nodes used RTX A5000 GPUs with about 24 GiB VRAM.

The deployed environment was Python 3.12.12, torch 2.13.0+cu126, Transformers
5.15.0, Accelerate 1.12.0, huggingface-hub 1.23.0, and PyArrow 21.0.0.
Reuse it for resumption. Setup/model download has already succeeded; do not rerun
setup merely to inspect outputs.

Runs pin input order/hash, generation configuration, shard count, generator bytes,
and per-shard runtime metadata. Keep `num_shards=64` for the larger run even when
resuming only failed shard indices. Never run a second writer on an active shard.
Do not edit the deployed generator/configuration or delete its journals mid-run.
The existing collector uses those journals to skip completed work.

Reference SHA256 values for this collection snapshot:

```text
generator:     421e593f7f2ab56bfac09adc84f7d887455ebeb0955685ed8c22abf9b43f6ac0
generation:    3a55740b91d29b6b1ec81b5731de8ea7a7f4088015e2af79a8b88361ca0cae70
pilot JSONL:   353ade1401738b544990fe8b0cbd3845c6bcaaa838e086ba973535db9127c1aa
50K manifest:  7b8347bdd614b601e1662aef6edc511381383172326b195bf5e0e970a210c6a9
```

The roughly 100M-parameter model is an **editor/drafter**, not a standalone
language model trained from scratch. Still needed: a matched DFlash checkpoint,
feature tensor schema, shared selected prefixes for teacher/editor, and labels.
The initial 32 contexts/response are planned, not produced. Keep prompt families
in one split and comparison initialization, order, and update budgets matched.

## Storage and automation follow-up

Read [DATA_PIPELINE_PLAN.md](DATA_PIPELINE_PLAN.md) and
[TRAINING_DATA.md](TRAINING_DATA.md). The proposed design is Nexus compute,
verified compressed files in private Hugging Face/object storage, and optional
Convex job/shard metadata. The uploader service and GitHub coordination are not
implemented or deployed. No data has been uploaded to Hugging Face or Convex by
this work. The existing prompt-export utility has an optional private Hub upload.

A GitHub repository secret named `CONVEX_API_KEY` was listed on November 5; its
value/type has not been retrieved or tested. The repository was public and had
no registered runners. A Convex key grants neither school VPN nor SSH access.
Missing deployment inputs include the storage destination/scoped credential,
Convex deployment URL/key type if used, and school-supported unattended access
if GitHub coordination is required. Use GitHub secrets inside intended workflows,
never export them through logs or artifacts to obtain their values locally.

School constraints: partition/account `class`; query live QoS limits. Published
aggregate caps are 32 CPU cores, four GPUs, and 256 GB RAM. Home quota is 30 GB;
additional project storage goes through the TA/instructor. Fall storage may be
removed from January 1, 2027 onward. Keep an independent copy of wanted outputs.
[Class limits](https://wiki.umiacs.umd.edu/umiacs/index.php/ClassAccounts).
Run inference, dataset scans, and substantial setup in SLURM allocations, not on
submission nodes. [Submission-node policy](https://wiki.umiacs.umd.edu/umiacs/index.php/Nexus/Submission_Node_Policy).

## Checks and final report

From the complete client checkout:

```bash
uv run --no-project --python 3.12 --with 'pyarrow==21.0.0' \
  python -m unittest discover -s tests -v
python3 scripts/validate_tours.py
```

The collection suite has 29 offline tests. They check data integrity, splits,
resumption, and progress accounting; they do not replace actual GPU validation.
Update the existing `.tours/06-prompt-data.tour` for meaningful code changes.
In VS Code: **CodeTour: Start Tour → 6 - Collect reproducible source prompts**.

Report the inspection timestamp, job states, unique prompt/response counts,
EOS/cap/skip counts, remaining shards, output paths, storage usage/quota, and
specific blockers. State exactly which artifacts were verified or uploaded.
Useful references: [source collection](DATA_COLLECTION.md),
[Qwen generation/resumption](QWEN_RESPONSES.md), and
[external storage plan](DATA_PIPELINE_PLAN.md).
