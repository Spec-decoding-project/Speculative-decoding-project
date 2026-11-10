# Collect Qwen3-8B responses

[`scripts/generate_qwen_responses.py`](../scripts/generate_qwen_responses.py) runs
the target model on the collected prompts and saves text **and exact token IDs**.
This follows [source preparation](TRAINING_DATA.md). It does not yet save DFlash
computations or construct editor/teacher continuation examples.

## Execution record: 2026-11-03

- Environment/model setup: job `8146461`, **COMPLETED**, exit `0:0`.
- First four responses: job `8146570` on `tron49`, **COMPLETED**, exit `0:0`.
  All four ended with EOS, totaling 2,134 output tokens in 90.8519 seconds of
  generation (the complete job, including loading, took 2 minutes 45 seconds).
  A copy was downloaded into local `data/qwen_responses_pilot/` and verified.
- Remaining original pilot: job `8146716`, continuing the same 384-prompt run
  while skipping the first four saved responses.
- Larger collection: array job `8146717`, 64 shards with three concurrent workers,
  covering **50,000 training + 1,024 validation prompts**. Together with the pilot,
  this uses at most four GPUs. The larger run has started generating responses.

The last two jobs were still running when this record was written; query live
state rather than treating submission as completion. At the four-prompt sample's
throughput, the larger corpus is a multi-day workload; prompt length and queue
availability make a precise completion estimate unreliable.

Remote outputs are under `/fs/classhomes/fpajak/speculative-decoding-project/data/`:
`qwen_responses_pilot/` and `qwen_responses_50k/`. The local pilot copy contains
only the four verified sample responses unless refreshed. Readable samples are
in the local gitignored `.context/qwen_sample_responses.md`.

## Fixed generation settings

[`configs/qwen_generation.json`](../configs/qwen_generation.json) pins
`Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218`:

- Full BF16 weights on CUDA, Transformers, SDPA attention, one prompt at a time.
- `enable_thinking=False` in the tokenizer's chat template.
- Greedy generation: `do_sample=False`, one beam, no sampling or repetition penalty.
- Up to 2,048 new tokens. A response ending at that cap is marked `length`, not
  counted as a naturally completed answer.
- Inputs over 6,144 tokens are explicitly skipped, without truncating their text.
- No extra system instruction or benchmark-specific math suffix is added.

The [official Qwen model card](https://huggingface.co/Qwen/Qwen3-8B) documents the
thinking switch. Greedy decoding is the experiment's requested setting; Qwen's
general-purpose sampling recommendations are not substituted for it. Greedy
does not guarantee identical tokens across different hardware or library versions.
Each worker records its runtime, tokenizer template hash, and EOS IDs.

## Run on Nexus

Use the [access handoff](CLOUD_AGENT_HANDOFF.md) for VPN and Duo. Submit from
`/fs/classhomes/fpajak/speculative-decoding-project` after copying the new scripts
and configuration there. The class nodes inspected for this run expose RTX A5000
GPUs with 24,564 MiB VRAM. The collector requires at least 22 GiB and BF16 support;
it never silently switches to quantized weights.

The CPU setup job creates `.venv-qwen`, installs pinned inference dependencies,
and downloads the pinned model to `data/models/qwen3-8b/`:

```bash
sbatch scripts/setup_qwen.sbatch
```

The environment uses PyTorch 2.13.0 with CUDA 12.6, Transformers 5.15.0,
Accelerate 1.12.0, Hugging Face Hub 1.23.0, and PyArrow 21.0.0. Allow roughly
16.4 GB for weights plus several GB for the environment and output. Installation
and model download occur on a compute node. The model's `pinned_revision.json`
marker is written after the download succeeds.

After setup succeeds, begin with four of the original 384 prompts:

```bash
sbatch scripts/generate_qwen_responses.sbatch --max-prompts 4
```

Continue the same run by omitting the per-invocation limit:

```bash
sbatch scripts/generate_qwen_responses.sbatch
```

The wrapper requests one RTX A5000, four CPU cores, 32 GiB RAM, and three hours.
Override SLURM time for larger runs as needed, within the account's current QoS
limits. The job survives closing the client SSH connection. `--max-prompts` limits
only new responses from that invocation; it does not change the run's identity.

For the larger Parquet export, pass `--input data/editor_prompts_50k` and a new
`--output-dir`. Both training and validation are included by default. `--split`
can select either explicitly. Never move contexts derived from one prompt into
the opposite split.

## Parallel workers and resuming

`--num-shards N --shard-index I` assigns input index `j` to worker `j % N`.
Choose N once per output directory; workers may execute sequentially or as a
SLURM array. All workers must use the same input, configuration, and script bytes.
Use an array concurrency limit that respects the student's aggregate GPU quota
and any other jobs already running. Do not change settings mid-run.

The larger collection was submitted on November 3 using the following command.
**Check its current state before submitting anything; do not duplicate active workers.**

```bash
QWEN_NUM_SHARDS=64 sbatch --array=0-63%3 --time=1-12:00:00 \
  --output=qwen-responses-%A_%a.log scripts/generate_qwen_responses.sbatch \
  --input data/editor_prompts_50k --output-dir data/qwen_responses_50k
```

Keep `QWEN_NUM_SHARDS=64` even when resubmitting only failed array indices. The
remaining pilot was submitted with a 12-hour limit; large array workers have a
36-hour limit. Successful workers exit early when their assigned prompts finish.

Every answer is flushed and fsynced to the worker's JSONL journal. On restart,
finished records are checked and skipped. Only an unfinished final line can be
discarded and regenerated. Corruption of completed records, changed inputs or
settings, a changed runtime in a resumed shard, or a second simultaneous writer
causes failure. Out-of-memory/model errors stop the worker while retaining prior
completed records; they are not converted into fabricated answers.

```text
data/qwen_responses_pilot/
  run.json                         input fingerprint, model/settings, script hash
  shard-00000.jsonl                 completed responses and explicit input skips
  shard-00000.runtime.json          GPU/library/template details
  shard-00000.summary.json          counts, tokens, timing, journal checksum
  *.lock                           worker/manifest locks
```

Each generated record retains the original source, messages, dataset split, and
input index. It adds prompt and response token IDs, readable response text,
`finish_reason`, timing, and integrity hashes. Response token IDs include the
terminal EOS when present. Do not infer training readiness solely from a JSONL
line count: distinguish EOS completions, length-capped outputs, and input skips.

Source manifests still describe source preparation and keep their own generation
counters at zero. Actual generation progress is in these separate journals,
summaries, and SLURM logs. A summary is refreshed when a worker exits normally or
is verified; during a running job the append-only journal and log are more current.

## Check outputs

After the worker stops, verification reads saved outputs without loading Qwen:

```bash
.venv-qwen/bin/python scripts/generate_qwen_responses.py --verify-only
```

Use the same input/output and shard arguments as generation for other runs.
The JSONL pilot verification uses only the standard library; a Parquet input also
requires PyArrow. Verification needs the corresponding original input collection.
Avoid verifying the same shard while its writer holds the lock.

Monitor the existing jobs and inspect a worker's log on the submission node:

```bash
squeue -u fpajak
sacct -j 8146716,8146717 --format=JobID,State,ExitCode,Elapsed
tail -20 qwen-responses-8146717_0.log
```

For counts from active journals, use the read-only
[`summarize_qwen_responses.py`](../scripts/summarize_qwen_responses.py) on an
allocated CPU. It checks completed-record hashes, reports EOS/caps/skips and
remaining work, and excludes incomplete tails without changing them. Passing both
run directories also reports distinct source prompt IDs so the overlapping pilot
does not inflate dataset diversity. This is a bounded observation, not full
verification against the original input dataset. The command and external storage
design are in [the pipeline plan](DATA_PIPELINE_PLAN.md).

Transformers may report unused sampling flags from the checkpoint. These are
ignored under the explicitly supplied `do_sample=False` configuration; they do
not enable sampling. The installed 5.15.0 configuration merge preserves explicitly
set values and only fills unset parameters from the checkpoint.

Tests cover interruption recovery, completed-record corruption, fixed shard
assignment, input/configuration changes, and runtime mismatches:

```bash
uv run --no-project --python 3.12 --with 'pyarrow==21.0.0' python -m unittest discover -s tests -v
python3 scripts/validate_tours.py
```

The tests use an offline fake backend; actual GPU execution is verified separately.
In VS Code, choose **CodeTour: Start Tour → 6 - Collect reproducible source prompts**
to follow the source pipeline through the added target-response steps.
