# Scale and store the editor experiment's data

The intended model is the roughly 100-million-parameter **editor/drafter in the
Qwen3/DFlash experiment**. This is not a plan to pretrain a standalone language
model. Parameter count alone does not determine the required number of examples.

## What exists and what still needs generation

The pinned source download contains **215,337 usable unique training prompts**:
7,473 GSM8K and 207,864 UltraChat. The original 384-record JSONL is a small sample
of that pool, not the entire download. The raw files occupy about 778 MiB.

The new entry point, [`scripts/prepare_training_data.py`](../scripts/prepare_training_data.py),
turns that pool into portable Parquet shards. Its default output has **50,000
training prompts + 1,024 validation prompts**. Source prompts are the first stage:

```text
GSM8K/UltraChat prompts
  -> Qwen3-8B greedy responses, thinking disabled
  -> matched DFlash draft contexts and saved computations
  -> shared selected beginnings and continuation targets
  -> teacher/editor training examples
```

Only the first stage is implemented by this script. Its source manifests record
zero generated responses and contexts. [The separate Qwen response collector](QWEN_RESPONSES.md)
now implements the next GPU stage and records actual progress in its own journals.
Neither the upstream dataset answers nor extra
copies of a prompt substitute for target-specific training examples. The exact
DFlash checkpoint, feature tensors, prefix choices, and label contract still need
to be fixed in the GPU collection implementation.

## How much to collect

Use the 384-prompt pilot to debug token alignment, labels, and feature capture.
Then compare progressively larger training subsets with the same held-out
validation set. The following is an experimental budget, not a claim that a
particular count guarantees sufficient training:

| Training responses | Intended use | Planned contexts at 32 per response |
| ---: | --- | ---: |
| 384 | Check collection and training plumbing | 12,288 |
| 3,000 | First useful learning-curve point | 96,000 |
| 10,000 | Check whether diversity improves validation | 320,000 |
| 50,000 | Larger initial editor experiment | 1,600,000 |
| 100,000 or the full pool | Scale if validation and compute justify it | 3,200,000 or more |

Contexts from the same response are correlated. More distinct prompts can add
coverage that repeatedly sampling one response cannot. Inspect held-out loss,
accepted-token length, actual end-to-end decoding speed, and separate math/chat
results. Keep initialization, example ordering, and update budgets matched across
the three methods at each dataset size. Retain the upstream test sets for final
evaluation rather than repeatedly choosing settings against them.

For context, the [DFlash paper](https://arxiv.org/html/2602.06036v1) reports about
800K target-generated training samples and 100K-sample ablations. That supports
trying larger, diverse samples, but it does not establish a minimum for this
different editor architecture. Do not apply language-model pretraining token
scaling rules directly to this conditional editor.

## Run locally or on a cloud CPU

Run from a complete copy of this repository, with [uv](https://docs.astral.sh/uv/)
installed. No GPU, VPN, account, or API key is needed to download the public sources.

```bash
uv run --script scripts/prepare_training_data.py build
```

On a fresh machine this downloads and verifies the locked raw source files into
`data/prompt_corpus/raw/`. On this Mac they already exist, so use:

```bash
uv run --script scripts/prepare_training_data.py build --offline
```

**The default build was already run and verified on this Mac on 2026-11-03** in
`data/editor_prompts_50k/`. An existing output directory is deliberately refused;
use verification or choose a different `--output-dir` for another build. The
original `data/prompt_corpus/prompts.jsonl` and its manifest remain unchanged.
The verified export totals 19,234,315 bytes (about 18.3 MiB). Its manifest SHA256
is `7b8347bdd614b601e1662aef6edc511381383172326b195bf5e0e970a210c6a9`.
All 51,024 exported prompts were independently matched to their original source
rows, and all 384 pilot IDs remain in the training split.

```bash
uv run --script scripts/prepare_training_data.py verify
```

The default training mixture is 6,000 math + 44,000 chat prompts. It is deliberately
not 50/50: GSM8K has only 7,473 unique training questions. Validation reserves 512
from each source, so the largest available GSM8K training sample is 6,961. Repeating
math examples does not create new unique data; any later sampling weights should
be recorded as a training decision.

To use the entire available pool, preserving the same validation split:

```bash
uv run --script scripts/prepare_training_data.py build --offline --all \
  --output-dir data/editor_prompts_all
```

With the pinned files and default validation size, this yields **214,313 training
prompts + 1,024 validation prompts**. Omit `--offline` if the raw cache is absent.
The source directory is adjustable with `--source-dir`; it is a collection root
containing `raw/`, not the `raw/` directory itself. `--all` cannot be combined with
explicit training counts.

Smaller nested runs with the same 12% math / 88% chat mixture can be prepared
without another download:

```bash
uv run --script scripts/prepare_training_data.py build --offline \
  --gsm8k-count 360 --ultrachat-count 2640 --output-dir data/editor_prompts_3k
uv run --script scripts/prepare_training_data.py build --offline \
  --gsm8k-count 1200 --ultrachat-count 8800 --output-dir data/editor_prompts_10k
```

Keep `--seed 42` and `--validation-per-source 512` unchanged across these runs.
Validation is selected from the **full eligible source pool before** choosing
training counts. The original seed-42 pilot is protected from becoming validation.
Training sets are nested when per-source counts increase. Changing the seed,
validation count, or source lock creates a different experiment. Do not concatenate
these exports as if they were disjoint datasets.
The original 384-prompt pilot is balanced 50/50 and is for debugging. Above
roughly 58K training prompts, GSM8K cannot sustain a 12% math share with this
validation reservation. Scaling farther changes the mixture unless another
math training source is added; record that difference in the experiment.

## File layout and integrity

```text
data/editor_prompts_50k/
  README.md                         dataset card and usage notes
  manifest.json                     source pins, counts, hashes, generation plan
  provenance/*-README.md            preserved upstream dataset cards
  data/train-00000.parquet ...      ordered user-only training records
  data/validation-00000.parquet     ordered user-only validation records
```

Parquet is compressed with Zstandard and contains `id`, `source`, and `messages`.
The default shard size is 10,000 records, adjustable with `--shard-rows`. The same
explicit schema is used in every shard, including nullable UltraChat `prompt_id`.
Read shards in filename order and rows in their stored order. Raw upstream files
stay in the source cache; the export can be verified or uploaded independently.

Verification checks hashes, row counts, schema, source provenance, deterministic
order, and duplicate/blank IDs or text across both splits. Building also excludes
normalized matches against both upstream test splits. This is exact NFC and
whitespace matching, not semantic decontamination. Incomplete builds are staged
in a temporary directory and only promoted after verification; ordinary failures
do not publish a partial export. Repeat a failed download/build to retry it.

Every downstream response, context, alternative prefix, and tensor must inherit
its source prompt's split. Splitting individual contexts would leak material
from the same response into both training and validation.

## Storage and sharing

The [Nexus compute and external storage plan](DATA_PIPELINE_PLAN.md) adds concrete
upload ordering, failure recovery, optional Convex metadata, and GitHub Actions
access prerequisites. Its progress inspector is implemented; the upload service
and runner integration are still proposed.

Use **GitHub for code/configuration** and a **Hugging Face dataset repository for
versioned data exports**. Raw Parquet shards already exceed GitHub's 100 MiB
ordinary-file limit. Git LFS is possible, but a dataset host provides a more useful
distribution format for this workflow. [GitHub large-file documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files).

No Convex database or key is needed. Training primarily reads sequential batches
from files; a database would add requests and another service to maintain.
Convex's current free limits include 0.5 GB database storage, 1 MiB per document,
and 1 GB file storage. That is a poor fit for growing activation dumps.
[Convex limits](https://docs.convex.dev/production/state/limits).

Hugging Face currently documents 100 GB private storage for free users, with
best-effort public storage. Check your account's remaining quota before producing
large tensors; this is not unlimited storage.
[Hugging Face storage limits](https://huggingface.co/docs/hub/storage-limits).

Keep text/metadata in Parquet. For later dense feature arrays, use sharded tensor
files such as Safetensors plus a manifest connecting tensor keys to prompt/context
IDs, shapes, and dtypes. If these grow beyond the dataset account budget, put the
tensor shards in university project storage or object storage near the GPU workers;
keep versioned manifests and small dataset exports on the Hub. Download/cache each
shard near the training job rather than fetching individual examples over an API.
A database can be added later for a dashboard's job metadata; it need not contain
the training tensors.

Estimate feature volume after the pilot. For example, **one hypothetical** BF16
tensor with 16 positions × 4,096 features costs 131,072 bytes per context. At
1.6 million contexts that single tensor alone would occupy about 195 GiB before
compression. Multiple layers, logits, or cached attention values can add much more.
These dimensions are an illustration, not the finalized editor feature schema.

## Optional private Hugging Face upload

Nothing has been uploaded by this change. Choose your own dedicated dataset repo
and authenticate with the Hugging Face CLI's interactive login or a runner-managed
`HF_TOKEN` secret. A Convex key does not authenticate to Hugging Face. The script
does not accept credentials as arguments and uploads only verified export files.

With the data dependencies installed in `.venv-data`, for example:

```bash
.venv-data/bin/hf auth login
.venv-data/bin/python scripts/prepare_training_data.py upload \
  --output-dir data/editor_prompts_50k --repo-id YOUR_HF_USERNAME/dflash-editor-prompts-50k
```

See [environment setup](DATA_COLLECTION.md#run-and-verify) if that environment does
not exist. The uploader creates a **private dataset repository**, refuses an
existing public destination or stale/unrelated remote files, and sends an exact
allowlist: shards, dataset cards, and manifest. It does not send the raw cache,
code worktree, attachments, or credentials. It uses the observed parent commit
to detect concurrent updates and prints the resulting commit URL. Record that
commit for subsequent downloads. Retrying the upload uses the Hub's transfer/cache
mechanism; choose a new dataset repository for incompatible export layouts.

The export includes source citations and pinned upstream cards. Those cards
declare MIT licensing; preserve attribution when sharing derived data. The GPU
collector must separately record its model revisions and generated-data provenance.

## Run on Nexus and check changes

Use the [cloud agent handoff](CLOUD_AGENT_HANDOFF.md) for VPN, SSH/Duo, and storage
paths. The 50K export, Python utilities, and data wrapper were copied there during
Qwen setup on 2026-11-03. The installed `.venv-qwen` also contains the source-data
dependencies. From the remote project root, use a new output directory because
the 50K export already exists:

```bash
export DATA_PYTHON="$PWD/.venv-qwen/bin/python"
sbatch scripts/prepare_training_data.sbatch build --offline --all --output-dir data/editor_prompts_all
```

The wrapper requests two CPUs, 4 GiB RAM, and 30 minutes in the `class` partition
and account; no GPU is used. Monitor the actual SLURM job ID and log. Do not perform
full scans, environment installation, or GPU generation on a submission node.

Run tests with the lightweight Parquet dependency, without installing inference:

```bash
uv run --no-project --python 3.12 --with 'pyarrow==21.0.0' python -m unittest discover -s tests -v
python3 scripts/validate_tours.py
```

Tests use synthetic source files and mock the Hub client. They check stable
validation across scaling, pilot preservation, held-out/duplicate filtering,
portable exports, corruption, insufficient data, and upload boundaries.
Open **CodeTour: Start Tour → 6 - Collect reproducible source prompts** in VS Code
to follow the extended [data tour](../.tours/06-prompt-data.tour).
