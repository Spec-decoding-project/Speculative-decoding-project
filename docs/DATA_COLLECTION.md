# Collect GSM8K and UltraChat source prompts

For a new agent taking over access and scaling, use the
[cloud agent handoff](CLOUD_AGENT_HANDOFF.md).
For the larger editor dataset, fixed validation splits, Parquet export, and
private Hugging Face upload, use [Scale and store training data](TRAINING_DATA.md).
For actual target-model outputs, use [Qwen response collection](QWEN_RESPONSES.md).

The entry point is `scripts/collect_prompt_data.py`. It downloads source data and
builds a fixed prompt list for later Qwen3-8B generation. It does not run inference
or training. The initial collection uses an equal mixture of 192 math and 192
general-instruction prompts, seed 42. This is a starting choice, adjustable with
`--gsm8k-count`, `--ultrachat-count`, and `--seed`.

## Sources and split boundaries

| Source | Training split | Held-out split | Prompt field |
| --- | --- | --- | --- |
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), `main` | `train`: 7,473 rows | `test`: 1,319 rows | `question` |
| [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), `default` | `train_sft`: 207,865 rows | `test_sft`: 23,110 rows | `messages[0].content` (first user turn) |

UltraChat-200k is the cleaned UltraChat variant. The original 1.4-million-dialogue
corpus and the separate `train_gen`/`test_gen` splits are not part of this collection.
Both selected datasets are public and their cards declare MIT licensing; the
download preserves those cards and the original Parquet files, including answers
and full conversations. No Hugging Face login is needed for these sources.

[`configs/prompt_sources.json`](../configs/prompt_sources.json) pins exact Hub
commits, file paths, sizes, row counts, and SHA256 hashes. The initial hashes were
checked against Hub LFS metadata (and Git blob IDs for the dataset cards).
The eight source files total 815,942,139 bytes, about 778 MiB. Allow additional
space for the Python environment and download cache.

The collector scans both held-out splits before sampling training rows. It
rejects blank prompts and excludes matches after NFC Unicode normalization and
whitespace collapsing, including duplicates across sources. This is an exact
normalized-text check, not semantic near-duplicate detection. Original prompt
text is retained in the output. A SHA256 ranking of the seed and stable source
row ID selects the sample across each full training split; a separate ranking
fixes the final mixed order.

The separate UltraChat `prompt` column sometimes differs in capitalization from
the first conversation message. The collector takes `messages[0].content` to
preserve the cleaned conversation text and requires its role to be `user`.
GSM8K answers and UltraChat assistant/later-user messages remain only in the raw
files. The output contains one user message per prompt. Do not use the existing
`dflash.benchmark` dataset loader for training selection: its GSM8K split is `test`.

## Run and verify

From the repository root, with [uv](https://docs.astral.sh/uv/) installed:

```bash
uv run --script scripts/collect_prompt_data.py
python3 scripts/collect_prompt_data.py --verify-only
```

The script's inline dependencies are `huggingface-hub==1.23.0` and
`pyarrow==21.0.0`, with Python 3.10–3.13. Without uv, use a separate environment:

```bash
python3.12 -m venv .venv-data
.venv-data/bin/python -m pip install 'huggingface-hub==1.23.0' 'pyarrow==21.0.0'
.venv-data/bin/python scripts/collect_prompt_data.py
```

Downloads can be resumed by repeating the command. `--offline` rebuilds the
prompt file using existing raw files after checking them against the source lock.
`--verify-only` needs only Python's standard library: it checks all source and
prompt checksums, IDs, counts, source splits, and message structure. It does not
contact Hugging Face or import inference libraries.

Missing files, checksum failures, unexpected row counts, and an insufficient
number of unique eligible prompts produce a nonzero exit. For a checksum error,
replace the named source file with a fresh download from its pinned revision;
cached downloads alone do not repair a manually modified local file. Rebuilding
replaces the derived `prompts.jsonl` and `manifest.json`. Choose a different
`--output-dir` if you want to retain a previous sample with different settings.

```text
data/prompt_corpus/
  raw/gsm8k/             original training/test Parquet and dataset card
  raw/ultrachat_200k/    original SFT training/test Parquet and dataset card
  prompts.jsonl         fixed ordered list of 384 user-only prompts
  manifest.json         source lock, selection statistics, checksums, next-stage plan
```

Each JSONL row has `id`, `source`, and `messages`. Source provenance includes the
dataset, configuration, revision, split, and zero-based row index across the
ordered split shards in the source lock. UltraChat also retains `prompt_id`.
The raw datasets and derived files are gitignored, so a fresh checkout must run
the collector or receive a copy of the entire collection directory.

The initial sample has SHA256
`353ade1401738b544990fe8b0cbd3845c6bcaaa838e086ba973535db9127c1aa`.
The scan found one blank UltraChat training prompt, no normalized training/test
overlap, and no duplicate training prompts. There are 24,429 unique held-out
prompts. The resulting 384 prompts occupy 354,435 bytes.

## Nexus student account

UMIACS requires the UMD GlobalProtect VPN before class-account SSH access.
Connect the VPN, then sign in to `fpajak@nexusclass.umiacs.umd.edu` with your UMD
directory credentials and complete Duo authentication. An open SSH tab does
not automatically authenticate a separate SSH or file-transfer process.
See the [official class-account guide](https://wiki.umiacs.umd.edu/umiacs/index.php/ClassAccounts).

The shared collection destination for this workspace is
`/fs/classhomes/fpajak/speculative-decoding-project/data/prompt_corpus`.
The collector, SLURM wrapper, and source lock belong in the corresponding
`scripts/` and `configs/` directories under `speculative-decoding-project`.

The collection was copied and verified on Nexus on 2026-11-03. SLURM job
`8143715` ran the wrapper's verification mode on `tron43` with one CPU and
1 GiB RAM, finishing in two seconds with state `COMPLETED` and exit code `0:0`.
All source checksums and the 384-prompt checksum matched the local collection.
The remote log is
`/fs/classhomes/fpajak/speculative-decoding-project/prompt-data-8143715.log`.
Only source data and prompts were collected; generation counters remain zero.

To copy the local collection into this existing directory structure:

```bash
scp -r data/prompt_corpus fpajak@nexusclass.umiacs.umd.edu:/fs/classhomes/fpajak/speculative-decoding-project/data/
```

To verify the transferred collection without installing dependencies, submit
from its project directory on Nexus:

```bash
cd /fs/classhomes/fpajak/speculative-decoding-project
sbatch --cpus-per-task=1 --mem=1G --time=00:05:00 scripts/collect_prompt_data.sbatch --verify-only
```

For fresh collection on Nexus, put a Python 3.10–3.13 environment with the two
data dependencies in shared home/project storage. From the project directory:

```bash
export DATA_PYTHON="$PWD/.venv-data/bin/python"
sbatch scripts/collect_prompt_data.sbatch
```

The wrapper defaults to the `class` partition/account, default QoS, two CPU
cores, 4 GiB RAM, and 30 minutes, with no GPU. The verification example overrides
those resource amounts to one CPU, 1 GiB RAM, and five minutes. Both write
`prompt-data-JOBID.log`; collector arguments pass through unchanged.
UMIACS prohibits computational jobs on submission nodes. Keep the data in shared
home/project storage so compute nodes can read it; the documented home quota is
30 GB, sufficient for this source collection.

## Next stage and checks

The manifest records the requested next-stage settings: `Qwen/Qwen3-8B`, thinking
disabled, greedy decoding, and 32 draft contexts per response (12,288 planned
contexts for this sample). These are plans, not generated results. Building those
training examples will also need the matched DFlash checkpoint, inference, and
instrumentation that saves draft calculations and shared selected beginnings.
Reuse this exact prompt order across the comparison methods; initialization and
training-update budgets are controlled later in the training pipeline.

```bash
uv run --no-project --python 3.12 --with 'pyarrow==21.0.0' python -m unittest discover -s tests -v
python3 scripts/validate_tours.py
```

The offline tests cover answer exclusion, split leakage, duplicate/blank handling,
deterministic sampling, checksum failures, and manifest round trips. Open
**CodeTour: Start Tour → 6 - Collect reproducible source prompts** in VS Code
to walk through the implementation and prediction questions.
