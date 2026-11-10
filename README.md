# DFlash — an independent starting point

This repository contains unmodified DFlash Python source from [`z-lab/dflash` at `07ebd93`](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a), ready for new experiments. It has fresh Git history under Frankstuff's repository. The earlier conditional-redrafting experiment is not included in this baseline.

**Start with [the code-tour guide](docs/CODE_TOURS.md).** Six JSON tours in [`.tours/`](.tours) teach the project layout, decoding loop, model internals, backends, benchmark interpretation, and prompt-data collection. [Tour prompts and research](docs/TOUR_PROMPTS.md) contains reusable prompts for generating and learning from future tours.

## What the code does

DFlash uses a draft model to propose several tokens in parallel. A larger target model verifies the proposals, retains a valid prefix, and supplies a correction or bonus token. The aim is faster generation while preserving the target model's decoding behavior.

| File | Role |
| --- | --- |
| [`dflash/cli.py`](dflash/cli.py) | `generate` and `benchmark` commands; chooses Transformers, MLX, or an HTTP server |
| [`dflash/model.py`](dflash/model.py) | PyTorch generation, draft attention, DFlash and DFlash2 model classes |
| [`dflash/model_mlx.py`](dflash/model_mlx.py) | Apple Silicon implementation, streaming, and cache/state rollback |
| [`dflash/benchmark.py`](dflash/benchmark.py) | Model loading, prompt formatting, datasets, server requests, and benchmark summaries |
| [`dflash/__init__.py`](dflash/__init__.py) | Lazy public imports so basic CLI help does not load model libraries |

The upstream package contains both original DFlash and DFlash2. The tours first explain original DFlash with greedy decoding, then identify the other paths. This checkout has no added refiner, conditional-redrafting runner, or training pipeline. Separate collection scripts prepare source prompts and Qwen target responses for future experiments.

## Collect prompt data

Moving workspaces or agents? Start with the [Nexus access and collection handoff](docs/CLOUD_AGENT_HANDOFF.md), including the branch to clone, Duo approval steps, existing job IDs, and data locations.

[The collection guide](docs/DATA_COLLECTION.md) downloads pinned GSM8K and UltraChat-200k training/test files and prepares 384 training prompts: 192 from each source, with a fixed seed and order. It records source revisions and checksums, excludes blank/duplicate/held-out prompts, and keeps dataset answers out of model inputs.

```bash
uv run --script scripts/collect_prompt_data.py
python3 scripts/collect_prompt_data.py --verify-only
```

The files live in gitignored `data/prompt_corpus/`. This stage uses CPU only and produces no Qwen responses, DFlash contexts, or model weights. See [tour 6](.tours/06-prompt-data.tour) for the data flow and the collection guide for Nexus/SLURM instructions.

For larger editor experiments, [the training-data guide](docs/TRAINING_DATA.md) explains dataset sizing, storage, and optional private Hugging Face uploads. The new script defaults to 50,000 training prompts plus 1,024 fixed validation prompts, exported as compressed Parquet shards. It can also use the full eligible pool with `--all`. Existing source files are reused; the original pilot remains intact.

```bash
uv run --script scripts/prepare_training_data.py build --offline
uv run --script scripts/prepare_training_data.py verify
```

The default export already exists on the original Mac worktree. Verify it there, or choose a new `--output-dir`; omit `--offline` on a fresh machine. These files contain source prompts; generated responses are written separately.

[The Qwen response collector](docs/QWEN_RESPONSES.md) performs that next GPU stage: pinned Qwen3-8B, full BF16 weights, thinking disabled, greedy decoding, saved token IDs, and resumable per-worker journals. It accepts the original pilot or the larger Parquet export. DFlash features and editor targets remain a later collection stage.

[The compute and storage plan](docs/DATA_PIPELINE_PLAN.md) explains external file storage, optional Convex metadata, GitHub Actions access requirements, and upload/recovery failure cases. The read-only `scripts/summarize_qwen_responses.py` reports saved-response counts without modifying live journals or double-counting pilot prompt IDs across runs.

## Read and inspect without model downloads

```bash
git clone https://github.com/Frankstuff/Speculative-decoding-project.git
cd Speculative-decoding-project
python3 -m dflash.cli --help
python3 -m dflash.cli generate --help
python3 -m dflash.cli benchmark --help
python3 scripts/validate_tours.py
```

These commands use the standard library. Open this folder in VS Code, install **CodeTour** (`vsls-contrib.codetour`), and run **CodeTour: Start Tour → 1 - Find your way around**. See [the guide](docs/CODE_TOURS.md) for navigation and learning exercises.

## Install for inference

Use a supported Python version for the pinned packages; the package CI uses Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[local]'
```

The `local` extra selects PyTorch/Transformers on Linux and MLX on Apple Silicon. The Transformers CLI selects a CUDA device. For an existing compatible inference server, use `python -m pip install -e .` for the lighter client installation. Installing the package does not itself download model checkpoints; running a local generation command can.

See the preserved [upstream usage guide](docs/UPSTREAM_README.md) for supported model pairs and generation/benchmark examples. Install this checkout with `-e` when modifying it; `pip install dflash` alone installs the published upstream package.

## Repository and source provenance

All five files under `dflash/` match the upstream snapshot byte for byte. Repository-specific additions are the prompt-data collector and its checks, learning materials, tour validator, editor recommendation, and project documentation. Package maintainer/URLs and ignore rules are adapted for this repository; the inherited PyPI publishing workflow is omitted. The original package build-check workflow remains.

The upstream [MIT license and copyright notice](LICENSE) are retained. [Repository history and everyday Git use](docs/REPOSITORY.md) explains the clean history, the local backup of the earlier experiment, and direct pushes without PRs. For the original project documentation and citations, see [UPSTREAM_README.md](docs/UPSTREAM_README.md).
