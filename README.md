# DFlash — an independent starting point

This repository starts from unmodified DFlash Python source from [`z-lab/dflash` at `07ebd93`](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a), with fresh Git history under Frankstuff's repository. It now includes an optional token-only autoregressive secondary drafter. The earlier conditional-redrafting experiment is not included.

**Start with [the code-tour guide](docs/CODE_TOURS.md).** Five JSON tours in [`.tours/`](.tours) teach the project layout, decoding loop, model internals, backends, and benchmark interpretation. [Tour prompts and research](docs/TOUR_PROMPTS.md) contains reusable prompts for generating and learning from future tours.

## What the code does

DFlash uses a draft model to propose several tokens in parallel. A larger target model verifies the proposals, retains a valid prefix, and supplies a correction or bonus token. The aim is faster generation while preserving the target model's decoding behavior.

| File | Role |
| --- | --- |
| [`dflash/cli.py`](dflash/cli.py) | `generate` and `benchmark` commands; chooses Transformers, MLX, or an HTTP server |
| [`dflash/model.py`](dflash/model.py) | PyTorch generation, draft attention, DFlash and DFlash2 model classes |
| [`dflash/model_mlx.py`](dflash/model_mlx.py) | Apple Silicon implementation, streaming, and cache/state rollback |
| [`dflash/benchmark.py`](dflash/benchmark.py) | Model loading, prompt formatting, datasets, server requests, and benchmark summaries |
| [`dflash/secondary.py`](dflash/secondary.py) | Pretrained secondary model loading and token-only suffix regeneration |
| [`dflash/validation.py`](dflash/validation.py) | Independent target-only greedy output check |
| [`dflash/__init__.py`](dflash/__init__.py) | Lazy public imports so basic CLI help does not load model libraries |

The upstream package contains both original DFlash and DFlash2. The tours first explain original DFlash with greedy decoding, then identify the other paths. Without `--secondary-model`, the original drafting path remains the default. There is no training pipeline.

## Try the secondary drafter

After installing the local dependencies on a CUDA machine, run:

```bash
dflash generate transformers \
  --model Qwen/Qwen3-4B --draft z-lab/Qwen3-4B-DFlash-b16 \
  --secondary-model Qwen/Qwen3-0.6B \
  --block-size 16 --max-new-tokens 128 \
  --temperature 0 --reasoning off --check-greedy \
  "Explain speculative decoding briefly."
```

The secondary model preserves the anchor and first DFlash proposal, then regenerates the remaining suffix from token IDs. The target verifies the resulting block. Read the [experiment guide](docs/SECONDARY_DRAFTER.md) for weight initialization, a three-way benchmark, implementation locations, CPU tests, and limitations. This is a correctness-first experiment; speedup must be measured.

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

The initial import (`cb44a76`) matches all five upstream `dflash/` files byte for byte. Later changes add the optional secondary drafter, output checks, CPU tests, and learning materials. Package maintainer/URLs and ignore rules are adapted for this repository; the inherited PyPI publishing workflow is omitted. CI checks package builds, code tours, and CPU inference correctness.

The upstream [MIT license and copyright notice](LICENSE) are retained. [Repository history and everyday Git use](docs/REPOSITORY.md) explains the clean history, the local backup of the earlier experiment, and direct pushes without PRs. For the original project documentation and citations, see [UPSTREAM_README.md](docs/UPSTREAM_README.md).

## Editor experiment (Shrinav and Frank, October 2026)

Our experiment on whether a small editor can reproduce a full second drafting pass's corrections cheaply, on Qwen3-4B. Start with [`editor-experiment/README.md`](editor-experiment/README.md) for the architecture, training, results, and plan, and [`results/SUMMARY.md`](results/SUMMARY.md) for the one-page project summary. Measured on the dev set: a full second pass adds 10.2% accepted positions at about 30% more round cost; the best 3M-parameter editor adds 3.7% at 5% to 6% more cost and does not break even. The run write-ups behind those numbers are in `results/`.
