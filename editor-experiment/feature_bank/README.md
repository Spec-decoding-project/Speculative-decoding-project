# Bounded Qwen3-8B feature bank

This collector implements the first gate of the [compact-editor research map](https://github.com/Spec-decoding-project/Speculative-decoding-project/issues/4). It does not train an editor or benchmark generation. The historical 4B/teacher-dependent collectors remain separate.

The target is Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`; the matching drafter is `z-lab/Qwen3-8B-DFlash-b16` revision `9b41424b7109f9c5413454f481b09a82b85333f4`. Both weights and source files are fingerprinted before loading. `prepare.py` downloads one copy of the pinned draft files, without a second Hugging Face cache copy. Establish quota headroom first.

The loader executes the released, hash-verified `dflash.py` unchanged. It extracts only the three pure helper function definitions from the released `utils.py` to avoid importing unrelated dataset benchmark utilities. It uses the installed Transformers Qwen3 and SDPA implementation; the runtime versions are saved. This is a native **uncached first-round forward** audit, not a validation of the released generation loop under that Transformers version.

## Run

Local CPU-only integrity tests:

```sh
python3 -m unittest discover -s editor-experiment/feature_bank -p 'test_*.py' -v
```

On Nexus, deploy this directory under `research/editor-feature-bank/feature_bank/` and the unchanged `compact_second_pass_prototype.py` alongside that directory. Use the existing `.venv-qwen` environment. Run downloads and file verification in a CPU allocation. Submit from `/fs/classhomes/fpajak/speculative-decoding-project`:

```sh
sbatch research/editor-feature-bank/feature_bank/collect.sbatch \
  --output data/editor_feature_banks/smoke-UNIQUE \
  --repairs 1 --preservation 0 --seconds 300
```

Only after the smoke audit passes:

```sh
sbatch research/editor-feature-bank/feature_bank/collect.sbatch \
  --output data/editor_feature_banks/bank-UNIQUE
```

Every output directory must be new. Inspect `audit.json` and job exit status; an existing directory or partial bank is not success. A one-block smoke success is not the ticket's full-bank success. Before training, independently verify the manifest and tensor hashes and the full-bank outcome.

## Selection and schema

Only the overlapping pilot's training IDs are considered. All pilot/main journal checksums are checked; shared IDs must agree on exact prompts, response tokens and finish status. Exact normalized message families must be disjoint across train/validation. This does not establish semantic near-duplicate independence. Capped responses are excluded, even if they contain some full windows.

Candidates use full, nonoverlapping 16-token windows, at most 32 per response, and at most 1,024 committed context tokens. Source groups alternate, shortest prompts first within each group; all first anchors are visited before second anchors. Selection stops at 128 repairable and 16 preservation blocks, or after 2,700 capture seconds. The SLURM allocation is one RTX A5000, four CPUs, 32 GiB RAM, at most one hour including setup. These are diagnostic examples, not an unbiased evaluation distribution.

The anchor occupies slot 0; the original first proposal occupies slot 1; slots 2–15 are editable. A repairable block has a correct first proposal and at least one later error. A preservation block has all 15 proposals correct. Wrong-first cases are counted but not included in this initial fitting bank.

Each safetensors file contains:

- `normalized_states`: BF16 `[1,16,4096]`, the actual output of the native draft's final normalization.
- `first_guess_embedding`: BF16 `[1,4096]`, the frozen target embedding of the actual first proposal.
- `original_ids`, `labels`, `valid_mask`: `[1,16]`; all positions are valid in this full-block bank.
- `prefix_ids`: `[1,anchor]` exact committed context tokens, for replay.
- `anchor`: `[1]` absolute anchor position.

The manifest records prompt ID, split, source, anchor, source-record hash, classification and tensor-file hash/size. Full vocabulary logits are computed transiently and never cached. Future target tokens enter causal label verification only, never the target feature prefix or masked draft inputs. Label replay disagreement stops the run rather than changing labels.

The first fixture checks deterministic native forwarding, exact capture of the final normalized states, future-label perturbation invariance of inputs, identity editor readout at native 15-row shape, and FP32 direct/folded nonzero-residual readout agreement. Direct/folded tolerance is `atol=0.002, rtol=0.0002`, with zero argmax changes required. This numerical algebra check does not validate a trained BF16 editor. Later tickets own trained deployment precision, target verification, cache behavior and costs of replacement/union.

`audit.json` includes source/model fingerprints, software/hardware, frozen parameter ownership, classifications, selection/source counts, capture time, tensor bytes and GPU peaks. `failure.json` preserves the failing fixture and exception. Keep raw manifests, response tokens and tensors on the server; publish aggregate evidence and code links in the decision ticket.
