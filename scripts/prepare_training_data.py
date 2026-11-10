# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["huggingface-hub==1.23.0", "pyarrow==21.0.0"]
# ///
"""Build, verify, or upload source prompts for the Qwen3/DFlash editor experiment.

This CPU stage produces no target responses, draft features, or training labels.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

if __package__:
    from . import collect_prompt_data as collector
else:
    import collect_prompt_data as collector


PILOT_PER_SOURCE = 192
DEFAULT_OUTPUT = Path("data/editor_prompts_50k")


def split_source(records, train_count, validation_count, seed):
    """Fix validation before choosing nested training subsets of the full pool."""
    ordered = sorted(records, key=lambda row: collector.rank(row, seed, "sample"))
    # Keep the original 192-per-source pilot in the training pool, even after scaling.
    validation_pool = ordered[PILOT_PER_SOURCE:]
    if validation_count > len(validation_pool):
        raise ValueError("Not enough unique prompts for validation after reserving the pilot")
    validation = sorted(
        validation_pool, key=lambda row: collector.rank(row, seed, "validation")
    )[:validation_count]
    validation_ids = {row["id"] for row in validation}
    training = [row for row in ordered if row["id"] not in validation_ids]
    if train_count is not None:
        if train_count > len(training):
            raise ValueError(
                f"Requested {train_count} training prompts, only {len(training)} available "
                "after reserving validation; prompts are never repeated"
            )
        training = training[:train_count]
    return training, validation


def parquet_schema():
    import pyarrow as pa

    return pa.schema([
        ("id", pa.string()),
        ("source", pa.struct([
            ("dataset", pa.string()), ("config", pa.string()),
            ("revision", pa.string()), ("split", pa.string()),
            ("row_index", pa.int64()), ("prompt_id", pa.string()),
        ])),
        ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ])


def artifact(root, relative, **metadata):
    path = root / relative
    return {
        "path": relative, "bytes": path.stat().st_size,
        "sha256": collector.sha256_file(path), **metadata,
    }


def write_shards(root, split, records, shard_rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = []
    (root / "data").mkdir(exist_ok=True)
    for number, start in enumerate(range(0, len(records), shard_rows)):
        batch = records[start:start + shard_rows]
        relative = f"data/{split}-{number:05d}.parquet"
        table = pa.Table.from_pylist(batch, schema=parquet_schema())
        pq.write_table(table, root / relative, compression="zstd", row_group_size=1024)
        files.append(artifact(root, relative, kind="parquet", split=split, rows=len(batch)))
    return files


def dataset_card(manifest):
    lines = [
        "---", "language: en", "license: mit", "task_categories:",
        "- text-generation", "configs:", "- config_name: default", "  data_files:",
        "  - split: train", "    path: data/train-*.parquet",
        "  - split: validation", "    path: data/validation-*.parquet", "---", "",
        "# Source prompts for the Qwen3/DFlash editor experiment", "",
        "User-only inputs, not a completed supervised training dataset. No Qwen responses,",
        "draft computations, alternative continuations, or editor targets have been generated.", "",
        "| Split | GSM8K | UltraChat | Total |", "| --- | ---: | ---: | ---: |",
    ]
    for split, counts in manifest["counts"].items():
        lines.append(f"| {split} | {counts['gsm8k']} | {counts['ultrachat_200k']} | {sum(counts.values())} |")
    lines.extend([
        "", f"Seed: {manifest['sampling']['seed']}. All records originate in upstream training splits.",
        "Upstream test prompts are excluded by NFC/whitespace-normalized matching. Validation",
        "is reserved from the full eligible training pool before sampling; it stays fixed when",
        "only training counts change. The original 192-per-source pilot is reserved for training",
        "at the same seed. This filter does not detect paraphrases or semantic near-duplicates.", "",
        "Split all downstream responses, contexts, prefix variants, and tensors by this prompt ID.",
        "Do not shuffle individual contexts between training and validation. Reuse the exact",
        "ordered inputs across comparison methods. A prompt ID's source row_index is zero-based",
        "across the ordered upstream shards in manifest.json; source.prompt_id is null for GSM8K.", "",
        "Parquet shard order and row order define the example order. manifest.json records all",
        "file hashes and source revisions. Original assistant answers are excluded. Raw upstream",
        "Parquet is available from the pinned sources and is not duplicated in this export.", "",
        "## Sources and attribution", "",
    ])
    for name, source in manifest["sources"].items():
        url = f"https://huggingface.co/datasets/{source['repo_id']}/tree/{source['revision']}"
        lines.append(f"- [{source['repo_id']}]({url}), split `{source['train_split']}`; "
                     f"preserved [dataset card](provenance/{name}-README.md).")
    lines.extend([
        "", "Both source cards declare MIT licensing. Preserve their notices and citations when",
        "redistributing this derived prompt selection. Consult the preserved cards for dataset",
        "limitations and authorship. This dataset makes no claim about target-model accuracy.", "",
        "## Next stage", "",
        "Generate Qwen/Qwen3-8B responses with thinking disabled and greedy decoding, then collect",
        "DFlash contexts and shared prefix choices for teacher/editor comparisons. Pin the model",
        "and tokenizer revisions, measure a small GPU pilot, and record actual successful examples",
        "separately from planned counts. Never pad short responses by duplicating contexts.", "",
    ])
    return "\n".join(lines)


def build_dataset(source_dir, output, counts, validation_count, seed, shard_rows, offline):
    if output.exists():
        raise ValueError(f"Output already exists: {output}; verify it or choose a new directory")
    if validation_count <= 0 or shard_rows <= 0 or any(
        count is not None and count <= 0 for count in counts.values()
    ):
        raise ValueError("Counts and shard size must be positive")
    sources = json.loads(collector.SOURCE_LOCK.read_text(encoding="utf-8"))
    if set(counts) != set(sources):
        raise ValueError("Training counts must specify every locked source")
    if not offline:
        collector.download_sources(source_dir, sources)
    collector.verify_raw(source_dir, sources)
    heldout = set()
    for name, source in sources.items():
        for _, row in collector.iter_rows(source_dir, name, source, source["test_split"]):
            key = collector.prompt_key(collector.extract_prompt(row, source))
            if key is not None:
                heldout.add(key)
    splits = {"train": [], "validation": []}
    split_counts = {split: {} for split in splits}
    stats, seen = {}, set()
    for name, source in sources.items():
        print(f"Scanning {name} training pool", flush=True)
        pool, stats[name] = collector.select_prompts(
            collector.iter_rows(source_dir, name, source, source["train_split"]),
            name, source, None, seed, heldout, seen,
        )
        training, validation = split_source(pool, counts[name], validation_count, seed)
        for split, rows in (("train", training), ("validation", validation)):
            splits[split].extend(rows)
            split_counts[split][name] = len(rows)
    for records in splits.values():
        records.sort(key=lambda row: collector.rank(row, seed, "order"))
    manifest = {
        "schema_version": 1, "kind": "editor_prompt_dataset", "stage": "source_prompts_only",
        "sources": sources, "counts": split_counts,
        "sampling": {
            "seed": seed, "requested_train_counts": counts,
            "validation_per_source": validation_count, "pilot_per_source": PILOT_PER_SOURCE,
            "selection": "reserve sample-ranked pilot; validation-rank remaining full pool; sample-rank training",
            "order": "SHA256(seed:order:id)", "source_order": list(sources),
            "deduplication": "NFC Unicode and collapsed whitespace; original input retained",
            "heldout_unique_prompts": len(heldout), "stats": stats,
        },
        "next_stage_plan": {
            "target_model": "Qwen/Qwen3-8B", "enable_thinking": False, "do_sample": False,
            "contexts_per_response": 32,
            "planned_training_contexts": len(splits["train"]) * 32,
            "planned_validation_contexts": len(splits["validation"]) * 32,
            "generated_responses": 0, "collected_draft_contexts": 0,
        },
        "files": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as directory:
        staging = Path(directory)
        for split, records in splits.items():
            manifest["files"].extend(write_shards(staging, split, records, shard_rows))
        (staging / "provenance").mkdir()
        for name in sources:
            relative = f"provenance/{name}-README.md"
            shutil.copyfile(source_dir / "raw" / name / "README.md", staging / relative)
            manifest["files"].append(artifact(staging, relative, kind="source_card"))
        (staging / "README.md").write_text(dataset_card(manifest), encoding="utf-8")
        manifest["files"].append(artifact(staging, "README.md", kind="dataset_card"))
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        verify_dataset(staging)
        if output.exists():
            raise ValueError(f"Output appeared during collection: {output}")
        staging.rename(output)
    return manifest


def verified_files(output, manifest):
    paths = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        path = output / relative
        if relative.is_absolute() or ".." in relative.parts or str(relative) in paths:
            raise ValueError(f"Invalid or duplicate artifact path: {relative}")
        if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
            raise ValueError(f"Artifact escapes dataset directory: {relative}")
        paths.add(str(relative))
        if not path.is_file() or path.stat().st_size != entry["bytes"] or collector.sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Artifact missing or checksum mismatch: {relative}")
    return paths


def verify_dataset(output):
    import pyarrow.parquet as pq

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["schema_version"], manifest["kind"], manifest["stage"]) != (1, "editor_prompt_dataset", "source_prompts_only"):
        raise ValueError("Unsupported prompt dataset manifest")
    verified_files(output, manifest)
    sources = {source["repo_id"]: (name, source) for name, source in manifest["sources"].items()}
    counts = {"train": Counter(), "validation": Counter()}
    ids, keys = set(), set()
    previous_rank = {split: None for split in counts}
    for entry in manifest["files"]:
        if entry["kind"] != "parquet":
            continue
        split = entry["split"]
        if split not in counts:
            raise ValueError(f"Unexpected output split: {split}")
        parquet = pq.ParquetFile(output / entry["path"])
        if parquet.metadata.num_rows != entry["rows"] or not parquet.schema_arrow.equals(parquet_schema()):
            raise ValueError(f"Unexpected Parquet rows or schema: {entry['path']}")
        for batch in parquet.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                source = row["source"]
                name, expected = sources[source["dataset"]]
                if any(source[field] != expected[field] for field in ("revision", "config")) or source["split"] != expected["train_split"]:
                    raise ValueError(f"Unexpected source provenance: {row['id']}")
                total = sum(item.get("rows", 0) for item in expected["files"] if item.get("split") == expected["train_split"])
                index = source["row_index"]
                if type(index) is not int or not 0 <= index < total or row["id"] != f"{name}:{source['split']}:{index}":
                    raise ValueError(f"Invalid source row ID: {row['id']}")
                messages = row["messages"]
                if not isinstance(messages, list) or len(messages) != 1 or messages[0]["role"] != "user":
                    raise ValueError(f"Expected one user message: {row['id']}")
                key = collector.prompt_key(messages[0]["content"])
                if key is None or key in keys or row["id"] in ids:
                    raise ValueError(f"Blank or duplicate prompt across dataset splits: {row['id']}")
                order = collector.rank(row, manifest["sampling"]["seed"], "order")
                if previous_rank[split] is not None and order < previous_rank[split]:
                    raise ValueError(f"Unexpected example order in {split}")
                previous_rank[split] = order
                keys.add(key)
                ids.add(row["id"])
                counts[split][name] += 1
    if {split: dict(count) for split, count in counts.items()} != manifest["counts"]:
        raise ValueError("Split counts do not match the manifest")
    return manifest


def upload_dataset(output, repo_id):
    """Upload only verified exported artifacts to a dedicated private dataset repo."""
    from huggingface_hub import HfApi

    manifest = verify_dataset(output)
    allowed = ["manifest.json"] + [entry["path"] for entry in manifest["files"]]
    api = HfApi()  # Uses HF_TOKEN or `hf auth login`; never accepts a token in command arguments.
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo_id, repo_type="dataset")
    if not info.private:
        raise ValueError("Destination is public; choose a dedicated private dataset repository")
    existing = set(api.list_repo_files(repo_id, repo_type="dataset", revision=info.sha))
    if existing - set(allowed) - {".gitattributes"}:
        raise ValueError("Destination contains unrelated or stale files; choose a new dataset repository")
    commit = api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=output,
        allow_patterns=allowed, parent_commit=info.sha,
        commit_message="Upload verified source prompts with fixed training and validation splits",
    )
    print(f"Uploaded private dataset: https://huggingface.co/datasets/{repo_id}/tree/{commit.oid}")
    return commit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Download/reuse sources and prepare Parquet prompt shards")
    build.add_argument("--source-dir", type=Path, default=Path("data/prompt_corpus"))
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--gsm8k-count", type=int, help="Training prompts, excluding validation (default: 6000)")
    build.add_argument("--ultrachat-count", type=int, help="Training prompts, excluding validation (default: 44000)")
    build.add_argument("--all", action="store_true", help="Use every eligible training prompt after reserving validation")
    build.add_argument("--validation-per-source", type=int, default=512)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--shard-rows", type=int, default=10000)
    build.add_argument("--offline", action="store_true", help="Use checksum-verified local source files only")
    verify = subparsers.add_parser("verify", help="Verify a portable export without source files or network")
    verify.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    upload = subparsers.add_parser("upload", help="Verify and upload the export to a private Hugging Face dataset")
    upload.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    upload.add_argument("--repo-id", required=True, help="Your Hugging Face namespace/dataset-name")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            if args.all and (args.gsm8k_count is not None or args.ultrachat_count is not None):
                raise ValueError("Use either --all or explicit training counts")
            counts = {
                "gsm8k": None if args.all else (6000 if args.gsm8k_count is None else args.gsm8k_count),
                "ultrachat_200k": None if args.all else (44000 if args.ultrachat_count is None else args.ultrachat_count),
            }
            manifest = build_dataset(args.source_dir, args.output_dir, counts, args.validation_per_source, args.seed, args.shard_rows, args.offline)
        elif args.command == "verify":
            manifest = verify_dataset(args.output_dir)
        else:
            upload_dataset(args.output_dir, args.repo_id)
            return
        for split, counts in manifest["counts"].items():
            print(f"Verified {split}: {sum(counts.values()):,} prompts ({counts})")
        print(f"Dataset: {args.output_dir}")
        print("Source prompts only. Generated responses: 0. Collected DFlash contexts: 0.")
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(1, f"Prompt dataset failed: {exc}\n")


if __name__ == "__main__":
    main()
