# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["huggingface-hub==1.23.0", "pyarrow==21.0.0"]
# ///
"""Download pinned source datasets and prepare prompts, without running models."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path


SOURCE_LOCK = Path(__file__).resolve().parents[1] / "configs/prompt_sources.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_key(text):
    """Compare Unicode/whitespace-normalized prompts without changing model input."""
    if not isinstance(text, str):
        raise ValueError("Expected a string prompt")
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else None


def extract_prompt(row, source):
    if source["prompt_column"] == "messages":
        messages = row["messages"]
        if (
            not isinstance(messages, list) or not messages
            or not isinstance(messages[0], dict) or messages[0].get("role") != "user"
        ):
            raise ValueError("UltraChat conversation must begin with a user message")
        # The separate `prompt` column can differ from the truecased conversation.
        return messages[0]["content"]
    return row[source["prompt_column"]]


def verify_raw(output, sources):
    for name, source in sources.items():
        for entry in source["files"]:
            path = output / "raw" / name / entry["path"]
            if not path.is_file():
                raise ValueError(f"Missing source file: {path}")
            if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Source checksum mismatch: {path}")


def download_sources(output, sources):
    from huggingface_hub import snapshot_download

    for name, source in sources.items():
        print(f"Fetching {source['repo_id']} at {source['revision']}", flush=True)
        snapshot_download(
            repo_id=source["repo_id"],
            repo_type="dataset",
            revision=source["revision"],
            allow_patterns=[entry["path"] for entry in source["files"]],
            local_dir=output / "raw" / name,
            max_workers=2,
        )


def iter_rows(output, name, source, split):
    import pyarrow.parquet as pq

    columns = [source["prompt_column"]]
    if "id_column" in source:
        columns.append(source["id_column"])
    row_index = 0
    for entry in source["files"]:
        if entry.get("split") != split:
            continue
        parquet = pq.ParquetFile(output / "raw" / name / entry["path"])
        if parquet.metadata.num_rows != entry["rows"]:
            raise ValueError(f"Unexpected row count in {name}/{entry['path']}")
        for batch in parquet.iter_batches(batch_size=1024, columns=columns):
            for row in batch.to_pylist():
                yield row_index, row
                row_index += 1


def select_prompts(rows, name, source, count, seed, heldout, seen):
    stats = Counter(rows_read=0, blank=0, heldout_overlap=0, duplicate=0, eligible=0)

    def candidates():
        for row_index, row in rows:
            stats["rows_read"] += 1
            text = extract_prompt(row, source)
            key = prompt_key(text)
            if key is None:
                stats["blank"] += 1
                continue
            if key in heldout:
                stats["heldout_overlap"] += 1
                continue
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            stats["eligible"] += 1
            provenance = {
                "dataset": source["repo_id"], "config": source["config"],
                "revision": source["revision"], "split": source["train_split"],
                "row_index": row_index,
            }
            if "id_column" in source:
                provenance["prompt_id"] = row[source["id_column"]]
            yield {
                "id": f"{name}:{source['train_split']}:{row_index}",
                "source": provenance,
                "messages": [{"role": "user", "content": text}],
            }

    selected = (
        list(candidates()) if count is None
        else heapq.nsmallest(count, candidates(), key=lambda row: rank(row, seed, "sample"))
    )
    if count is not None and len(selected) != count:
        raise ValueError(f"{name}: requested {count} unique prompts, only {len(selected)} eligible")
    return selected, dict(stats)


def rank(row, seed, purpose):
    return hashlib.sha256(f"{seed}:{purpose}:{row['id']}".encode("utf-8")).digest()


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(output, sources, counts, seed):
    if any(count <= 0 for count in counts.values()):
        raise ValueError("Prompt counts must be positive")
    verify_raw(output, sources)
    heldout = set()
    for name, source in sources.items():
        for _, row in iter_rows(output, name, source, source["test_split"]):
            key = prompt_key(extract_prompt(row, source))
            if key is not None:
                heldout.add(key)

    prompts, stats, seen = [], {}, set()
    for name, source in sources.items():
        selected, stats[name] = select_prompts(
            iter_rows(output, name, source, source["train_split"]),
            name, source, counts[name], seed, heldout, seen,
        )
        prompts.extend(selected)
    prompts.sort(key=lambda row: rank(row, seed, "order"))
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompts).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "stage": "source_prompts_only",
        "sources": sources,
        "sampling": {
            "seed": seed, "counts": counts,
            "selection": "lowest SHA256(seed:sample:id), then SHA256(seed:order:id)",
            "deduplication": "NFC Unicode and collapsed whitespace; original input retained",
            "source_order": list(sources),
            "heldout_unique_prompts": len(heldout),
            "stats": stats,
        },
        "prompts": {
            "path": "prompts.jsonl", "rows": len(prompts), "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
        "next_stage_plan": {
            "target_model": "Qwen/Qwen3-8B", "enable_thinking": False,
            "do_sample": False, "contexts_per_response": 32,
            "planned_contexts": len(prompts) * 32,
            "generated_responses": 0, "collected_draft_contexts": 0,
        },
    }
    atomic_write(output / "prompts.jsonl", content)
    atomic_write(output / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return manifest


def verify_collection(output):
    """Check transferred bytes and prompt structure without network or ML libraries."""
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["stage"] != "source_prompts_only":
        raise ValueError("Unsupported collection manifest")
    verify_raw(output, manifest["sources"])
    metadata = manifest["prompts"]
    path = output / metadata["path"]
    if path.stat().st_size != metadata["bytes"] or sha256_file(path) != metadata["sha256"]:
        raise ValueError(f"Prompt checksum mismatch: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    counts, ids, keys = Counter(), set(), set()
    sources = {source["repo_id"]: (name, source) for name, source in manifest["sources"].items()}
    for row in records:
        source = row["source"]
        name, expected = sources[source["dataset"]]
        for field in ("config", "revision"):
            if source[field] != expected[field]:
                raise ValueError(f"Wrong source {field}: {row['id']}")
        if source["split"] != expected["train_split"]:
            raise ValueError(f"Non-training split: {row['id']}")
        total = sum(entry.get("rows", 0) for entry in expected["files"] if entry.get("split") == source["split"])
        index = source["row_index"]
        if type(index) is not int or not 0 <= index < total:
            raise ValueError(f"Invalid source row index: {row['id']}")
        if row["id"] != f"{name}:{source['split']}:{index}":
            raise ValueError(f"Inconsistent prompt ID: {row['id']}")
        messages = row["messages"]
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError(f"Expected one user message: {row['id']}")
        key = prompt_key(messages[0]["content"])
        if key is None or key in keys or row["id"] in ids:
            raise ValueError(f"Blank or duplicate prompt: {row['id']}")
        ids.add(row["id"])
        keys.add(key)
        counts[name] += 1
    if len(records) != metadata["rows"] or dict(counts) != manifest["sampling"]["counts"]:
        raise ValueError("Prompt counts do not match manifest")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/prompt_corpus"))
    parser.add_argument("--gsm8k-count", type=int, default=192)
    parser.add_argument("--ultrachat-count", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true", help="Prepare from checksum-verified local source files")
    parser.add_argument("--verify-only", action="store_true", help="Verify an existing collection; no network or third-party imports")
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            manifest = verify_collection(args.output_dir)
        else:
            counts = {"gsm8k": args.gsm8k_count, "ultrachat_200k": args.ultrachat_count}
            if any(count <= 0 for count in counts.values()):
                raise ValueError("Prompt counts must be positive")
            sources = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
            if not args.offline:
                download_sources(args.output_dir, sources)
            manifest = prepare(args.output_dir, sources, counts, args.seed)
        print(f"Verified {manifest['prompts']['rows']} prompts: {args.output_dir / 'prompts.jsonl'}")
        print(f"SHA256: {manifest['prompts']['sha256']}")
        print("Source data only; Qwen responses and DFlash contexts have not been generated.")
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(1, f"Data collection failed: {exc}\n")


if __name__ == "__main__":
    main()
