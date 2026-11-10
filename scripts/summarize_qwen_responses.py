"""Read Qwen journals without changing them; count completed records, not log lines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def snapshot_records(path):
    """Bound the read at the opening size; leave a partial final record untouched."""
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        remaining = size
        while remaining:
            line = stream.readline(remaining)
            if not line:
                raise ValueError(f"Journal shrank during inspection; retry: {path}")
            remaining -= len(line)
            if not line.endswith(b"\n"):
                yield None, len(line)
                break
            record = json.loads(line)
            checksum = record.pop("record_sha256")
            if digest(record) != checksum:
                raise ValueError(f"Response checksum mismatch: {path}")
            yield record, len(line)


def summarize_run(root):
    plan = json.loads((root / "run.json").read_text())
    if plan["stage"] != "qwen_target_responses" or plan["schema_version"] != 1:
        raise ValueError(f"Unsupported run: {root}")
    total, shards = plan["input_rows"], plan["num_shards"]
    if type(total) is not int or type(shards) is not int or total <= 0 or shards <= 0:
        raise ValueError(f"Invalid run dimensions: {root}")
    expected_names = {f"shard-{index:05d}.jsonl" for index in range(shards)}
    if any(path.name not in expected_names for path in root.glob("shard-*.jsonl")):
        raise ValueError(f"Unexpected journal shard: {root}")
    run_hash = digest(plan)
    seen_ids, seen_indices, generated_ids = set(), set(), set()
    totals, splits, per_shard = Counter(), {}, []
    for index in range(shards):
        path = root / f"shard-{index:05d}.jsonl"
        counts = Counter()
        assigned = len(range(index, total, shards))
        if path.exists():
            for record, byte_count in snapshot_records(path):
                counts["journal_bytes"] += byte_count
                if record is None:
                    counts["partial_tail_bytes"] += byte_count
                    continue
                identifier, position = record["id"], record["prompt_index"]
                if (not isinstance(identifier, str) or type(position) is not int
                        or not 0 <= position < total or position % shards != index
                        or identifier in seen_ids or position in seen_indices
                        or record["run_sha256"] != run_hash):
                    raise ValueError(f"Duplicate or inconsistent response identity: {path}")
                split = record["dataset_split"]
                if split not in ("train", "validation"):
                    raise ValueError(f"Invalid dataset split: {path}")
                seen_ids.add(identifier)
                seen_indices.add(position)
                split_counts = splits.setdefault(split, Counter())
                counts["processed"] += 1
                split_counts["processed"] += 1
                if record["status"] == "generated":
                    reason, tokens = record["finish_reason"], record["response_token_ids"]
                    if reason not in ("eos", "length") or not tokens or record["output_tokens"] != len(tokens):
                        raise ValueError(f"Invalid generated response: {path}")
                    generated_ids.add(identifier)
                    for counter in (counts, split_counts):
                        counter["responses"] += 1
                        counter[reason] += 1
                        counter["output_tokens"] += len(tokens)
                elif record["status"] == "skipped_input_too_long":
                    counts["skipped_input_too_long"] += 1
                    split_counts["skipped_input_too_long"] += 1
                else:
                    raise ValueError(f"Invalid response status: {path}")
        totals.update(counts)
        per_shard.append({
            "shard_index": index, "present": path.exists(), "assigned": assigned,
            "processed": counts["processed"], "remaining": assigned - counts["processed"],
            "responses": counts["responses"], "eos": counts["eos"], "length": counts["length"],
            "skipped_input_too_long": counts["skipped_input_too_long"],
            "journal_bytes": counts["journal_bytes"], "partial_tail_bytes": counts["partial_tail_bytes"],
        })
    summary = {
        "directory": str(root), "run_sha256": run_hash, "generation": plan["generation"],
        "input_prompts": total, "num_shards": shards,
        **{key: totals[key] for key in (
            "processed", "responses", "eos", "length", "skipped_input_too_long", "output_tokens",
            "journal_bytes", "partial_tail_bytes",
        )},
        "remaining": total - totals["processed"],
        "complete_shards": sum(row["remaining"] == 0 and not row["partial_tail_bytes"] for row in per_shard),
        "splits": splits, "shards": per_shard,
    }
    return summary, generated_ids


def summarize_runs(roots):
    roots = [root.resolve() for root in roots]
    if len(set(roots)) != len(roots):
        raise ValueError("The same output directory was supplied more than once")
    started = datetime.now(timezone.utc).isoformat()
    runs, unique_ids = [], set()
    for root in roots:
        summary, identifiers = summarize_run(root)
        runs.append(summary)
        unique_ids.update(identifiers)
    responses = sum(run["responses"] for run in runs)
    return {
        "snapshot_started_at": started, "snapshot_finished_at": datetime.now(timezone.utc).isoformat(),
        "note": "Read-only per-file snapshots; live writers may advance. Partial tails are excluded. "
                "Record checksums and run/shard identities checked; original inputs and model outputs "
                "are not independently verified. Unique counts use source prompt IDs across runs.",
        "responses_across_runs": responses, "unique_prompt_ids_with_responses": len(unique_ids),
        "repeated_prompt_responses_across_runs": responses - len(unique_ids), "runs": runs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Qwen output directories containing run.json")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(summarize_runs(args.runs), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Cannot summarize Qwen responses: {exc}\n")


if __name__ == "__main__":
    main()
