"""Collect resumable Qwen3-8B greedy responses, with thinking disabled, on CUDA."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

if __package__:
    from . import collect_prompt_data as collector
else:
    import collect_prompt_data as collector


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen_generation.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@contextmanager
def locked(path, nonblocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        try:
            fcntl.flock(stream, flags)
        except BlockingIOError as exc:
            raise ValueError(f"Another worker is writing this shard: {path}") from exc
        yield


def load_prompts(path, split):
    if path.is_dir():
        if __package__:
            from .prepare_training_data import verify_dataset
        else:
            from prepare_training_data import verify_dataset
        import pyarrow.parquet as pq

        manifest = verify_dataset(path)
        rows = []
        for entry in manifest["files"]:
            if entry["kind"] == "parquet" and (split == "all" or entry["split"] == split):
                for row in pq.read_table(path / entry["path"]).to_pylist():
                    rows.append(dict(row, dataset_split=entry["split"]))
    else:
        if split == "validation":
            raise ValueError("The original JSONL pilot contains training prompts only")
        manifest = collector.verify_collection(path.parent)
        if path.name != manifest["prompts"]["path"]:
            raise ValueError("JSONL input is not the collection's verified prompt file")
        rows = [dict(json.loads(line), dataset_split="train") for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("No prompts selected")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate prompt IDs")
    return rows


def open_plan(output, prompts, config, num_shards):
    plan = {
        "schema_version": 1, "stage": "qwen_target_responses",
        "input_rows": len(prompts), "input_sha256": digest(prompts),
        "generation": config, "num_shards": num_shards,
        "generator_sha256": collector.sha256_file(Path(__file__)),
    }
    with locked(output / "plan.lock"):
        path = output / "run.json"
        if path.exists():
            if json.loads(path.read_text()) != plan:
                raise ValueError("Run settings, inputs, shard count, or generator changed; use a new output directory")
        else:
            collector.atomic_write(path, (json.dumps(plan, indent=2) + "\n").encode())
    return plan


def read_journal(path, expected, run_sha256, repair_tail):
    """Only an unfinished final line can be discarded; completed corruption fails."""
    completed = {}
    if not path.exists():
        return completed
    with path.open("r+b" if repair_tail else "rb") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                if not repair_tail:
                    raise ValueError(f"Incomplete final record: {path}; resume the worker to repair it")
                stream.truncate(start)
                stream.flush()
                os.fsync(stream.fileno())
                break
            record = json.loads(line)
            checksum = record.pop("record_sha256")
            if digest(record) != checksum:
                raise ValueError(f"Response checksum mismatch: {path}")
            identifier = record["id"]
            if identifier not in expected or identifier in completed:
                raise ValueError(f"Unexpected or duplicate response ID: {identifier}")
            index, prompt = expected[identifier]
            if record["prompt_index"] != index or record["prompt_sha256"] != digest(prompt) or record["run_sha256"] != run_sha256:
                raise ValueError(f"Response provenance mismatch: {identifier}")
            if record["status"] not in ("generated", "skipped_input_too_long"):
                raise ValueError(f"Unexpected response status: {identifier}")
            if record["status"] == "generated":
                if record["finish_reason"] not in ("eos", "length") or not record["response_token_ids"]:
                    raise ValueError(f"Invalid generated response: {identifier}")
            completed[identifier] = record
    return completed


def append_record(stream, record):
    payload = dict(record, record_sha256=digest(record))
    stream.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
    stream.flush()
    os.fsync(stream.fileno())


class QwenBackend:
    def __init__(self, model_path, config):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("A CUDA GPU with BF16 support is required; no quantized/CPU fallback is used")
        if torch.cuda.get_device_properties(0).total_memory < 22 * 1024 ** 3:
            raise ValueError("This collector needs a GPU with at least 22 GiB VRAM for full BF16 weights")
        torch.manual_seed(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        self.torch = torch
        self.config = config
        if model_path:
            metadata = json.loads((model_path / "pinned_revision.json").read_text())
            if metadata != {"model_id": config["model_id"], "revision": config["revision"]}:
                raise ValueError("Local model snapshot does not match the pinned model revision")
        location = str(model_path) if model_path else config["model_id"]
        kwargs = {"revision": config["revision"], "local_files_only": bool(model_path), "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(location, **kwargs)
        if "enable_thinking" not in str(self.tokenizer.chat_template):
            raise ValueError("Tokenizer template does not support disabling thinking")
        self.model = AutoModelForCausalLM.from_pretrained(
            location, dtype=torch.bfloat16, attn_implementation=config["attention"],
            device_map={"": "cuda:0"}, **kwargs,
        ).eval()
        source_config = self.model.generation_config
        self.eos_ids = source_config.eos_token_id
        if isinstance(self.eos_ids, int):
            self.eos_ids = [self.eos_ids]
        self.generation_config = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=config["max_new_tokens"],
            bos_token_id=source_config.bos_token_id,
            eos_token_id=self.eos_ids, pad_token_id=source_config.pad_token_id,
            use_cache=True,
        )
        self.runtime = {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "dtype": str(self.model.dtype), "eos_token_ids": self.eos_ids,
            "chat_template_sha256": hashlib.sha256(str(self.tokenizer.chat_template).encode()).hexdigest(),
        }

    def generate(self, prompt):
        rendered = self.tokenizer.apply_chat_template(
            prompt["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        prompt_ids = inputs["input_ids"][0].tolist()
        base = {
            "prompt_token_ids": prompt_ids, "input_tokens": len(prompt_ids),
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        }
        if len(prompt_ids) > self.config["max_input_tokens"]:
            return dict(base, status="skipped_input_too_long", reason="Input exceeds configured token limit; no truncation applied")
        inputs = inputs.to("cuda:0")
        self.torch.cuda.synchronize()
        started = time.monotonic()
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, generation_config=self.generation_config)
        self.torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        ids = generated[0, len(prompt_ids):].tolist()
        return dict(
            base, status="generated", response_token_ids=ids,
            response=self.tokenizer.decode(ids, skip_special_tokens=True),
            output_tokens=len(ids), finish_reason="eos" if ids and ids[-1] in self.eos_ids else "length",
            generation_seconds=round(elapsed, 4),
        )


def collect(output, prompts, config, num_shards, shard_index, max_prompts, model_path, verify_only=False, backend_factory=QwenBackend):
    if verify_only and not (output / "run.json").is_file():
        raise ValueError("No existing response run to verify")
    plan = open_plan(output, prompts, config, num_shards)
    run_sha256 = digest(plan)
    expected = {row["id"]: (index, row) for index, row in enumerate(prompts) if index % num_shards == shard_index}
    shard = output / f"shard-{shard_index:05d}"
    with locked(shard.with_suffix(".lock"), nonblocking=True):
        journal = shard.with_suffix(".jsonl")
        completed = read_journal(journal, expected, run_sha256, repair_tail=not verify_only)
        pending = [(index, row) for identifier, (index, row) in expected.items() if identifier not in completed]
        if not verify_only and pending:
            backend = backend_factory(model_path, config)
            runtime_path = shard.with_suffix(".runtime.json")
            if completed and not runtime_path.exists():
                raise ValueError("Runtime record is missing for existing responses")
            if runtime_path.exists() and json.loads(runtime_path.read_text()) != backend.runtime:
                raise ValueError("Runtime differs from previous responses in this shard; use the original environment")
            collector.atomic_write(runtime_path, (json.dumps(backend.runtime, indent=2) + "\n").encode())
            with journal.open("ab") as stream:
                for index, row in pending[:max_prompts]:
                    record = {
                        "id": row["id"], "prompt_index": index, "dataset_split": row["dataset_split"],
                        "source": row["source"], "messages": row["messages"],
                        "prompt_sha256": digest(row), "run_sha256": run_sha256,
                        **backend.generate(row),
                    }
                    append_record(stream, record)
                    completed[row["id"]] = record
                    print(f"{len(completed)}/{len(expected)} {row['id']} {record['status']} "
                          f"tokens={record.get('output_tokens', 0)} finish={record.get('finish_reason', 'skipped')}", flush=True)
        statuses = Counter(record["status"] for record in completed.values())
        reasons = Counter(record.get("finish_reason", "skipped") for record in completed.values())
        summary = {
            "shard_index": shard_index, "assigned_prompts": len(expected), "processed": len(completed),
            "remaining": len(expected) - len(completed), "statuses": dict(statuses), "finish_reasons": dict(reasons),
            "output_tokens": sum(record.get("output_tokens", 0) for record in completed.values()),
            "generation_seconds": round(sum(record.get("generation_seconds", 0) for record in completed.values()), 4),
            "journal_sha256": collector.sha256_file(journal) if journal.exists() else None,
        }
        collector.atomic_write(shard.with_suffix(".summary.json"), (json.dumps(summary, indent=2) + "\n").encode())
        print(json.dumps(summary, indent=2), flush=True)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/prompt_corpus/prompts.jsonl"))
    parser.add_argument("--split", choices=("all", "train", "validation"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("data/qwen_responses_pilot"))
    parser.add_argument("--model-path", type=Path, help="Downloaded pinned snapshot with pinned_revision.json")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-prompts", type=int, help="Maximum NEW responses this invocation; rerun to resume")
    parser.add_argument("--verify-only", action="store_true", help="Verify this shard's saved responses without loading the model")
    args = parser.parse_args(argv)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards or (args.max_prompts is not None and args.max_prompts <= 0):
        parser.error("Invalid shard assignment or max-prompts")
    try:
        config = json.loads(CONFIG.read_text())
        if config["enable_thinking"] or config["do_sample"] or config["dtype"] != "bfloat16":
            raise ValueError("This collector requires non-thinking, greedy, BF16 generation")
        prompts = load_prompts(args.input, args.split)
        collect(args.output_dir, prompts, config, args.num_shards, args.shard_index,
                args.max_prompts, args.model_path, args.verify_only)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(1, f"Qwen response collection failed: {exc}\n")


if __name__ == "__main__":
    main()
