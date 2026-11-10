"""Exercise real Parquet exports and mock only the network-facing Hub client."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import collect_prompt_data as collector
from scripts import prepare_training_data as dataset


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.output = self.root / "export"
        self.sources = {}
        for name in ("gsm8k", "ultrachat_200k"):
            source = {
                "repo_id": f"fixture/{name}", "config": "main", "revision": "a" * 40,
                "prompt_column": "question", "train_split": "train", "test_split": "test", "files": [],
            }
            raw = self.cache / "raw" / name
            raw.mkdir(parents=True)
            for split, questions in (
                ("train", [f"{name} question {i}" for i in range(300)] + ["Held out", " ", "Shared"]),
                ("test", ["Held out"]),
            ):
                path = raw / f"{split}.parquet"
                table = pa.Table.from_pylist([{"question": value, "answer": "SECRET ANSWER"} for value in questions])
                pq.write_table(table, path)
                source["files"].append({
                    "path": path.name, "split": split, "rows": len(questions),
                    "bytes": path.stat().st_size, "sha256": collector.sha256_file(path),
                })
            card = raw / "README.md"
            card.write_text("---\nlicense: mit\n---\nFixture source card\n", encoding="utf-8")
            source["files"].append({"path": card.name, "bytes": card.stat().st_size, "sha256": collector.sha256_file(card)})
            self.sources[name] = source
        self.lock = self.root / "sources.json"
        self.lock.write_text(json.dumps(self.sources), encoding="utf-8")
        patcher = patch.object(collector, "SOURCE_LOCK", self.lock)
        patcher.start()
        self.addCleanup(patcher.stop)

    def build(self, output=None, train_count=200):
        return dataset.build_dataset(
            self.cache, output or self.output, dict.fromkeys(self.sources, train_count),
            validation_count=8, seed=42, shard_rows=37, offline=True,
        )

    def rows(self, output, split):
        return [row for path in sorted((output / "data").glob(f"{split}-*.parquet")) for row in pq.read_table(path).to_pylist()]

    def test_expansion_keeps_validation_fixed_and_pilot_in_training(self):
        manifest = self.build()
        bigger = self.root / "bigger"
        self.build(bigger, train_count=250)
        training = self.rows(self.output, "train")
        validation = self.rows(self.output, "validation")
        ids = lambda rows: {row["id"] for row in rows}
        self.assertEqual(validation, self.rows(bigger, "validation"))
        self.assertLess(ids(training), ids(self.rows(bigger, "train")))
        self.assertFalse(ids(training) & ids(validation))
        pilot = self.root / "pilot"
        pilot.mkdir()
        (pilot / "raw").symlink_to(self.cache / "raw", target_is_directory=True)
        collector.prepare(pilot, self.sources, dict.fromkeys(self.sources, 192), 42)
        pilot_rows = [json.loads(line) for line in (pilot / "prompts.jsonl").read_text().splitlines()]
        self.assertLessEqual(ids(pilot_rows), ids(training))
        self.assertNotIn("SECRET ANSWER", json.dumps(training + validation))
        self.assertNotIn("Held out", [row["messages"][0]["content"] for row in training + validation])
        self.assertEqual(manifest["sampling"]["stats"]["ultrachat_200k"]["duplicate"], 1)
        self.assertEqual(manifest["next_stage_plan"]["collected_draft_contexts"], 0)
        self.assertEqual(dataset.verify_dataset(self.output), manifest)

    def test_all_uses_unique_pool_and_export_is_portable(self):
        manifest = self.build(train_count=None)
        self.assertEqual(manifest["counts"], {"train": {"gsm8k": 293, "ultrachat_200k": 292}, "validation": dict.fromkeys(self.sources, 8)})
        self.cache.rename(self.root / "moved-cache")
        self.assertEqual(dataset.verify_dataset(self.output), manifest)

    def test_repeated_build_is_byte_identical_and_existing_output_is_preserved(self):
        self.build()
        second = self.root / "second"
        self.build(second)
        self.assertEqual((second / "manifest.json").read_bytes(), (self.output / "manifest.json").read_bytes())
        before = collector.sha256_file(self.output / "manifest.json")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.build()
        self.assertEqual(collector.sha256_file(self.output / "manifest.json"), before)

    def test_insufficient_count_and_invalid_args_do_not_publish_partial_output(self):
        with self.assertRaisesRegex(ValueError, "only 293 available"):
            self.build(train_count=1000)
        self.assertFalse(self.output.exists())
        with patch.object(collector, "download_sources") as download:
            for args in (["build", "--gsm8k-count", "0"], ["build", "--validation-per-source", "0"], ["build", "--all", "--gsm8k-count", "200"]):
                with self.subTest(args=args), self.assertRaises(SystemExit) as failure:
                    dataset.main(args + ["--output-dir", str(self.output)])
                self.assertEqual(failure.exception.code, 1)
            download.assert_not_called()

    def test_checksum_failure_blocks_export_verification(self):
        manifest = self.build()
        path = self.output / manifest["files"][0]["path"]
        path.write_bytes(path.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            dataset.verify_dataset(self.output)

    def test_duplicate_across_splits_is_rejected_even_with_updated_file_hash(self):
        manifest = self.build()
        training = self.rows(self.output, "train")
        entry = next(item for item in manifest["files"] if item.get("split") == "validation")
        path = self.output / entry["path"]
        validation = pq.read_table(path).to_pylist()
        validation[0] = training[0]
        pq.write_table(pa.Table.from_pylist(validation, schema=dataset.parquet_schema()), path)
        entry.update(bytes=path.stat().st_size, sha256=collector.sha256_file(path))
        (self.output / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "duplicate prompt across"):
            dataset.verify_dataset(self.output)

    def test_explicit_schema_preserves_ultrachat_id_after_math_records(self):
        rows = [
            {"id": "math", "source": {"dataset": "math", "row_index": 0}, "messages": [{"role": "user", "content": "Math"}]},
            {"id": "chat", "source": {"dataset": "chat", "row_index": 1, "prompt_id": "original-chat-id"}, "messages": [{"role": "user", "content": "Chat"}]},
        ]
        dataset.write_shards(self.root, "train", rows, 10)
        result = self.rows(self.root, "train")
        self.assertIsNone(result[0]["source"]["prompt_id"])
        self.assertEqual(result[1]["source"]["prompt_id"], "original-chat-id")

    def test_upload_uses_allowlist_private_repo_and_parent_commit(self):
        manifest = self.build()
        (self.output / "unrelated-secret.txt").write_text("not for upload")
        api = MagicMock()
        api.repo_info.return_value = types.SimpleNamespace(private=True, sha="parent-commit")
        api.list_repo_files.return_value = [".gitattributes"]
        api.upload_folder.return_value = types.SimpleNamespace(oid="new-commit")
        module = types.SimpleNamespace(HfApi=lambda: api)
        with patch.dict("sys.modules", huggingface_hub=module):
            dataset.upload_dataset(self.output, "user/fixture")
            api.create_repo.assert_called_once_with("user/fixture", repo_type="dataset", private=True, exist_ok=True)
            kwargs = api.upload_folder.call_args.kwargs
            self.assertEqual(kwargs["parent_commit"], "parent-commit")
            self.assertEqual(set(kwargs["allow_patterns"]), {"manifest.json", *(file["path"] for file in manifest["files"])})
            self.assertNotIn("unrelated-secret.txt", kwargs["allow_patterns"])
            for private, files, error in (
                (False, [], "Destination is public"),
                (True, ["old-data.parquet"], "unrelated or stale"),
            ):
                api.upload_folder.reset_mock()
                api.repo_info.return_value.private = private
                api.list_repo_files.return_value = files
                with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                    dataset.upload_dataset(self.output, "user/fixture")
                api.upload_folder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
