"""Offline checks for response provenance, interruption recovery, and shard assignment."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts import generate_qwen_responses as generation


class FakeBackend:
    calls = []
    runtime = {"backend": "test-fixture"}

    def __init__(self, model_path, config):
        pass

    def generate(self, prompt):
        self.calls.append(prompt["id"])
        return {
            "status": "generated", "response": "Fixture answer", "response_token_ids": [7, 9],
            "prompt_token_ids": [3, 4], "input_tokens": 2, "output_tokens": 2,
            "finish_reason": "eos", "generation_seconds": 1.0,
        }


class ResponseCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "responses"
        self.prompts = [
            {"id": f"fixture:{i}", "source": {"row_index": i},
             "messages": [{"role": "user", "content": f"Question {i}"}],
             "dataset_split": "train" if i < 4 else "validation"}
            for i in range(6)
        ]
        self.config = {"model_id": "fixture", "do_sample": False, "enable_thinking": False}
        FakeBackend.calls = []

    def collect(self, **kwargs):
        args = dict(output=self.output, prompts=self.prompts, config=self.config,
                    num_shards=1, shard_index=0, max_prompts=None, model_path=None,
                    backend_factory=FakeBackend)
        args.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return generation.collect(**args)

    def test_resume_skips_completed_records_and_preserves_splits(self):
        first = self.collect(max_prompts=2)
        self.assertEqual(first["remaining"], 4)
        second = self.collect()
        self.assertEqual(second["remaining"], 0)
        self.assertEqual(second["statuses"], {"generated": 6})
        self.assertEqual(FakeBackend.calls, [row["id"] for row in self.prompts])
        self.collect(verify_only=True)
        self.collect()
        self.assertEqual(len(FakeBackend.calls), 6)
        records = [json.loads(line) for line in (self.output / "shard-00000.jsonl").read_text().splitlines()]
        self.assertEqual([row["dataset_split"] for row in records], [row["dataset_split"] for row in self.prompts])

    def test_shards_cover_input_exactly_once(self):
        for index in range(4):
            self.collect(num_shards=4, shard_index=index)
        self.assertCountEqual(FakeBackend.calls, [row["id"] for row in self.prompts])
        self.assertEqual(len(set(FakeBackend.calls)), 6)

    def test_changed_inputs_or_configuration_cannot_resume(self):
        self.collect(max_prompts=1)
        before = (self.output / "shard-00000.jsonl").read_bytes()
        for kwargs in ({"config": dict(self.config, do_sample=True)}, {"prompts": self.prompts[::-1]}, {"num_shards": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "Run settings"):
                self.collect(**kwargs)
        self.assertEqual((self.output / "shard-00000.jsonl").read_bytes(), before)

    def test_only_incomplete_tail_is_repaired(self):
        self.collect(max_prompts=1)
        path = self.output / "shard-00000.jsonl"
        with path.open("ab") as stream:
            stream.write(b'{"interrupted":')
        with self.assertRaisesRegex(ValueError, "Incomplete final record"):
            self.collect(verify_only=True)
        self.collect()
        self.assertEqual(len(FakeBackend.calls), 6)
        content = path.read_text().replace("Fixture answer", "Tampered answer", 1)
        path.write_text(content)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.collect()

    def test_model_failure_leaves_earlier_response_durable(self):
        class FailingBackend(FakeBackend):
            def generate(self, prompt):
                if prompt["id"] == "fixture:1":
                    raise RuntimeError("simulated GPU failure")
                return super().generate(prompt)
        with self.assertRaisesRegex(RuntimeError, "GPU failure"):
            self.collect(backend_factory=FailingBackend)
        summary = self.collect()
        self.assertEqual(summary["processed"], 6)
        self.assertEqual(FakeBackend.calls.count("fixture:0"), 1)

    def test_runtime_change_or_missing_run_fails(self):
        with self.assertRaisesRegex(ValueError, "No existing response run"):
            self.collect(verify_only=True)
        self.collect(max_prompts=1)
        class ChangedBackend(FakeBackend):
            runtime = {"backend": "different-library-version"}
        with self.assertRaisesRegex(ValueError, "Runtime differs"):
            self.collect(backend_factory=ChangedBackend)


if __name__ == "__main__":
    unittest.main()
