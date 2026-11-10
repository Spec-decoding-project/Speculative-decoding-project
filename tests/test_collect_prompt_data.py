"""Training-prompt integrity checks; all fixtures are synthetic and offline."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.collect_prompt_data import (
    extract_prompt, main, prepare, prompt_key, select_prompts, verify_collection, verify_raw,
)


SOURCE = {
    "repo_id": "fixture/math", "config": "main", "revision": "a" * 40,
    "prompt_column": "question", "train_split": "train", "test_split": "test",
    "files": [{"path": "fixture.parquet", "split": "train", "rows": 100}],
}


class PromptDataTests(unittest.TestCase):
    def test_answers_duplicates_and_heldout_questions_are_excluded(self):
        rows = list(enumerate([
            {"question": "Held out", "answer": "SECRET"},
            {"question": "Allowed\nquestion", "answer": "SECRET"},
            {"question": "Allowed question", "answer": "SECRET"},
            {"question": "  ", "answer": "SECRET"},
            {"question": "Another question", "answer": "SECRET"},
        ]))
        selected, stats = select_prompts(rows, "math", SOURCE, 2, 42, {prompt_key("Held out")}, set())
        self.assertEqual({r["source"]["row_index"] for r in selected}, {1, 4})
        self.assertNotIn("SECRET", json.dumps(selected))
        self.assertEqual(stats, {"rows_read": 5, "blank": 1, "heldout_overlap": 1, "duplicate": 1, "eligible": 2})
        self.assertIn("Allowed\nquestion", [r["messages"][0]["content"] for r in selected])

    def test_ultrachat_uses_only_initial_prompt(self):
        source = dict(SOURCE, repo_id="fixture/chat", prompt_column="messages", id_column="prompt_id", train_split="train_sft")
        rows = [(8, {"prompt": "first request", "prompt_id": "original-id", "messages": [
            {"role": "user", "content": "First request"},
            {"role": "assistant", "content": "SECRET ANSWER"},
            {"role": "user", "content": "Later request"},
        ]})]
        selected, _ = select_prompts(rows, "chat", source, 1, 42, set(), set())
        self.assertEqual(selected[0]["messages"], [{"role": "user", "content": "First request"}])
        self.assertEqual(selected[0]["source"]["prompt_id"], "original-id")
        self.assertEqual(selected[0]["source"]["split"], "train_sft")

    def test_ultrachat_rejects_missing_or_non_user_first_turn(self):
        source = dict(SOURCE, prompt_column="messages")
        for messages in ([], None, [{"role": "assistant", "content": "Answer"}]):
            with self.subTest(messages=messages), self.assertRaisesRegex(ValueError, "begin with a user"):
                extract_prompt({"messages": messages}, source)

    def test_seed_is_repeatable_and_samples_beyond_start_of_dataset(self):
        rows = [(i, {"question": f"Question {i}"}) for i in range(100)]
        def sample(seed):
            return select_prompts(rows, "math", SOURCE, 10, seed, set(), set())[0]
        self.assertEqual(sample(42), sample(42))
        self.assertNotEqual(sample(42), sample(43))
        self.assertTrue(any(r["source"]["row_index"] >= 50 for r in sample(42)))

    def test_insufficient_unique_data_fails_instead_of_repeating(self):
        with self.assertRaisesRegex(ValueError, "only 1 eligible"):
            select_prompts([(i, {"question": "Same"}) for i in range(3)], "math", SOURCE, 2, 42, set(), set())

    def test_cross_source_duplicates_and_unicode_are_excluded(self):
        seen = {prompt_key("Café question")}
        rows = [(0, {"question": "Cafe\u0301  question"}), (1, {"question": "Different"})]
        selected, stats = select_prompts(rows, "math", SOURCE, 1, 42, set(), seen)
        self.assertEqual(selected[0]["source"]["row_index"], 1)
        self.assertEqual(stats["duplicate"], 1)

    def test_missing_or_corrupt_raw_data_fails_checksum(self):
        content = b"original"
        source = dict(SOURCE, files=[{"path": "file", "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(ValueError, "Missing source"):
                verify_raw(output, {"math": source})
            path = output / "raw/math/file"
            path.parent.mkdir(parents=True)
            path.write_bytes(content)
            verify_raw(output, {"math": source})
            path.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_raw(output, {"math": source})

    def test_manifest_roundtrip_and_corrupt_prompt_detection(self):
        def rows(output, name, source, split):
            questions = ["Test question"] if split == "test" else ["Train A", "Test question", "Train B"]
            return iter((i, {"question": value}) for i, value in enumerate(questions))
        with tempfile.TemporaryDirectory() as directory, patch("scripts.collect_prompt_data.verify_raw"), patch("scripts.collect_prompt_data.iter_rows", side_effect=rows):
            output = Path(directory)
            expected = prepare(output, {"math": SOURCE}, {"math": 2}, 42)
            self.assertEqual(verify_collection(output), expected)
            self.assertEqual(expected["next_stage_plan"]["collected_draft_contexts"], 0)
            before = (output / "prompts.jsonl").read_bytes()
            prepare(output, {"math": SOURCE}, {"math": 2}, 42)
            self.assertEqual((output / "prompts.jsonl").read_bytes(), before)
            (output / "prompts.jsonl").write_bytes(before + b"\n")
            with self.assertRaisesRegex(ValueError, "Prompt checksum mismatch"):
                verify_collection(output)

    def test_invalid_count_does_not_download(self):
        with patch("scripts.collect_prompt_data.download_sources") as download:
            with self.assertRaises(SystemExit) as failure:
                main(["--gsm8k-count", "0"])
            self.assertEqual(failure.exception.code, 1)
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
