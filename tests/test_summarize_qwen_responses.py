"""Progress inspection must not repair writers' files or inflate response counts."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts import generate_qwen_responses as generation
from scripts import summarize_qwen_responses as progress


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prompts = [dict(id=f"fixture:{i}") for i in range(4)]

    def make_run(self, name="first", shards=2):
        root = self.root / name
        plan = generation.open_plan(root, self.prompts, {"model_id": "fixture"}, shards)
        return root, plan

    def append(self, run, index, status="generated", reason="eos"):
        root, plan = run
        record = dict(id=f"fixture:{index}", prompt_index=index,
                      run_sha256=generation.digest(plan), dataset_split="train" if index < 3 else "validation",
                      status=status, finish_reason=reason, response_token_ids=[7, 9], output_tokens=2)
        path = root / f"shard-{index % plan['num_shards']:05d}.jsonl"
        with path.open("ab") as stream:
            generation.append_record(stream, record)
        return path

    def test_counts_eos_caps_skips_and_missing_work_separately(self):
        run = self.make_run()
        self.append(run, 0)
        self.append(run, 1, reason="length")
        self.append(run, 3, status="skipped_input_too_long")
        summary = progress.summarize_runs([run[0]])["runs"][0]
        self.assertEqual([summary[k] for k in ("processed", "responses", "eos", "length", "remaining")], [3, 2, 1, 1, 1])
        self.assertEqual(summary["skipped_input_too_long"], 1)
        self.assertEqual(summary["complete_shards"], 1)
        self.assertEqual(summary["splits"]["validation"]["skipped_input_too_long"], 1)

    def test_partial_tail_is_reported_without_modifying_live_journal(self):
        run = self.make_run()
        path = self.append(run, 0)
        with path.open("ab") as stream:
            stream.write(b'{"partial":')
        before = path.read_bytes()
        summary = progress.summarize_runs([run[0]])["runs"][0]
        self.assertEqual(summary["responses"], 1)
        self.assertEqual(summary["partial_tail_bytes"], len(b'{"partial":'))
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(summary["shards"][1]["present"])

    def test_duplicate_pilot_prompts_are_not_new_unique_data(self):
        first, second = self.make_run(), self.make_run("second")
        self.append(first, 0)
        self.append(second, 0)
        self.append(second, 1)
        summary = progress.summarize_runs([first[0], second[0]])
        self.assertEqual(summary["responses_across_runs"], 3)
        self.assertEqual(summary["unique_prompt_ids_with_responses"], 2)
        self.assertEqual(summary["repeated_prompt_responses_across_runs"], 1)

    def test_snapshot_does_not_chase_new_appends(self):
        run = self.make_run(shards=1)
        path = self.append(run, 0)
        records = progress.snapshot_records(path)
        self.assertEqual(next(records)[0]["prompt_index"], 0)
        self.append(run, 1)
        self.assertEqual(list(records), [])
        self.assertEqual(progress.summarize_runs([run[0]])["responses_across_runs"], 2)

    def test_corruption_and_duplicate_records_fail_instead_of_overcounting(self):
        run = self.make_run()
        path = self.append(run, 0)
        original = path.read_bytes()
        path.write_bytes(original.replace(b'"output_tokens": 2', b'"output_tokens": 3'))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            progress.summarize_runs([run[0]])
        path.write_bytes(original + original)
        with self.assertRaisesRegex(ValueError, "inconsistent response"):
            progress.summarize_runs([run[0]])

    def test_valid_checksum_cannot_hide_wrong_shard_or_run(self):
        run = self.make_run()
        path = self.append(run, 0)
        record = json.loads(path.read_text())
        for change in ({"prompt_index": 1}, {"run_sha256": "different"}):
            altered = {**record, **change}
            altered.pop("record_sha256")
            path.write_bytes(b"")
            with path.open("ab") as stream:
                generation.append_record(stream, altered)
            with self.assertRaisesRegex(ValueError, "inconsistent response"):
                progress.summarize_runs([run[0]])


if __name__ == "__main__":
    unittest.main()
