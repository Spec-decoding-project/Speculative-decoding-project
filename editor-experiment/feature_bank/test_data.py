import copy
import json
from pathlib import Path
import tempfile
import unittest

from data import audit_sources, block_inputs, candidates, digest, records


def row(identifier='a', **changes):
    result = dict(id=identifier, dataset_split='train', source='gsm8k',
                  status='generated', finish_reason='eos',
                  prompt_token_ids=[10, 11], response_token_ids=list(range(100, 164)),
                  messages=[dict(role='user', content=identifier)])
    result.update(changes)
    return result


def journal(directory, rows):
    directory.mkdir()
    run = {'test': True}
    (directory / 'run.json').write_text(json.dumps(run))
    with (directory / 'shard-00000.jsonl').open('w') as f:
        for r in rows:
            body = dict(r, run_sha256=digest(run))
            f.write(json.dumps(dict(body, record_sha256=digest(body))) + '\n')


class DataTests(unittest.TestCase):
    def test_future_labels_cannot_enter_inputs(self):
        r = row()
        prefix, noise, labels = block_inputs(r, 18, 999)
        modified = copy.deepcopy(r)
        modified['response_token_ids'][17:32] = [777] * 15
        p2, n2, l2 = block_inputs(modified, 18, 999)
        self.assertEqual((prefix, noise), (p2, n2))
        self.assertNotEqual(labels, l2)
        self.assertEqual(len(prefix), 18)
        self.assertEqual(noise, [116] + [999] * 15)

    def test_selection_budget_exclusions_and_round_robin(self):
        rows = [row('a'), row('b', source='ultrachat_200k'),
                row('c', finish_reason='length'), row('d', dataset_split='validation')]
        schedule, excluded = candidates(rows, anchors_per_prompt=2)
        self.assertEqual([(r['id'], a) for r, a in schedule], [('a', 2), ('b', 2), ('a', 18), ('b', 18)])
        self.assertEqual({e['reason'] for e in excluded}, {'output_capped', 'not_training'})
        with self.assertRaises(ValueError): candidates(rows, anchors_per_prompt=33)
        with self.assertRaises(ValueError): block_inputs(row(), 60, 999)

    def test_checksums_duplicates_and_partial_tail_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data'
            journal(path, [row()])
            self.assertEqual(len(list(records(path))), 1)
            file = path / 'shard-00000.jsonl'
            text = file.read_text()
            file.write_text(text + text)
            with self.assertRaisesRegex(ValueError, 'Duplicate'): list(records(path))
            file.write_text(text.rstrip())
            with self.assertRaisesRegex(ValueError, 'Partial'): list(records(path))
            file.write_text(text.replace('gsm8k', 'other'))
            with self.assertRaisesRegex(ValueError, 'checksum'): list(records(path))

    def test_overlap_is_not_new_diversity(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot, main = Path(tmp) / 'pilot', Path(tmp) / 'main'
            journal(pilot, [row()])
            journal(main, [row(), row('v', dataset_split='validation')])
            selected, audit = audit_sources(pilot, main)
            self.assertEqual(len(selected), 1)
            self.assertEqual(audit['pilot_overlap'], 1)
            self.assertEqual(audit['counts'], {'train': 1, 'validation': 1})

    def test_real_source_metadata_groups_by_dataset_not_row(self):
        rows = [row('chat1', source={'dataset': 'chat', 'row_index': 1}),
                row('chat2', source={'dataset': 'chat', 'row_index': 2}),
                row('math1', source={'dataset': 'math', 'row_index': 1}),
                row('math2', source={'dataset': 'math', 'row_index': 2})]
        schedule, _ = candidates(rows, anchors_per_prompt=1)
        self.assertEqual([r['id'] for r, _ in schedule], ['chat1', 'math1', 'chat2', 'math2'])

    def test_family_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot, main = Path(tmp) / 'pilot', Path(tmp) / 'main'
            journal(pilot, [row()])
            journal(main, [row(), row('v', dataset_split='validation', messages=[dict(role='user', content=' A ')])])
            with self.assertRaisesRegex(ValueError, 'family overlap'): audit_sources(pilot, main)


if __name__ == '__main__':
    unittest.main()
