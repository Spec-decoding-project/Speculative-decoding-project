from copy import deepcopy
import unittest

from one_block_fit.test_fit import fixture
from sixteen_block_fit.data import all_rows, select


class NativeAudit(unittest.TestCase):
    def examples(self):
        inputs, greedy, native = fixture()
        for f in inputs['fixtures']:
            f['source'] = 'test'
        return inputs, greedy, native

    def test_removes_old_four_block_cap_without_weakening_acceptance(self):
        a, b, c = self.examples()
        rows, excluded = all_rows(a, b, c)
        self.assertEqual(len(rows), 9)
        self.assertFalse(excluded)
        c['blocks'][0]['num_sampled'] = 1
        with self.assertRaisesRegex(ValueError, 'acceptance'):
            all_rows(a, b, c)

    def test_rejects_corrupted_prefix_duplicate_and_family(self):
        for fault in ('prefix', 'duplicate', 'family'):
            a, b, c = self.examples()
            if fault == 'prefix':
                c['blocks'][0]['positions'][0] += 1
            elif fault == 'duplicate':
                c['blocks'].append(deepcopy(c['blocks'][0]))
            else:
                a['fixtures'][1]['family_id'] = a['fixtures'][0]['family_id']
            with self.assertRaises(ValueError):
                all_rows(a, b, c)


def selection_fixture():
    inputs, rows, development = {'fixtures': []}, [], []
    for i in range(12):
        split = 'train' if i < 4 else 'validation'
        inputs['fixtures'].append(dict(id=str(i), split=split))
        for a in range(8 if i < 4 else 4):
            row = dict(prompt_id=str(i), family_id=str(i), split=split, anchor=a*20,
                       kind='repairable')
            rows.append(row)
            if i >= 4:
                development.append(row)
    rows.append(dict(prompt_id='4', family_id='4', split='validation', anchor=100, kind='preservation'))
    return inputs, rows, {'held_out': development}


class FixedSelection(unittest.TestCase):
    def test_selects_sixteen_distinct_repairs_and_keeps_controls_excluded(self):
        a, b, c = selection_fixture()
        selected = select(a, b, c)
        self.assertEqual(len(selected['training']), 16)
        self.assertEqual(selected['development'], c['held_out'])
        self.assertEqual(len(selected['preservation']), 1)
        self.assertTrue(all(r['split'] == 'train' and r['kind'] == 'repairable' for r in selected['training']))
        self.assertEqual(selected['training_unique_editable_positions'], 224)
        self.assertEqual(selected, select(a, list(reversed(b)), c))

    def test_insufficient_repairs_and_split_leakage_fail(self):
        a, b, c = selection_fixture()
        with self.assertRaisesRegex(ValueError, 'Fewer than four'):
            select(a, [r for r in b if r['prompt_id'] != '0' or r['anchor'] < 60], c)
        a, b, c = selection_fixture()
        b[-1]['family_id'] = '0'
        with self.assertRaisesRegex(ValueError, 'overlap'):
            select(a, b, c)


if __name__ == '__main__':
    unittest.main()
