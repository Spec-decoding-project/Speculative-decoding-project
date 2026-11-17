"""CPU checks for the diagnostic's token alignment and enforced target-call cap."""
import unittest
from unittest.mock import patch

import torch
import exact_session as diagnostic

diagnostic.torch = torch


class ExactDiagnosticTests(unittest.TestCase):
    def test_divergent_row_and_margin_are_not_shifted(self):
        labels = [10, 11, 12946, 12]
        logits = torch.zeros(1, 4, 20000, dtype=torch.bfloat16)
        for i, token in enumerate(labels):
            logits[0, i, token] = 37.5
        logits[0, 2, 16984] = 37.75
        result = diagnostic.logits_summary(logits, labels, 2)
        self.assertEqual(result['mismatches'], [dict(position=2, expected=12946, actual=16984)])
        self.assertEqual(result['focus']['signed_margin'], -0.25)
        logits[0, 2, 16984] = 37.5
        result = diagnostic.logits_summary(logits, labels, 2)
        self.assertEqual(result['matches'], 4)
        self.assertEqual(result['focus']['maximum_tie_ids'], [12946, 16984])
        with self.assertRaisesRegex(ValueError, 'Misaligned'):
            diagnostic.logits_summary(logits, labels[:-1], 2)

    def test_control_alternate_is_a_different_token_even_at_a_tie(self):
        logits = torch.zeros(1, 1000)
        logits[0, 374] = logits[0, 646] = 31.25
        result = diagnostic.logits_summary(logits, [374], 0)['focus']
        self.assertEqual(result['actual'], 374)
        self.assertEqual(result['alternate'], 646)
        self.assertEqual(result['signed_margin'], 0.0)
        self.assertEqual(result['maximum_tie_ids'], [374, 646])

    def test_forward_cap_stops_before_the_next_model_execution(self):
        harness = object.__new__(diagnostic.Harness)
        harness.calls, harness.limit = 299, 300
        harness.seconds, harness.max_seconds = 0.0, 600
        with patch.object(torch.cuda, 'synchronize'):
            harness._before(None, (), {})
            self.assertEqual(harness.calls, 300)
            with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
                harness._before(None, (), {})
            harness.calls, harness.seconds = 0, 600
            with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
                harness._before(None, (), {})


if __name__ == '__main__':
    unittest.main()
