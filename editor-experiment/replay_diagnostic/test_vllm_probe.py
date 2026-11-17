"""Guard the diagnostic against treating different contexts as equal logits."""
import unittest
from vllm_probe import compare, first_difference


class ComparisonTests(unittest.TestCase):
    def trial(self, token, prefix, logits):
        return dict(tokens={'fixture': [token]}, trace=[dict(prefix_sha256=prefix,
                    logits_sha256=logits)])

    def test_equal_tokens_do_not_hide_logit_difference(self):
        result = compare(self.trial(1, 'same', 'a'), self.trial(1, 'same', 'b'))
        self.assertIsNone(result['token_differences']['fixture'])
        self.assertEqual(result['matching_prefix_rows'], 1)
        self.assertEqual(len(result['unequal_full_logit_rows']), 1)

    def test_different_prefixes_are_not_logit_comparisons(self):
        result = compare(self.trial(1, 'one', 'a'), self.trial(2, 'two', 'a'))
        self.assertEqual(result['matching_prefix_rows'], 0)
        self.assertEqual(result['token_differences']['fixture']['position'], 0)

    def test_shorter_output_is_failure(self):
        self.assertEqual(first_difference([1, 2], [1]),
                         dict(position=1, length_mismatch=True))

    def test_recomputed_prefix_cannot_hide_an_earlier_difference(self):
        left = self.trial(1, 'same', 'different')
        left['trace'].extend(self.trial(1, 'same', 'matching')['trace'])
        result = compare(left, self.trial(1, 'same', 'matching'))
        self.assertEqual(len(result['unequal_full_logit_rows']), 1)


if __name__ == '__main__':
    unittest.main()
