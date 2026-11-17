"""CPU checks for intervention scope/restoration, not GPU numerical equivalence."""
import unittest

import torch

from rowwise import single_row_linears


class Cache:
    def get_seq_length(self):
        return 3


class ObservedLinear(torch.nn.Linear):
    def __init__(self):
        super().__init__(4, 4)
        self.seen = []

    def forward(self, x):
        self.seen.append((tuple(x.shape), tuple(x.stride())))
        return super().forward(x)


class Target(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = ObservedLinear()
        self.lm_head = ObservedLinear()

    def forward(self, ids, past_key_values=None):
        x = torch.nn.functional.one_hot(ids % 4, 4).float()
        return self.lm_head(torch.tanh(self.body(x)))


class SingleRowTests(unittest.TestCase):
    def test_prefill_unchanged_block_rows_and_head_are_single_token(self):
        target = Target()
        weights = {k: v.clone() for k, v in target.state_dict().items()}
        with single_row_linears(target, block_size=4) as state:
            target(torch.tensor([[0, 1, 2]]))
            actual = target(torch.tensor([[1, 2, 3, 0]]), past_key_values=Cache())
        self.assertEqual(state['linear_calls'], 8)
        self.assertEqual(dict(state['split_modules']), {'body': 1, 'lm_head': 1})
        for module in [target.body, target.lm_head]:
            self.assertEqual([x[0] for x in module.seen], [(1, 3, 4)] + [(1, 1, 4)] * 4)
            self.assertTrue(all(x[1] == (4, 4, 1) for x in module.seen[1:]))
            self.assertNotIn('forward', module.__dict__)
        expected = target(torch.tensor([[1, 2, 3, 0]]), past_key_values=Cache())
        torch.testing.assert_close(actual, expected)
        self.assertTrue(all(torch.equal(v, target.state_dict()[k]) for k, v in weights.items()))

    def test_invalid_layout_restores_methods_and_hooks(self):
        target = Target()
        with self.assertRaisesRegex(ValueError, 'cached native'):
            with single_row_linears(target, block_size=4):
                target(torch.tensor([[0, 1, 2]]))
                target(torch.tensor([[0, 1]]), past_key_values=Cache())
        self.assertFalse(target._forward_pre_hooks)
        self.assertFalse(target._forward_hooks)
        self.assertNotIn('forward', target.body.__dict__)
        self.assertNotIn('forward', target.lm_head.__dict__)
        target(torch.tensor([[0, 1]]))


if __name__ == '__main__':
    unittest.main()
