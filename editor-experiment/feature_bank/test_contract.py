import copy
import unittest

from native_replay import LEGACY, NATIVE, validate_contract


class ContractTests(unittest.TestCase):
    def test_legacy_is_identified_and_contract_mismatches_rejected(self):
        self.assertEqual(validate_contract({}, {}), LEGACY)
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            validate_contract({'replay_contract': NATIVE}, {})
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            validate_contract({'replay_contract': 'skip-mismatches'}, {'replay_contract': 'skip-mismatches'})

    def test_native_requires_all_real_fixture_regressions(self):
        regressions = [dict(id=i, anchor=a, target_features_equal=True,
                           draft_normalized_states_equal=True, actual_proposal_equal=True,
                           saved_labels_match_native_verification=True)
                       for i,a in [('gsm8k:train:3821',38),('ultrachat_200k:train_sft:164610',19),
                                   ('ultrachat_200k:train_sft:61759',22)]]
        audit = dict(replay_contract=NATIVE, native_regressions=regressions)
        manifest = dict(replay_contract=NATIVE)
        self.assertEqual(validate_contract(audit, manifest), NATIVE)
        for invalid in [[], regressions[:-1]]:
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                validate_contract(dict(audit,native_regressions=invalid),manifest)
        for key in ('target_features_equal','draft_normalized_states_equal',
                    'actual_proposal_equal','saved_labels_match_native_verification'):
            bad=copy.deepcopy(audit)
            bad['native_regressions'][2][key]=False
            with self.assertRaisesRegex(ValueError, 'failed'):
                validate_contract(bad,manifest)


if __name__ == '__main__':
    unittest.main()
