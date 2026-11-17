import unittest

from audit_vllm_probe import audit_trial, token_hash
from vllm_contract import require_exact


class IntegrityTests(unittest.TestCase):
    def fixture(self):
        return {'fixture': dict(prompt=[10,20])}

    def trial(self):
        return dict(tokens={'fixture':[30]}, details={'fixture':dict(request_id='0')},
            trace=[dict(request_id='0-internal-uuid',prefix_sha256=token_hash([10,20]),
                        position=1,argmax=30,vocabulary_size=151936)])

    def test_internal_request_suffix_is_resolved_from_authenticated_prompt(self):
        result=audit_trial(self.fixture(),self.trial())
        self.assertEqual(result['generated_positions_verified'],1)
        self.assertEqual(result['request_mapping']['fixture']['internal'],'0-internal-uuid')

    def test_wrong_context_cannot_authenticate_generated_token(self):
        trial=self.trial();trial['trace'][0]['prefix_sha256']=token_hash([10,21])
        with self.assertRaises(ValueError):audit_trial(self.fixture(),trial)

    def test_wrong_argmax_cannot_authenticate_generated_token(self):
        trial=self.trial();trial['trace'][0]['argmax']=31
        with self.assertRaises(ValueError):audit_trial(self.fixture(),trial)

    def test_matching_tokens_with_missing_or_different_logits_fail_gate(self):
        good=dict(token_differences={'fixture':None},matching_prefix_rows=1,
                  unequal_full_logit_rows=[])
        require_exact(good,1)
        for bad in [dict(good,matching_prefix_rows=0),dict(good,unequal_full_logit_rows=[{}])]:
            with self.assertRaises(ValueError):require_exact(bad,1)


if __name__ == '__main__':unittest.main()
