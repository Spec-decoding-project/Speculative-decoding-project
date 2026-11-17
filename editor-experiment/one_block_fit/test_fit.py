from copy import deepcopy
import unittest

from one_block_fit.collect import make_bank
from one_block_fit.train import passed


def fixture():
    inputs, greedy, native = {'fixtures':[]}, {'tokens':{}}, {'blocks':[]}
    for i in range(9):
        name = f'prompt-{i}'
        prefix, labels = [100+i], list(range(20,36))
        draft = list(labels)
        draft[5] = 99
        inputs['fixtures'].append(dict(id=name,prompt=prefix,family_id=name,
            split='train' if i==0 else 'validation',record_sha256='fixture'))
        greedy['tokens'][name] = labels
        native['blocks'].append(dict(prefix=prefix,anchor=1,positions=list(range(1,17)),
            draft=draft,native_draft_count=15,num_sampled=5,num_rejected=11))
    return inputs,greedy,native


class CaptureIntegrity(unittest.TestCase):
    def test_native_acceptance_and_prompt_separation(self):
        result=make_bank(*fixture())
        self.assertEqual(len(result['training']),1)
        self.assertEqual(len(result['held_out']),8)
        self.assertEqual(result['training'][0]['native_accepted'],4)

    def test_wrong_context_is_rejected(self):
        a,b,c=fixture()
        c['blocks'][0]['prefix'].append(12)
        with self.assertRaisesRegex(ValueError,'prefix'):
            make_bank(a,b,c)

    def test_acceptance_disagreement_is_rejected(self):
        a,b,c=fixture()
        c['blocks'][0]['num_sampled']=4
        with self.assertRaisesRegex(ValueError,'acceptance'):
            make_bank(a,b,c)

    def test_duplicate_and_family_leakage_are_rejected(self):
        a,b,c=fixture()
        c['blocks'].append(deepcopy(c['blocks'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            make_bank(a,b,c)
        a,b,c=fixture()
        a['fixtures'][1]['family_id']='prompt-0'
        with self.assertRaisesRegex(ValueError,'overlap'):
            make_bank(a,b,c)

    def test_wrong_first_proposals_stay_in_evaluation(self):
        a,b,c=fixture()
        c['blocks'][1]['draft'][1]=99
        c['blocks'][1].update(num_sampled=1,num_rejected=15)
        result=make_bank(a,b,c)
        self.assertEqual(result['held_out'][0]['kind'],'wrong_first')

    def test_low_teacher_forced_loss_alone_cannot_pass(self):
        metrics=dict(teacher_forced_ce=0.01,free_running_exact_suffix=0,
                     free_running_token_accuracy=0.9)
        self.assertFalse(passed(metrics))
        metrics.update(free_running_exact_suffix=1.0,free_running_token_accuracy=1.0)
        self.assertTrue(passed(metrics))
        metrics['teacher_forced_ce']=0.05
        self.assertFalse(passed(metrics))


if __name__=='__main__':
    unittest.main()
