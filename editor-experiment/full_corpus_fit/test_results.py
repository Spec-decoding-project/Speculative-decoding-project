from copy import deepcopy
import unittest

from full_corpus_fit.audit_results import audit_predictions


class SavedPredictionAudit(unittest.TestCase):
    def setUp(self):
        labels=list(range(16))
        original=[0,1,99]+labels[3:]
        self.examples=[dict(prompt_id='test-a',anchor=20,source='source-a',
                            kind='repairable',draft=original,labels=labels)]
        self.result=dict(blocks=1,teacher_forced_ce=.1,wrong_first=0,
                         free_running_exact_suffix=1.,free_running_token_accuracy=1.,
                         mean_original_progress=1.,mean_edited_progress=15.,mean_union_progress=15.,
                         repaired=1,damaged=0,rows=[dict(prompt_id='test-a',anchor=20,
                         original_tokens=original,edited_tokens=labels.copy(),label_tokens=labels,
                         kind='repairable',original_progress=1,edited_progress=15,union_progress=15,
                         repaired=True,damaged=False,wrong_first=False,exact_suffix=True)])

    def test_recounts_actual_tokens(self):
        result=audit_predictions(self.result,self.examples)
        self.assertEqual(result['editable_matches'],14)
        self.assertEqual(result['prompt_weighted_mean_delta'],14)

    def test_rejects_wrong_summary_and_per_row_metrics(self):
        for location in ('summary','row'):
            changed=deepcopy(self.result)
            if location=='summary': changed['mean_edited_progress']=14
            else: changed['rows'][0]['edited_progress']=14
            with self.assertRaises(ValueError):
                audit_predictions(changed,self.examples)

    def test_rejects_changed_labels_seed_and_population(self):
        for mutation in ('labels','seed','population','identity'):
            changed=deepcopy(self.result)
            if mutation=='labels': changed['rows'][0]['label_tokens'][5]=100
            elif mutation=='seed': changed['rows'][0]['edited_tokens'][1]=100
            elif mutation=='population': changed['rows']=[]
            else: changed['rows'][0]['prompt_id']='test-b'
            with self.assertRaises(ValueError):
                audit_predictions(changed,self.examples)


if __name__=='__main__':
    unittest.main()
