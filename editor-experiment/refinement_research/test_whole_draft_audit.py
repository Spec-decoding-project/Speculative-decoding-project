from copy import deepcopy

import pytest

from refinement_research.whole_draft_audit import recount, interval


def fixture():
    labels=list(range(16));draft=labels.copy();draft[3]=19
    edited=labels[:4]+[19]*12
    bank={('p',3):dict(draft=draft,labels=labels,source='fixture',kind='repairable')}
    result=dict(mean_edited_progress=3,free_running_token_accuracy=2/14,repaired=1,damaged=0,
        rows=[dict(prompt_id='p',anchor=3,original_tokens=draft,label_tokens=labels,edited_tokens=edited,
            original_progress=2,edited_progress=3)])
    return bank,result


def test_prefix_can_improve_while_all_position_accuracy_falls():
    bank,result=fixture();_,metrics=recount(result,bank)
    assert metrics['edited_prefix']>metrics['original_prefix']
    assert metrics['edited_token_accuracy']<metrics['original_token_accuracy']
    assert metrics['prefix_gained']==1 and metrics['prefix_lost']==0


def test_recount_rejects_changed_labels_and_seed():
    bank,result=fixture()
    changed=deepcopy(result);changed['rows'][0]['label_tokens'][4]=19
    with pytest.raises(AssertionError):recount(changed,bank)
    changed=deepcopy(result);changed['rows'][0]['edited_tokens'][1]=19
    with pytest.raises(AssertionError):recount(changed,bank)


def test_paired_interval_counts_prompt_clusters_and_separate_token_metric():
    bank,result=fixture();rows,_=recount(result,bank)
    baseline=[dict(r,edited=r['original'],edited_tokens=r['original_tokens']) for r in rows]
    observed=interval(rows,baseline,draws=100)
    assert observed['prefix_delta_ci95']==[1.,1.]
    assert observed['token_accuracy_delta_ci95'][1]<0
