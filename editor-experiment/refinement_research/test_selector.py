import copy
import pytest

from refinement_research.selector import compile_first_only, features, fit, fit_tree, label_shadow, predict, selected, select_prompts


def row(i, x, gain, prefix='fit'):
    return dict(prompt_id=f'{prefix}-{i}',family_id=f'{prefix}-{i}',split='train',
                features=[x]+[0.]*13,gain=gain)


def test_features_have_no_labels_and_reject_invalid_confidence():
    x=features(list(range(15)),[float(i) for i in range(15)],3,100,121)
    assert x[:7]==[0.,1.,2.,3.,1.,7.5,1.5]
    assert x[7:]==[15,0,1,3,100,21,0]
    assert features([2]*15,[1.]*15,1,100,101)[7:]==[1,14,15,1,100,1,1]
    with pytest.raises(ValueError):features([2]*15,[float('nan')]*15,1,100,101)


def test_tree_learns_repair_and_damage_and_threshold_reloads_exactly():
    rows=[row(i, i%2, 3 if i%2 else -2) for i in range(256)]
    tree=fit_tree(rows)
    assert predict(tree,[0.]*14)<0 < predict(tree,[1.]+[0.]*13)
    calibration=[row(i,i%2,3 if i%2 else -2,'cal') for i in range(128)]
    model=fit(rows,calibration)
    import json
    restored=json.loads(json.dumps(model))
    assert selected(restored,[1.]+[0.]*13)
    assert not selected(restored,[0.]*14)


def test_calibration_can_reject_editing_and_prevents_split_leakage():
    training=[row(i,i%2,1) for i in range(256)]
    cal=[row(i,i%2,0,'cal') for i in range(128)]
    assert fit(training,cal)['threshold'] is None
    with pytest.raises(ValueError,match='leakage'):fit(training,training)
    cal[0]['split']='development'
    with pytest.raises(ValueError,match='training-origin'):fit(training,cal)


def test_shadow_labels_exclude_unknown_tail_and_count_damage():
    prompt=dict(id='p',family_id='f',split='train',source='s',prompt=[50,51])
    full=[50,51,*range(30)]
    edits=[];decisions=[]
    for position in (3,28):
        original=[full[position-1],*full[position:position+15]]
        original += [99]*(16-len(original))
        edited=original.copy();edited[3]=99
        edits.append(dict(anchor=position-1,prefix=full[:position-1],original=original,edited=edited))
        decisions.append(dict(length=position,features=[0.]*14))
    record=dict(generated=dict(tokens=full[2:]),observed=dict(selector_decisions=decisions,edits=edits))
    result=label_shadow(prompt,record)
    assert len(result)==1 and result[0]['gain']==-13
    overshoot=copy.deepcopy(record)
    overshoot['observed']['edits'].append(dict(anchor=len(full)+4,prefix=full+[80]*4,
        original=[80]*16,edited=[80]*16))
    overshoot['observed']['selector_decisions'].append(dict(length=len(full)+5,features=[0.]*14))
    assert label_shadow(prompt,overshoot)==result
    overshoot['observed']['edits'][-1]['prefix'][0]=999
    with pytest.raises(ValueError,match='canonical baseline'):label_shadow(prompt,overshoot)
    broken=copy.deepcopy(record);broken['observed']['edits'][0]['edited'][1]=99
    with pytest.raises(ValueError,match='changed'):label_shadow(prompt,broken)


def test_skip_hook_never_copies_full_context_or_invokes_editor():
    import torch
    from types import SimpleNamespace
    from refinement_research.selector_worker import SelectorWorker
    worker=SelectorWorker()
    worker.mode='selector_probe'
    worker.selector_model=dict(tree=dict(value=1.),threshold=0.)
    worker.selector_gaps=torch.ones(15)
    worker.selector_decisions=[]
    worker.selector_prompt_length=100
    worker.model_runner=SimpleNamespace(req_states=SimpleNamespace(
        req_id_to_index={'r':0},total_len=SimpleNamespace(gpu=torch.tensor([101]))))
    args=[None]*6;args[5]=torch.tensor([1])
    # Deliberately no all_token_ids or editor: a skip must not touch either.
    assert not worker.choose_edit(SimpleNamespace(use_local_argmax_reduction=False),
                                  torch.arange(15).reshape(1,15),SimpleNamespace(req_ids=['r']),args)
    assert not worker.selector_decisions[0]['selected']
    assert worker.selector_gaps is None


def test_selection_is_disjoint_reproducible_and_excludes_previous_native_prompts():
    from copy_action_fit.data import SOURCES
    from refinement_research.native_best import select as previous
    rows=[dict(id=f'{s}-{split}-{i}',family_id=f'{s}-{split}-{i}',source=s,
               split=split,prompt=[1]*(i+10))
          for s in SOURCES for split,n in [('train',64),('development',256),('test',256)]
          for i in range(n)]
    chosen=select_prompts(rows)
    assert chosen==select_prompts(list(reversed(rows)))
    assert {k:len(v) for k,v in chosen.items()}==dict(fit=96,calibration=32,evaluation=64)
    old={p['id'] for p in previous(rows)}
    assert not old & {p['id'] for p in chosen['evaluation']}
    assert all(p['split']=='train' for p in chosen['fit']+chosen['calibration'])
    assert all(p['split']=='development' for p in chosen['evaluation'])
    assert len({p['family_id'] for group in chosen.values() for p in group})==192


def test_compilation_proves_first_only_policy_and_rejects_other_decisions():
    model=dict(threshold=.5,tree=dict(feature=12,threshold=1.5,left=dict(value=1.),
        right=dict(feature=4,threshold=.25,left=dict(value=.1),right=dict(value=.3))))
    assert compile_first_only(model)['policy']=='first_proposal_only'
    for generated in (1,2,16,64):
        for gap in (0.,10.):
            x=[0.]*14;x[12]=generated;x[4]=gap
            assert selected(model,x)==(generated==1)
    model['tree']['right']['right']['value']=.6
    with pytest.raises(ValueError,match='not exactly equivalent'):compile_first_only(model)
