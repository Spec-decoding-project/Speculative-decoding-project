import pytest
import torch
from torch.nn import functional as F

from autoregressive_editor.model import Config, TokenEditor
from copy_action_fit.data import SOURCES
from copy_action_fit.learning import inputs
from refinement_research.whole_draft_data import choose_training
from refinement_research.whole_draft_fit import position_weights, rate, checks


def test_weights_protect_only_surviving_prefix_and_first_error():
    labels=torch.tensor([1,2,3,4,5])
    draft=torch.tensor([1,2,9,4,8])
    early=torch.exp(-torch.arange(5).float()/7)
    assert torch.equal(position_weights(draft,labels,'uniform'),torch.ones(5))
    assert torch.equal(position_weights(draft,labels,'early'),early)
    assert torch.equal(position_weights(draft,labels,'protect_repair'),early*torch.tensor([2,2,4,1,1]))
    assert torch.equal(position_weights(labels,labels,'protect_repair'),early*2)
    assert torch.equal(position_weights(torch.zeros(5).long(),labels,'protect_repair'),early*torch.tensor([4,1,1,1,1]))


def test_uniform_loss_and_gradient_match_original_token_objective():
    torch.manual_seed(4)
    logits=torch.randn(14,20,requires_grad=True)
    labels=torch.arange(14);draft=labels.roll(2)
    weight=position_weights(draft,labels,'uniform')
    new=(F.cross_entropy(logits,labels,reduction='none')*weight).sum()/weight.sum()
    old=F.cross_entropy(logits,labels)
    assert torch.allclose(new,old)
    assert torch.allclose(torch.autograd.grad(new,logits,retain_graph=True)[0],torch.autograd.grad(old,logits)[0])


def test_preservation_and_first_error_receive_more_gradient_than_late_errors():
    logits=torch.zeros(5,10,requires_grad=True);labels=torch.arange(5)
    draft=torch.tensor([0,1,9,3,9])
    weight=position_weights(draft,labels,'protect_repair')
    loss=(F.cross_entropy(logits,labels,reduction='none')*weight).sum()/weight.sum()
    loss.backward()
    magnitudes=logits.grad.norm(dim=-1)
    assert magnitudes[2]>magnitudes[0]>magnitudes[3]>magnitudes[4]


def test_selection_excludes_calibration_prompts_and_duplicate_families():
    candidates=[]
    for source in SOURCES:
        for kind in ('preservation','repairable'):
            for i in range(10):
                candidates.append(dict(source=source,kind=kind,prompt_id=f'{source}/{kind}/{i}',
                    family_id=f'{source}/{i}',anchor=i))
    calibration=[candidates[0],candidates[20]]
    chosen=choose_training(candidates,calibration,per_source=4,preservation=2)
    assert len(chosen)==8
    assert len({r['family_id'] for r in chosen})==8
    assert not {r['family_id'] for r in chosen}&{r['family_id'] for r in calibration}
    assert all(sum(r['source']==s and r['kind']==k for r in chosen)==2
               for s in SOURCES for k in ('preservation','repairable'))


def test_full_proposal_can_inform_first_output_but_future_labels_cannot():
    torch.manual_seed(3)
    model=TokenEditor(Config(vocab_size=20,width=16,layers=1,heads=2,max_length=64))
    labels=list(range(16));draft=labels.copy();draft[3]=19
    row=dict(prefix=[2,3,4],draft=draft,labels=labels)
    checks(model,[row])
    ids,segments,start,_,_=inputs(model,row)
    before,_=model(ids,segments,logit_start=start)
    changed=dict(row,draft=draft[:-1]+[18])
    ids,segments,start,_,_=inputs(model,changed)
    after,_=model(ids,segments,logit_start=start)
    assert not torch.equal(before[:,0],after[:,0])


def test_schedule_and_objective_validation():
    assert rate(1)==1e-4/16 and rate(16)==1e-4 and rate(768)==1e-5
    with pytest.raises(ValueError):
        position_weights(torch.ones(4),torch.ones(4),'unknown')
