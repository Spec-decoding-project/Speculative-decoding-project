from dataclasses import replace
import copy
import pytest
import torch
from full_proposal_teacher import visible_ids,forward_scores,suffix_loss,fit,evaluate_example
from test_teacher_context_intervention import make_native
from conditional_redraft_model import make_anchored_example

def fixture():
    torch.manual_seed(619);torch.set_num_threads(1)
    _,model=make_native();model.config._attn_implementation='sdpa';model.requires_grad_(True)
    emb=torch.nn.Embedding(80,32).eval().requires_grad_(False)
    head=torch.nn.Linear(32,80,bias=False).eval().requires_grad_(False)
    bank=dict(ids=torch.arange(32),features=torch.randn(32,64))
    ex=make_anchored_example(bank['ids'],bank['features'],anchor=8,block_size=16,mask_token_id=0)
    proposal=torch.tensor([[8,9]+[40+i for i in range(14)]])
    return model,emb,head,bank,ex,proposal

def test_exact_input_difference_and_no_label_leak():
    model,emb,head,bank,ex,p=fixture()
    a=visible_ids(ex,p,mask_token_id=0,mode='first_guess')
    b=visible_ids(ex,p,mask_token_id=0,mode='full_proposal')
    assert a.tolist()==[[8,9]+[0]*14] and torch.equal(b,p)
    # Perturb target answers, which are not proposal inputs.
    changed=replace(ex,labels=ex.labels.roll(1,-1))
    for mode in ('first_guess','full_proposal'):
        assert torch.equal(forward_scores(model,emb,head,ex,p,mode=mode),forward_scores(model,emb,head,changed,p,mode=mode))
    assert torch.equal(ex.noise_ids[:,1:],torch.zeros(1,15,dtype=torch.long))

def test_future_feature_rows_never_enter_teacher():
    model,emb,head,bank,ex,p=fixture()
    changed=bank['features'].clone();changed[8:]+=1000
    ex2=make_anchored_example(bank['ids'],changed,anchor=8,block_size=16,mask_token_id=0)
    assert torch.equal(ex.context_features,ex2.context_features)
    assert torch.equal(forward_scores(model,emb,head,ex,p,mode='full_proposal'),forward_scores(model,emb,head,ex2,p,mode='full_proposal'))

def test_control_exactly_reproduces_historical_loss_and_gradients():
    from teacher_support_training import suffix_loss as previous
    model,emb,head,bank,ex,p=fixture()
    actual,w=suffix_loss(model,emb,head,ex,ex.valid,p,mode='first_guess');actual.backward()
    gradients={k:v.grad.clone() for k,v in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    expected,w2=previous(model,emb,head,ex,ex.valid,feedback=True);expected.backward()
    assert torch.equal(actual,expected) and torch.equal(w,w2)
    assert all(torch.equal(gradients[k],v.grad) for k,v in model.named_parameters())

def test_full_proposal_reaches_private_weights_only():
    model,emb,head,bank,ex,p=fixture()
    loss,_=suffix_loss(model,emb,head,ex,ex.valid,p,mode='full_proposal');loss.backward()
    assert model.fc.weight.grad.abs().sum()>0
    assert all(v.grad is None for m in (emb,head) for v in m.parameters())
    # Later draft tokens can affect an earlier output through native attention.
    q=p.clone();q[0,-1]+=1
    assert not torch.equal(forward_scores(model,emb,head,ex,p,mode='full_proposal')[:,0],forward_scores(model,emb,head,ex,q,mode='full_proposal')[:,0])
    assert torch.equal(forward_scores(model,emb,head,ex,p,mode='first_guess'),forward_scores(model,emb,head,ex,q,mode='first_guess'))

def test_inference_keeps_wrong_first_guess_and_never_uses_answer():
    model,emb,head,bank,ex,p=fixture();p=p.clone();p[0,1]=60
    tokens,scores=evaluate_example(model,emb,head,ex,p,mode='full_proposal')
    assert tokens[0,:2].tolist()==[8,60]
    changed=replace(ex,labels=torch.full_like(ex.labels,75))
    tokens2,scores2=evaluate_example(model,emb,head,changed,p,mode='full_proposal')
    assert torch.equal(tokens,tokens2) and torch.equal(scores,scores2)

@pytest.mark.parametrize('bad',['shape','anchor','padding','mode'])
def test_invalid_proposals_rejected(bad):
    model,emb,head,bank,ex,p=fixture();mode='full_proposal'
    if bad=='shape':p=p[:,:-1]
    if bad=='anchor':p=p.clone();p[0,0]=2
    if bad=='padding':p=p.clone();p[0,-1]=-100
    if bad=='mode':mode='unknown'
    with pytest.raises(ValueError):visible_ids(ex,p,mask_token_id=0,mode=mode)

def test_actual_fit_control_replays_prior_worker_and_full_input_updates():
    from teacher_support_worker import fit as old_fit
    from teacher_support_training import make_schedule,ARMS
    from test_teacher_support_training import metadata
    from conditional_full_reference_precision import FullReferenceFirstPass,FullConditionalReference
    native,emb,head,bank,ex,p=fixture();ordinary=FullReferenceFirstPass(native,split=1)
    base=FullConditionalReference(ordinary,'feedback')
    models=[copy.deepcopy(base) for _ in range(3)]
    meta=metadata()
    for m in meta:
        for a in m['anchors']:a.update(anchor=8,labels=ex.labels[0].tolist(),original=p[0].tolist(),valid=[True]*16)
    schedule=make_schedule(meta);batches=schedule['schedules'][ARMS[0]][:2]
    banks={m['id']:bank for m in meta};den=schedule['prompt_denominators']
    cfg=dict(learning_rate=1e-4,betas=[.9,.95],adam_epsilon=1e-8,weight_decay=.01,clip_norm=1.,warmup_fraction=.05,feedback=True)
    results=[]
    results.append(old_fit(models[0],emb,head,meta,banks,batches,den,cfg,lambda:None,lambda *a:None))
    for model,mode in zip(models[1:],('first_guess','full_proposal')):
        results.append(fit(model,emb,head,meta,banks,batches,den,dict(cfg,mode=mode),lambda:None,lambda *a:None))
    assert all(r['updates']==2 for r in results)
    assert len({r['supervision_weight'] for r in results})==1
    assert all(torch.equal(v,models[1].state_dict()[k]) for k,v in models[0].state_dict().items())
    assert any(not torch.equal(v,models[2].state_dict()[k]) for k,v in models[1].state_dict().items())
    assert all(torch.equal(v,base.state_dict()[k]) for k,v in models[0].state_dict().items() if not v.is_floating_point())
