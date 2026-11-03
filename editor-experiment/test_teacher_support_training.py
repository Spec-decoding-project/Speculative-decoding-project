import copy,math
import pytest
import torch
from teacher_support_training import make_schedule,training_example,suffix_loss,ARMS
from test_teacher_context_intervention import make_native

def metadata():
    result=[]
    for pi in range(32):
        anchors=[]
        for ai in range(32):
            labels=list(range(16));original=list(labels)
            if ai%3==0:original[1]=20
            valid=[True]*(16 if ai%5 else 9)+[False]*(0 if ai%5 else 7)
            anchors.append(dict(labels=labels,original=original,valid=valid,anchor=ai+1))
        result.append(dict(id=str(pi),split='train',anchors=anchors))
    return result

def test_same_updates_prompts_supervised_positions_and_no_excluded_controls():
    meta=metadata();plan=make_schedule(meta)
    assert plan==make_schedule(meta)
    assert all(len(s)==32 for s in plan['schedules'].values())
    for batch_index in range(32):
        batches=[plan['schedules'][a][batch_index] for a in ARMS]
        for old,all_,no in zip(*batches):
            assert all_==no
            assert old['prompt_index']==all_['prompt_index']
            assert old['valid_length']==all_['valid_length']
            m=meta[old['prompt_index']];a=m['anchors'][old['source_index']]
            assert a['original'][1]==a['labels'][1]
            assert sum(a['valid'])>=old['valid_length']
    meta[0]['split']='development'
    with pytest.raises(ValueError,match='split'):make_schedule(meta)

def test_only_first_label_is_input_future_features_and_labels_do_not_leak():
    _,model=make_native();model.requires_grad_(True);model.config._attn_implementation='sdpa'
    embedding=torch.nn.Embedding(80,32).eval().requires_grad_(False)
    head=torch.nn.Linear(32,80,bias=False).eval().requires_grad_(False)
    bank=dict(ids=torch.arange(32),features=torch.randn(32,64))
    a=dict(anchor=8,labels=list(range(8,24)),valid=[True]*16)
    meta=dict(anchors=[a]);assignment=dict(source_index=0,valid_length=9)
    ex,valid=training_example(bank,meta,assignment,0,'cpu')
    seen=[];hook=model.register_forward_pre_hook(lambda m,args,kw:seen.append({k:v.clone() for k,v in kw.items() if torch.is_tensor(v)}),with_kwargs=True)
    loss,weight=suffix_loss(model,embedding,head,ex,valid,feedback=True)
    assert seen[0]['target_hidden'].shape[1]==8
    assert torch.equal(seen[0]['noise_embedding'][:,1],embedding(ex.labels[:,1]))
    assert torch.equal(seen[0]['noise_embedding'][:,2:],embedding(torch.zeros(1,14,dtype=torch.long)))
    loss.backward();assert model.fc.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in embedding.parameters()) and all(p.grad is None for p in head.parameters())
    assert abs(float(weight)-sum(math.exp(-j/7) for j in range(7)))<1e-6
    newbank=dict(ids=bank['ids'].clone(),features=bank['features'].clone());newbank['features'][8:]+=999
    newbank['ids'][10:24]+=20;newmeta=copy.deepcopy(meta);newmeta['anchors'][0]['labels']=newbank['ids'][8:24].tolist()
    ex2,valid2=training_example(newbank,newmeta,assignment,0,'cpu')
    suffix_loss(model,embedding,head,ex2,valid2,feedback=True)
    assert torch.equal(seen[0]['noise_embedding'],seen[1]['noise_embedding'])
    assert torch.equal(seen[0]['target_hidden'],seen[1]['target_hidden'])
    suffix_loss(model,embedding,head,ex,valid,feedback=False);hook.remove()
    assert torch.equal(seen[2]['noise_embedding'][:,1],embedding(torch.zeros(1,dtype=torch.long)))

def test_suffix_loss_matches_explicit_manual_mask_and_gradients():
    _,model=make_native();model.requires_grad_(True);model.config._attn_implementation='sdpa'
    embedding=torch.nn.Embedding(80,32).eval().requires_grad_(False)
    head=torch.nn.Linear(32,80,bias=False).eval().requires_grad_(False)
    bank=dict(ids=torch.arange(32),features=torch.randn(32,64))
    meta=dict(anchors=[dict(anchor=8,labels=list(range(8,24)),valid=[True]*16)])
    ex,valid=training_example(bank,meta,dict(source_index=0,valid_length=9),0,'cpu')
    actual,_=suffix_loss(model,embedding,head,ex,valid,feedback=True)
    actual.backward();gradient=model.fc.weight.grad.clone();model.zero_grad(set_to_none=True)
    ids=ex.noise_ids.clone();ids[:,1]=ex.labels[:,1]
    hidden=model(noise_embedding=embedding(ids),target_hidden=ex.context_features,position_ids=ex.positions,is_causal=False)
    logp=head(hidden[:,2:]).log_softmax(-1)
    manual=sum(-logp[0,j-2,ex.labels[0,j]]*math.exp(-(j-2)/7) for j in range(2,9))
    manual.backward()
    torch.testing.assert_close(actual,manual)
    torch.testing.assert_close(gradient,model.fc.weight.grad,atol=2e-6,rtol=1e-5)

def test_actual_worker_fit_updates_only_teacher_and_preserves_matched_exposure():
    from teacher_support_worker import fit
    from conditional_full_reference_precision import FullReferenceFirstPass,FullConditionalReference
    torch.set_num_threads(1)
    _,native=make_native();native.config._attn_implementation='sdpa'
    ordinary=FullReferenceFirstPass(native,split=1)
    model=FullConditionalReference(ordinary,'feedback')
    embedding=torch.nn.Embedding(80,32).eval().requires_grad_(False)
    head=torch.nn.Linear(32,80,bias=False).eval().requires_grad_(False)
    bank=dict(ids=torch.arange(32),features=torch.randn(32,64))
    meta=metadata()
    for m in meta:
        for a in m['anchors']:
            a.update(anchor=8,labels=list(range(8,24)),original=list(range(8,24)),valid=[True]*16)
    plan=make_schedule(meta);schedule=plan['schedules'][ARMS[1]][:2]
    before=model.drafter.fc.weight.detach().clone();base=ordinary.drafter.fc.weight.detach().clone();e=embedding.weight.clone()
    cfg=dict(learning_rate=1e-4,betas=[.9,.95],adam_epsilon=1e-8,weight_decay=.01,clip_norm=1.,warmup_fraction=.05,feedback=True)
    result=fit(model,embedding,head,meta,{m['id']:bank for m in meta},schedule,plan['prompt_denominators'],cfg,lambda:None,lambda *args:None)
    assert result['updates']==2 and result['supervision_weight']>0
    assert not torch.equal(before,model.drafter.fc.weight)
    assert torch.equal(base,ordinary.drafter.fc.weight) and torch.equal(e,embedding.weight)
    assert all(p.grad is None for p in head.parameters())

def test_worker_evaluation_clamps_each_branch_and_reuses_only_unconditioned_scores():
    from teacher_support_worker import evaluate
    from conditional_full_reference_precision import FullReferenceFirstPass,FullConditionalReference
    _,native=make_native();native.config._attn_implementation='sdpa'
    ordinary=FullReferenceFirstPass(native,split=1);model=FullConditionalReference(ordinary,'feedback')
    embedding=torch.nn.Embedding(80,32).eval().requires_grad_(False);head=torch.nn.Linear(32,80,bias=False).eval().requires_grad_(False)
    bank=dict(ids=torch.arange(32),features=torch.randn(32,64))
    a=dict(anchor=8,labels=list(range(8,24)),valid=[True]*16)
    meta=dict(id='fixture',anchors=[a]);ids=torch.arange(16).repeat(15,1).tolist();ids[0]=list(range(4,20))
    r=dict(labels=a['labels'],valid=a['valid'],original_tokens=[8,4]+[7]*14,candidate_ids=ids)
    records={('fixture',0):r};banks={'fixture':bank};calls=[]
    hook=model.drafter.register_forward_pre_hook(lambda *args:calls.append(1))
    off=evaluate(model,embedding,head,[meta],banks,records,False,lambda:None,lambda rows:None)
    assert len(calls)==1
    assert off[0]['tokens']['1'][:2]==[8,4] and off[0]['tokens']['2'][:2]==[8,5]
    assert off[0]['candidate_logs']['1']==off[0]['candidate_logs']['2']
    on=evaluate(model,embedding,head,[meta],banks,records,True,lambda:None,lambda rows:None)
    hook.remove();assert len(calls)==3
    assert on[0]['candidate_logs']['1']!=on[0]['candidate_logs']['2']
    r.update(teacher_tokens=on[0]['tokens']['1'],teacher_logs=[[0.]*16]+on[0]['candidate_logs']['1'])
    evaluate(model,embedding,head,[meta],banks,records,True,lambda:None,lambda rows:None,historical=True)
    r['teacher_tokens'][2]=(r['teacher_tokens'][2]+1)%80
    with pytest.raises(ValueError,match='historical|recorded'):
        evaluate(model,embedding,head,[meta],banks,records,True,lambda:None,lambda rows:None,historical=True)
