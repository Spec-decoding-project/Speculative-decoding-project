"""Matched first-guess versus completed-proposal teacher inputs; no cloud calls.

This is preparation for a teacher-value experiment, not a trained compact editor.
Never substitute recorded target labels for proposal tokens in either condition.
"""
import time
import torch
from torch.nn import functional as F

MODES=('first_guess','full_proposal')

def visible_ids(example, proposal_tokens, *, mask_token_id, mode):
    if mode not in MODES:raise ValueError('Unknown proposal-input mode')
    ids=example.noise_ids
    if proposal_tokens.shape!=ids.shape or proposal_tokens.dtype!=torch.long:
        raise ValueError('Proposal must contain the anchor and full first draft')
    if proposal_tokens.device!=ids.device:raise ValueError('Proposal device differs')
    if not torch.equal(proposal_tokens[:,:1],ids[:,:1]):raise ValueError('Proposal anchor differs')
    if not bool((ids[:,1:]==mask_token_id).all()):raise ValueError('Expected masked future input')
    if bool((proposal_tokens<0).any()):raise ValueError('Proposal contains label padding or invalid IDs')
    result=ids.clone()
    if mode=='first_guess':result[:,1]=proposal_tokens[:,1]
    else:result[:,1:]=proposal_tokens[:,1:]
    return result

def forward_scores(model,embedding,head,example,proposal_tokens,*,mode):
    ids=visible_ids(example,proposal_tokens,mask_token_id=model.mask_token_id,mode=mode)
    # Detaching inputs preserves gradients through all private teacher weights,
    # while vocabulary embeddings and recorded committed features stay frozen.
    hidden=model(noise_embedding=embedding(ids).detach(),target_hidden=example.context_features.detach(),
                 position_ids=example.positions,is_causal=False)
    return head(hidden[:,2:])

def suffix_loss(model,embedding,head,example,valid,proposal_tokens,*,mode):
    scores=forward_scores(model,embedding,head,example,proposal_tokens,mode=mode)
    labels=example.labels[:,2:].masked_fill(~valid[:,2:],0)
    nll=F.cross_entropy(scores.float().transpose(1,2),labels,reduction='none')
    weight=torch.exp(-torch.arange(scores.shape[1],device=scores.device).float()/7.)[None]*valid[:,2:]
    if not torch.isfinite(nll).all():raise ValueError('Nonfinite training loss')
    return (nll*weight).sum(),weight.sum()

def fit(model,embedding,head,metadata,banks,schedule,denominators,cfg,guard,progress):
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from teacher_support_training import training_example
    from conditional_redraft_learning import optimizer_step,amp
    if cfg['mode'] not in MODES:raise ValueError('Unknown proposal-input mode')
    if any(m['split']!='train' for m in metadata):raise ValueError('Fit requires training-only metadata')
    if any(p.requires_grad for module in (embedding,head) for p in module.parameters()):
        raise ValueError('Shared embedding and output head must be frozen')
    model.eval().requires_grad_(True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['betas']),
        eps=cfg['adam_epsilon'],weight_decay=cfg['weight_decay'])
    device=next(model.parameters()).device;rows=[];started=time.monotonic()
    for step,batch in enumerate(schedule):
        guard();optimizer.zero_grad(set_to_none=True);total=weight=0.
        for item in batch:
            pi=item['prompt_index'];meta=metadata[pi];a=meta['anchors'][item['source_index']]
            ex,valid=training_example(banks[meta['id']],meta,item,model.drafter.mask_token_id,device)
            # This one study retains the historical eligible-first support and
            # matched schedule. Labels gate supervision, never manufacture inputs.
            proposal=torch.tensor(a['original'],device=device,dtype=torch.long)[None]
            if int(proposal[0,1])!=int(ex.labels[0,1]):raise ValueError('Expected matched eligible-first schedule')
            with amp(device),sdpa_kernel(SDPBackend.MATH):
                numerator,denominator=suffix_loss(model.drafter,embedding,head,ex,valid,proposal,mode=cfg['mode'])
                (numerator/denominators[pi]).backward()
            total+=float(numerator.detach());weight+=float(denominator)
        row=optimizer_step(model,optimizer,1.,step,len(schedule),config=cfg)
        if not row['updated']:raise ValueError('Missing optimizer update')
        row.update(step=step+1,weighted_nll=total,supervision_weight=weight);rows.append(row)
        if (step+1)%32==0:progress(step+1,rows)
    return dict(updates=len(rows),rows=rows,seconds=time.monotonic()-started,
                supervision_weight=sum(r['supervision_weight'] for r in rows))

@torch.no_grad()
def evaluate_example(model,embedding,head,example,proposal_tokens,*,mode):
    """One original-input redraft; first guess is retained, not answer-selected."""
    scores=forward_scores(model,embedding,head,example,proposal_tokens,mode=mode)
    if not torch.isfinite(scores).all():raise ValueError('Nonfinite evaluation scores')
    return torch.cat((proposal_tokens[:,:2],scores.argmax(-1)),dim=1),scores
