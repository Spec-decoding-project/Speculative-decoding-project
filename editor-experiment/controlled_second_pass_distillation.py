"""Controlled research losses. Legacy objective remains in second_pass_distillation.py.

The CE coefficient and all-valid denominator are identical across arms. Eligibility
is training-only. No run, loading, or cloud activity occurs on import.
"""
import torch
import torch.nn.functional as F

def controlled_correction_loss(student_logits,teacher_logits,labels,valid,*,arm='teacher_kl',teacher_weight=.5,temperature=1.):
    if arm not in {'target_only','eligible_ce','teacher_kl'}:raise ValueError('Unknown arm')
    if student_logits.shape!=teacher_logits.shape or student_logits.shape[:-1]!=labels.shape or labels.shape!=valid.shape:raise ValueError('Misaligned tensors')
    if valid.dtype!=torch.bool or not bool(valid.any()):raise ValueError('Require valid rows')
    if temperature!=1.:raise ValueError('This fixed comparison uses temperature1; other temperatures change its gradient interpretation')
    if teacher_weight<0:raise ValueError('Nonnegative auxiliary weight required')
    dtype=torch.float64 if student_logits.dtype==torch.float64 else torch.float32
    s=student_logits[valid].to(dtype);t=teacher_logits.detach()[valid].to(dtype);y=labels[valid]
    logp=F.log_softmax(s,-1);logq=F.log_softmax(t,-1)
    ce=F.nll_loss(logp,y,reduction='none');eligible=t.argmax(-1).eq(y)
    if arm=='target_only':aux=s.sum()*0
    elif arm=='eligible_ce':aux=(ce*eligible).sum()/len(y)
    else:aux=(F.kl_div(logp,logq,reduction='none',log_target=True).sum(-1)*eligible).sum()/len(y)
    loss=ce.mean()+teacher_weight*aux
    return loss,{'ce':ce.mean().detach(),'auxiliary':aux.detach(),'eligible_rows':int(eligible.sum()),'valid_rows':len(y)}
