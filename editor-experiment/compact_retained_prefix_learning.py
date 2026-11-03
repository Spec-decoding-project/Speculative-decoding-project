"""Existing retained-prefix weighting on the compact editor; no cloud operations.

Uniform control delegates unchanged to the established compact CE update.
The candidate tests the full masked, smoothed, normalized objective package.
"""
from contextlib import nullcontext
import math
import torch
from torch.nn import functional as F
from compact_second_pass_learning import update as uniform_update
from conditional_redraft_objectives import conditional_objective


def objective(scores,batch,*,smoothing=.5):
    original=batch['labels'].clone()
    original[:,1:]=batch['base_scores'].argmax(-1)
    # Known anchor and fixed x1 have no trainable score here; zero padding only
    # satisfies the established full-block objective interface.
    return conditional_objective(F.pad(scores,(0,0,2,0)),original,
        batch['labels'],batch['valid'],objective='marginal',cut=1,smoothing=smoothing)


def update(editor,optimizer,batches,output_weight,*,arm,autocast=lambda:nullcontext()):
    if arm=='uniform_ce':
        return uniform_update(editor,optimizer,batches,output_weight,arm='target_only',autocast=autocast)
    if arm!='retained_prefix':raise ValueError('Unknown arm')
    if output_weight.requires_grad:raise ValueError('Frozen output head required')
    valid_rows=sum(int(b['valid'][:,2:].sum()) for b in batches)
    if valid_rows<=0:raise ValueError('No valid editable positions')
    for batch in batches:
        for name in ('states','visible','base_scores','teacher_scores'):
            if batch[name].requires_grad:raise ValueError('Frozen input required: '+name)
    editor.train();optimizer.zero_grad(set_to_none=True)
    total_weight=0.;numerator=0.;useful=0.;eligible=0
    for batch in batches:
        if not bool(batch['valid'][:,2:].any()):continue
        with autocast():
            folded=editor.folded_weight(output_weight)
            scores=editor.folded_scores(batch['states'],batch['visible'],batch['base_scores'][:,1:],folded)
            result=objective(scores,batch)
        if not torch.isfinite(result.numerator):raise ValueError('Nonfinite objective')
        weight=float(result.denominator)
        if not math.isfinite(weight) or weight<0:raise ValueError('Invalid objective weight')
        if weight>0:result.numerator.backward()
        total_weight+=weight;numerator+=float(result.numerator.detach())
        useful+=float(result.useful_surrogate.detach().sum());eligible+=int(result.eligible.sum())
    if total_weight==0:
        # No gradient, weight decay, momentum update or Adam step on an empty
        # objective. Keep this distinct from an exposure counter in a runner.
        return dict(loss=0.,valid_rows=valid_rows,weight=0.,eligible_blocks=eligible,
                    useful_surrogate=useful,optimizer_step=False)
    for p in editor.parameters():
        if p.grad is not None:
            p.grad.div_(total_weight)
            if not torch.isfinite(p.grad).all():raise ValueError('Nonfinite normalized gradient')
    coefficient_grad=sum(float(p.grad.detach().double().square().sum())
        for name,p in editor.named_parameters() if name!='up' and p.grad is not None)**.5
    up_grad=float(editor.up.grad.detach().double().norm())
    optimizer.step()
    return dict(loss=numerator/total_weight,valid_rows=valid_rows,weight=total_weight,
        eligible_blocks=eligible,useful_surrogate=useful,optimizer_step=True,
        coefficient_gradient_norm=coefficient_grad,up_gradient_norm=up_grad)
