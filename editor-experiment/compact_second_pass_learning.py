"""Matched compact-editor updates and fixed-context evaluation; no cloud calls."""
from contextlib import nullcontext
import torch
from controlled_second_pass_distillation import controlled_correction_loss
from conditional_redraft_objectives import branch_progress

ARMS=('target_only','eligible_ce','teacher_kl')


def update(editor, optimizer, batches, output_weight, *, arm, teacher_weight=.5,
           autocast=lambda: nullcontext()):
    """One effective batch, normalized over every valid editable position.

    A batch contains states[B,16,H], visible[B,H], base_scores[B,15,V],
    teacher_scores[B,14,V], labels/valid[B,16]. These are frozen inputs.
    W U is differentiable and refreshed per microbatch, never a stale cache.
    """
    if arm not in ARMS:raise ValueError('Unknown arm')
    if output_weight.requires_grad:raise ValueError('Output head must remain frozen')
    total=sum(int(b['valid'][:,2:].sum()) for b in batches)
    if total<=0:raise ValueError('No valid editable positions')
    for b in batches:
        for name in ('states','visible','base_scores','teacher_scores'):
            if b[name].requires_grad:raise ValueError('Frozen inputs required: '+name)
    editor.train()
    optimizer.zero_grad(set_to_none=True)
    sums={'loss':0.,'ce':0.,'auxiliary':0.}
    eligible=0
    for b in batches:
        count=int(b['valid'][:,2:].sum())
        if not count:continue
        with autocast():
            folded=editor.folded_weight(output_weight)
            scores=editor.folded_scores(b['states'],b['visible'],b['base_scores'][:,1:],folded)
            loss,metrics=controlled_correction_loss(scores,b['teacher_scores'],
                b['labels'][:,2:],b['valid'][:,2:],arm=arm,teacher_weight=teacher_weight)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        (loss*(count/total)).backward()
        sums['loss']+=float(loss.detach())*count/total
        for name in ('ce','auxiliary'):sums[name]+=float(metrics[name])*count/total
        eligible+=metrics['eligible_rows']
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in editor.parameters()):
        raise ValueError('Nonfinite gradient')
    # No normalization or fixed-magnitude rescaling of optimizer updates.
    coefficient_grad=sum(float(p.grad.detach().double().square().sum())
        for name,p in editor.named_parameters() if name!='up' and p.grad is not None)**.5
    up_grad=float(editor.up.grad.detach().double().norm())
    optimizer.step()
    return dict(**sums,valid_rows=total,eligible_rows=eligible,
                coefficient_gradient_norm=coefficient_grad,up_gradient_norm=up_grad)


@torch.no_grad()
def evaluate_batch(editor, batch, output_weight, *, folded_weight=None):
    """Caller supplies a frozen deployment copy and matching precision/row shapes."""
    if editor.training or any(p.requires_grad for p in editor.parameters()):
        raise ValueError('Use a separate frozen deployment copy')
    if batch['states'].shape[0]!=1:
        raise ValueError('Evaluation fixes serving batch one; do not silently change GEMM shape')
    folded=editor.folded_weight(output_weight) if folded_weight is None else folded_weight
    scores=editor.folded_scores(batch['states'],batch['visible'],batch['base_scores'][:,1:],folded)
    labels,valid=batch['labels'],batch['valid']
    original=labels.clone()
    original[:,1:]=batch['base_scores'].argmax(-1)
    edited=original.clone()
    edited[:,2:]=scores.argmax(-1)
    metric=branch_progress(original,edited,labels,valid)
    a,b=int(metric['original'][0]),int(metric['alternative'][0])
    capacity=int(valid.sum())-1
    return dict(original_progress=1+a,edited_progress=1+b,union_progress=1+max(a,b),
        repaired=b>a,damaged=b<a,fully_correct_original=a==capacity,
        damaged_fully_correct=a==capacity and b<a,x1_unreachable=a<1,
        original_tokens=original[0].tolist(),edited_tokens=edited[0].tolist(),
        scope='Fixed-context recorded-label coverage; no speedup or native output-fidelity claim')
