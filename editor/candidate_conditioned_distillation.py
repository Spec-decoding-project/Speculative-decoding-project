"""Objectives for a fixed-support probability editor; no cloud operations."""
from contextlib import nullcontext
import torch
import torch.nn.functional as F

ARMS = ('full_teacher_kl', 'candidate_teacher_kl', 'candidate_target_ce')


def candidates(base_scores, k=4):
    """Descending scores, lowest vocabulary ID first on ties; matches saved rules."""
    work = base_scores.detach().clone()
    selected = []
    for _ in range(k):
        _, index = work.max(-1)
        selected.append(index)
        work.scatter_(-1, index[..., None], -float('inf'))
    return torch.stack(selected, -1)


def objective(scores, teacher, base, labels, valid, *, arm):
    if arm not in ARMS or scores.shape != teacher.shape or scores.shape != base.shape:
        raise ValueError('Unknown arm or score-shape mismatch')
    if scores.shape[:-1] != labels.shape or labels.shape != valid.shape:
        raise ValueError('Label shape mismatch')
    if valid.dtype != torch.bool or not bool(valid.any()):
        raise ValueError('No valid editable rows')
    dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
    s, t, q = scores[valid].to(dtype), teacher.detach()[valid].to(dtype), base.detach()[valid].to(dtype)
    y = labels[valid]
    ids = candidates(q)
    base_logp = q - q.logsumexp(-1, keepdim=True)
    mass = base_logp.gather(-1, ids).exp().sum(-1).clamp(max=1.)
    logp = F.log_softmax(s.gather(-1, ids), -1)
    logt = F.log_softmax(t.gather(-1, ids), -1)
    inside = ids.eq(y[:, None])
    covered = inside.any(-1)
    if arm == 'full_teacher_kl':
        loss = F.kl_div(F.log_softmax(s,-1), F.log_softmax(t,-1),
                        reduction='none', log_target=True).sum(-1).mean()
    elif arm == 'candidate_teacher_kl':
        loss = (mass * F.kl_div(logp, logt, reduction='none', log_target=True).sum(-1)).mean()
    else:
        # Outside-support target loss is constant in the student parameters.
        # Omit that constant, but retain the all-valid denominator. No silent
        # reweighting by the number of covered labels or dropping whole blocks.
        loss = -(logp * inside.to(logp.dtype)).sum(-1).mean()
    return loss, dict(valid_rows=len(y), candidate_label_rows=int(covered.sum()),
                      outside_candidate_label_rows=int((~covered).sum()),
                      mean_original_candidate_mass=float(mass.mean()),
                      conditional_teacher_kl=float((mass*F.kl_div(logp.detach(),logt,reduction='none',log_target=True).sum(-1)).mean()))


def update(editor, optimizer, batches, output_weight, *, arm, autocast=lambda: nullcontext()):
    """Same full score forward in all arms, isolating the supervised objective."""
    total = sum(int(b['valid'][:,2:].sum()) for b in batches)
    if total <= 0 or output_weight.requires_grad:
        raise ValueError('Invalid batch or mutable frozen head')
    editor.train()
    optimizer.zero_grad(set_to_none=True)
    sums = dict(loss=0., conditional_teacher_kl=0., mean_original_candidate_mass=0.)
    inside = outside = 0
    for b in batches:
        n = int(b['valid'][:,2:].sum())
        if not n:
            continue
        if any(b[k].requires_grad for k in ('states','visible','base_scores','teacher_scores')):
            raise ValueError('Inputs must be frozen')
        with autocast():
            folded = editor.folded_weight(output_weight)
            scores = editor.folded_scores(b['states'],b['visible'],b['base_scores'][:,1:],folded)
            loss, m = objective(scores,b['teacher_scores'],b['base_scores'][:,1:],
                                b['labels'][:,2:],b['valid'][:,2:],arm=arm)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        (loss*(n/total)).backward()
        sums['loss'] += float(loss.detach())*n/total
        for name in ('conditional_teacher_kl','mean_original_candidate_mass'):
            sums[name] += m[name]*n/total
        inside += m['candidate_label_rows']
        outside += m['outside_candidate_label_rows']
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in editor.parameters()):
        raise ValueError('Nonfinite gradients')
    up_grad = float(editor.up.grad.detach().double().norm())
    coefficients = sum(float(p.grad.detach().double().square().sum()) for name,p in editor.named_parameters()
                       if name!='up' and p.grad is not None)**.5
    optimizer.step()
    return dict(**sums,valid_rows=total,candidate_label_rows=inside,outside_candidate_label_rows=outside,
                up_gradient_norm=up_grad,coefficient_gradient_norm=coefficients)
