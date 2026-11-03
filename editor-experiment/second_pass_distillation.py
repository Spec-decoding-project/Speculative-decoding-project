"""Offline second-pass distillation objective; no cloud or model loading.

Student receives only first-pass serving inputs. Teacher logits and true target
labels are training-only information. This objective does not certify speedup.
"""
import torch
import torch.nn.functional as F


def correction_distillation(student_logits, teacher_logits, labels, valid,
                            *, teacher_weight=0.5, temperature=1.0):
    """Target CE plus teacher KL where the teacher's winning token is correct.

    Target CE covers every valid position, including already-correct predictions.
    Correctness filtering prevents explicit imitation of teacher's wrong winners;
    it does not guarantee that soft probabilities or student updates are harmless.
    Filtering is training-only, never an inference-time oracle.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError('Student and teacher must use identical slots/vocabulary')
    if labels.shape != student_logits.shape[:-1] or valid.shape != labels.shape:
        raise ValueError('Labels and mask must align with logits')
    if valid.dtype != torch.bool or not valid.any():
        raise ValueError('A nonempty boolean supervision mask is required')
    if temperature <= 0 or not 0 <= teacher_weight <= 1:
        raise ValueError('Invalid temperature or loss weight')
    s = student_logits[valid].float()
    t = teacher_logits.detach()[valid].float()
    y = labels[valid]
    ce = F.cross_entropy(s, y)
    eligible = t.argmax(-1).eq(y)
    if eligible.any():
        kl = F.kl_div(F.log_softmax(s[eligible] / temperature, -1),
                      F.softmax(t[eligible] / temperature, -1),
                      reduction='batchmean') * temperature ** 2
        loss = (1 - teacher_weight) * ce + teacher_weight * kl
    else:
        kl = s.sum() * 0
        loss = ce
    return loss, {'ce': ce.detach(), 'teacher_kl': kl.detach(),
                  'teacher_eligible': int(eligible.sum()),
                  'valid_positions': int(valid.sum())}
