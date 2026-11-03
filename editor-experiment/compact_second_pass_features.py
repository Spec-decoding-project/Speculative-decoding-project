"""Frozen feature extraction for the controlled editor screen; no cloud calls."""
import torch
from compact_second_pass_bank import restore_compact_example


@torch.no_grad()
def extract_record(target, ordinary, teacher, bank, metadata, index):
    state, visible, scores, labels, valid=restore_compact_example(
        target,ordinary,bank,metadata,index)
    x1=scores[:,0].argmax(-1)
    actual_visible=target.get_input_embeddings()(x1)
    if not torch.equal(actual_visible,visible):raise ValueError('Mixed reconstruction and x1 inputs')
    hidden=teacher(state,actual_visible[:,None])
    if not torch.isfinite(hidden).all() or not torch.isfinite(state.normalized).all():
        raise ValueError('Nonfinite frozen features')
    return dict(prompt_id=metadata['id'],split=metadata['split'],anchor_index=index,
        states=state.normalized.detach().cpu(),teacher_states=hidden[:,2:].detach().cpu(),
        x1=x1.detach().cpu(),original=scores.argmax(-1).detach().cpu(),
        teacher_winners=target.get_output_embeddings()(hidden[:,2:]).argmax(-1).detach().cpu(),
        labels=labels.detach().cpu(),valid=valid.detach().cpu(),
        input_scope='Committed target features, masked first pass and its own x1; labels and teacher outputs are training-only')


@torch.no_grad()
def materialize(records, embedding, head, device):
    """Project each record with the frozen serving row shape before stacking.

    Do not combine rows across examples in the vocabulary GEMM: numerical
    changes could alter x1 or teacher eligibility. No full logits stored on disk.
    """
    if not records:raise ValueError('Empty feature batch')
    if any(p.requires_grad for m in (embedding,head) for p in m.parameters()):
        raise ValueError('Frozen embedding and head required')
    batches=[]
    for r in records:
        states=r['states'].to(device);teacher_states=r['teacher_states'].to(device)
        if states.ndim!=3 or states.shape[0]!=1 or teacher_states.shape!=(1,states.shape[1]-2,states.shape[2]):
            raise ValueError('Invalid feature rows')
        base=head(states[:,1:]);teacher_scores=head(teacher_states)
        if not torch.equal(teacher_scores.argmax(-1).cpu(),r['teacher_winners']):
            raise ValueError('Teacher readout reconstruction changed eligibility inputs')
        actual=base.argmax(-1)
        if not torch.equal(actual.cpu(),r['original']) or not torch.equal(actual[:,0].cpu(),r['x1']):
            raise ValueError('Readout reconstruction changed token decisions')
        batches.append(dict(states=states,visible=embedding(actual[:,0]),base_scores=base,
            teacher_scores=teacher_scores,labels=r['labels'].to(device),valid=r['valid'].to(device)))
    return {name:torch.cat([b[name] for b in batches],dim=0) for name in batches[0]}
