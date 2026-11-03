"""Matched support intervention; data-only scheduling and native teacher training.

Correct first tokens are supervised TRAINING inputs, never evaluation inputs.
No model loading or paid execution happens on import.
"""
import hashlib,math

ARMS=('old_support_feedback','all_support_feedback','all_support_no_feedback')

def make_schedule(metadata,seed=20261004):
    from conditional_prompt_batches import batches
    if any(m['split']!='train' for m in metadata):raise ValueError('Training split only')
    replacements={};denominators={};details=[]
    for pi,m in enumerate(metadata):
        eligible=[i for i,a in enumerate(m['anchors']) if a['valid'][1] and a['original'][1]==a['labels'][1]]
        if not eligible:raise ValueError('No within-prompt old-support control')
        denominators[pi]=sum(sum(math.exp(-(j-2)/7.) for j in range(2,sum(a['valid']))) for a in m['anchors'])
        for ai,a in enumerate(m['anchors']):
            n=sum(a['valid'])
            if n<3:raise ValueError('No editable training label')
            if ai in eligible:choice=ai
            else:
                candidates=[i for i in eligible if sum(m['anchors'][i]['valid'])>=n]
                if not candidates:raise ValueError('Cannot match supervision count within prompt')
                def key(i):return hashlib.sha256(f'{seed}:{m["id"]}:{ai}:{i}'.encode()).hexdigest()
                choice=min(candidates,key=key)
            replacements[pi,ai]=choice
            details.append(dict(prompt=m['id'],anchor_index=ai,old_control_index=choice,supervised_suffix=n-2))
    schedules={a:[] for a in ARMS}
    for batch in batches(metadata,0,seed,'mixed'):
        for arm in ARMS:
            schedules[arm].append([dict(prompt_index=pi,source_index=replacements[pi,ai] if arm==ARMS[0] else ai,
                original_index=ai,valid_length=sum(metadata[pi]['anchors'][ai]['valid'])) for pi,ai in batch])
    return dict(schedules=schedules,prompt_denominators=denominators,assignments=details)

def training_example(bank,meta,assignment,mask_token_id,device):
    import torch
    from conditional_redraft_model import make_anchored_example
    a=meta['anchors'][assignment['source_index']]
    # The native example slices context strictly before the anchor. No future
    # feature row is made available to the model.
    anchor=a['anchor'];ids=bank['ids'].to(device)
    features=bank['features'].to(device)
    ex=make_anchored_example(ids,features,anchor=anchor,block_size=16,mask_token_id=mask_token_id)
    if ex.labels[0].tolist()!=a['labels'] or ex.valid[0].tolist()!=a['valid']:
        raise ValueError('Recorded training labels changed')
    keep=assignment['valid_length']
    if not 3<=keep<=int(ex.valid.sum()):raise ValueError('Invalid matched supervision length')
    valid=ex.valid.clone();valid[:,keep:]=False
    return ex,valid

def suffix_loss(model,embedding,head,example,valid,*,feedback):
    import torch
    import torch.nn.functional as F
    ids=example.noise_ids.clone()
    if not bool((ids[:,1:]==model.mask_token_id).all()):raise ValueError('Future guesses must start masked')
    if feedback:ids[:,1]=example.labels[:,1]
    hidden=model(noise_embedding=embedding(ids).detach(),target_hidden=example.context_features.detach(),
                 position_ids=example.positions,is_causal=False)
    logits=head(hidden[:,2:])
    labels=example.labels[:,2:].masked_fill(~valid[:,2:],0)
    nll=F.cross_entropy(logits.float().transpose(1,2),labels,reduction='none')
    weights=torch.exp(-torch.arange(logits.shape[1],device=logits.device).float()/7.)[None]*valid[:,2:]
    if not torch.isfinite(nll).all():raise ValueError('Nonfinite suffix loss')
    return (nll*weights).sum(),weights.sum()
