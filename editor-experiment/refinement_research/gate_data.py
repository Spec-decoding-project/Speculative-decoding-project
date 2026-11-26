"""Authentic split selection and cached, teacher-forced editor-only features."""
from collections import defaultdict
import json

import torch

from autoregressive_editor.data import condition
from copy_action_fit.data import digest, read_bank
from refinement_research.gate import supervision


def select(path):
    training, development, metadata = read_bank(path, per_group=512)
    groups = defaultdict(list)
    for row in training:
        groups[row['kind'],row['source']].append(row)
    fit, calibration = [], []
    for rows in groups.values():
        if len(rows) != 512:
            raise ValueError('Expected four groups of 512 distinct authentic blocks')
        fit.extend(rows[:384]); calibration.extend(rows[384:])
    if len(fit) != 1536 or len(calibration) != 512:
        raise ValueError('Unexpected optimization/calibration populations')
    for field in ('prompt_id','family_id'):
        if {r[field] for r in fit} & {r[field] for r in calibration}:
            raise ValueError('Gate calibration overlaps optimization')
    metadata.update(fit_blocks=1536, calibration_blocks=512,
                    fit_sha256=digest(fit), calibration_sha256=digest(calibration))
    return fit, calibration, development, metadata


@torch.no_grad()
def extract(model, rows, progress=None):
    """Cached one-position execution matches the actual autoregressive readout."""
    features, generated = [], []
    device = model.token.weight.device
    for index, row in enumerate(rows):
        ids, segments = condition(row['prefix'], row['draft'], 'draft')
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        types = torch.tensor([segments], dtype=torch.long, device=device)
        source = torch.tensor([row['draft']], dtype=torch.long, device=device)
        labels = torch.tensor([row['labels']], dtype=torch.long, device=device)
        past = dict(kv=None, original=source, seed=len(ids)-1)
        feature_parts, generated_parts = [], []
        for position in range(2,16):
            result = model.readout(tokens, types, past=past, use_cache=True,
                                  logit_start=tokens.shape[1]-1)
            feature_parts.append(result['features'][0,-1])
            generated_parts.append(result['best'][0])
            # Only the previous correct token becomes the next decoder input.
            tokens, types, past = labels[:,position:position+1], torch.full_like(tokens[:,:1],2), result['cache']
        features.append(torch.stack(feature_parts).cpu())
        generated.append(torch.cat(generated_parts).cpu())
        if progress and (index+1) % 128 == 0:
            progress(dict(extracted=index+1, blocks=len(rows)))
    original = torch.tensor([r['draft'][2:] for r in rows])
    labels = torch.tensor([r['labels'][2:] for r in rows])
    generated = torch.stack(generated)
    targets, eligible, weights = supervision(original, generated, labels)
    return dict(features=torch.stack(features), generated=generated, original=original,
                labels=labels, targets=targets, eligible=eligible, weights=weights,
                rows_sha256=digest(rows))


def describe(data):
    eligible = data['eligible']
    return dict(blocks=data['features'].shape[0], positions=eligible.numel(),
        eligible_decisions=int(eligible.sum()), copy_targets=int((eligible & data['targets'].bool()).sum()),
        generate_targets=int((eligible & ~data['targets'].bool()).sum()),
        feature_dimensions=data['features'].shape[-1], rows_sha256=data['rows_sha256'])
