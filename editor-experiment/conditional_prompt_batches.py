"""Frozen grouped and mixed schedules; no models, downloads, or execution."""
import math
import random


def batches(metadata, epoch, seed, mode, size=32):
    if mode not in ('grouped', 'mixed') or size != 32:
        raise ValueError('This comparison fixes two schedules and 32 anchors/update')
    if len(metadata) % size or any(len(m['anchors']) != size for m in metadata):
        raise ValueError('Require equal 32-anchor responses and complete mixed batches')
    rng = random.Random(seed + 1009 * epoch)
    order = list(range(len(metadata)))
    rng.shuffle(order)
    anchor_orders = {}
    for pi in order:
        anchor_orders[pi] = list(range(size))
        rng.shuffle(anchor_orders[pi])
    if mode == 'grouped':
        return [[(pi, ai) for ai in anchor_orders[pi]] for pi in order]
    result = []
    for round_index in range(size):
        prompt_order = list(order)
        rng.shuffle(prompt_order)
        for start in range(0, len(prompt_order), size):
            result.append([(pi, anchor_orders[pi][round_index]) for pi in prompt_order[start:start + size]])
    return result


def prompt_denominators(metadata, *, cut=1, decay=7.0):
    """Mathematical equal-prompt normalizers; inference never sees this mask."""
    result = {}
    for pi, m in enumerate(metadata):
        value = 0.0
        for d in m['anchors']:
            matches = all(d['original'][j] == d['labels'][j] for j in range(1, cut + 1))
            if matches:
                value += sum(math.exp(-(j - cut - 1) / decay)
                    for j, valid in enumerate(d['valid']) if valid and j > cut)
        if not value > 0 or not math.isfinite(value):
            raise ValueError('A response has no finite conditional supervision')
        result[pi] = value
    return result


def signature(batch, metadata):
    return [(metadata[pi]['id'], ai) for pi, ai in batch]
