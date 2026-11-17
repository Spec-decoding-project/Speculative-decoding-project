"""Diagnostic: use single-row linear calls only in one native verification block.

This does not change attention, weights, dtype, prefill or acceptance logic.
It is deliberately opt-in and is not a validated serving implementation.
"""
from collections import Counter
from contextlib import contextmanager

import torch


@contextmanager
def single_row_linears(target, block_size=16):
    state = dict(active=False, target_calls=0, split_modules=Counter(), linear_calls=0)
    originals = []
    handles = []

    def before(module, args, kwargs):
        state['target_calls'] += 1
        state['active'] = state['target_calls'] == 2
        if state['target_calls'] > 2:
            raise ValueError('This diagnostic supports exactly one native round')
        if state['active']:
            ids = args[0] if args else kwargs['input_ids']
            cache = kwargs.get('past_key_values')
            if ids.shape != (1, block_size) or cache is None or cache.get_seq_length() <= 0:
                raise ValueError('Expected one cached native verification block')

    def after(module, args, kwargs, output):
        state['active'] = False

    try:
        for name, module in target.named_modules():
            if isinstance(module, torch.nn.Linear):
                method = module.forward
                originals.append((module, 'forward' in module.__dict__, method))

                def forward(x, original=method, key=name):
                    if not state['active']:
                        return original(x)
                    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != block_size:
                        raise ValueError(f'Unsupported linear input shape at {key}: {tuple(x.shape)}')
                    state['split_modules'][key] += 1
                    state['linear_calls'] += block_size
                    # clone gives the same contiguous [1, 1, width] layout as
                    # cached decoding, rather than retaining a block view stride.
                    return torch.cat([
                        original(x[:, i:i+1].clone(memory_format=torch.contiguous_format))
                        for i in range(block_size)
                    ], dim=1)

                module.forward = forward
        handles.append(target.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(target.register_forward_hook(after, with_kwargs=True, always_call=True))
        yield state
    finally:
        state['active'] = False
        for handle in handles:
            handle.remove()
        for module, had_override, method in originals:
            if had_override:
                module.forward = method
            else:
                del module.forward
