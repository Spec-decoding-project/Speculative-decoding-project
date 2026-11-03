"""Full native references with dtype conversion that preserves numerical buffers."""
import torch
from conditional_full_reference import (
    ARMS, RawFeatureState, FullReferenceFirstPass, restore_reference_example,
    FullConditionalReference as OriginalFullConditionalReference,
)


class FullConditionalReference(OriginalFullConditionalReference):
    def _apply(self, fn, recurse=True):
        # HF initializes nonpersistent RoPE frequencies in FP32 even when the
        # pretrained weights are BF16. Casting these constants is not a weight
        # precision conversion: it changes the model's positional calculation.
        preserved = {name:value.detach().clone() for name,value in self.named_buffers()
                     if value.is_floating_point()}
        super()._apply(fn, recurse=recurse)
        current = dict(self.named_buffers())
        for name, value in preserved.items():
            path, _, leaf = name.rpartition('.')
            owner = self.get_submodule(path) if path else self
            owner._buffers[leaf] = value.to(device=current[name].device)
        return self


def make_full_references(ordinary, *, dtype=torch.float32):
    models = {arm:FullConditionalReference(ordinary,arm).to(dtype=dtype) for arm in ARMS}
    first = models[ARMS[0]].state_dict()
    assert all(torch.equal(v,models[ARMS[1]].state_dict()[k]) for k,v in first.items())
    owners = [{id(p) for p in m.parameters()} for m in models.values()]
    assert not owners[0]&owners[1]
    assert not any(ids&{id(p) for p in ordinary.parameters()} for ids in owners)
    assert all(p.requires_grad for m in models.values() for p in m.parameters())
    for model in models.values():
        check_numerical_buffers(ordinary,model)
    return models


def check_numerical_buffers(ordinary, model):
    original = dict(ordinary.drafter.named_buffers())
    current = dict(model.drafter.named_buffers())
    assert set(original)==set(current)
    rows=[]
    for name,value in original.items():
        other=current[name]
        exact=value.dtype==other.dtype and torch.equal(value,other)
        rows.append(dict(name=name,native_dtype=str(value.dtype),deployed_dtype=str(other.dtype),
                         exact=exact,elements=value.numel()))
    if not all(r['exact'] for r in rows):
        raise ValueError('Private numerical buffers differ from native model')
    return rows
