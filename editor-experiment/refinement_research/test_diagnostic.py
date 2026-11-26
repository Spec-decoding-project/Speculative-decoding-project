from copy import deepcopy
from types import MethodType

import pytest
import torch

from autoregressive_editor.model import Config
from copy_action_fit.model import CopyEditor
from refinement_research.diagnose_copy import probe, skipped_cases, vocabulary_only


def test_probe_feeds_recorded_past_and_never_current_reference():
    model = CopyEditor(Config(vocab_size=20, width=16, layers=1, heads=2, max_length=32))
    seen = []
    def scripted(self, ids, segments, **kwargs):
        seen.append(ids.tolist())
        scores = torch.zeros((1,1,21))
        scores[0,0,10] = 4
        scores[0,0,20] = 5
        return scores, None
    model.forward = MethodType(scripted, model)
    example = dict(prefix=[1,2], draft=[4,5,6,7], labels=[4,5,8,10])
    result = probe(model, example, [4,5,9,7], 3)
    assert seen[-1] == [[9]]  # The recorded token, not the teacher-forced 8.
    assert result['vocabulary_argmax'] == 10 and result['action_argmax'] == 20
    assert result['reference_rank'] == 1 and result['copy_over_reference'] == 1
    changed = deepcopy(example); changed['labels'][3] = 11
    assert probe(model, changed, [4,5,9,7], 3)['action_argmax'] == result['action_argmax']


def test_vocabulary_ablation_removes_only_the_copy_action():
    model = CopyEditor(Config(vocab_size=20, width=16, layers=1, heads=2, max_length=32))
    with torch.no_grad():
        for parameter in model.parameters(): parameter.zero_()
        model.copy_head.bias.fill_(100)
    plain = vocabulary_only(model)
    assert model.propose([1,2], [4,5,6,7]) == [4,5,6,7]
    assert plain.propose([1,2], [4,5,6,7], 'draft') == [4,5,0,0]
    assert all(torch.equal(value, plain.state_dict()[key]) for key,value in model.state_dict().items()
               if not key.startswith('copy_head.'))
    assert not any(p.requires_grad for p in plain.parameters())


def test_skipped_repair_selection_checks_identity_and_exact_prefix():
    labels = list(range(16)); original = labels.copy(); original[3] = 99
    common = dict(prompt_id='p', anchor=10, original_tokens=original, label_tokens=labels)
    control = dict(rows=[dict(common, repaired=True, edited_tokens=labels.copy())])
    copied = dict(rows=[dict(common, edited_progress=2, original_progress=2,
                            edited_tokens=original.copy(), actions=[151936]*14)])
    cases = skipped_cases(control, copied)
    assert len(cases) == 1 and cases[0]['position'] == 3
    changed = deepcopy(copied); changed['rows'][0]['edited_tokens'][2] = 99
    with pytest.raises(ValueError, match='preceding prefix'):
        skipped_cases(control, changed)
    changed = deepcopy(copied); changed['rows'][0]['prompt_id'] = 'other'
    with pytest.raises(ValueError, match='Unpaired'):
        skipped_cases(control, changed)
