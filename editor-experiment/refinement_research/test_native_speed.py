import pytest

from refinement_research.native_speed import compare, select, summarize
from refinement_research.native_speed_worker import editor_input
from refinement_research.editor_kernels import editor_kernels


def test_native_anchor_is_excluded_from_context_and_preserved_in_draft():
    context, draft = editor_input(list(range(1100)), list(range(15)))
    assert context == list(range(75,1099))
    assert draft == [1099,*range(15)]
    short, draft = editor_input([4,5,6], list(range(15)))
    assert short == [4,5] and draft[:2] == [6,0]
    with pytest.raises(ValueError): editor_input([1],list(range(15)))


def test_prompt_selection_has_fixed_strata_and_excludes_training_and_final_test():
    rows = [dict(id=f'{split}-{length}-{i}',split=split,prompt=[1]*length)
            for split in ('train','development','test') for length in (20,1200) for i in range(8)]
    chosen = select(rows)
    assert select(list(reversed(rows))) == chosen
    assert len(chosen)==8 and all(p['split']=='development' for p in chosen)
    assert [p['length_group'] for p in chosen] == ['short']*4+['long']*4


@pytest.mark.parametrize('field,value',[('prompt',[2]),('tokens',[4]),('finish_reason','stop')])
def test_correctness_gate_rejects_token_and_termination_changes(field,value):
    original=dict(prompt=[1],tokens=[3],finish_reason='length')
    compare(original,dict(original,request_id='different_request'))
    with pytest.raises(ValueError):compare(original,dict(original,**{field:value}))


def test_summary_uses_paired_prompt_medians_and_reports_slowdown():
    rows=[]
    for group in ('short','long'):
        for mode,seconds in [('off',2.),('editor',3.)]:
            for repeat in range(5):
                rows.append(dict(phase='timing',length_group=group,prompt_id=group,mode=mode,
                    precision='fp32',
                    seconds=seconds if repeat<4 else 99.,generated=dict(tokens=[1]*64),
                    observed=dict(rounds=[{}]*10,editor_ms=[6.7],peak_allocated_bytes=100)))
    result=summarize(rows)
    assert result['groups']['all']['speedup']==pytest.approx(2/3)
    assert result['groups']['short']['total_median_seconds']==dict(off=2.,editor=3.)


def test_scoped_dispatch_restores_target_after_editor_exception():
    class Library:
        destroyed = False
        def _destroy(self):self.destroyed=True
    class Invariant:
        _batch_invariant_MODE = True
        _batch_invariant_LIB = Library()
        def enable_batch_invariant_mode(self):
            assert not self._batch_invariant_MODE
            self._batch_invariant_MODE=True
            self._batch_invariant_LIB=Library()
    state=Invariant();old=state._batch_invariant_LIB
    with pytest.raises(ValueError,match='editor failure'):
        with editor_kernels('standard',state):
            assert old.destroyed and not state._batch_invariant_MODE
            raise ValueError('editor failure')
    assert state._batch_invariant_MODE and state._batch_invariant_LIB is not old
    with editor_kernels('invariant',state):assert state._batch_invariant_MODE
