import pytest
from refinement_research.native_best import aggregate, counts, select
from copy_action_fit.data import SOURCES


def test_visible_commit_count_excludes_prefill_and_terminal_overshoot():
    samples=[dict(speculative=False,sampled=0),dict(speculative=False,sampled=1),
             dict(speculative=True,sampled=4),dict(speculative=True,sampled=16)]
    result=counts(samples,8)
    assert result==dict(verification_rounds=2,verification_tokens=7,prefill_tokens=1,
                        committed_per_round=[4,3])
    with pytest.raises(ValueError):counts(samples,100)


def test_selection_balances_sources_and_lengths_without_train_or_test():
    prompts=[dict(id=f'{source}-{split}-{i}',source=source,split=split,prompt=[0]*(i+1))
             for source in SOURCES for split in ('train','development','test') for i in range(256)]
    chosen=select(prompts)
    assert chosen==select(list(reversed(prompts)))
    assert len(chosen)==128 and all(p['split']=='development' for p in chosen)
    for source in SOURCES:
        for q in ('Q1','Q2','Q3','Q4'):
            assert sum(p['source']==source and p['length_group']==q for p in chosen)==16


def test_economics_uses_actual_commits_and_all_editor_calls():
    rows=[]
    for mode,rounds,decode_ms,seconds,editor_ms in [('off',20,1000,1.2,[]),('editor',18,1080,1.3,[10.]*19)]:
        rows.append(dict(mode=mode,seconds=seconds,generated=dict(tokens=[0]*73),
            metrics=dict(verification_rounds=rounds,verification_tokens=72,decode_ms=decode_ms,prefill_ms=200),
            observed=dict(editor_ms=editor_ms,edit_calls=len(editor_ms),graph_hits=len(editor_ms),graph_misses=0)))
    result=aggregate(rows)
    assert result['arms']['off']['average_committed_per_round']==3.6
    assert result['arms']['editor']['average_committed_per_round']==4.
    assert result['editing_budget_ms_per_round']==pytest.approx(50*(4/3.6-1))
    assert result['measured_editor_ms_per_round']==pytest.approx(190/18)
    assert result['required_extra_committed_tokens_per_round']==pytest.approx(3.6*(190/18)/50)
    assert result['latency_increase_percent']==pytest.approx(100*(1.3/1.2-1))
