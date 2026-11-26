from collections import Counter
import pytest
from copy_action_fit.data import SOURCES
from refinement_research.broader_data import choose_broader


def examples():
    result=[]
    for source in SOURCES:
        for kind in ('preservation','repairable'):
            for i in range(6):
                labels=list(range(16));draft=labels.copy()
                if kind=='repairable':draft[2]=19
                identity=f'{source}/{kind}/{i}'
                result.append(dict(split='train',source=source,kind=kind,prompt_id=identity,family_id=identity,
                    anchor=2,context_start=0,prefix=[1,2],draft=draft,labels=labels,target_replay_verified=True))
    return result


def test_expanded_selection_excludes_calibration_prompts_and_families():
    rows=examples();cal=[rows[0],dict(prompt_id='other',family_id=rows[7]['family_id'])]
    selected=choose_broader(rows,cal,preservation=2,repairable=3)
    assert selected==choose_broader(list(reversed(rows)),cal,preservation=2,repairable=3)
    assert len(selected)==10
    assert not {r['prompt_id'] for r in selected}&{r['prompt_id'] for r in cal}
    assert not {r['family_id'] for r in selected}&{r['family_id'] for r in cal}
    assert Counter(r['kind'] for r in selected)=={'preservation':4,'repairable':6}


def test_insufficient_controls_fail_without_silent_replacement_or_label_changes():
    rows=examples()
    with pytest.raises(ValueError,match='Insufficient'):choose_broader(rows,[],preservation=7,repairable=3)
