from refinement_research.short_edits import trim


def test_short_edit_can_keep_a_repair_without_a_later_destructive_edit():
    labels=list(range(16));draft=labels.copy();draft[2]=19
    full=labels.copy();full[3]=19
    example=dict(prompt_id='fixture',anchor=0,draft=draft,labels=labels,kind='repairable')
    saved=dict(rows=[dict(prompt_id='fixture',anchor=0,original_tokens=draft,label_tokens=labels,
                         edited_tokens=full,actions=full[2:],kind='repairable',wrong_first=False)])
    original=trim(saved,[example],0);short=trim(saved,[example],1);long=trim(saved,[example],14)
    assert original['mean_edited_progress']==1
    assert short['mean_edited_progress']==15 and short['repaired']==1 and short['damaged']==0
    assert short['rows'][0]['actions']==[2]+[151936]*13
    assert long['mean_edited_progress']==2
