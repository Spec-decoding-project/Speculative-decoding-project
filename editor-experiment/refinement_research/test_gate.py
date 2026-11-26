from copy import deepcopy

import pytest
import torch

from autoregressive_editor.data import teacher_forcing
from autoregressive_editor.model import Config, TokenEditor
from refinement_research.gate import GateEditor, generator_hash, generator_state, simulated_progress, supervision


torch.set_num_threads(1)


def model(kind='linear'):
    return GateEditor(Config(vocab_size=20, width=16, layers=1, heads=2, max_length=32), kind)


@pytest.mark.parametrize('kind', ['scores', 'linear', 'mlp'])
def test_copy_initialization_generation_ablation_and_cache_alignment(kind):
    editor = model(kind)
    prefix, draft = [1,2], [4,5,6,7]
    assert editor.propose(prefix,draft) == editor.propose(prefix,draft,cached=False) == draft
    plain = TokenEditor(editor.config); plain.load_state_dict(generator_state(editor))
    with torch.no_grad():
        (editor.gate_head[-1] if kind == 'mlp' else editor.gate_head).bias.fill_(-100)
    assert editor.propose(prefix,draft) == plain.propose(prefix,draft,'draft')
    assert editor.propose(prefix,draft,cached=False) == plain.propose(prefix,draft,'draft')


def test_mixture_normalization_causal_mask_and_position_alignment():
    editor = model()
    row = dict(prefix=[1,2], draft=[4,5,6,7], labels=[4,5,8,9])
    ids, segments, start, _ = teacher_forcing(row, 'draft')
    x, s = torch.tensor([ids]), torch.tensor([segments])
    part = editor.readout(x,s,logit_start=start)
    assert part['source'].tolist() == [[6,7]]
    assert part['features'].shape == (1,2,24)
    logits,_ = editor(x,s,logit_start=start)
    torch.testing.assert_close(logits.exp().sum(-1),torch.ones(1,2))
    changed=x.clone();changed[0,-1]=19
    other,_ = editor(changed,s,logit_start=start)
    torch.testing.assert_close(logits[:,0],other[:,0],rtol=0,atol=0)


def test_supervision_stops_after_unrepairable_position_and_weights_prefix():
    labels=torch.tensor([[1,2,3,4,5]])
    original=torch.tensor([[1,9,9,4,5]])
    generated=torch.tensor([[9,2,8,4,9]])
    target,eligible,weight=supervision(original,generated,labels)
    assert eligible.tolist()==[[True,True,False,False,False]]
    assert weight.tolist()==[[2,1,0,0,0]]
    assert target[0,:2].tolist()==[1,0]
    assert simulated_progress(torch.tensor([[1.,-1.,1.,1.,1.]]),original,generated,labels).tolist()==[3]


def test_optimizer_updates_only_gate_and_can_learn_a_known_decision():
    torch.manual_seed(0)
    editor=model('scores');before=generator_hash(editor)
    features=torch.randn(128,24);target=(features[:,-8]>0).float()
    optimizer=torch.optim.AdamW(editor.gate_head.parameters(),lr=.05)
    for _ in range(100):
        optimizer.zero_grad()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(editor.decision(features),target)
        loss.backward();optimizer.step()
    assert ((editor.decision(features)>=0)==target.bool()).float().mean()>.95
    assert generator_hash(editor)==before
    assert all(p.grad is None and not p.requires_grad for name,p in editor.named_parameters()
               if not name.startswith('gate_head.'))


@pytest.mark.parametrize('kind', ['scores','linear','mlp'])
def test_limited_edits_match_full_causal_prefix_and_preserve_remainder(kind):
    torch.manual_seed(71)
    editor=GateEditor(Config(vocab_size=20,width=16,layers=1,heads=2,max_length=64),kind)
    with torch.no_grad():
        (editor.gate_head[-1] if kind=='mlp' else editor.gate_head).bias.fill_(-100)
    prefix,draft=[1,2],list(range(16))
    full,actions=editor.propose_actions(prefix,draft)
    for limit in (0,1,2,4,14):
        expected=full[:2+limit]+draft[2+limit:]
        result,limited_actions=editor.propose_actions(prefix,draft,max_edits=limit)
        assert result==expected==editor.propose(prefix,draft,max_edits=limit,cached=False)
        assert limited_actions==actions[:limit]+[20]*(14-limit)
    for invalid in (-1,15,True,1.5):
        with pytest.raises(ValueError):editor.propose(prefix,draft,max_edits=invalid)


def test_tensor_decode_needs_no_host_scalar_or_list_reads(monkeypatch):
    from autoregressive_editor.data import condition
    editor=model();prefix,draft=[1,2],[4,5,6,7]
    expected,expected_actions=editor.propose_actions(prefix,draft)
    ids,segments=condition(prefix,draft,'draft')
    def forbidden(*args,**kwargs):raise AssertionError('Host read inside device-only decode')
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor,'item',forbidden)
        patch.setattr(torch.Tensor,'tolist',forbidden)
        result,actions=editor.decode_tensors(torch.tensor([ids]),torch.tensor([segments]),
                                             torch.tensor([draft]),limit=2,record_actions=True)
    assert result.flatten().tolist()==expected and actions.flatten().tolist()==expected_actions
