from copy import deepcopy
import math
from types import MethodType

import pytest
import torch

from autoregressive_editor.learning import Trainer
from autoregressive_editor.model import Config
from copy_action_fit.data import SOURCES, choose, sha
from copy_action_fit.learning import PilotTrainer, evaluate, inputs, passed, weights_hash
from copy_action_fit.model import CopyEditor, losses


torch.set_num_threads(1)


def config():
    return Config(vocab_size=20, width=16, layers=1, heads=2, max_length=32)


def rows():
    return [dict(prompt_id='p0', anchor=2, prefix=[1,2], draft=[4,5,6,7],
                 labels=[4,5,6,7], kind='preservation', source=SOURCES[0]),
            dict(prompt_id='p1', anchor=2, prefix=[1,3], draft=[4,5,12,13],
                 labels=[4,5,8,9], kind='repairable', source=SOURCES[1])]


def test_induced_token_probability_adds_copy_and_vocabulary_mass():
    logits = torch.tensor([[.1,.2,.3,.4],[.1,.2,.3,.4]]).log().requires_grad_()
    action, token = losses(logits, torch.tensor([1,2]), torch.tensor([1,1]), copy_enabled=True)
    assert float(action.detach()) == pytest.approx(-math.log(.4)-math.log(.3), abs=1e-6)
    assert float(token.detach()) == pytest.approx(-math.log(.6)-math.log(.3), abs=1e-6)
    action.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('cached', [False, True])
def test_copy_action_feeds_the_actual_token_to_the_next_step(cached):
    model=CopyEditor(config())
    observed=[]

    def scripted(self, ids, segments, **kwargs):
        observed.append(ids.clone())
        logits=torch.full((1,1,self.config.vocab_size+1),-100.)
        logits[0,0,self.config.vocab_size if len(observed)==1 else 9]=100.
        return logits, ('synthetic-cache',) if kwargs['use_cache'] else None

    model.forward=MethodType(scripted,model)
    proposal,actions=model.propose_actions([1,2],[4,5,6,7],cached=cached)
    assert actions==[20,9] and proposal==[4,5,6,9]
    assert int(observed[1][0,-1])==6
    assert all(int(ids.max()) < 20 for ids in observed)


def test_real_copy_cache_matches_full_forward_and_always_copy_fails_repair_gate():
    model=CopyEditor(config())
    with torch.no_grad(): model.copy_head.bias.fill_(100.)
    for row in rows():
        assert model.propose(row['prefix'],row['draft']) == row['draft']
        assert model.propose(row['prefix'],row['draft'],cached=False) == row['draft']
    result=evaluate(model,rows(),'copy')
    assert result['rows'][0]['exact_suffix'] is True
    assert result['rows'][1]['exact_suffix'] is False
    assert result['copy_actions']==4 and not passed(result)


def test_copy_logits_respect_future_label_mask():
    model=CopyEditor(config())
    a,b=deepcopy(rows()[1]),deepcopy(rows()[1])
    b['labels'][2]=19
    x,s,start,_,_=inputs(model,a); y,t,_,_,_=inputs(model,b)
    left,_=model(x,s,logit_start=start); right,_=model(y,t,logit_start=start)
    torch.testing.assert_close(left[:,0],right[:,0],rtol=0,atol=0)


def test_warm_start_preserves_shared_weights_resets_optimizer_and_trains_copy_head(tmp_path):
    old=Trainer(config(),'draft',seed=2,lr=.01)
    old.update(rows())
    path=tmp_path/'original.pt';old.save(path,{'original':True})
    before=sha(path)
    new=PilotTrainer(config(),'copy')
    assert new.warm_start(path)==1
    assert weights_hash(new.model,shared=True)==weights_hash(old.model)
    assert not new.optimizer.state and new.step==0
    assert torch.count_nonzero(new.model.copy_head.weight)==0
    assert sum(p.numel() for p in new.model.parameters())-sum(p.numel() for p in old.model.parameters())==17
    result=new.update(rows())
    assert result['copy_head_gradient_norm'] > 0
    assert torch.count_nonzero(new.model.copy_head.weight)>0
    assert weights_hash(new.model,shared=True)!=weights_hash(old.model)
    assert sha(path)==before


@pytest.mark.parametrize('variant',['baseline','copy'])
def test_mixed_repair_and_preservation_fit_and_exact_checkpoint_resume(variant,tmp_path):
    trainer=PilotTrainer(config(),variant,lr=.02,seed=3)
    for _ in range(100): trainer.update(rows())
    result=evaluate(trainer.model,rows(),variant)
    assert passed(result) and result['damaged']==0
    path=tmp_path/'checkpoint.pt'
    trainer.save(path,{'consecutive':1,'run_sha256':'fixture'})
    restored,state=PilotTrainer.load(path)
    assert state=={'consecutive':1,'run_sha256':'fixture'}
    assert evaluate(restored.model,rows(),variant)==result
    for obj in (trainer,restored): obj.update(rows())
    assert all(torch.equal(value,restored.model.state_dict()[name])
               for name,value in trainer.model.state_dict().items())


def native_rows():
    result=[]
    for kind in ('preservation','repairable'):
        for source in SOURCES:
            for i in range(3):
                identity=f'{kind}-{source}-{i}'
                labels=list(range(16)); draft=labels.copy()
                if kind=='repairable': draft[2]=18
                result.append(dict(prompt_id=identity,family_id=identity,source=source,
                    kind=kind,split='train',target_replay_verified=True,anchor=102,
                    context_start=100,prefix=[1,2],draft=draft,labels=labels))
    return result


def test_native_selection_is_balanced_distinct_and_order_independent():
    candidates=native_rows()
    a=choose(candidates,per_group=1)
    assert choose(reversed(candidates),per_group=1)==a
    assert len(a)==4 and len({r['family_id'] for r in a})==4
    assert {r['kind'] for r in a}=={'repairable','preservation'}
    assert {r['source'] for r in a}==set(SOURCES)


def test_selection_rejects_heldout_rows_and_silent_shortfall():
    candidates=native_rows();candidates[0]['split']='development'
    with pytest.raises(ValueError,match='Only training'): choose(candidates,per_group=1)
    with pytest.raises(ValueError,match='Insufficient'): choose(native_rows()[:3],per_group=1)


def test_paired_uncertainty_keeps_prompt_blocks_together():
    from copy_action_fit.train import uncertainty
    left=[dict(prompt_id=prompt,anchor=i,edited_progress=delta)
          for prompt,delta in [('a',2),('b',4)] for i in range(4)]
    right=[dict(row,edited_progress=0) for row in left]
    result=uncertainty(left,right,draws=1000)
    assert result['prompts']==2 and result['mean_delta']==3
    assert result['ci95']==[2,4]
    assert uncertainty(left,left,draws=100)['ci95']==[0,0]
    with pytest.raises(ValueError,match='identities'):
        uncertainty(left,list(reversed(right)),draws=10)


def test_action_audit_rejects_false_copy_trace():
    from copy_action_fit.train import audit
    from full_corpus_fit.test_results import SavedPredictionAudit
    fixture=SavedPredictionAudit();fixture.setUp()
    result=fixture.result
    result['rows'][0]['actions']=fixture.examples[0]['labels'][2:]
    result['copy_actions']=0
    assert audit(result,fixture.examples,'copy',151936)['passed']
    result['rows'][0]['actions'][0]=151936
    result['copy_actions']=1
    with pytest.raises(ValueError,match='wrong token'):
        audit(result,fixture.examples,'copy',151936)
