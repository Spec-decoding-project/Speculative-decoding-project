import torch

from autoregressive_editor.model import Config
from copy_action_fit.learning import progress
from refinement_research.gate import GateEditor, generator_state, simulated_progress
from refinement_research.gate_data import extract
from refinement_research.train_gates import restore, save_checkpoint


def test_cached_teacher_prefixes_predict_actual_autoregressive_prefix_length():
    torch.manual_seed(11)
    model=GateEditor(Config(vocab_size=20,width=16,layers=1,heads=2,max_length=64),'linear')
    labels=list(range(16));draft=labels.copy();draft[4]=19
    rows=[dict(prefix=[1,2],draft=draft,labels=labels),
          dict(prefix=[3,4],draft=labels,labels=labels)]
    data=extract(model,rows)
    for bias in (-100.,0.,100.):
        with torch.no_grad():model.gate_head.bias.fill_(bias)
        predicted=simulated_progress(model.decision(data['features']),data['original'],
                                     data['generated'],data['labels']).tolist()
        actual=[progress(model.propose(r['prefix'],r['draft']),r['labels']) for r in rows]
        assert predicted==actual


def test_gate_checkpoint_restores_predictions_and_next_optimizer_update(tmp_path):
    torch.manual_seed(12)
    config=Config(vocab_size=20,width=16,layers=1,heads=2,max_length=32)
    original=GateEditor(config,'linear')
    optimizer=torch.optim.AdamW(original.gate_head.parameters(),lr=.001,weight_decay=.01)
    features=torch.randn(32,24);labels=(features[:,0]>0).float()
    def update(model,opt):
        opt.zero_grad();torch.nn.functional.binary_cross_entropy_with_logits(model.decision(features),labels).backward();opt.step()
    update(original,optimizer)
    path=tmp_path/'gate.pt';state=dict(step=1,epoch=0,cursor=32)
    save_checkpoint(path,original,optimizer,state,'fixture')
    restored=GateEditor(config,'linear')
    restored.load_state_dict(generator_state(original),strict=False)
    other=torch.optim.AdamW(restored.gate_head.parameters(),lr=.001,weight_decay=.01)
    assert restore(path,restored,other,'fixture')==state
    assert torch.equal(original.decision(features),restored.decision(features))
    update(original,optimizer);update(restored,other)
    assert all(torch.equal(value,restored.gate_head.state_dict()[name])
               for name,value in original.gate_head.state_dict().items())
