"""Three frozen-generator gate fits, selected only on training-origin calibration."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from autoregressive_editor.model import Config, TokenEditor
from copy_action_fit.data import CHECKPOINT_SHA256, SHARED_WEIGHTS_SHA256, digest, save, sha
from copy_action_fit.learning import aggregate, evaluate
from copy_action_fit.train import audit, require, timing, uncertainty
from refinement_research.gate import GateEditor, generator_hash, generator_state, simulated_progress
from refinement_research.gate_data import describe, extract, select


def progress(output, **fields):
    value = dict(utc_epoch=time.time(), **fields)
    save(output/'progress.json', value)
    print(json.dumps(value), flush=True)


def initialize(path, kind):
    torch.manual_seed(0)
    model = GateEditor(Config(), kind)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(payload['model'], strict=False)
    require(set(missing) == {k for k in model.state_dict() if k.startswith('gate_head.')} and not unexpected,
            'Frozen generator state changed')
    del payload
    require(generator_hash(model) == SHARED_WEIGHTS_SHA256, 'Incorrect frozen generator initialization')
    return model.to('cuda').eval()


def move(data, device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in data.items()}


@torch.no_grad()
def integration_controls(model, rows):
    """Check full-size COPY and generator behavior before fitting any gate."""
    plain = TokenEditor(model.config).to(model.token.weight.device).eval()
    plain.load_state_dict(generator_state(model), strict=True)
    selected = [rows[i*(len(rows)-1)//15] for i in range(16)]
    for row in selected:
        require(model.propose(row['prefix'], row['draft']) == row['draft'],
                'Initial gate must preserve the entire proposal')
    model.gate_head.bias.fill_(-20)
    for row in selected:
        require(model.propose(row['prefix'], row['draft']) ==
                plain.propose(row['prefix'], row['draft'], 'draft'),
                'Forced-generation gate differs from original generator')
    model.gate_head.bias.zero_()
    del plain
    return dict(blocks=16, initial_copy='passed', forced_generator='passed')


@torch.no_grad()
def calibration_metrics(model, data):
    values = model.decision(data['features'])
    loss = (F.binary_cross_entropy_with_logits(values, data['targets'], reduction='none')*data['weights']).sum()/data['weights'].sum()
    prefixes = simulated_progress(values, data['original'], data['generated'], data['labels'])
    original = 1+(data['original'] == data['labels']).long().cumprod(-1).sum(-1)
    return dict(mean_original_progress=float(original.float().mean()),
                mean_edited_progress=float(prefixes.float().mean()), weighted_decision_ce=float(loss),
                repaired=int((prefixes > original).sum()), damaged=int((prefixes < original).sum()),
                copy_rate=float((values >= 0).float().mean()))


def save_checkpoint(path, model, optimizer, state, contract):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(dict(schema='frozen-generator-gate-v1', kind=model.kind, config=asdict(model.config),
        base_checkpoint_sha256=CHECKPOINT_SHA256, generator_weights_sha256=SHARED_WEIGHTS_SHA256,
        gate=model.gate_head.state_dict(), optimizer=optimizer.state_dict(),
        state=state, contract_sha256=contract), temporary)
    temporary.replace(path)


def restore(path, model, optimizer, contract):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(payload['schema'] == 'frozen-generator-gate-v1' and payload['kind'] == model.kind
            and payload['contract_sha256'] == contract
            and payload['config'] == asdict(model.config)
            and payload['base_checkpoint_sha256'] == CHECKPOINT_SHA256, 'Gate checkpoint provenance mismatch')
    model.gate_head.load_state_dict(payload['gate'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    return payload['state']


def fit(args, kind, fit_data, cal_data, cal_rows, development, contract, deadline):
    output = args.output/kind
    output.mkdir(exist_ok=True)
    if (output/'complete.json').exists():
        result = json.loads((output/'complete.json').read_text())
        require(result['contract_sha256'] == contract and sha(output/'best.pt') == result['checkpoint_sha256'],
                'Completed gate provenance changed')
        return result
    model = initialize(args.initial, kind)
    optimizer = torch.optim.AdamW(model.gate_head.parameters(), lr=.001, weight_decay=.01)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    require(trainable == {id(p) for group in optimizer.param_groups for p in group['params']},
            'Only gate parameters may be optimized')
    expected = dict(scores=9, linear=265, mlp=17025)[kind]
    require(sum(p.numel() for p in model.parameters() if p.requires_grad) == expected, 'Gate parameter count changed')
    latest, best = output/'latest.pt', output/'best.pt'
    if latest.exists():
        state = restore(latest, model, optimizer, contract)
    else:
        initial = calibration_metrics(model, cal_data)
        state = dict(step=0, epoch=0, cursor=0, selected_epoch=0, selected_step=0,
                     best_key=[initial['mean_edited_progress'], -initial['weighted_decision_ce']],
                     initial_calibration=initial, history=[], outcome=None)
        save_checkpoint(best,model,optimizer,state,contract)
        save_checkpoint(latest,model,optimizer,state,contract)
    eligible = fit_data['eligible']
    x, y, weight = fit_data['features'][eligible], fit_data['targets'][eligible], fit_data['weights'][eligible]
    require(len(x) > 0 and int((y == 0).sum()) >= 20, 'Insufficient meaningful generator-choice supervision')
    mean_weight = weight.mean()
    attempt = str(time.time_ns())
    started = time.monotonic()
    with (output/f'updates-{attempt}.jsonl').open('w') as log:
        while state['step'] < 2000 and state['epoch'] < 200 and state['outcome'] is None:
            epoch = state['epoch']
            order = torch.randperm(len(x), generator=torch.Generator().manual_seed(epoch)).to(x.device)
            while state['cursor'] < len(x):
                if state['step'] >= 2000 or time.time() >= deadline:
                    state['outcome'] = 'update_limit' if state['step'] >= 2000 else 'time_limit'
                    break
                indices = order[state['cursor']:state['cursor']+256]
                optimizer.zero_grad(set_to_none=True)
                values = model.decision(x[indices])
                loss = (F.binary_cross_entropy_with_logits(values,y[indices],reduction='none')*weight[indices]).mean()/mean_weight
                require(bool(torch.isfinite(loss)), 'Nonfinite gate loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.gate_head.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                state['step'] += 1; state['cursor'] += len(indices)
                log.write(json.dumps(dict(step=state['step'], epoch=epoch+1, loss=float(loss.detach()),
                                          gradient_norm=float(norm)))+'\n'); log.flush()
                if state['step'] % 25 == 0:
                    save_checkpoint(latest,model,optimizer,state,contract)
            if state['cursor'] == len(x):
                state['epoch'] += 1; state['cursor'] = 0
                if state['epoch'] % 5 == 0:
                    metric = calibration_metrics(model, cal_data)
                    key = [metric['mean_edited_progress'], -metric['weighted_decision_ce']]
                    state['history'].append(dict(epoch=state['epoch'], step=state['step'], **metric))
                    if key > state['best_key']:
                        state.update(best_key=key, selected_epoch=state['epoch'], selected_step=state['step'])
                        save_checkpoint(best,model,optimizer,state,contract)
                    elif state['epoch']-state['selected_epoch'] >= 50:
                        state['outcome'] = 'calibration_plateau'
                    progress(args.output, phase='gate_training', kind=kind, epoch=state['epoch'],
                             step=state['step'], selected_epoch=state['selected_epoch'], **metric)
                    save(output/'calibration-history.json',state['history'])
                save_checkpoint(latest,model,optimizer,state,contract)
    if state['outcome'] is None:
        state['outcome'] = 'update_limit' if state['step'] >= 2000 else 'epoch_limit'
    save_checkpoint(latest,model,optimizer,state,contract)
    final_state = dict(state)
    restore(best,model,optimizer,contract)
    selected = calibration_metrics(model,cal_data)
    require([selected['mean_edited_progress'], -selected['weighted_decision_ce']] == final_state['best_key'],
            'Selected checkpoint reload changed calibration metrics')
    require(generator_hash(model) == SHARED_WEIGHTS_SHA256, 'Gate optimization modified the generator')
    progress(args.output,phase='gate_evaluation',kind=kind,selected_epoch=state['selected_epoch'])
    actual_cal = evaluate(model,cal_rows,'copy')
    save(output/'calibration.json',actual_cal)
    simulated = simulated_progress(model.decision(cal_data['features']),cal_data['original'],
                                   cal_data['generated'],cal_data['labels']).tolist()
    require(simulated == [r['edited_progress'] for r in actual_cal['rows']],
            'Cached calibration simulation differs from actual autoregressive prefix decisions')
    result = evaluate(model,development,'copy')
    save(output/'development.json',result)
    save(output/'audit.json',dict(calibration=audit(actual_cal,cal_rows,'copy',151936),
                                  development=audit(result,development,'copy',151936)))
    timed = timing(model,[development[i*(len(development)-1)//63] for i in range(64)])
    save(output/'timing.json',timed)
    original = [dict(r,edited_progress=r['original_progress']) for r in result['rows']]
    interval = uncertainty(result['rows'],original)
    complete = dict(kind=kind, contract_sha256=contract, optimizer_updates=final_state['step'],
        epochs=final_state['epoch'], outcome=final_state['outcome'],
        selected_epoch=final_state['selected_epoch'], selected_step=final_state['selected_step'],
        trainable_parameters=expected, frozen_parameters=40750336,
        generator_weights_sha256=generator_hash(model), checkpoint_sha256=sha(best),
        checkpoint_reload='passed', calibration_simulation='all 512 actual prefixes match',
        cached_full='passed on 64 development blocks', audit='passed',
        training_prefix_simulation=calibration_metrics(model,fit_data),
        calibration=selected, development=aggregate(result,development), uncertainty=interval,
        median_editor_ms=timed['median_editor_ms'], elapsed_seconds=time.monotonic()-started,
        scope='Exploratory development transfer and isolated FP32 latency; no generation speedup claim')
    save(output/'complete.json',complete)
    del model,optimizer,x,y,weight
    gc.collect();torch.cuda.empty_cache()
    return complete


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bank','initial','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--broader-from',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=True)
    require(sha(args.initial) == CHECKPOINT_SHA256,'Starting generator checkpoint changed')
    budget=args.output/'budget.json'
    if not budget.exists():save(budget,dict(started_epoch=time.time(),maximum_seconds=7200))
    deadline=json.loads(budget.read_text())['started_epoch']+6900
    if args.broader_from:
        from refinement_research.broader_data import select_broader
        fit_rows,cal_rows,development,selection=select_broader(args.bank,args.broader_from)
    else:
        fit_rows,cal_rows,development,selection=select(args.bank)
    save(args.output/'selection.json',selection)
    save(args.output/'fit-examples.json',fit_rows);save(args.output/'calibration-examples.json',cal_rows)
    root=Path(__file__).parent
    contract=dict(schema='frozen-generator-gates-v1',selection=selection,
        initial_checkpoint_sha256=CHECKPOINT_SHA256,generator_weights_sha256=SHARED_WEIGHTS_SHA256,
        kinds=['scores','linear','mlp'],seed=0,lr=.001,weight_decay=.01,clip_norm=1,batch_size=256,
        max_updates=2000,max_epochs=200,calibration_interval_epochs=5,patience_epochs=50,
        objective='Reachable exclusive candidate choices, weighted by remaining attainable prefix',
        group_balancing=bool(args.broader_from),
        selection_rule='Maximum internal-calibration mean prefix, then minimum weighted decision CE',
        source_hashes={p.name:sha(p) for p in sorted(root.glob('*.py'))},
        torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        final_test_loaded=False)
    path=args.output/'run.json'
    require(not path.exists() or json.loads(path.read_text()) == contract,'Changed run contract on resume')
    save(path,contract);contract_hash=digest(contract)
    cache=args.output/'features.pt'
    if cache.exists():
        datasets=torch.load(cache,map_location='cpu',weights_only=True)
        require(datasets['contract_sha256'] == contract_hash,'Feature cache provenance changed')
    else:
        model=initialize(args.initial,'scores')
        save(args.output/'integration-controls.json',integration_controls(model,cal_rows))
        datasets=dict(contract_sha256=contract_hash)
        for split,rows in (('fit',fit_rows),('calibration',cal_rows)):
            progress(args.output,phase='feature_extraction',split=split,blocks=len(rows))
            datasets[split]=extract(model,rows,lambda p:progress(args.output,phase='feature_extraction',split=split,**p))
            if split=='fit' and args.broader_from:
                factors=torch.tensor([selection['group_weights'][r['kind']+'/'+r['source']] for r in rows])
                datasets[split]['weights']*=factors[:,None]
        require(generator_hash(model) == SHARED_WEIGHTS_SHA256,'Feature extraction changed generator')
        temporary=cache.with_suffix('.pt.tmp');torch.save(datasets,temporary);temporary.replace(cache)
        del model;gc.collect();torch.cuda.empty_cache()
    save(args.output/'feature-summary.json',{k:describe(datasets[k]) for k in ('fit','calibration')})
    fit_data,cal_data=move(datasets['fit'],'cuda'),move(datasets['calibration'],'cuda')
    del datasets
    results={}
    for kind in ('scores','linear','mlp'):
        require(time.time()<deadline-180,'Insufficient remaining evaluation budget; no silent extension')
        results[kind]=fit(args,kind,fit_data,cal_data,cal_rows,development,contract_hash,deadline-180)
    require(sha(args.initial) == CHECKPOINT_SHA256,'Input checkpoint changed')
    save(args.output/'complete.json',dict(passed_checks=True,arms=results,final_test_loaded=False,
         elapsed_seconds=time.time()-json.loads(budget.read_text())['started_epoch'],
         scope='Three predeclared exploratory comparisons; fresh confirmation required for a success claim'))
    progress(args.output,phase='complete',prefixes={k:r['development']['mean_edited_progress'] for k,r in results.items()})


if __name__ == '__main__':
    main()
