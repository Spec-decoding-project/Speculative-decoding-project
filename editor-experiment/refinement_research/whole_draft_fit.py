"""Three fixed whole-proposal objectives; calibration selects, development reports."""
import argparse
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from autoregressive_editor.model import Config, TokenEditor
from copy_action_fit.data import CHECKPOINT_SHA256, digest, save, sha
from copy_action_fit.learning import inputs, evaluate, aggregate
from copy_action_fit.train import audit, require
from refinement_research.gate import GateEditor, generator_hash, generator_state, simulated_progress
from refinement_research.gate_data import extract, describe
from refinement_research.graph_replay import CapturedEditor, measure
from refinement_research.train_gates import calibration_metrics, progress
from refinement_research.whole_draft_data import populations

ARMS = ('uniform', 'early', 'protect_repair')
GATE_SHA = '5ab048b622beffb10ab8d282027cb2510f5439e4621793d5a82e687e696d679b'


def position_weights(original, labels, arm):
    if arm not in ARMS or original.ndim != 1 or original.shape != labels.shape:
        raise ValueError('Expected an objective and aligned suffix vectors')
    weight = torch.ones_like(labels, dtype=torch.float32)
    if arm != 'uniform':
        weight = torch.exp(-torch.arange(len(labels), device=labels.device).float()/7)
    if arm == 'protect_repair':
        correct_prefix = (original == labels).long().cumprod(0).bool()
        reachable = torch.cat((torch.ones(1, dtype=torch.bool, device=labels.device), correct_prefix[:-1]))
        first_error = reachable & ~correct_prefix
        weight = weight * torch.where(correct_prefix, 2., torch.where(first_error, 4., 1.))
    return weight


def rate(step, total=768):
    if step <= 16:
        return 1e-4*step/16
    return 1e-5+9e-5*(1+math.cos(math.pi*(step-16)/(total-16)))/2


def score(result):
    return (result['mean_edited_progress'], result['free_running_token_accuracy'])


def save_weights(path, model, **metadata):
    temporary = path.with_suffix('.tmp.pt')
    torch.save(dict(config=asdict(model.config), model=model.state_dict(), **metadata), temporary)
    temporary.replace(path)


def load_generator(path, device='cuda'):
    data = torch.load(path, map_location='cpu', weights_only=True)
    model = TokenEditor(Config(**data['config'])).to(device)
    model.load_state_dict(data['model'], strict=True)
    return model


def checks(model, rows):
    for row in rows:
        expected = model.propose(row['prefix'], row['draft'], 'draft')
        require(expected == model.propose(row['prefix'], row['draft'], 'draft', cached=False),
                'Cached/full output mismatch')
        ids, segments, start, _, _ = inputs(model, row)
        with torch.no_grad():
            original, _ = model(ids, segments, logit_start=start)
            altered = ids.clone(); altered[0,-1] = (altered[0,-1]+1)%model.config.vocab_size
            changed, _ = model(altered, segments, logit_start=start)
        require(torch.equal(original[:,:-1], changed[:,:-1]), 'Future label leaked through causal mask')


def checkpoint(path, model, optimizer, state, contract):
    save_weights(path, model, optimizer=optimizer.state_dict(), state=state,
                 contract=contract, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())


def train_generator(args, arm, training, calibration, contract, deadline):
    out = args.output/arm; out.mkdir(exist_ok=True)
    directory = args.checkpoints/arm; directory.mkdir(exist_ok=True)
    best, latest = directory/'generator-best.pt', directory/'generator-latest.pt'
    if (out/'generator-complete.json').exists():
        result = json.loads((out/'generator-complete.json').read_text())
        require(result['contract'] == contract and sha(best) == result['checkpoint_sha256'], 'Generator result changed')
        return best
    torch.manual_seed(0)
    model = load_generator(args.initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=True)
        require(saved['contract'] == contract, 'Resume contract mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        state = saved['state']; torch.set_rng_state(saved['rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved
    else:
        checks(model, calibration[:4])
        initial = evaluate(model, calibration, 'baseline')
        save(out/'calibration-0000.json', initial)
        state = dict(step=0, best_step=0, best_score=list(score(initial)))
        save_weights(best, model, selected_step=0, contract=contract)
        checkpoint(latest, model, optimizer, state, contract)
    require({id(p) for p in model.parameters()} ==
            {id(p) for g in optimizer.param_groups for p in g['params']}, 'Optimizer ownership mismatch')
    order = None; epoch = -1; durations = []
    with (out/f'updates-{time.time_ns()}.jsonl').open('w') as log:
        while state['step'] < 768:
            if time.time() > deadline-120:
                checkpoint(latest, model, optimizer, state, contract)
                raise TimeoutError('Experiment budget reached; checkpoint saved')
            current = state['step']//256
            if current != epoch:
                epoch = current
                order = torch.randperm(len(training), generator=torch.Generator().manual_seed(epoch)).tolist()
            batch = [training[i] for i in order[(state['step']%256)*32:(state['step']%256+1)*32]]
            denominator = sum(float(position_weights(torch.tensor(r['draft'][2:]),
                torch.tensor(r['labels'][2:]), arm).sum()) for r in batch)
            for group in optimizer.param_groups:
                group['lr'] = rate(state['step']+1)
            torch.cuda.synchronize(); began = time.perf_counter()
            model.train(); optimizer.zero_grad(set_to_none=True); loss_total = 0.
            for row in batch:
                ids, segments, start, labels, original = inputs(model, row)
                logits, _ = model(ids, segments, logit_start=start)
                loss = (F.cross_entropy(logits[0].float(), labels, reduction='none')
                        *position_weights(original, labels, arm)).sum()/denominator
                require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
                loss.backward(); loss_total += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); state['step'] += 1
            torch.cuda.synchronize(); seconds = time.perf_counter()-began; durations.append(seconds)
            log.write(json.dumps(dict(step=state['step'], weighted_ce=loss_total,
                gradient_norm=float(norm), seconds=seconds, lr=rate(state['step'])))+'\n'); log.flush()
            if state['step']%32 == 0:
                progress(args.output, phase='generator_training', arm=arm, step=state['step'],
                    maximum_updates=768, weighted_ce=loss_total,
                    recent_seconds_per_update=statistics.mean(durations[-32:]),
                    estimated_remaining_this_arm_update_seconds=(768-state['step'])*statistics.mean(durations[-32:]))
            if state['step']%128 == 0:
                result = evaluate(model, calibration, 'baseline')
                save(out/f"calibration-{state['step']:04d}.json", result)
                if score(result) > tuple(state['best_score']):
                    state.update(best_step=state['step'], best_score=list(score(result)))
                    save_weights(best, model, selected_step=state['step'], contract=contract)
                checkpoint(latest, model, optimizer, state, contract)
                progress(args.output, phase='generator_calibration', arm=arm, **state)
    del model, optimizer; gc.collect(); torch.cuda.empty_cache()
    selected = load_generator(best)
    restored = evaluate(selected, calibration, 'baseline')
    recorded = json.loads((out/f"calibration-{state['best_step']:04d}.json").read_text())
    require(restored == recorded, 'Reloaded generator changed calibration predictions')
    checks(selected, calibration[:8])
    save(out/'generator-complete.json', dict(contract=contract, **state,
        checkpoint_sha256=sha(best), generator_weights_sha256=generator_hash(selected),
        reload='passed', cached_full='passed', causal='passed'))
    del selected; gc.collect(); torch.cuda.empty_cache()
    return best


def gated_generator(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    model = GateEditor(Config(**payload['config']), 'linear').cuda()
    missing, extra = model.load_state_dict(payload['model'], strict=False)
    require(set(missing) == {'gate_head.weight', 'gate_head.bias'} and not extra, 'Gate/base alignment changed')
    return model


def fit_gate(args, arm, base, fit_rows, cal_rows, contract, deadline):
    out = args.output/arm
    selected_path = args.checkpoints/arm/'gate-best.pt'
    model = gated_generator(base); original_hash = generator_hash(model)
    if (out/'gate-complete.json').exists():
        result = json.loads((out/'gate-complete.json').read_text())
        require(result['contract'] == contract and result['base_sha256'] == sha(base), 'Changed gate base')
        require(result['checkpoint_sha256'] == sha(selected_path), 'Changed selected gate')
        model.gate_head.load_state_dict(torch.load(selected_path, weights_only=True)['gate'])
        return model
    def tick(value):
        require(time.time() < deadline-60, 'Gate feature budget exhausted')
        progress(args.output, phase='gate_features', arm=arm, **value)
    fit = extract(model, fit_rows, tick); cal = extract(model, cal_rows, tick)
    save(out/'gate-features.json', dict(fit=describe(fit), calibration=describe(cal)))
    eligible = fit['eligible']
    x, y, weights = [v[eligible].cuda() for v in (fit['features'], fit['targets'], fit['weights'])]
    cal = {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in cal.items()}
    require(len(x)>0, 'No meaningful gate training positions')
    optimizer = torch.optim.AdamW(model.gate_head.parameters(), lr=.001, weight_decay=.01)
    require({id(p) for p in model.parameters() if p.requires_grad} ==
        {id(p) for g in optimizer.param_groups for p in g['params']}, 'Gate optimizer includes generator')
    metric = calibration_metrics(model, cal)
    best_key = (metric['mean_edited_progress'], -metric['weighted_decision_ce'])
    best_epoch = best_step = step = 0; history = []
    torch.save(dict(gate=model.gate_head.state_dict(), selected_step=0), selected_path)
    for epoch in range(1,201):
        require(time.time()<deadline-60, 'Gate fitting budget exhausted')
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(epoch-1)).cuda()
        for indices in order.split(256):
            if step >= 2000:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = (F.binary_cross_entropy_with_logits(model.decision(x[indices]), y[indices],
                    reduction='none')*weights[indices]).mean()/weights.mean()
            require(bool(torch.isfinite(loss)), 'Nonfinite gate loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.gate_head.parameters(),1.,error_if_nonfinite=True)
            optimizer.step(); step += 1
        if epoch%5 == 0 or step>=2000:
            metric = calibration_metrics(model, cal)
            history.append(dict(epoch=epoch, step=step, **metric))
            key = (metric['mean_edited_progress'], -metric['weighted_decision_ce'])
            if key > best_key:
                best_key, best_epoch, best_step = key, epoch, step
                torch.save(dict(gate=model.gate_head.state_dict(), selected_step=step), selected_path)
            if epoch-best_epoch >= 50:
                break
        if step>=2000:
            break
    model.gate_head.load_state_dict(torch.load(selected_path, weights_only=True)['gate'])
    require(generator_hash(model)==original_hash, 'Gate fitting changed generator')
    actual = evaluate(model, cal_rows, 'copy')
    predicted = simulated_progress(model.decision(cal['features']),cal['original'],cal['generated'],cal['labels'])
    require(predicted.tolist()==[r['edited_progress'] for r in actual['rows']], 'Gate simulation mismatch')
    reload = gated_generator(base)
    reload.gate_head.load_state_dict(torch.load(selected_path, weights_only=True)['gate'])
    require(evaluate(reload,cal_rows,'copy')==actual, 'Gate reload mismatch')
    checks(reload,cal_rows[:8])
    save(out/'gate-calibration.json',actual)
    save(out/'gate-history.json',history)
    save(out/'gate-complete.json', dict(contract=contract, base_sha256=sha(base),
        checkpoint_sha256=sha(selected_path), updates=step, selected_step=best_step,
        generator_weights_sha256=original_hash, reload='passed', simulation='passed', cached_full='passed'))
    del model, optimizer, x, y, weights, fit, cal; gc.collect(); torch.cuda.empty_cache()
    return reload


def timing(model, rows):
    records = []
    for index in [i*(len(rows)-1)//15 for i in range(16)]:
        row = rows[index]; expected = model.propose(row['prefix'],row['draft'])
        torch.cuda.synchronize(); begin=time.perf_counter()
        graph=CapturedEditor(model,row['prefix'],row['draft'])
        torch.cuda.synchronize(); setup=1000*(time.perf_counter()-begin)
        changed=list(row['draft']); changed[3]=(changed[3]+1)%model.config.vocab_size
        require(graph.propose(row['prefix'],changed)==model.propose(row['prefix'],changed), 'Graph stale input')
        samples=[]
        for _ in range(5):
            actual,ms=measure(lambda:graph.propose(row['prefix'],row['draft']))
            require(actual==expected,'Graph/eager mismatch');samples.append(ms)
        records.append(dict(index=index,prefix_length=len(row['prefix']),setup_ms=setup,samples_ms=samples))
        del graph;gc.collect();torch.cuda.empty_cache()
    return dict(records=records,median_warm_ms=statistics.median(statistics.median(r['samples_ms']) for r in records),
                median_setup_ms=statistics.median(r['setup_ms'] for r in records),
                scope='Isolated graph editor with input/output copies; no native generation speed claim')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bank','initial','incumbent_gate','output','checkpoints'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);args.checkpoints.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(0)
    deadline=time.time()+110*60
    require(not (args.output/'complete.json').exists(),'Completed experiment must not be overwritten')
    require(sha(args.initial)==CHECKPOINT_SHA256 and sha(args.incumbent_gate)==GATE_SHA,'Initial checkpoint changed')
    training,gate_fit,calibration,development,metadata=populations(args.bank)
    contract=dict(schema='whole-draft-objectives-v1',arms=ARMS,metadata=metadata,
        initial_sha256=CHECKPOINT_SHA256,incumbent_gate_sha256=GATE_SHA,updates=768,batch=32,
        source_hashes={str(p.relative_to(Path(__file__).parents[1])):sha(p)
            for folder in ('autoregressive_editor','copy_action_fit','full_corpus_fit','refinement_research')
            for p in sorted((Path(__file__).parents[1]/folder).glob('*.py'))})
    identity=digest(contract)
    if (args.output/'run.json').exists():
        require(json.loads((args.output/'run.json').read_text())==json.loads(json.dumps(contract)), 'Run contract changed')
    else:
        save(args.output/'run.json',contract)
        save(args.output/'generator-fit-identities.json',[dict(prompt_id=r['prompt_id'],anchor=r['anchor']) for r in training])
    progress(args.output,phase='validated_inputs',contract=identity,**metadata)
    selected={}
    for arm in ARMS:
        selected[arm]=train_generator(args,arm,training,calibration,identity,deadline)
    # All generator checkpoint choices are fixed before development is consulted.
    results={}
    for arm in ARMS:
        model=fit_gate(args,arm,selected[arm],gate_fit,calibration,identity,deadline)
        plain=load_generator(selected[arm])
        for label,network,variant in [('plain',plain,'baseline'),('gated',model,'copy')]:
            result=evaluate(network,development,variant)
            save(args.output/arm/f'development-{label}.json',result)
            audit(result,development,variant,network.config.vocab_size)
            results[f'{arm}/{label}']=aggregate(result,development)
        del plain
        timed=timing(model,development);save(args.output/arm/'timing.json',timed)
        results[f'{arm}/gated']['median_warm_editor_ms']=timed['median_warm_ms']
        del model;gc.collect();torch.cuda.empty_cache()
    original=load_generator(args.initial)
    original_result=evaluate(original,development,'baseline')
    save(args.output/'original-generator-development.json',original_result)
    results['original_generator']=aggregate(original_result,development)
    del original;gc.collect();torch.cuda.empty_cache()
    incumbent=gated_generator(args.initial)
    incumbent.gate_head.load_state_dict(torch.load(args.incumbent_gate,map_location='cpu',weights_only=True)['gate'])
    old=evaluate(incumbent,development,'copy')
    require(old['mean_edited_progress']==3.69970703125 and old['free_running_token_accuracy']==0.3065708705357143,
            'Incumbent development replay changed')
    save(args.output/'incumbent-development.json',old);results['incumbent']=aggregate(old,development)
    save(args.output/'incumbent-timing.json',timing(incumbent,development))
    save(args.output/'complete.json',dict(passed_checks=True,contract=identity,results=results,
        original_token_accuracy=sum(sum(a==b for a,b in zip(r['draft'][2:],r['labels'][2:])) for r in development)/(2048*14),
        final_test_loaded=False,scope='Exploratory saved-context comparison; not native generation speed'))
    progress(args.output,phase='complete',contract=identity)


if __name__=='__main__':
    main()
