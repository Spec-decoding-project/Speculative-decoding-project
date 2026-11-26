"""Fixed, shorter edit budgets for all three frozen gate checkpoints."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from autoregressive_editor.learning import progress as correct_prefix
from copy_action_fit.data import CHECKPOINT_SHA256, SHARED_WEIGHTS_SHA256, digest, save, sha
from copy_action_fit.learning import aggregate
from copy_action_fit.train import require, timing, uncertainty
from full_corpus_fit.audit_results import prefix as independent_prefix
from refinement_research.gate import generator_hash
from refinement_research.gate_data import select
from refinement_research.train_gates import initialize, progress


def trim(saved, examples, limit):
    require(len(saved['rows'])==len(examples), 'Changed evaluation population')
    rows=[]
    for old, example in zip(saved['rows'],examples):
        require((old['prompt_id'],old['anchor'])==(example['prompt_id'],example['anchor'])
                and old['original_tokens']==example['draft'] and old['label_tokens']==example['labels'],
                'Saved output differs from immutable bank')
        row=dict(old)
        row['edited_tokens']=old['edited_tokens'][:2+limit]+old['original_tokens'][2+limit:]
        row['actions']=old['actions'][:limit]+[151936]*(14-limit)
        a=correct_prefix(row['original_tokens'],row['label_tokens'])
        b=correct_prefix(row['edited_tokens'],row['label_tokens'])
        require(a==independent_prefix(row['original_tokens'],row['label_tokens']) and
                b==independent_prefix(row['edited_tokens'],row['label_tokens']), 'Prefix recount differs')
        row.update(original_progress=a,edited_progress=b,union_progress=max(a,b),
                   repaired=b>a,damaged=b<a,exact_suffix=row['edited_tokens'][2:]==row['label_tokens'][2:])
        for position,action in enumerate(row['actions'],2):
            emitted=row['original_tokens'][position] if action==151936 else action
            require(emitted==row['edited_tokens'][position], 'Action emission differs')
        rows.append(row)
    result=dict(rows=rows,blocks=len(rows),repaired=sum(r['repaired'] for r in rows),
        damaged=sum(r['damaged'] for r in rows),wrong_first=sum(r['wrong_first'] for r in rows),
        mean_original_progress=statistics.mean(r['original_progress'] for r in rows),
        mean_edited_progress=statistics.mean(r['edited_progress'] for r in rows),
        mean_union_progress=statistics.mean(r['union_progress'] for r in rows),
        free_running_exact_suffix=sum(r['exact_suffix'] for r in rows)/len(rows),
        free_running_token_accuracy=sum(sum(a==b for a,b in zip(r['edited_tokens'][2:],r['label_tokens'][2:]))
                                        for r in rows)/(14*len(rows)),
        copy_actions=sum(a==151936 for r in rows for a in r['actions']),editable_tokens=14*len(rows))
    return result


class Limited:
    def __init__(self,model,limit):self.model,self.limit=model,limit
    def propose(self,prefix,draft,conditioning='draft',*,cached=True):
        return self.model.propose(prefix,draft,conditioning,cached=cached,max_edits=self.limit)


@torch.no_grad()
def check_actual(model,examples,result,limit):
    for example,row in zip(examples,result['rows']):
        tokens,actions=model.propose_actions(example['prefix'],example['draft'],max_edits=limit)
        require(tokens==row['edited_tokens'] and actions==row['actions'],
                'Short autoregressive execution differs from saved causal prefix')
    return dict(blocks=len(examples),tokens_and_actions='exact')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bank','initial','parent','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4);args.output.mkdir(parents=True,exist_ok=True)
    budget=args.output/'budget.json'
    if not budget.exists():save(budget,dict(started_epoch=time.time(),maximum_seconds=1200))
    deadline=json.loads(budget.read_text())['started_epoch']+1140
    parent=json.loads((args.parent/'complete.json').read_text())
    require(parent['passed_checks'] and not parent['final_test_loaded'], 'Parent run incomplete')
    require(sha(args.initial)==CHECKPOINT_SHA256,'Starting generator changed')
    _,calibration,development,selection=select(args.bank)
    require(selection==json.loads((args.parent/'selection.json').read_text()),'Selection changed')
    parent_files={str(p.relative_to(args.parent)):sha(p) for kind in ('scores','linear','mlp')
                  for p in [args.parent/kind/name for name in ('best.pt','calibration.json','development.json')]}
    contract=dict(schema='fixed-short-edits-v1',parent_complete_sha256=sha(args.parent/'complete.json'),
        parent_files=parent_files,selection=selection,kinds=['scores','linear','mlp'],budgets=[1,2,4,14],
        optimizer_updates=0,final_test_loaded=False,
        source_hashes={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    run=args.output/'run.json'
    require(not run.exists() or json.loads(run.read_text())==contract,'Run contract changed')
    save(run,contract);contract_hash=digest(contract)
    results={}
    for kind in ('scores','linear','mlp'):
        model=initialize(args.initial,kind)
        checkpoint=args.parent/kind/'best.pt'
        require(sha(checkpoint)==parent['arms'][kind]['checkpoint_sha256'],'Gate checkpoint changed')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        require(payload['generator_weights_sha256']==SHARED_WEIGHTS_SHA256 and
                payload['contract_sha256']==parent['arms'][kind]['contract_sha256'],'Gate provenance mismatch')
        model.gate_head.load_state_dict(payload['gate'],strict=True)
        saved={split:json.loads((args.parent/kind/(split+'.json')).read_text())
               for split in ('calibration','development')}
        for limit in (1,2,4,14):
            name=f'{kind}-{limit}';directory=args.output/name;directory.mkdir(exist_ok=True)
            if (directory/'complete.json').exists():
                done=json.loads((directory/'complete.json').read_text())
                require(done['contract_sha256']==contract_hash,'Completed budget provenance changed')
                results[name]=done;continue
            require(time.time()<deadline-45,'Insufficient remaining budget; no automatic extension')
            progress(args.output,phase='short_edit_evaluation',kind=kind,edit_limit=limit)
            metrics={};checks={};values={}
            for split,examples in (('calibration',calibration),('development',development)):
                result=trim(saved[split],examples,limit)
                if limit<14:checks[split]=check_actual(model,examples,result,limit)
                save(directory/(split+'.json'),result)
                metrics[split]=aggregate(result,examples);values[split]=result
            samples=[development[i*(len(development)-1)//63] for i in range(64)]
            timed=timing(Limited(model,limit),samples);save(directory/'timing.json',timed)
            original=[dict(r,edited_progress=r['original_progress']) for r in values['development']['rows']]
            interval=uncertainty(values['development']['rows'],original)
            delta=interval['mean_delta'];base=metrics['development']['mean_original_progress']
            required=timed['median_editor_ms']*(base+1)/delta if delta>0 else None
            require(generator_hash(model)==SHARED_WEIGHTS_SHA256,'Frozen generator changed')
            done=dict(kind=kind,edit_limit=limit,contract_sha256=contract_hash,checkpoint_sha256=sha(checkpoint),
                calibration=metrics['calibration'],development=metrics['development'],uncertainty=interval,
                actual_checks=checks,cached_full='passed on 64 development blocks',
                median_editor_ms=timed['median_editor_ms'],minimum_baseline_round_ms_for_break_even=required,
                screening_assumption='Same verification cost and prefix distribution, plus one target token per round',
                scope='Exploratory saved-prefix comparison and isolated FP32 timing; not measured serving speedup')
            save(directory/'complete.json',done);results[name]=done
        del model,payload;gc.collect();torch.cuda.empty_cache()
    require(all(sha(args.parent/name)==expected for name,expected in parent_files.items()),'Parent artifacts changed')
    save(args.output/'complete.json',dict(passed_checks=True,comparisons=results,optimizer_updates=0,
        final_test_loaded=False,elapsed_seconds=time.time()-json.loads(budget.read_text())['started_epoch']))
    progress(args.output,phase='complete',comparisons=len(results))


if __name__=='__main__':main()
