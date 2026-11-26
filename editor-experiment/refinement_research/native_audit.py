"""Independently recount completed native-pilot artifacts; emit no prompt IDs."""
import argparse
import json
from pathlib import Path
import statistics

from copy_action_fit.data import save, sha
from full_corpus_fit.data import acceptance, audit_chunk


def audit(path):
    prompts=json.loads((path/'prompts.json').read_text())
    refs=json.loads((path/'reference-results.json').read_text())
    rows=json.loads((path/'candidate-results.json').read_text())
    complete=json.loads((path/'complete.json').read_text())
    assert len(prompts)==8 and len(refs)==12 and complete['passed_correctness']
    mapping={p['id']:(i,p,r['generated']) for i,(p,r) in enumerate(zip(prompts,refs))}
    positions, audited, changes, graph_calls, logits_compared = 0,0,[],0,0
    baseline_hashes={}
    for r in rows:
        index,p,ref=mapping[r['prompt_id']]
        if r['phase']=='stop-fixture':
            ref=next(x['generated'] for x in refs[8:] if
                     (x['budget'],x['stop_token'])==(r['budget'],r['stop_token']))
        assert all(r['generated'][k]==ref[k] for k in ('prompt','tokens','finish_reason'))
        observed=r['observed']
        if observed['native']:
            hashed=any('logits_sha256' in x for x in observed['native']['trace'])
            checked=audit_chunk([p],[r['generated']],observed['native'],hashes=hashed)
            positions+=checked['positions_checked'];audited+=1
            if r['phase']=='correctness' and hashed:
                if r['mode']=='off':baseline_hashes[index]=checked['aligned_hashes']
                else:
                    assert checked['aligned_hashes']==baseline_hashes[index]
                    logits_compared+=len(checked['aligned_hashes'])
            if r['phase']=='bf16-correctness' and hashed:
                assert checked['aligned_hashes']==baseline_hashes[index]
                logits_compared+=len(checked['aligned_hashes'])
        if r['phase']=='timing' and r['mode']=='editor':
            assert observed['graph_misses']==observed['graph_evictions']==0
            graph_calls+=observed['graph_hits']
        if r['phase']=='correctness' and r['mode']=='editor':
            full=p['prompt']+ref['tokens']
            for edit in observed['edits']:
                known=min(edit['anchor'],len(full))
                assert edit['prefix'][:known]==full[:known]
                assert edit['original'][:2]==edit['edited'][:2]
                label=full[edit['anchor']:edit['anchor']+16]
                if len(label)==16:
                    changes.append(acceptance(edit['edited'],label)-acceptance(edit['original'],label))
    cases=[]
    for i,p in enumerate(prompts):
        timing={mode:[r for r in rows if r['phase']=='timing' and r['prompt_id']==p['id'] and r['mode']==mode]
                for mode in ('off','editor')}
        assert all(len(v)==5 for v in timing.values())
        seconds={k:statistics.median(r['seconds'] for r in v) for k,v in timing.items()}
        assert seconds==complete['groups']['all']['per_prompt'][p['id']]['seconds']
        cold=next(r for r in rows if r['phase']=='cold' and r['prompt_id']==p['id'])
        cases.append(dict(case=i,input_tokens=len(p['prompt']),group=p['length_group'],
            seconds=seconds,speedup=seconds['off']/seconds['editor'],cold_seconds=cold['seconds'],
            graph_preparation_ms=cold['observed']['graph_preparation_ms'],
            graph_misses=cold['observed']['graph_misses'],
            rounds={k:len(v[0]['observed']['rounds']) for k,v in timing.items()},
            speculative_round_median_ms={k:statistics.median(x['gpu_ms'] for r in v
                for x in r['observed']['rounds'] if x['speculative']) for k,v in timing.items()},
            editor_median_ms=statistics.median(x for r in timing['editor'] for x in r['observed']['editor_ms']),
            repeat_seconds={k:[r['seconds'] for r in v] for k,v in timing.items()}))
    return dict(passed=True,candidate_runs=len(rows),audited_native_runs=audited,
        audited_positions=positions,full_target_logit_hash_comparisons=logits_compared,
        warm_graph_hits=graph_calls,warm_graph_misses=0,complete_editor_blocks=len(changes),
        repaired_blocks=sum(x>0 for x in changes),damaged_blocks=sum(x<0 for x in changes),
        prefix_delta_sum=sum(changes),cases=cases,
        groups={k:{a:b for a,b in v.items() if a!='per_prompt'} for k,v in complete['groups'].items()},
        peak_allocated_bytes=complete['peak_allocated_bytes'],
        cold_seconds=complete['cold_seconds'],
        source_artifact_hashes={name:sha(path/name) for name in
            ('prompts.json','run.json','reference-results.json','candidate-results.json','complete.json')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();save(args.output,audit(args.directory))
