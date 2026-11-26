"""Read-only reconstruction of selector split, labels, decisions and native timings."""
import argparse
from collections import Counter
import json
from pathlib import Path

from copy_action_fit.data import CHECKPOINT_SHA256, SOURCES, digest, save, sha
from full_corpus_fit.data import audit_chunk
from refinement_research.native_best import GATE_SHA
from refinement_research.native_best_audit import round_economics


def read(path):return json.loads(path.read_text())


def score(tree, features):
    if 'feature' not in tree:return tree['value']
    side='left' if features[tree['feature']] <= tree['threshold'] else 'right'
    return score(tree[side],features)


def progress(a,b):
    n=0
    for x,y in zip(a,b):
        if x!=y:break
        n+=1
    return n


def audit(path):
    groups=read(path/'prompts.json');manifest=read(path/'run.json')
    model=read(path/'selector.json');complete=read(path/'complete.json')
    assert digest(groups)==manifest['prompt_digest']
    assert read(path/'frozen-selector.json')==dict(sha256=sha(path/'selector.json'),digest=digest(model))
    assert read(path/'selector-loaded.json')['selector_digest']==digest(model)
    for phase in ('collection','evaluation'):
        loaded=read(path/(phase+'-editor-loaded.json'))
        assert loaded['initial_sha256']==CHECKPOINT_SHA256 and loaded['gate_sha256']==GATE_SHA
        assert loaded['precision']=='fp32' and loaded['kernels']=='standard' and loaded['parameters']==40750601
    ids=set();families=set()
    for phase,n,split in [('fit',96,'train'),('calibration',32,'train'),('evaluation',64,'development')]:
        ps=groups[phase]
        assert len(ps)==n and all(p['split']==split for p in ps)
        assert Counter((p['source'],p['length_group']) for p in ps)=={
            (s,q):n//8 for s in SOURCES for q in ('Q1','Q2','Q3','Q4')}
        assert len({p['family_id'] for p in ps})==n
        assert not ids & {p['id'] for p in ps} and not families & {p['family_id'] for p in ps}
        ids.update(p['id'] for p in ps);families.update(p['family_id'] for p in ps)
    collection=[read(p) for p in sorted((path/'records').glob('collection-*.json'))]
    label_summary={};positions=hashes=0
    for phase in ('fit','calibration'):
        mapping={p['id']:p for p in groups[phase]}
        expected=[];discarded=0
        for r in (r for r in collection if r['phase']==phase):
            p=mapping[r['prompt_id']];full=p['prompt']+r['generated']['tokens']
            assert r['mode']=='shadow' and len(r['observed']['edits'])==len(r['observed']['selector_decisions'])
            for e,d in zip(r['observed']['edits'],r['observed']['selector_decisions']):
                at=e['anchor']+1
                known=min(at-1,len(full))
                assert at==d['length'] and d['selected'] and len(e['prefix'])==at-1 and e['prefix'][:known]==full[:known]
                assert e['original'][:2]==e['edited'][:2]
                if at+15>len(full):discarded+=1;continue
                assert e['original'][0]==full[at-1]
                original=progress(e['original'][1:],full[at:at+15]);edited=progress(e['edited'][1:],full[at:at+15])
                expected.append(dict(prompt_id=p['id'],family_id=p['family_id'],split=p['split'],
                    source=p['source'],anchor=e['anchor'],features=d['features'],original=original,edited=edited,gain=edited-original))
            if r['observed']['native']:
                check=audit_chunk([p],[r['generated']],r['observed']['native'],hashes=True)
                positions+=check['positions_checked'];hashes+=len(check['aligned_hashes'])
        assert expected==read(path/(phase+'-labels.json'))
        assert digest(expected)==model[phase+'_digest']
        chosen=[r for r in expected if model['threshold'] is not None and score(model['tree'],r['features'])>=model['threshold']]
        label_summary[phase]=dict(prompts=len(mapping),rows=len(expected),incomplete_windows_discarded=discarded,
            original_prefix=sum(r['original'] for r in expected)/len(expected),
            edited_prefix=sum(r['edited'] for r in expected)/len(expected),selected_rows=len(chosen),
            selected_gain=sum(r['gain'] for r in chosen),repairs=sum(r['gain']>0 for r in chosen),
            damage=sum(r['gain']<0 for r in chosen),selected_net=sum(r['gain']-.45 for r in chosen))
    mapping={p['id']:p for p in groups['evaluation']}
    rows=[read(p) for p in sorted((path/'records').glob('evaluation-*.json'))]
    baselines={r['prompt_id']:r for r in rows if r['mode']=='off' and r['phase']=='correctness'}
    assert len(baselines)==64
    references=read(path/'reference-results.json');assert len(references)==8
    for ref in references:
        p=mapping[ref['prompt_id']]
        for key in ('prompt','tokens','finish_reason'):assert ref['generated'][key]==baselines[p['id']]['generated'][key]
        positions+=audit_chunk([p],[ref['generated']],ref['observed']['native'])['positions_checked']
    for r in rows:
        b=baselines[r['prompt_id']];obs=r['observed'];p=mapping[r['prompt_id']]
        assert r['generated']['prompt']==p['prompt']
        for key in ('prompt','tokens','finish_reason'):assert r['generated'][key]==b['generated'][key]
        remaining=len(r['generated']['tokens']);commits=[];prefill=0
        for sample in obs['all_samples']:
            visible=min(remaining,sample['sampled']);remaining-=visible
            if sample['speculative']:commits.append(visible)
            else:prefill+=visible
        assert remaining==0 and prefill==1 and commits==r['metrics']['committed_per_round']
        assert len(commits)==r['metrics']['verification_rounds'] and sum(commits)==r['metrics']['verification_tokens']
        if r['mode'] in ('off','selector_probe'):assert obs['edit_calls']==0
        if r['mode'] in ('selected','selector_probe'):
            assert len(obs['selector_decisions'])==len(commits)+1
            for d in obs['selector_decisions']:
                assert d['score']==score(model['tree'],d['features'])
                assert d['selected']==(r['mode']=='selected' and model['threshold'] is not None and d['score']>=model['threshold'])
        if r['mode']=='selected':assert obs['edit_calls']==sum(d['selected'] for d in obs['selector_decisions'])
        if obs['native']:
            checked=audit_chunk([p],[r['generated']],obs['native'],hashes=True)
            positions+=checked['positions_checked'];hashes+=len(checked['aligned_hashes'])
            assert checked['aligned_hashes']==b['observed']['audit']['aligned_hashes']
        if r['phase']=='timing':assert obs['graph_misses']==0
    timing=[r for r in rows if r['phase']=='timing'];assert len(timing)==64*4*3
    economic={}
    for source in ('all',*SOURCES):
        subset=[r for r in timing if source=='all' or r['source']==source]
        aggregate={}
        for mode in ('off','selector_probe','editor','selected'):
            arm=[r for r in subset if r['mode']==mode]
            for pid in {r['prompt_id'] for r in arm}:
                repeats=[r for r in arm if r['prompt_id']==pid]
                assert sorted(r['repeat'] for r in repeats)==[0,1,2]
                assert len({digest(r['metrics']['committed_per_round']) for r in repeats})==1
            seconds=sum(r['seconds'] for r in arm);rounds=sum(len(r['metrics']['committed_per_round']) for r in arm)
            ms=sum(r['metrics']['decode_ms'] for r in arm);committed=sum(sum(r['metrics']['committed_per_round']) for r in arm)
            aggregate[mode]=dict(tokens_per_second=sum(len(r['generated']['tokens']) for r in arm)/seconds,
                average_round_ms=ms/rounds,committed_per_round=committed/rounds,verification_rounds=rounds)
            for key,value in aggregate[mode].items():assert abs(value-complete['groups'][source]['arms'][mode][key])<1e-9
        b=aggregate['off']
        economic[source]={mode:round_economics(b['average_round_ms'],a['average_round_ms'],
            b['committed_per_round'],a['committed_per_round']) for mode,a in aggregate.items() if mode!='off'}
    return dict(passed=True,labels=label_summary,native_positions_checked=positions,
        aligned_target_hashes=hashes,timed_generations=len(timing),exact_output_generations=len(rows)+len(references),
        selector_sha256=sha(path/'selector.json'),threshold=model['threshold'],economics=economic,
        gate_sha256=GATE_SHA,editor_optimizer_updates=0,final_test_evaluated=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('path',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();save(args.output,audit(args.path))


if __name__=='__main__':main()
