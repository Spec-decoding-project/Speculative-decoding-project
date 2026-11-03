"""Descriptive temporal measurements; never infer a causal repair from replay."""
from pathlib import Path
import json
import math
import statistics

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notes/second-pass-profiling-2026-10-02/context-optimization-replay/attempt1'

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    source=BASE/'recovered'
    terminal=read(source/'terminal.json')
    if terminal['status']!='complete':
        result={'status':'incomplete_replay','terminal':terminal,
                'interpretation':'No explanation of the historical spike is established by an incomplete or mismatched replay.'}
        (BASE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result,indent=2));return
    summary=read(source/'summary.json')
    assert summary['updates']==896 and summary['loss_trace_matches_exactly']
    assert summary['historical_epoch_weight_matches']==2 and summary['observed_updates']==132
    updates=read(source/'replay-updates.json');trace=read(source/'diagnostics.json')
    assert len(updates)==896 and len(trace)==132
    historical=read(ROOT/'notes/second-pass-profiling-2026-10-02/context-access-training/attempt1/recovered/updates.json')
    historical=[r for r in historical if r['arm']=='aligned_context'][:896]
    assert len(historical)==896
    for actual,expected in zip(updates,historical):
        assert all(actual[k]==expected[k] for k in ('epoch','step','loss','valid_rows')), 'Historical trace mismatch'
    losses={(r['epoch'],r['step']):r['loss'] for r in updates}
    assert len(losses)==896
    seen=set();series=[]
    for row in trace:
        key=(row['epoch'],row['step']);assert key not in seen;seen.add(key)
        parts=row['microbatches'];assert len(parts)==8
        n=sum(m['supervised_output']['rows'] for m in parts)
        ce=sum(m['supervised_output']['mean_ce']*m['supervised_output']['rows'] for m in parts)/n
        assert abs(ce-losses[key])<1e-5, (key,ce,losses[key])
        def avg(get):return sum(get(m)*m['supervised_output']['rows'] for m in parts)/n
        gradients={};steps={};weights={}
        for name,p in row['parameters'].items():
            group='context' if name.startswith('context_') else 'shared_editor'
            gradients[group]=gradients.get(group,0)+(p['gradient_l2'] or 0)**2
            steps[group]=steps.get(group,0)+p['update_l2']**2
            weights[group]=weights.get(group,0)+p['weight_l2_before']**2
        r={'epoch':key[0],'step':key[1],'loss':losses[key],
            'valid_ce_reconstructed':ce,
            'attention_normalized_entropy':avg(lambda m:m['attention']['supervised_queries']['mean_normalized_entropy']),
            'attention_mean_max_probability':avg(lambda m:m['attention']['supervised_queries']['mean_largest_probability']),
            'attention_max_span':max(m['attention']['supervised_queries']['max_row_span'] for m in parts),
            'worst_valid_ce':max(m['supervised_output']['max_ce'] for m in parts),
            'mean_correct_margin':avg(lambda m:m['supervised_output']['mean_correct_margin']),
            'correction_rms':math.sqrt(avg(lambda m:m['supervised_output']['correction']['rms']**2)),
            'context_residual_rms':math.sqrt(avg(lambda m:m['activations']['context_out']['supervised_queries']['rms']**2)),
            'last_layer_rms':math.sqrt(avg(lambda m:m['activations']['layers.1']['supervised_queries']['rms']**2)),
            'gradient_l2':{g:math.sqrt(v) for g,v in gradients.items()},
            'relative_update_l2':{g:math.sqrt(steps[g]/max(weights[g],1e-30)) for g in steps}}
        series.append(r)
    assert {r['step'] for r in series if r['epoch']==3}==set(range(1,129))
    windows={}
    for name,(lo,hi) in {'early':(1,32),'before_observed_burst':(65,86),'observed_burst':(87,124),'after_burst':(125,128)}.items():
        rows=[r for r in series if r['epoch']==3 and lo<=r['step']<=hi]
        keys=['loss','attention_normalized_entropy','attention_mean_max_probability','attention_max_span','worst_valid_ce',
              'mean_correct_margin','correction_rms','context_residual_rms','last_layer_rms']
        windows[name]={'steps':[lo,hi], 'count':len(rows),
                      'means':{k:statistics.mean(r[k] for r in rows) for k in keys},
                      'maxima':{k:max(r[k] for r in rows) for k in keys}}
    result={'status':'historical_trace_reproduced','updates':896,'independent_exact_historical_loss_matches':896,
        'observed_updates':132,'windows':windows,'series':series,
        'limitations':['Descriptive temporal associations; no causal intervention or repair',
            'FP32 reconstructed attention, not native BF16 probabilities',
            'Attention distributions cover the added context lookup only; trunk self-attention distributions are not recorded',
            'Low attention entropy alone is not a defect: useful attention can be concentrated',
            'Window definitions use historical loss episode, not an independent confirmatory panel',
            'Weighted RMS aggregates supervised rows; no claim about all model activations',
            'No new generation, speedup, or novel method established']}
    (BASE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'windows':windows},indent=2))

if __name__=='__main__':main()
