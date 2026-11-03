"""Independent token-based recount. Incomplete panels get no aggregate verdict."""
from collections import Counter
import json
from pathlib import Path
import sys


def prefix(tokens, labels, valid):
    if len(tokens)!=len(labels) or len(valid)!=len(labels) or tokens[0]!=labels[0]:
        raise ValueError('Invalid aligned block')
    n=0
    for a,b,v in zip(tokens,labels,valid):
        if not v or a!=b:break
        n+=1
    return n


def analyze(rows, expected_keys):
    keys=[(r['prompt'],r['anchor_index']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected_keys):
        raise ValueError('Incomplete, duplicate or changed panel: no full-panel conclusion')
    summary={}
    for r in rows:
        y,v=r['labels'],r['valid']; a=prefix(r['original_tokens'],y,v)
        if a!=r['original_progress'] or not r['same_trace_neutral_exact']:
            raise ValueError('Original or neutral control changed')
        for name,t in r['tokens'].items():
            if t[:2]!=r['original_tokens'][:2] or prefix(t,y,v)!=r['progress'][name]:
                raise ValueError('Token-level recount failed')
        for name in ('conditioned','unconditioned'):
            if r['tokens']['native_'+name]!=r['tokens']['traced_'+name] or not r['parity'][name]['tokens_exact']:
                raise ValueError('Instrumentation token disagreement')
        c=r['progress']['native_conditioned'];u=r['progress']['native_unconditioned']
        category=('conditioned_does_not_repair' if c<=a else
                  'unconditioned_already_reaches_conditioned_endpoint' if u>=c else
                  'unconditioned_repairs_but_conditioning_extends_further' if u>a else 'only_conditioned_repairs')
        if category!=r['category']:raise ValueError('Historical stratum changed')
        for group in ('all',r['domain'],category):
            g=summary.setdefault(group,{})
            for mode in ('route','mass','raw'):
                p=r['progress'][mode];s=g.setdefault(mode,Counter())
                s['contexts']+=1;s['teacher_repair_contexts']+=c>a
                s['teacher_added_positions']+=max(0,c-a)
                s['teacher_positions_retained']+=max(0,min(p,c)-a)
                s['teacher_repair_complete']+=c>a and p>=c
                s['teacher_repair_lost_entirely']+=c>a and p<=a
                s['union_positions_lost_vs_conditioned']+=max(0,max(a,c)-max(a,p))
                s['union_positions_gained_vs_conditioned']+=max(0,max(a,p)-max(a,c))
                s['edited_prefix_shorter_than_original']+=p<a
    return dict(contexts=len(rows),prompts=len({k[0] for k in keys}),counts=summary,
                scope='Exposed fixed contexts, pooled descriptive counts. No speedup, population inference or compact-editor feasibility claim.')


def main(folder):
    folder=Path(folder);plan=json.loads((folder/'runner-plan.json').read_text())
    attempt=folder/'attempt1';recovered=attempt/'recovered'
    audit=json.loads((recovered/'final-audit.json').read_text())
    if not audit['teacher_unchanged'] or not audit['versions_unchanged']:raise ValueError('Weights changed')
    rows=json.loads((recovered/'predictions.json').read_text())
    result=analyze(rows,[(p['id'],a) for p in plan['panels'] for a in plan['anchor_indices']])
    with (attempt/'analysis.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main(sys.argv[1])
