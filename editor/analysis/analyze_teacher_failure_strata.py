"""Separate observed teacher failure locations without another model run."""
from collections import defaultdict
from pathlib import Path
import json
import hashlib
import sys


def count(tokens, labels, valid):
    n = 0
    for a, b, enabled in zip(tokens, labels, valid, strict=True):
        if not enabled or a != b:
            break
        n += 1
    return n


def run(folder):
    folder = Path(folder)
    sources = {}

    def read(path):
        sources[path.relative_to(folder).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    if read(folder/'terminal.json')['status'] != 'complete':
        raise ValueError('Complete comparison required')
    names = ['initial_first_guess', 'initial_full_proposal', 'first_guess', 'full_proposal',
             'first_guess_singlepass', 'full_proposal_singlepass']
    panels = {}
    for name in names:
        panel = read(folder/(name+'-predictions.json'))
        panels[name] = {(r['prompt'], r['anchor_index']): r for r in panel}
        if len(panel) != 256 or len(panels[name]) != 256:
            raise ValueError('Incomplete panel')
    rows = []
    for file in sorted((folder/'controls').glob('*.json')):
        proposals = {r['anchor_index']: r for r in read(folder/'proposals'/file.name)}
        for c in read(file):
            key = file.stem, c['anchor_index']
            original = proposals[key[1]]
            labels, valid = original['labels'], original['valid']
            old = count(original['tokens'], labels, valid)
            size = sum(valid)
            stratum = ('locked_first_wrong' if old == 1 else 'initially_complete' if old == size
                       else 'first_error_2_5' if old <= 5 else 'first_error_6_10' if old <= 10
                       else 'first_error_11_15')
            row = dict(prompt=key[0], anchor_index=key[1], stratum=stratum,
                       original_chain=old, valid_positions=size,
                       released_conditional=c['coverage']['conditional64'])
            for name, panel in panels.items():
                r = panel[key]
                if r['labels'] != labels or r['valid'] != valid:
                    raise ValueError('Labels differ')
                tokens = r['original'] if name.endswith('singlepass') else r['tokens']
                row[name+'_chain'] = count(tokens, labels, valid)
                row[name+'_coverage'] = r['coverage']['conditional64'] if name.endswith('singlepass') else r['union64']
            rows.append(row)
    if len(rows) != 256:
        raise ValueError('Missing contexts')
    summaries = {}
    for group in ['all', 'first_guess_correct', *sorted({r['stratum'] for r in rows})]:
        rr = [r for r in rows if group == 'all' or
              (group == 'first_guess_correct' and r['original_chain'] >= 2) or r['stratum'] == group]
        summary = dict(contexts=len(rr), prompts=len({r['prompt'] for r in rr}),
                       original_chain_positions=sum(r['original_chain'] for r in rr),
                       released_conditional_positions=sum(r['released_conditional'] for r in rr), methods={})
        for name in names:
            deltas = [r[name+'_chain']-r['original_chain'] for r in rr]
            per = defaultdict(lambda: [0, 0])
            for r in rr:
                per[r['prompt']][0] += r[name+'_coverage']
                per[r['prompt']][1] += r['released_conditional']
            summary['methods'][name] = dict(
                raw_chain_positions=sum(r[name+'_chain'] for r in rr),
                coverage_positions=sum(r[name+'_coverage'] for r in rr),
                repaired=sum(x > 0 for x in deltas), damaged=sum(x < 0 for x in deltas),
                gained=sum(max(x, 0) for x in deltas), lost=sum(max(-x, 0) for x in deltas),
                mean_prompt_coverage_difference_percent=sum(100*(a/b-1) for a,b in per.values())/len(per))
        summaries[group] = summary
    # Paired raw-chain edits distinguish effects hidden by aggregate coverage.
    contrasts = {}
    for newer, older in [('full_proposal', 'initial_full_proposal'),
                         ('first_guess', 'initial_first_guess'),
                         ('full_proposal', 'full_proposal_singlepass'),
                         ('first_guess', 'first_guess_singlepass')]:
        delta = [r[newer+'_chain']-r[older+'_chain'] for r in rows]
        contrasts[newer+'_vs_'+older] = dict(better=sum(x>0 for x in delta),worse=sum(x<0 for x in delta),
                                           tied=sum(x==0 for x in delta), gained=sum(max(x,0) for x in delta),
                                           lost=sum(max(-x,0) for x in delta))
    return dict(scope='Post-result descriptive strata; not a new advancement criterion or causal mediation test',
                summaries=summaries, raw_chain_paired_contrasts=contrasts, rows=rows, source_hashes=sources,
                limitations=['Difficulty strata use reference answers and cannot gate deployment.',
                             'Training and input effects interact; contrasts do not identify an internal cause.',
                             'The ordinary-input trained-weight control is not independently trained for that input.',
                             'No generation, timing, statistical independence of tokens, or student limit claim.'])


if __name__ == '__main__':
    folder = Path(sys.argv[1]); result = run(folder)
    output = folder.parent/'failure-strata.json'
    if output.exists() and json.loads(output.read_text()) != result:
        raise ValueError('Preserve earlier result')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_hashes')},indent=2))
