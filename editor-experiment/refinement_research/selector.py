"""Small pre-editor regression tree. No editor activations or target labels at inference."""
import math

from copy_action_fit.data import SOURCES, digest

FEATURES = ['gap_first', 'gap_second', 'gap_third', 'gap_fourth',
            'gap_tail_min', 'gap_tail_mean', 'gap_first_four_mean',
            'unique_proposals', 'adjacent_repeats', 'first_token_repeats',
            'previous_committed', 'prompt_length', 'generated_length', 'first_round']
THRESHOLDS = [0., .25, .5, .75, 1., 1.5, 2., None]
EDIT_TOKEN_COST = .45


def features(proposal, gaps, previous, prompt_length, length):
    if len(proposal) != 15 or len(gaps) != 15 or not all(math.isfinite(v) and v >= 0 for v in gaps):
        raise ValueError('Require 15 finite nonnegative draft logit gaps')
    return [*gaps[:4], min(gaps[1:]), sum(gaps[1:])/14, sum(gaps[:4])/4,
            len(set(proposal)), sum(a == b for a,b in zip(proposal,proposal[1:])),
            proposal.count(proposal[0]), previous, prompt_length,
            length-prompt_length, int(length == prompt_length+1)]


def predict(tree, x):
    while 'feature' in tree:
        tree = tree['left'] if x[tree['feature']] <= tree['threshold'] else tree['right']
    return tree['value']


def selected(model, x):
    return model['threshold'] is not None and predict(model['tree'], x) >= model['threshold']


def fit_tree(rows, depth=3, minimum=64, shrinkage=32):
    import numpy as np
    x = np.asarray([r['features'] for r in rows], dtype=float)
    y = np.asarray([r['gain'] for r in rows], dtype=float)
    if not len(rows) or x.shape != (len(rows), len(FEATURES)) or not np.isfinite(x).all():
        raise ValueError('Invalid selector fit matrix')

    def build(indices, remaining):
        target = y[indices]
        leaf = dict(value=float(target.sum()/(len(indices)+shrinkage)),
                    samples=len(indices), mean_gain=float(target.mean()))
        if not remaining or len(indices) < 2*minimum:
            return leaf
        best = None
        baseline = float((target**2).sum()-target.sum()**2/len(target))
        for j in range(x.shape[1]):
            order = indices[np.argsort(x[indices,j], kind='stable')]
            values, ys = x[order,j], y[order]
            sums, squares = np.cumsum(ys), np.cumsum(ys**2)
            for k in range(minimum, len(order)-minimum+1):
                if values[k-1] == values[k]:
                    continue
                loss = (squares[k-1]-sums[k-1]**2/k + squares[-1]-squares[k-1]
                        -(sums[-1]-sums[k-1])**2/(len(order)-k))
                if loss < baseline-1e-9 and (best is None or loss < best[0]-1e-9):
                    best = (float(loss), j, float((values[k-1]+values[k])/2), order[:k], order[k:])
        if best is None:
            return leaf
        _,j,threshold,left,right = best
        return dict(feature=j, name=FEATURES[j], threshold=threshold,
                    left=build(left,remaining-1), right=build(right,remaining-1), **leaf)
    return build(np.arange(len(rows)), depth)


def fit(fit_rows, calibration):
    for field in ('prompt_id','family_id'):
        if {r[field] for r in fit_rows} & {r[field] for r in calibration}:
            raise ValueError('Selector fit/calibration leakage')
    if any(r['split'] != 'train' for r in fit_rows+calibration):
        raise ValueError('Selector only learns from training-origin prompts')
    tree = fit_tree(fit_rows)
    trials=[]
    for threshold in THRESHOLDS:
        chosen=[r for r in calibration if threshold is not None and predict(tree,r['features']) >= threshold]
        gain=sum(r['gain'] for r in chosen)
        trials.append(dict(threshold=threshold, edits=len(chosen), gain=gain,
                           token_net=gain-EDIT_TOKEN_COST*len(chosen)))
    # Ties favor fewer edits, including the explicit zero-edit policy.
    winner=max(trials, key=lambda r:(r['token_net'],-r['edits'],r['threshold'] is None))
    return dict(schema='cheap-selector-v1', features=FEATURES, tree=tree,
                threshold=winner['threshold'], calibration=trials,
                fit_rows=len(fit_rows), calibration_rows=len(calibration),
                edit_token_cost=EDIT_TOKEN_COST, fit_digest=digest(fit_rows),
                calibration_digest=digest(calibration), depth=3, minimum_leaf=64, shrinkage=32)


def prefix_length(proposal, labels):
    for i,(a,b) in enumerate(zip(proposal, labels)):
        if a != b:
            return i
    return min(len(proposal),len(labels))


def label_shadow(prompt, record):
    full = prompt['prompt'] + record['generated']['tokens']
    decisions = record['observed']['selector_decisions']
    edits = record['observed']['edits']
    if len(decisions) != len(edits):
        raise ValueError('Missing shadow editor result')
    rows=[]
    for decision,edit in zip(decisions,edits):
        position=edit['anchor']+1
        if decision['length'] != position or edit['prefix'] != full[:position-1]:
            raise ValueError('Shadow context does not follow canonical baseline')
        if full[position-1] != edit['original'][0] or edit['original'][:2] != edit['edited'][:2]:
            raise ValueError('Shadow anchor/first proposal changed')
        if position+15 > len(full):
            continue  # The unobserved tail must never become a guessed label.
        labels=full[position:position+15]
        original=prefix_length(edit['original'][1:], labels)
        edited=prefix_length(edit['edited'][1:], labels)
        rows.append(dict(prompt_id=prompt['id'],family_id=prompt['family_id'],split=prompt['split'],
            source=prompt['source'],anchor=edit['anchor'],features=decision['features'],
            original=original,edited=edited,gain=edited-original))
    return rows


def select_prompts(prompts):
    from refinement_research.native_best import select as old_select
    excluded={p['id'] for p in old_select(prompts)}
    result={'fit':[], 'calibration':[], 'evaluation':[]}
    used_families=set()
    for split in ('train','development'):
        for source in SOURCES:
            population=sorted((p for p in prompts if p['split']==split and p['source']==source
                               and len(p['prompt'])+80 <= 4096), key=lambda p:(len(p['prompt']),digest(p['id'])))
            for q in range(4):
                group=population[q*len(population)//4:(q+1)*len(population)//4]
                candidates=sorted(group,key=lambda p:digest(['cheap-selector-v1',p['id']]))
                chosen=[]
                for p in candidates:
                    if p['id'] in excluded or p['family_id'] in used_families:
                        continue
                    chosen.append(dict(p,length_group=f'Q{q+1}'))
                    used_families.add(p['family_id'])
                    if len(chosen)==(16 if split=='train' else 8):break
                if len(chosen)!=(16 if split=='train' else 8):
                    raise ValueError('Insufficient disjoint selector population')
                if split=='train':
                    result['fit'].extend(chosen[:12]);result['calibration'].extend(chosen[12:])
                else:result['evaluation'].extend(chosen)
    for phase in result:
        result[phase].sort(key=lambda p:digest(['selector-order-v1',p['id']]))
    return result
