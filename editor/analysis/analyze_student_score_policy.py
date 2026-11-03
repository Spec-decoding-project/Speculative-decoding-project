"""Frozen-student probability comparison with unchanged selection rules."""
from pathlib import Path
from collections import defaultdict
import hashlib
import heapq
import json
import math
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'notes/second-pass-profiling-2026-10-03'
OUT = BASE / 'student-score-policy/attempt1'
sys.path.insert(0, str(ROOT / 'experiments'))
from sparse_edit_policy import choose
from original_draft_tree_reference import matching_prefix
from analyze_teacher_edit_score import independent_choice


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree(original, ids, logs, budget=30):
    """Same mandatory original chain and fixed x1; permit unsorted score sources."""
    assert len(original) == 16 and len(ids) == len(logs) == 15
    assert all(len(row) == 4 for row in ids + logs)
    assert all(math.isfinite(x) and x <= 1e-6 for row in logs for x in row)
    assert all(sum(math.exp(x) for x in row) <= 1.00001 for row in logs)
    nodes = {tuple(original[:i]) for i in range(1, 17)}
    mass = {tuple(original[:i]): sum(logs[j-1][0] for j in range(1, i)) for i in range(1, 17)}
    frontier = []

    def offer(parent):
        depth = len(parent)
        if depth < 2 or depth == 16:
            return
        for token, logp in zip(ids[depth-1], logs[depth-1]):
            child = parent + (token,)
            if child not in nodes:
                heapq.heappush(frontier, (-mass[parent]-logp, child))

    for n in sorted(nodes):
        offer(n)
    while len(nodes) < budget and frontier:
        negative, n = heapq.heappop(frontier)
        if n in nodes:
            continue
        assert n[:-1] in nodes
        nodes.add(n)
        mass[n] = -negative
        offer(n)
    assert len(nodes) == budget
    assert all(len(n) == 1 or n[:-1] in nodes for n in nodes)
    return nodes


def prefix(seq, y, valid):
    return next((i for i, (a, b, v) in enumerate(zip(seq, y, valid)) if not v or a != b), len(seq))


def main():
    receipt = read(OUT / 'recovery.json')
    for name in ('rows.json', 'summary.json', 'terminal.json'):
        assert sha(OUT / 'recovered' / name) == receipt[name]['sha256']
    summary = read(OUT / 'recovered/summary.json')
    assert summary['status'] == read(OUT / 'recovered/terminal.json')['status'] == 'complete'
    assert summary['model_hashes_before'] == summary['model_hashes_after']
    rows = read(OUT / 'recovered/rows.json')
    b = read(OUT.parent / 'frozen/local-bundle.json')
    spec = b['inputs']['original-score-reference.json']
    p = ROOT / spec['local_path']
    assert sha(p) == spec['sha256']
    originals = {(r['prompt'], r['anchor_index']): r for r in read(p)}
    teacher_dir = BASE / 'teacher-edit-score/attempt2'
    p = teacher_dir / 'recovered/rows.json'
    assert sha(p) == read(teacher_dir / 'recovery.json')['rows.json']['sha256']
    teachers = {(r['prompt'], r['anchor_index']): r for r in read(p)}
    prior_trees = {(r['prompt'], r['anchor_index']): r for r in read(teacher_dir / 'tree-score-rows.json')}
    keys = [(r['prompt'], r['anchor_index']) for r in rows]
    assert len(keys) == len(set(keys)) == 2048 and set(keys) == set(originals) == set(teachers)
    refs = {}
    for arm in summary['arms']:
        spec = b['inputs'][arm + '-reference.json']
        p = ROOT / spec['local_path']
        assert sha(p) == spec['sha256']
        refs[arm] = {(r['prompt'], r['anchor_index']): r for r in read(p)}
    per = defaultdict(lambda: defaultdict(int))
    counts = defaultdict(int)
    details = []
    for r in rows:
        key = r['prompt'], r['anchor_index']
        q, t = originals[key], teachers[key]
        p = per[r['prompt']]
        p['original'] += q['original_progress']
        sources = {'q0': q['log_probabilities'], 'teacher': t['teacher_candidate_logs']}
        sources.update({arm: row['candidate_logs'] for arm, row in r['students'].items()})
        item = dict(prompt=key[0], anchor_index=key[1], progress={})
        for arm, logs in sources.items():
            assert logs[0] == q['log_probabilities'][0]
            action = choose(q['original_tokens'], q['candidate_ids'], logs)
            j, rank, proposal = independent_choice(q['original_tokens'], q['candidate_ids'], logs)
            assert (j, rank, proposal) == (action['position'], action['rank'], action['proposal'])
            one = max(q['original_progress'], prefix(proposal, q['labels'], q['valid']))
            nodes = tree(q['original_tokens'], q['candidate_ids'], logs)
            selected = matching_prefix(nodes, q['labels'], q['valid'])
            if arm in ('q0', 'teacher'):
                assert one == t[arm + '_policy']['union_progress']
                assert selected == prior_trees[key][arm + '_tree']
            else:
                assert r['students'][arm]['edited_tokens'] == refs[arm][key]['result']['edited_tokens']
                old = max(q['original_progress'], prefix(r['students'][arm]['edited_tokens'], q['labels'], q['valid']))
                assert old == refs[arm][key]['result']['union_progress']
                p[arm + '_argmax'] += old
            for rule, value in (('one_edit', one), ('tree', selected)):
                name = arm + '_' + rule
                p[name] += value
                item['progress'][name] = value
        for arm in summary['arms']:
            for rule in ('one_edit', 'tree'):
                name = arm + '_' + rule
                a, z = item['progress'][name], item['progress']['q0_' + rule]
                counts[name + '_better'] += a > z
                counts[name + '_worse'] += a < z
                counts[name + '_tied'] += a == z
                counts[name + '_gained_positions'] += max(0, a-z)
                counts[name + '_lost_positions'] += max(0, z-a)
        details.append(item)
    prompts = sorted(per)
    assert len(prompts) == 64
    rng = random.Random(202610031)
    domains = [[p for p in prompts if p.startswith(d+'-')] for d in ('gsm8k', 'ultrachat')]
    draws = [rng.choices(domains[0], k=32) + rng.choices(domains[1], k=32) for _ in range(10000)]

    def interval(values):
        ordered = sorted(sum(values[p] for p in sample)/64 for sample in draws)
        return [ordered[249], ordered[9749]]

    result = {}
    for name in per[prompts[0]]:
        if name == 'original':
            continue
        rule = 'tree' if name.endswith('_tree') else 'one_edit'
        values = {p: 100*(per[p][name]/per[p]['original']-1) for p in prompts}
        delta = {p: 100*(per[p][name]-per[p]['q0_'+rule])/per[p]['original'] for p in prompts}
        ratios = {p: per[p]['q0_'+rule]/per[p][name] for p in prompts}
        result[name] = dict(coverage_gain_percent=statistics.mean(values.values()), interval95=interval(values),
            difference_from_q0_rule_pp=statistics.mean(delta.values()), difference_interval95=interval(delta),
            modeled_break_even_extra_round_cost_percent=100*(1/statistics.mean(ratios.values())-1),
            domains={d: statistics.mean(values[p] for p in prompts if p.startswith(d+'-')) for d in ('gsm8k','ultrachat')})
    contrasts = {}
    for rule in ('one_edit', 'tree'):
        delta = {p: 100*(per[p]['teacher_only_'+rule]-per[p]['gated_mixture_'+rule])/per[p]['original'] for p in prompts}
        contrasts[rule] = dict(teacher_only_minus_mixture_pp=statistics.mean(delta.values()), interval95=interval(delta))
    analysis = dict(status='PASS', contexts=2048, responses=64, conditions=result, objective_contrasts=contrasts,
        counts=dict(counts), per_prompt=dict(per), source_hashes=dict(rows=sha(OUT/'recovered/rows.json'), analyzer=sha(Path(__file__))),
        limitations=['Reused development evidence, no population confirmation',
                     'No new training or generated trajectories; no measured runtime saving',
                     'Scores use full-vocabulary normalization; same original candidates, fixed x1 and original preservation',
                     'Model state and native historical prediction parity checked before interpretation'])
    (OUT/'analysis.json').write_text(json.dumps(analysis, indent=2)+'\n')
    (OUT/'selected-progress.json').write_text(json.dumps(details)+'\n')
    print(json.dumps({k:v for k,v in analysis.items() if k!='per_prompt'}, indent=2))


if __name__ == '__main__':
    main()
