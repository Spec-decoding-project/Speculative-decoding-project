"""Recount both intervention precisions without dropping sensitive contexts."""
import copy
import json
from pathlib import Path
import sys
from analyze_teacher_context import analyze


def analyze_native(rows, keys):
    for r in rows:
        if not r['all_neutral_modes_both_precisions_exact']:
            raise ValueError('Neutral validation incomplete')
        if any(v['max_score_error'] != 0 for v in r['parity'].values()):
            raise ValueError('Native scores changed')
        for mode in ('route', 'mass', 'raw'):
            audit = r['precision_audit'][mode]
            if (audit['tokens_exact'] != (r['tokens'][mode] == r['tokens'][mode+'_fp64'])
                    or audit['prefix_exact'] != (r['progress'][mode] == r['progress'][mode+'_fp64'])):
                raise ValueError('Precision sensitivity recount failed')
    primary = analyze(rows, keys)
    double = copy.deepcopy(rows)
    for r in double:
        for mode in ('route', 'mass', 'raw'):
            r['tokens'][mode] = r['tokens'][mode+'_fp64']
            r['progress'][mode] = r['progress'][mode+'_fp64']
    sensitive = [dict(prompt=r['prompt'], anchor_index=r['anchor_index'], category=r['category'],
                      modes=[m for m in ('route', 'mass', 'raw') if not r['precision_audit'][m]['prefix_exact']])
                 for r in rows if any(not a['prefix_exact'] for a in r['precision_audit'].values())]
    return dict(fp32=primary, fp64=analyze(double, keys), prefix_sensitive_cases=sensitive,
                interpretation='Keep every case. Precision-sensitive prefix effects are unresolved; no majority vote or favorable-precision selection. These interventions do not prove economical compression.')


if __name__ == '__main__':
    folder = Path(sys.argv[1]); attempt = folder/'attempt1'
    plan = json.loads((folder/'runner-plan.json').read_text())
    audit = json.loads((attempt/'recovered/final-audit.json').read_text())
    if not audit['teacher_unchanged'] or not audit['versions_unchanged']:
        raise ValueError('Final weight audit failed')
    rows = json.loads((attempt/'recovered/predictions.json').read_text())
    result = analyze_native(rows, [(p['id'], a) for p in plan['panels'] for a in plan['anchor_indices']])
    with (attempt/'analysis.json').open('x') as f: json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
