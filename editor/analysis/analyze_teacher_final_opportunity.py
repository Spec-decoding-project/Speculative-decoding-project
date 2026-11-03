"""Descriptive teacher comparison on the same fresh contexts; no endpoint selection."""
from pathlib import Path
import hashlib
import json
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'notes/second-pass-profiling-2026-10-02/teacher-unseen-comparison'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    attempt = BASE / 'final-attempt1'
    state = read(attempt / 'execution.json')
    if state['status'] != 'compute_closed_archive_retained' or state['scientific_status'] != 'fresh_evaluation_complete':
        raise ValueError('Wait for audited completion and recovery')
    metadata = attempt / 'archive-metadata'
    manifest = read(attempt / 'archive-metadata-recovery.json')

    def verified(name):
        raw = (metadata / name).read_bytes()
        row = manifest[name]
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256'] or row['local_sha256'] != row['sha256']:
            raise ValueError('Recovered evidence differs')
        return json.loads(raw)

    audit = verified('final-audit.json')
    audited = {row['id']: row for row in audit['per_prompt']}
    plan = read(BASE / 'final-plan.json')
    teacher = {}
    domains = {'gsm8k': [], 'ultrachat': []}
    for directory, prompts in plan['groups'].items():
        for prompt in prompts:
            pid = prompt['id']
            record = verified(f'{directory}/teacher-opportunity/{pid}.json')
            if record['prompts'] != 1 or set(record['per_prompt']) != {pid} or pid in teacher:
                raise ValueError('Teacher prompt coverage differs')
            t = record['per_prompt'][pid]
            for metrics in audited[pid]['evaluations'].values():
                if (metrics['original'], metrics['contexts']) != (t['original'], t['contexts']):
                    raise ValueError('Teacher and editor contexts differ')
            teacher[pid] = t
            domains[prompt['domain']].append(pid)
    if len(teacher) != 64 or set(teacher) != set(audited) or any(len(ids) != 32 for ids in domains.values()):
        raise ValueError('Incomplete or different final panel')
    gains = {pid: 100 * (t['union'] / t['original'] - 1) for pid, t in teacher.items()}
    rng = random.Random(20261003)
    boot = sorted(statistics.mean(gains[rng.choice(ids)] for ids in domains.values() for _ in ids) for _ in range(10000))
    mean = statistics.mean(gains.values())
    result = dict(
        status='complete', analysis_role='Supplementary descriptive teacher comparison; does not change the frozen primary contrast',
        prompts=64, contexts=sum(t['contexts'] for t in teacher.values()),
        teacher_coverage_gain_percent=mean, interval95_percent=[boot[250], boot[9750]],
        by_domain_percent={d: statistics.mean(gains[p] for p in ids) for d, ids in domains.items()},
        teacher_repairs=sum(t['repairs'] for t in teacher.values()),
        teacher_damage=sum(t['damage'] for t in teacher.values()),
        reachable_errors=sum(t['reachable'] for t in teacher.values()),
        editor_to_teacher_additional_coverage_ratio={
            name: statistics.mean(row['evaluations'][name]['gain_pp'] for row in audited.values()) / mean
            for name in next(iter(audited.values()))['evaluations']},
        limitations='Ratios compare average additional coverage, not recovery of identical teacher corrections. Teacher scores came from the frozen teacher, with runtime controls and authenticated files; this supplementary summary does not independently rerun its forward pass. Fixed-context coverage is not generation speed, an information limit, or a universal student ceiling.')
    (attempt / 'teacher-opportunity-analysis.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
