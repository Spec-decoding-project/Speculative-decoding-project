"""Independent token audit of frozen editor predictions on sampled training data."""
from pathlib import Path
import hashlib
import json
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
BASE = ROOT / 'notes/second-pass-profiling-2026-10-02/teacher-unseen-comparison/training-fit-diagnostic'
ARMS = ['new_train--' + a for a in ('target_only', 'eligible_ce', 'teacher_kl')]


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_prompt(rows, metadata, prompt):
    from teacher_data_final_audit import prefix
    anchors = metadata['anchors']
    if metadata['id'] != prompt['id'] or metadata['split'] != prompt['split'] or metadata['domain'] != prompt['domain']:
        raise ValueError('Wrong reference metadata')
    if [r['anchor_index'] for r in rows] != list(range(len(anchors))):
        raise ValueError('Missing, duplicate, or reordered anchors')
    native = prompt['input_ids'] + metadata['output_ids']
    totals = {name: dict(original=0, union=0, repairs=0, damage=0, damaged_fully_correct=0, teacher_positions_captured=0) for name in ARMS}
    teacher = dict(original=0, union=0, contexts=len(rows), reachable=0, repairs=0, damage=0)
    for row, anchor in zip(rows, anchors):
        labels, valid, original = anchor['labels'], anchor['valid'], anchor['original']
        start = anchor['anchor']
        length = min(16, len(native) - start)
        if start < 0 or length < 1 or labels[:length] != native[start:start+length] or valid != [True]*length + [False]*(16-length):
            raise ValueError('Labels do not match saved native continuation')
        if row['prompt'] != prompt['id'] or row['labels'] != labels or row['valid'] != valid or set(row['evaluations']) != set(ARMS):
            raise ValueError('Prediction references or arms differ')
        a = prefix(original, labels, valid)
        teacher_tokens = row['teacher_tokens']
        if teacher_tokens[:2] != original[:2]:
            raise ValueError('Teacher changed the fixed prefix')
        t = prefix(teacher_tokens, labels, valid)
        teacher['original'] += a
        teacher['union'] += max(a, t)
        teacher['reachable'] += 2 <= a < length
        teacher['repairs'] += t > a
        teacher['damage'] += t < a
        for name, v in row['evaluations'].items():
            if v['original_tokens'] != original or v['edited_tokens'][:2] != original[:2]:
                raise ValueError('Editor changed original or fixed prefix')
            b = prefix(v['edited_tokens'], labels, valid)
            if (a, b, max(a, b)) != (v['original_progress'], v['edited_progress'], v['union_progress']):
                raise ValueError('Independent prefix recount differs')
            if v['fully_correct_original'] != (a == length) or v['damaged_fully_correct'] != (a == length and b < a):
                raise ValueError('Damage count differs')
            q = totals[name]
            q['original'] += a
            q['union'] += max(a, b)
            q['repairs'] += b > a
            q['damage'] += b < a
            q['damaged_fully_correct'] += a == length and b < a
            q['teacher_positions_captured'] += max(0, min(b, t) - a)
    return dict(id=prompt['id'], domain=prompt['domain'], contexts=len(rows), teacher=teacher, arms=totals)


def main():
    attempt = BASE / 'attempt1'
    state = read(attempt / 'execution.json')
    if state['status'] != 'closed' or state.get('result', {}).get('status') != 'complete' or not state['scratch_absent'] or not state['shutdown']['stopped'] or state['shutdown']['active_tasks']:
        raise ValueError('Diagnostic incomplete or unclosed')
    recovered = attempt / 'recovered'
    for name, row in read(attempt / 'recovery.json').items():
        path = (recovered / name).resolve()
        if not path.is_relative_to(recovered.resolve()) or path.stat().st_size != row['bytes'] or sha(path) != row['sha256'] or row['local_sha256'] != row['sha256']:
            raise ValueError('Recovered evidence differs')
    plan = read(BASE / 'plan.json')
    if sha(BASE / 'plan.json') != read(BASE / 'bundle.json')['plan_sha256']:
        raise ValueError('Frozen diagnostic changed')
    summary = read(recovered / 'summary.json')
    if summary['status'] != 'complete' or summary['prompts'] != 64 or summary['contexts'] != plan['expected_contexts'] or summary['training_updates'] or summary['checkpoint_selection']:
        raise ValueError('Incomplete or changed scope')
    identities = read(recovered / 'checkpoint-identities.json')
    if set(identities) != set(ARMS):
        raise ValueError('Checkpoint set differs')
    for name, spec in plan['checkpoints'].items():
        row = identities[name]
        if row['file_sha256'] != spec['sha256'] or row['initial_state_sha256'] != spec['state_sha256'] or row['final_state_sha256'] != spec['state_sha256']:
            raise ValueError('Checkpoint changed')
    controls = read(recovered / 'compatibility.json')
    if len(controls) != 6 or not all(r['exact'] for r in controls):
        raise ValueError('Known predictions not reproduced')
    reports = []
    for p in plan['prompts']:
        reference = plan['reference_metadata'][p['id']]
        path = ROOT / reference['source']
        if sha(path) != reference['sha256']:
            raise ValueError('Native reference changed')
        recorded = read(recovered / 'predictions' / (p['id'] + '.json'))
        report = audit_prompt(recorded['records'], read(path), p)
        if report['teacher'] != plan['teacher_sample'][p['id']]:
            raise ValueError('Teacher opportunity no longer matches existing bank audit')
        reports.append(report)
    fresh = read(BASE.parent / 'final-attempt1/analysis.json')
    if sha(BASE.parent / 'final-attempt1/analysis.json') != plan['completed_fresh_analysis_sha256']:
        raise ValueError('Historical fresh comparison changed')
    arms = {}
    for name in ARMS:
        arms[name] = dict(training_sample_coverage_gain_percent=100*statistics.mean(r['arms'][name]['union']/r['arms'][name]['original'] - 1 for r in reports),
            fresh_coverage_gain_percent=fresh['per_arm'][name]['equal_prompt_coverage_gain_percent'],
            repairs=sum(r['arms'][name]['repairs'] for r in reports), damage=sum(r['arms'][name]['damage'] for r in reports),
            teacher_positions_captured=sum(r['arms'][name]['teacher_positions_captured'] for r in reports),
            by_domain_percent={d:100*statistics.mean(r['arms'][name]['union']/r['arms'][name]['original']-1 for r in reports if r['domain']==d) for d in ('gsm8k','ultrachat')})
    result = dict(status='complete',per_arm=arms,prompts=64,contexts=summary['contexts'],
        teacher_coverage_gain_percent=100*statistics.mean(r['teacher']['union']/r['teacher']['original']-1 for r in reports),
        teacher_extra_positions=sum(r['teacher']['union']-r['teacher']['original'] for r in reports),per_prompt=reports,
        scope='Descriptive training/fresh comparison. Different prompts and teacher opportunities prevent a causal diagnosis of memorization, capacity or optimization; no new learning, checkpoint selection, or latency claim.')
    (attempt/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='per_prompt'},indent=2))


if __name__ == '__main__':
    main()
