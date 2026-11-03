"""Post hoc saved-token audit; no fitting or checkpoint selection."""
import json
from pathlib import Path
from collections import Counter
import statistics
import torch

root = Path('notes/second-pass-profiling-2026-10-01/compact-training-completion/attempt2')
folder = root / 'recovered'
arms = ('target_only', 'eligible_ce', 'teacher_kl')
evaluations = {(a,e): {(r['prompt'],r['anchor_index']):r['result'] for r in json.loads((folder/f'{a}-epoch-{e}-evaluation.json').read_text())} for a in arms for e in (1,2,3)}
features = {}
for file in (folder/'features').glob('*.pt'):
    for r in torch.load(file, map_location='cpu', weights_only=True):
        if r['split'] == 'development':
            key = (r['prompt_id'],r['anchor_index'])
            assert key not in features
            features[key] = r
assert len(features) == 2048
output = {}
for arm in arms:
    counts = Counter(); examples = {}; totals = {}
    for key,r in features.items():
        label = r['labels'][0].tolist(); valid = r['valid'][0].tolist()
        result = evaluations[arm,3][key]
        original = result['original_tokens']; a = result['original_progress']
        teacher = original[:2] + r['teacher_winners'][0].tolist()
        def progress(tokens):
            n = 1
            for i in range(1,len(tokens)):
                if not valid[i] or tokens[i] != label[i]: break
                n += 1
            return n
        assert progress(original) == a
        assert progress(result['edited_tokens']) == result['edited_progress']
        t = progress(teacher)
        best = max(evaluations[arm,e][key]['union_progress'] for e in (1,2,3))
        total = totals.setdefault(key[0],[0,0]); total[0] += a; total[1] += best
        if t <= a: continue
        counts['teacher_repaired_blocks'] += 1
        edited = result['edited_tokens']; b = result['edited_progress']
        if b > a: category = 'repaired'
        elif edited[a] == label[a]: category = 'corrects_first_error_but_breaks_earlier_prefix'
        elif edited[a] == original[a]: category = 'keeps_original_first_error'
        else: category = 'changes_first_error_to_another_wrong_token'
        counts[category] += 1
        counts['earlier_prefix_damage_in_teacher_repair_blocks'] += b < a
        counts['teacher_repair_blocks_helped_at_any_saved_epoch'] += best > a
        counts['matching_teacher_extra_positions_without_prefix_requirement'] += sum(edited[i] == label[i] for i in range(a,t))
        counts['teacher_extra_positions'] += t-a
        counts['teacher_extra_positions_in_accepted_prefix'] += max(0,min(b,t)-a)
        if len(examples.setdefault(category,[])) < 3:
            examples[category].append({'prompt':key[0],'anchor':key[1],'original_progress':a,'teacher_progress':t,'edited_progress':b,'original_tokens':original,'edited_tokens':edited,'labels':label})
    assert sum(counts[c] for c in ('repaired','corrects_first_error_but_breaks_earlier_prefix','keeps_original_first_error','changes_first_error_to_another_wrong_token')) == counts['teacher_repaired_blocks'] == 353
    output[arm] = {'counts':dict(counts),'unimplementable_per_block_best_of_three_epochs_gain_percent':statistics.mean(100*(b/a-1) for a,b in totals.values()),'examples':examples}
output['limitations'] = ['Exposed development data, post hoc descriptive audit, one seed.', 'Token matches after a rejection are not accepted progress.', 'Best epoch chosen separately for each block is an oracle, not a deployable selector.', 'Error categories do not identify capacity, information loss, or a causal training defect.']
(root/'teacher-miss-anatomy.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({a:{k:v for k,v in output[a].items() if k!='examples'} for a in arms},indent=2))
