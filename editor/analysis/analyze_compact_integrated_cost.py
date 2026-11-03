"""Analyze completed integrated-cost evidence without treating random edits as learning."""
from pathlib import Path
from collections import defaultdict
import json
import statistics

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notes/second-pass-profiling-2026-10-01/compact-integrated-cost'


def main():
    folder=BASE/'attempt1/recovered'
    terminal=json.loads((folder/'terminal.json').read_text())
    rows=json.loads((folder/'rounds.json').read_text()) if (folder/'rounds.json').exists() else []
    grouped=defaultdict(list)
    for r in rows:
        if 'wall_ms' in r:grouped[(r['context']['id'],r['context']['anchor_index'],r['arm'])].append(r)
    arms=('original','teacher_revised','teacher_union','compact_union')
    contexts=[]
    for prompt,anchor in sorted({(p,a) for p,a,k in grouped}):
        if not all(len(grouped[prompt,anchor,k])==20 for k in arms):continue
        times={k:statistics.median(r['wall_ms'] for r in grouped[prompt,anchor,k]) for k in arms}
        progress={k:grouped[prompt,anchor,k][0]['result']['committed_positions'] for k in arms}
        for k in arms:
            assert len({json.dumps(r['result'],sort_keys=True) for r in grouped[prompt,anchor,k]})==1
        ratio=times['compact_union']/times['original']
        contexts.append(dict(prompt=prompt,anchor=anchor,median_round_ms=times,
            progress_untrained_compact_not_quality_evidence=progress,
            compact_added_ms=times['compact_union']-times['original'],
            compact_added_round_cost_pct=100*(ratio-1),
            required_progress_gain_for_5pct_saving_pct=100*(ratio/.95-1),
            compact_vs_teacher_union_round_change_pct=100*(times['compact_union']/times['teacher_union']-1)))
    out=dict(status=terminal['status'],complete_contexts=len(contexts),contexts=contexts,
        limitations=['Untrained compact editor: observed random proposals cannot establish quality or learned gains.',
        'Four fixed contexts from two development prompts; no population confidence interval.',
        'All draft networks eager; graph optimization may change the comparison.',
        'Required progress is conditional algebra using measured round-cost ratio, not a prediction.',
        'Cost includes different downstream proposals, so it is not isolated correction-kernel overhead.'])
    (BASE/'attempt1/analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
