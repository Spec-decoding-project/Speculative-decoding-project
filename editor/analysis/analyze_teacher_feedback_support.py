"""Reanalyze saved 64-prompt teacher controls; no new inference."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/results/conditional-full-reference-precision-2026-09-22/evaluation'


def main():
    feedback=json.loads((BASE/'feedback-epoch-3.json').read_text())
    other={r['id']:r for r in json.loads((BASE/'no_feedback-epoch-3.json').read_text())}
    rows=[]
    blocks=[]
    for f in feedback:
        n=other[f['id']]
        assert f['original_progress']==n['original_progress']
        assert len(f['details'])==len(n['details'])==32
        rows.append(dict(id=f['id'],domain=f['domain'],original=f['original_progress'],
                         feedback_gain=f['union_progress']-f['original_progress'],
                         no_feedback_gain=n['union_progress']-n['original_progress']))
        for a,b in zip(f['details'],n['details']):
            assert (a['anchor'],a['original'])==(b['anchor'],b['original'])
            assert a['union']==max(a['original'],a['alternative'])
            assert b['union']==max(b['original'],b['alternative'])
            blocks.append(dict(prompt=f['id'],anchor=a['anchor'],original=a['original'],
                feedback_gain=a['union']-a['original'],no_feedback_gain=b['union']-b['original']))
    def summarize(items):
        original=sum(r['original'] for r in items)
        return dict(prompts=len(items),original_progress=original,
            feedback_extra=sum(r['feedback_gain'] for r in items),
            no_feedback_extra=sum(r['no_feedback_gain'] for r in items),
            feedback_pooled_gain=sum(r['feedback_gain'] for r in items)/original,
            no_feedback_pooled_gain=sum(r['no_feedback_gain'] for r in items)/original,
            feedback_equal_prompt_gain=statistics.mean(r['feedback_gain']/r['original'] for r in items),
            no_feedback_equal_prompt_gain=statistics.mean(r['no_feedback_gain']/r['original'] for r in items),
            prompts_with_feedback_gain=sum(r['feedback_gain']>0 for r in items),
            feedback_beats_no_feedback=sum(r['feedback_gain']>r['no_feedback_gain'] for r in items),
            feedback_loses_to_no_feedback=sum(r['feedback_gain']<r['no_feedback_gain'] for r in items))
    ranked=sorted(rows,key=lambda r:(-r['feedback_gain'],r['id']))
    both=[b for b in blocks if b['feedback_gain']>0 and b['no_feedback_gain']>0]
    result=dict(status='complete',overall=summarize(rows),
        domains={d:summarize([r for r in rows if r['domain']==d]) for d in sorted({r['domain'] for r in rows})},
        largest_two_prompt_gain_share=sum(r['feedback_gain'] for r in ranked[:2])/sum(r['feedback_gain'] for r in rows),
        excluding_two_largest=summarize([r for r in rows if r not in ranked[:2]]),
        block_overlap=dict(total=len(blocks),both_repair=len(both),
            feedback_only_repairs=sum(b['feedback_gain']>0 and b['no_feedback_gain']==0 for b in blocks),
            no_feedback_only_repairs=sum(b['no_feedback_gain']>0 and b['feedback_gain']==0 for b in blocks),
            feedback_better=sum(b['feedback_gain']>b['no_feedback_gain'] for b in blocks),
            no_feedback_better=sum(b['feedback_gain']<b['no_feedback_gain'] for b in blocks)),
        limitations=['Historical exposed development set, fixed-context recorded-label comparison, not native generation.',
          'Both models are independently trained under matched recipes; contrast measures feedback-enabled training/inference package, not an inference-only intervention.',
          'Neither overlap nor feedback benefit proves student compressibility.',
          'Removing top gain prompts is a sensitivity diagnostic, not a new primary estimate.'])
    out=ROOT/'notes/second-pass-profiling-2026-10-01/gpu-attempt4/teacher-feedback-support.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
