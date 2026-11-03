# Can a drafter learn from its own draft?

Shrinav Loka and Frank. Speculative decoding on Qwen3 with the DFlash block
drafter. Question: can a small network, reusing the first drafting pass's
computation, reproduce the corrections a full second drafting pass makes,
cheaply enough to speed up generation?

## Layout

| Folder | What it is |
|---|---|
| `dflash/` | The DFlash drafter we build on, unmodified snapshot. See `dflash/SNAPSHOT.md`. |
| `editor-experiment/` | Our experiment: the 3M-parameter editor, the three training losses, the full second-pass teacher, feature banks, complete-round decode and cost measurement, and the analysis scripts. Start with `editor-experiment/README.md`. |
| `results/` | `SUMMARY.md` (one-page project summary), the D2SD paper note, and the ten run write-ups the numbers below come from. |

## Setup

Target Qwen3-4B, greedy. First drafter z-lab/Qwen3-4B-DFlash-b16, frozen,
16-token blocks. Teacher: a full copy of the drafter trained to redraft with
the first token guess visible; its draft is kept beside the original and the
target verifies both. Editor: 3,024,384 parameters, two width-256 attention
layers over the first pass's hidden states plus the first token, emitting a
rank-256 residual through the drafter's frozen vocabulary head. Trained on
384 prompts x 32 contexts for 1,152 updates; evaluated on 64 held-back
prompts x 32 contexts. GSM8K and UltraChat.

## Results (measured on the dev set)

| Setup | Extra accepted positions | Extra round cost |
|---|---|---|
| Teacher, full second pass | +10.2% | about +30% (derived) |
| Editor, correct-token loss | +3.0% | +5% to +6% |
| Editor, teacher-correct weighting | +3.7% | +5% to +6% |
| Editor, teacher KL | +3.5% | +5% to +6% |

- The editor needed about +6% to break even and +10.6% to +11.9% for a 5%
  net speedup. No arm reached it.
- A free 24-node tree built from the first drafter's own confidence scores
  beat every editor.
- The teacher repairs 79 and damages 33 blocks on its training prompts, but
  repairs 35 and damages 41 on unseen prompts. It memorizes.
- A frozen drafter re-reading its own completed draft loses 49% of its
  correct positions.

Full numbers and the per-run evidence: `results/SUMMARY.md`.

## Plan

1. First real end-to-end speed measurement, on Qwen3-8B: plain DFlash vs
   DFlash + second pass vs DFlash + editor.
2. Scale the editor's training data 10x with the existing frozen teacher,
   keeping only redrafts the target verified as better. If the gain does not
   move, data dose is not the limit.
3. If it does move, train a D2SD-style second drafter for Qwen3-8B, since
   none is released, and repeat.

Compute: UMIACS Nexus class cluster, RTX A5000.
