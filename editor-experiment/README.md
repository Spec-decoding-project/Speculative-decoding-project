# Editor experiment

Shrinav Loka and Frank, October 2026.

## Question

A drafter proposes a block of 16 tokens; the target checks them in one pass. Re-drafting the block with a second, trained drafter adds accepted tokens but costs almost a full extra pass. Can a small editor that reuses the first pass's computation recover that gain cheaply?

## Setup

- Target: Qwen3-4B, greedy. Drafter: z-lab/Qwen3-4B-DFlash-b16, frozen.
- Teacher: a 537M copy of the drafter, trained to redraft with the first token guess visible. Its draft is kept beside the original; the target verifies both.
- Editor: 3,024,384 parameters. Two width-256 attention layers over the first pass's hidden states and the first token's embedding, emitting a rank-256 residual through the drafter's frozen vocabulary head. Starts as the identity. Never changes the first guess or the original draft.
- Data: 384 train and 64 dev prompts, 32 contexts each, GSM8K and UltraChat.
- Training: 3 epochs, 1,152 AdamW updates, batch 32, lr 1e-3. Three losses, same init and data order: correct tokens; correct tokens with extra weight where the teacher was right; correct tokens plus KL to the teacher.

## Results (dev set, measured)

| | Extra accepted positions | Extra round cost |
|---|---|---|
| Teacher | +10.2% | about +30% |
| Editor, correct tokens | +3.0% | +5% to +6% |
| Editor, teacher-weighted | +3.7% | +5% to +6% |
| Editor, teacher KL | +3.5% | +5% to +6% |

- Break-even needed about +6%. No arm reached it.
- A free 24-node tree from the drafter's own scores beat every editor.
- The teacher repairs 79 and damages 33 training blocks, but 35 and 41 on unseen prompts. It memorizes, so the editor imitates prompt-specific repairs.
- A frozen drafter re-reading its own draft loses 49% of its correct positions.

## Plan

1. Measure real end-to-end speed on Qwen3-8B: DFlash, DFlash + second pass, DFlash + editor.
2. Scale editor data 10x with the existing frozen teacher, keeping only redrafts the target verified as better. If the gain stays near 3.7%, data is not the limit.
3. If it moves, train a D2SD-style second drafter for Qwen3-8B and repeat.

## Files

- `compact_second_pass_prototype.py`: the editor.
- `compact_second_pass_learning.py`: the three losses.
- `compact_second_pass_full_training.py`: training loop.
- `compact_second_pass_features.py`, `compact_second_pass_bank.py`: feature banks.
- `full_proposal_teacher.py`, `teacher_support_training.py`: the teacher.
- `compact_second_pass_decode.py`, `second_pass_profile.py`: one verification round and its cost.
- `conditional_redraft_*.py`: shared harness.
- `analysis/`: scripts that produce the tables above.

Needs torch, transformers, the two checkpoints, and a 24 GB GPU. The cloud launchers are not included.
