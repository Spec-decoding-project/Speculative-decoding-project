# November 25 project update

We found a small positive correction signal and a substantial way to reduce editor execution overhead. **The project has not yet demonstrated net generation speedup.** Six gate-training runs and the completed diagnostics clarify the next step: native correctness and complete-round cost, rather than blindly increasing data or model size.

The underlying whole-draft autoregressive editor remains **40,750,336 parameters**. It reads accepted context and the complete DFlash token proposal, preserves the anchor and first proposal token, and generates fourteen editable tokens one at a time. It uses its own embeddings/representations; no target/drafter hidden states are inputs. This session freezes that generator and trains only a COPY-versus-generate decision gate: 9, 265 or 17,025 parameters.

| Experiment | What it established |
|---|---|
| Frozen COPY diagnosis | Joint COPY fine-tuning both overrides some good answers and weakens the generator. Preserve the original generator. |
| Three gates on 1,536 training prompts | All three improve development prefix quality modestly. Best observed mean: **3.6997**, versus DFlash **3.5776** (+3.4%). |
| One/two/four-edit budgets | Eight of nine shortened variants lose quality. The one positive short gain is tiny. |
| Same gates on 9,392 training prompts | More data helps the score gate slightly and does not reliably improve the others. Best earlier result remains higher. |
| Exact-output CUDA graph prototype | On a fixed broader linear checkpoint, warm editor latency falls **25.45 → 6.73 ms (3.78×)**, preserving every tested output. |

Both training comparisons select checkpoints using the same 512 training-origin calibration prompts. Development uses 2,048 blocks from 512 separate prompts, excluded from optimization. These development prompts have been inspected repeatedly; this is exploratory evidence, not fresh confirmation. The first fit supplies 991 decisive training positions and the larger fit 8,300. Only the gate receives gradient updates; the original full-trained generator is unchanged.

Training computes which candidate is correct under the reference preceding tokens, then minimizes weighted binary decision loss. `backward()` computes gate gradients; `optimizer.step()` updates the gate. At evaluation, the editor consumes its own selected tokens. Checkpoints are selected by calibration prefix length, not by lower training loss alone. The detailed process and limits are in [HISTORY.md](../../../HISTORY.md).

The latency result is for a **warm, fixed-shape editor invocation**. Graph preparation costs about 102 ms per captured shape; a serving reuse strategy remains to be implemented. The graphed checkpoint's development prefix is 3.6797, and its warm break-even screening threshold is about 302 ms of baseline round time under stated assumptions. Target/drafter were unloaded during editor timing. These numbers must not be described as a 3.78× faster DFlash generator.

All five GPU jobs completed successfully and all artifacts are backed up. The final 54-test local suite passes. Full-size GPU controls, reloads, generator hashes, calibration simulations, cache/output checks and independent saved-result audits passed. Original data, old experiments and original workspace changes are preserved. No new target determinism investigation or final-test evaluation ran.

The next useful milestone is native verification and complete-round timing with graph reuse, correct stopping/cache behavior and precision differences recorded. The existing serving-fidelity ticket carries that work. Nothing additional is needed from the user to understand or reproduce this research checkpoint.

Detailed evidence: [diagnostic](DIAGNOSTIC.md), [first gate fits](GATES.md), [short edits](SHORT_EDITS.md), [broader training](BROADER_DATA.md), [CUDA graph result](GRAPH_REPLAY.md). Code and reports are on `work/ar-refinement-research`; private checkpoints, raw predictions and logs are mirrored under `~/.local/share/wayfinder-artifacts/`.
