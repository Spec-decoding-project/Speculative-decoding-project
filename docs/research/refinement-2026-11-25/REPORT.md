# November 26 project update

**Cheap first-block selection now gives a small native warm-cache win on GSM8K:
+2.73% throughput, with UltraChat effectively unchanged at −0.08%.** A small tree
fitted on training/calibration data simplifies exactly to calling the frozen
autoregressive editor only on the first proposal, then skipping it with a flag.
Across 64 development prompts and three repeats, this removes 94.63% of editor
calls versus always-on editing while preserving every tested final sequence.
GSM8K's prompt-bootstrap 95% speed ratio is 1.00971–1.04654; UltraChat's spans 1.
Cold graph setup still makes the selected path slower, and no production cache
hit rate or long-response benefit is established. All jobs/audits/backups are
complete; no new editor training ran. [Complete selector result](CHEAP_SELECTOR.md).

## Earlier always-on comparison

**The exact earlier 3.6997 linear gate now has a completed native benchmark on
128 development prompts. It lowers end-to-end throughput by 8.86% on UltraChat
and 8.59% on GSM8K.** All 640 timing pairs preserve final tokens; full target-logit,
acceptance and graph-reuse checks pass. At batch one, commits per round improve
only 2.15%/2.32%, permitting 1.50/1.62 ms of extra work, while measured complete
rounds cost 8.75/8.67 ms more. All eight length groups lose on average.
[Complete matched benchmark, repeats, length groups and break-even results](NATIVE_BEST.md).

The earlier native pilot used a different, broader-data gate and only eight
UltraChat prompts. Its best BF16 arm was 7.5% slower overall; those results are
historical and are not substituted for the requested earlier-gate comparison.
That pilot also identified and reduced a process-wide editor-kernel penalty.
[Earlier native integration and kernel investigation](NATIVE_SPEED.md).

All jobs, independent audits and backups are complete; no more training ran.
Four natural EOS cases now pass for replacement. Union, concurrency and fresh
confirmation remain open. The present always-on execution design fails the speed
criterion; reduce its cost or change when editing runs before scaling it further.
Earlier training and isolated timing findings follow.

The underlying whole-draft autoregressive editor remains **40,750,336 parameters**. It reads accepted context and the complete DFlash token proposal, preserves the anchor and first proposal token, and generates fourteen editable tokens one at a time. It uses its own embeddings/representations; no target/drafter hidden states are inputs. The earlier gate experiments froze that generator and trained only a COPY-versus-generate decision gate: 9, 265 or 17,025 parameters.

| Experiment | What it established |
|---|---|
| Frozen COPY diagnosis | Joint COPY fine-tuning both overrides some good answers and weakens the generator. Preserve the original generator. |
| Three gates on 1,536 training prompts | All three improve development prefix quality modestly. Best observed mean: **3.6997**, versus DFlash **3.5776** (+3.4%). |
| One/two/four-edit budgets | Eight of nine shortened variants lose quality. The one positive short gain is tiny. |
| Same gates on 9,392 training prompts | More data helps the score gate slightly and does not reliably improve the others. Best earlier result remains higher. |
| Native earlier-gate benchmark | Exact outputs, but 8.86%/8.59% lower UltraChat/GSM8K throughput with warm graphs; the acceptance gain buys much less time than editing adds. |
| Exact-output CUDA graph prototype | On a fixed broader linear checkpoint, warm editor latency falls **25.45 → 6.73 ms (3.78×)**, preserving every tested output. |

Both training comparisons select checkpoints using the same 512 training-origin calibration prompts. Development uses 2,048 blocks from 512 separate prompts, excluded from optimization. These development prompts have been inspected repeatedly; this is exploratory evidence, not fresh confirmation. The first fit supplies 991 decisive training positions and the larger fit 8,300. Only the gate receives gradient updates; the original full-trained generator is unchanged.

Training computes which candidate is correct under the reference preceding tokens, then minimizes weighted binary decision loss. `backward()` computes gate gradients; `optimizer.step()` updates the gate. At evaluation, the editor consumes its own selected tokens. Checkpoints are selected by calibration prefix length, not by lower training loss alone. The detailed process and limits are in [HISTORY.md](../../../HISTORY.md).

The earlier 3.78× latency result is for a **warm, fixed-shape editor invocation**. Graph preparation costs about 102 ms per captured shape; that prototype lacked a serving cache. The graphed checkpoint's development prefix is 3.6797, and its warm break-even screening threshold is about 302 ms of baseline round time under stated assumptions. Target/drafter were unloaded during that timing. These numbers must not be described as a 3.78× faster DFlash generator. The native report now measures bounded graph reuse, cold cost and complete generation directly.

All five earlier GPU jobs completed successfully and all artifacts are backed up. Their 54-test local suite passes. Full-size GPU controls, reloads, generator hashes, calibration simulations, cache/output checks and independent saved-result audits passed. The subsequent native GPU jobs, CPU pin audits and independent native audits also pass their correctness checks; their always-on aggregate performance result is negative; the newer first-block result is summarized above. Original data, old experiments and original workspace changes are preserved. The final test split remains unused.

The earlier proposed native milestone has now run for replacement; see the current report above. The serving-fidelity ticket remains open for its untested requirements. Nothing additional is needed from the user to reproduce the completed measurements.

Detailed evidence: [diagnostic](DIAGNOSTIC.md), [first gate fits](GATES.md), [short edits](SHORT_EDITS.md), [broader training](BROADER_DATA.md), [CUDA graph result](GRAPH_REPLAY.md). Code and reports are on `work/ar-refinement-research`; private checkpoints, raw predictions and logs are mirrored under `~/.local/share/wayfinder-artifacts/`.
