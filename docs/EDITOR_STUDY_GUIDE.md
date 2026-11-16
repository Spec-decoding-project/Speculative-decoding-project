# Practical preparation for the DFlash editor experiments

You do not need another broad survey of inference before starting. The immediate goal is to be able to trace one example from saved tokens through an optimizer update and then through sequential generation and target verification. Work through these topics in order, using tiny synthetic examples first. Real-data training is still gated by target replay validation.

## Topics, code, and practical checkpoints

| Topic | Read in this repository | Exercise that demonstrates understanding |
| --- | --- | --- |
| Git and reproducible experiments | The two `work/ar-*` branches, `experiment.json`, experiment reports | Show that the branches use identical model/trainer code and differ only in conditioning configuration. Record a commit, environment, seed and data checksum without changing the original workspace. |
| Token positions and speculative decoding | `GLOSSARY.md`, `dflash/secondary.py`, `feature_bank/data.py` | On paper, draw the committed prefix, target anchor, first DFlash proposal and 14 editable positions. Identify which logit predicts which token. Explain why a wrong preserved first proposal cannot be repaired by changing the suffix. |
| Tensors, embeddings and vocabulary projection | `autoregressive_editor/model.py` | Trace the shapes of token IDs, embeddings, Q/K/V and logits through a toy forward pass. Count tied parameters once. Explain why a vocabulary of 151,936 makes a width-256 table about 38.9M parameters. |
| Conditional autoregressive modeling | `autoregressive_editor/data.py` (`condition`, `teacher_forcing`) | Write both input sequences for the same example. Explain why the whole original draft may be visible, but future correct output tokens may not. Predict the effect of replacing the original draft with unrelated tokens. |
| Teacher forcing, shifts and masks | `autoregressive_editor/test_editor.py` | Hand-compute the input/label pairing for a three-token suffix. Perturb a later correct token and demonstrate that an earlier logit stays unchanged. Explain how parallel training can still produce an autoregressive model. |
| Cross-entropy and autograd | `autoregressive_editor/learning.py` (`Trainer.update`) | Compute cross-entropy for a tiny three-class example. Snapshot one weight before forward, after `backward`, and after `optimizer.step`. Show that only the last operation changes it. Inspect the gradient and explain its sign. |
| Optimizers, batches and numerical behavior | `learning.py`, `run.py` | Explain learning rate, AdamW state, weight decay, gradient clipping and accumulation. Compare token-weighted microbatch accumulation with a single equivalent batch on a deterministic toy case. Diagnose missing `zero_grad`, a detached loss and an omitted optimizer parameter. |
| Training versus generalization | `learning.py` (`evaluate`), saved bank metadata | Memorize two synthetic examples, then test different examples. Explain why falling teacher-forced loss can coexist with poor free-running exact match. Distinguish optimization failure, insufficient information, capacity and overfitting. |
| KV cache and inference state | `model.py` (`Layer.forward`, `propose`) | Compare cached and full-sequence logits. Draw the non-square causal mask for one query after a cached prefix. Run a second independent example and prove that its cache starts empty. |
| Data integrity and leakage | `feature_bank/data.py`, `autoregressive_editor/export_bank.py` | Trace where prefix, original proposal and target labels come from. Explain why 50k target responses do not already constitute 50k editor examples. Reject a corrupted record and a prompt that appears in both splits. Explain why the 384-prompt pilot is not 384 additional unique examples. |
| GPUs, precision and operations | SLURM scripts, runtime reports | Submit only a tiny synthetic job, read its log/status, and distinguish allocated memory from peak memory. Explain FP32 versus BF16, asynchronous CUDA timing, warmup and synchronization. Explain why reading a logit margin matters when argmax differs across execution paths. |
| Experimental comparisons and uncertainty | Historical retained-prefix report, new research reports | Distinguish percentage from percentage points; explain a confidence interval crossing zero. Design a paired prompt-level comparison and separate repairs from damage. Explain why selecting only repairable blocks biases an estimate of real deployment benefit. |
| Complete inference cost | `docs/AUTOREGRESSIVE_EDITOR_PLAN.md` | Build a budget for context/draft prefill, serial editor decoding, vocabulary projection, cache work and target verification. Calculate whether a 5% increase in useful tokens pays for 10% more round time. Explain why full generation can differ from a fixed-context estimate. |

Paths beginning with `autoregressive_editor/` and `feature_bank/` are relative to `editor-experiment/`. Use the actual code as the authority, and distinguish implemented behavior from proposed experiments.

## A small lab sequence

1. Run `describe` and the synthetic tests using the commands in the editor README. Explain the parameter count and each test's failure mode. Do not mistake passing tests for measured proposal quality.
2. Manually construct one prefix, a four-position draft and target labels. Write the exact sequences for both conditioning modes and identify the suffix loss positions before running code.
3. Use vocabulary 20, width 16 and one layer. Inspect logits, loss, gradients and one changed parameter. Make one deliberate bug at a time in a disposable branch: shift labels, expose a future label, omit `optimizer.step`, or retain a stale cache. Predict which checks detect it.
4. Overfit one example, then two, using free-running generation as well as CE. Save/reload the checkpoint and compare the next optimizer update with uninterrupted training.
5. Describe a prompt-disjoint learning curve at 384, 3k, 10k and 50k responses. Specify both unique-example count and token/update budget; more optimizer exposure and more data are different interventions. Reserve a final test set before tuning.
6. Design a timing experiment before running it. Measure the full pipeline, keep replacement and union verification separate, and include a no-editor baseline. Write down the result that would make you stop.

The order matters: verify the mechanics with tiny examples before spending compute on 50k responses. Failure to memorize is a diagnostic signal, not automatic evidence that more parameters are needed.

## Prompt for another model

Copy the following into a model that can inspect this repository. If it cannot inspect files, provide this guide, the editor README, the autoregressive plan and the relevant source files; it must label code-specific claims as unverified until it sees them.

```text
Act as a demanding practical examiner for my DFlash autoregressive token-editor
research project. I have studied the inference pipeline but have little hands-on
training experience. Your objective is to find gaps in my ability to implement,
debug and evaluate the experiment, not to reassure me or lecture at length.

First read AGENTS.md, GLOSSARY.md, docs/EDITOR_STUDY_GUIDE.md,
docs/AUTOREGRESSIVE_EDITOR_PLAN.md, editor-experiment/autoregressive_editor/
{README.md,model.py,data.py,learning.py,run.py,test_editor.py}, and the linked
experiment reports. Cite file/function evidence for code-specific corrections.
Distinguish the historical parallel hidden-state editor from the new token-only
autoregressive variants. Do not assume that a proposed or synthetic experiment
has passed on real data. Do not launch real-data training or bypass the replay gate.

Ask ONE main question at a time and wait for my answer. Begin with a short
diagnostic across data flow, tensor shapes, causal masking, gradients, optimizer
updates, free-running inference, verification and timing. Do not show the answer
or a checklist containing the answer before I attempt it.

At least two-thirds of the examination must be practical: predicting exact
tensor shapes or outputs, tracing a toy token sequence, writing a small function,
finding a bug, interpreting a training curve/log, or designing a falsifiable
experiment. Require concrete calculations, code or evidence rather than accepting
vocabulary. Use tiny CPU/synthetic cases and disposable files for exercises.

Score each response 0–4: 0 no usable understanding; 1 terminology only; 2 partly
correct but cannot execute; 3 correct with a minor gap; 4 correct implementation
and explanation of a failure case. Explain the exact gap. If I fail, give one
small hint, let me retry, then explain and assign a different transfer problem.
Keep a visible compact scorecard and revisit weak topics after several questions.

Include these traps, without announcing the answer:
- backward() versus optimizer.step(), gradient accumulation and zero_grad;
- next-token shifts, suffix-only loss, causal masks and cached query offsets;
- why a decoder can train in parallel yet generate autoregressively;
- why whole-draft conditioning is allowed but future correct labels are leakage;
- tied embedding parameter counts and frozen versus optimizer-owned weights;
- teacher-forced CE versus sequential exact suffix accuracy;
- a preserved wrong first proposal and its effect on consecutive acceptance;
- original, replacement and union paths, including direct prefix damage;
- why target responses alone lack the original DFlash proposals;
- prompt/family leakage, overlapping pilot data, and repairable-only selection;
- BF16 argmax differences, logit margins and faithful target replay;
- checkpoint optimizer/RNG state and cache isolation;
- CUDA timing, warmup, synchronization and all components of editor overhead;
- percentage points, paired prompt uncertainty and practical significance;
- why failed memorization is not proof of inadequate capacity;
- more unique data versus more updates, and train/dev/final-test discipline.

Require a capstone: I must trace one real code path, implement and debug a tiny
synthetic fit, and propose a controlled comparison of both autoregressive input
variants that could conclude "no benefit." I must specify correctness gates,
baselines, data splits, metrics, time/memory budgets and stopping rules. Assess
evidence, not confidence. End each session with my three weakest skills and one
small practical exercise for each. Never certify readiness solely from verbal
answers or invent an experiment result.
```

## Small amount of supporting theory

Use the [PyTorch optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) for autograd and optimizer mechanics, and the [Transformers causal language-modeling guide](https://huggingface.co/docs/transformers/tasks/language_modeling) for next-token training. Read a small section, then demonstrate it in this repository's tiny model. These are supporting references, not evidence that this project's editor works.
