# Can pretrained initialization make compact redrafting practical?

## Decision

Pretrained compression is a defensible comparison family, but copying a teacher layer into the current editor is not a compatible initialization. Do not launch another small-editor fit on the claim that this is a straightforward pretrained replacement. No GPU work, parameter changes, or new architecture were introduced.

## What is actually different in our models

The existing compact module has two width-256 PyTorch TransformerEncoderLayers, default ReLU feed-forward blocks, LayerNorm, and block-only self-attention. It consumes final normalized first-pass states and a broadcast explicit guess embedding. Its vocabulary correction is rank256 and additive.

The teacher has five width2560 native DFlash layers, grouped-query attention with query/key normalization and rotary positions, gated SiLU MLPs, RMSNorm, and direct access to committed target features. It begins from token/noise inputs and privately projects the context. The input interface and nonlinear operations differ, not just matrix sizes.

Configuration arithmetic independently reproduces 537,427,200 teacher-private parameters. One native layer has 100,930,816 parameters; its context projection alone has 32,768,000. The current editor has 3,024,384 parameters. The private parameter ratio is about177.7, but this is NOT a claim of177.7-fold total compression: the editor still relies on the frozen pretrained first drafter and shared vocabulary head. It is learning a narrower residual task. Parameter counts are not latency estimates.

Selecting rows/columns of teacher weights does not preserve this different computation automatically. A mapped initialization would need explicit mappings for the residual stream, attention, MLP gates, normalization and input interface, and would be a changed architecture or an approximation. It cannot be described as a pure initialization control of the existing module without demonstrating that correspondence.

## What published evidence supports

[Minitron, Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/html/2407.14679v1) compresses15.6B models to8B/4B. Its lightweight recovery ablations generally use1.8B tokens. This establishes practical pruning-plus-distillation at those scales, not cheap recovery of our correction function.

[Minitron in practice](https://arxiv.org/html/2408.11796v1), section4 and Figure6, compares random initialization, random pruning, and its pruning/distillation recipe. It reports better starting behavior from retaining pretrained weights. Its final distillation budgets are94B and380B tokens. These are examples, not lower bounds on our data requirements. The measured advantage is language-model quality rather than consecutive accepted-prefix progress.

[MiniLMv2](https://aclanthology.org/2021.findings-acl.188.pdf) transfers attention relations across different student widths/head counts. Its BERT students are randomly initialized and trained400,000steps with batch256 on Wikipedia/BookCorpus. It supports internal-relation supervision as an established technique; it is not evidence that this loss will fix our model at384responses. Applying relation losses would also require a justified alignment between our block-only attention and the teacher's larger context.

[SliceGPT's official implementation](https://github.com/microsoft/TransformerCompression) exposes rotation/slicing and recovery fine-tuning. DFlash's custom architecture is not listed as a supported model on the inspected page. Integration would require validating its special context-attention path, not merely passing our checkpoint to an existing command.

## Important local counterevidence

We have already tested pretrained-derived214,976,512-parameter conditional models. The September22 comparison found4.90%progress gain from reuse and7.11%from recomputation, but reuse cost per accepted token was8.57%worse. Thus pretraining is not an unexplored universal fix. The3M editor's specific initialization remains unisolated, as do the effects of interface and capacity.

The current full-teacher gain also leaves little cost margin: the previous complete-panel economics allowed approximately4.387%uniform added round overhead for5%modeled savings even with perfect teacher-prefix recovery. A larger inherited layer must meet a cost budget, not merely improve learning. The current3M editor's measured production time was about2.44–2.49ms on four contexts; retaining fewer weights alone does not predict proportional time savings.

## Research implication

The defensible design objective is to retain a useful pretrained computation while removing measured expensive operations. It is not to assume that any tiny learned network can inherit the teacher's behavior after a few epochs.

Before funding such a design, specify which existing operations it retains, which it replaces, the resulting input contract, and its complete-round cost model. Compare inherited versus random initialization within that SAME candidate architecture and exposure. Match retained-original verification and score-readout costs. Improvement would establish a method at the measured dose, not initialization as the sole explanation of past failures.

No candidate currently satisfies those requirements just from the reviewed literature. This assessment narrows the next design work; it does not rename ordinary pruning/distillation as a novel algorithm or close the user's cheap-second-pass research question.

Local evidence: architecture.json; ../compact-cost-isolation/attempt1/RESULTS.md; ../../conditional-architecture-cublas-2026-09-22/results.md; ../teacher-target-economics/RESULTS.md.
