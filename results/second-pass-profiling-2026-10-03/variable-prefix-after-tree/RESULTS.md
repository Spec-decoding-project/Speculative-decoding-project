# Richer-prefix teachers still leave a small competitive margin

Completed locally with no model calls or paid execution. All 2,048 saved contexts from 64 development responses are included. These are the already-trained September 23 fixed-prefix and variable-prefix checkpoints, not new fits or a reproduction of published D²SD.

## Comparison

Retain the ordinary top-16-candidate, free-first 30-node tree and add the saved teacher proposal(s). Within each inference policy, give both teachers the larger of their two union sizes, filling unused places with ordinary candidates. The ordinary reference receives exactly the same number of nodes. Proposal construction and the node allowance use no reference labels.

| Inference policy | Mean nodes | Fixed-trained teacher: modeled savings vs ordinary | Variable-trained teacher: modeled savings vs ordinary |
|---|---:|---:|---:|
| Keep one original guess, propose one suffix | 39.05 | 1.321% | 0.717% |
| One confidence-selected prefix | 39.16 | 0.862% | 1.340% |
| Two confidence-selected prefixes | 47.66 | 0.227% | 1.423% |

These numbers assume equal complete-round time and derive from saved matching-prefix coverage. They do not include the additional teacher inference cost and are not measured speedups. Node allowances match models within a policy, not one- versus two-branch policies.

The primary two-prefix variable-teacher estimate is **1.423%**, with descriptive 95% prompt-bootstrap interval **0.356–2.472%**. Matching-prefix coverage improves 1.767% [0.664–2.873%]. Across contexts it gains 756 positions and loses 350 compared with ordinary expansion. The modeled saving is 4.941% on math but **-2.095%** on general instructions. Do not select the math slice as a replacement success criterion.

Variable training does outperform matched fixed training under the two-prefix policy: 1.273% additional coverage [0.637–1.909%], corresponding to 1.184% modeled equal-round-cost savings. This positive learning effect remains part of the record. It is insufficient to establish a useful advantage over the ordinary alternative, and it does not establish that the small student can preserve it.

## Decision

Our existing richer-prefix checkpoints do not supply the substantial competitive teacher benefit needed to justify another compression fit. A second proposal improves on a single original chain, but inexpensive expansion of the original scores absorbs much of that opportunity. This is why a better teacher-imitation loss alone is currently a weak next bet.

The result is not a general limit on conditioning, redrafting or training at larger scale. The original experiment used one seed, 384 responses, positive prefix lengths and a 4B target. The paper permits a zero-length guessed prefix and normally uses four branches on larger targets; our negative result does not test that full system. [D²SD, Sections 3.2–4.1](https://arxiv.org/html/2606.04446v1).

The previous successful goal turn tested single-token conditioning with fixed weights. This turn completes a separate competitive audit of already-trained richer-prefix models. Both old training results remain unchanged; no checkpoint selection, new training, generation or published-performance claim occurred.

## Checks and limits

20,480 construction/count checks passed. All 64 metadata files, 64 policy files and 128 final prediction files were authenticated, plus the wider-candidate bank. Token arrays, label masks and anchor mappings join exactly. All historical retained-progress totals reproduce, including fixed/variable one-selected-cut totals 12,038/12,139 and two-cut totals 12,316/12,548. Ordinary trees agree under independent heap/frontier construction; unions preserve prefix closure and exact budgets. Inputs and hashes, complete per-response results and every comparison are in `results.json`.

Repeatedly exposed prompts, fixed recorded references and teacher mode/exposure differences limit conclusions. Equal node counts are not equal runtime. Candidate support beyond the original top 16 is not tested. No actual generation speedup or cheap editor is established.

## A newly available comparison resource, not a new claimed method

The official xPress repository now lists a **Qwen3-4B** checkpoint. The public Hugging Face metadata records creation on September 30, after our September 22 xPress experiments, which used Qwen3-8B. The pinned revision is `fce84732637e1fb70bea3c70ca341131b8d49b61`. Only metadata, config and a 7,064-byte tensor header were read; no tensor payload was downloaded. [Official repository](https://github.com/Supercomputing-System-AI-Lab/xPress/blob/main/README.md), [checkpoint](https://huggingface.co/UIUC-SSAIL/Qwen3-4B-XPress-b16).

The config specifies width 2,560, five draft layers, block size 16, target layers 1/9/17/25/33 and six refinement passes. The header contains 617,577,728 parameters, including the refiner. This is a separately co-trained backbone/head, not a head we can safely attach to our unchanged q0 features. Shape compatibility is not released-weight execution validation or evidence of a gain on our workload.

It is now possible to assess an already-trained refiner in the same target family before paying to create a stronger teacher. That assessment must compare its co-trained backbone with refinement off, the released original drafter, and ordinary candidate trees, then charge complete refinement and verification costs. Success as a reference would not be our contribution, and compressing an already cheap refiner may have too little removable cost. Our old 8B profiles placed refinement near 1.2–1.3% of instrumented round time, so a large saving from removing it cannot be presumed for this 4B checkpoint. The old reuse/cache/readout variants stay closed.

The new checkpoint is a concrete prerequisite to investigate, not a training recommendation or funded launch. Current unallocated funds remain $0.035066676845026734, with no active paid jobs. Broad cheap-redrafting research remains active and unmet. Source receipts are in `xpress-4b-source/`.
