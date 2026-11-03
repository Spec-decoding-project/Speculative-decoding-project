# Cheap conditional redrafting research record

Research remains active on replacing one expensive conditional second drafting pass with a cheap editor. The useful contribution is unproven. This record consolidates the October 1 profiling, local analysis, and three independent reviews. No new cloud work or training was performed during these reviews.

## Evidence and its scope

The completed diagnostic covered 48 fixed contexts from 24 exposed development prompts, with 20 paired timing repetitions per condition. The original draft committed 222 positions; revised-only committed 240; retaining both committed 251. Mean complete-round times were 27.10, 34.38, and 34.95 ms. Pooled cost per accepted position increased 17.36% and 14.05% for revised-only and union. These are not generation-speedup results.

Revision improved seven contexts, damaged six, and left 35 unchanged. Two repairs supplied 22 of 29 gross additional positions. Of those 29 positions, three already matched at the same location in the original proposal. Other changes include realignment: they do not establish missing information or a need for a complex rewriter. Lower-ranked original candidates were not saved by this timing runner.

The teacher is a separately trained five-layer model receiving committed target features and the first guessed token, x1. It does not receive later draft guesses. The older matched training study found a 2.157 percentage-point feedback advantage over its no-feedback control, while both trained models improved over the original. Overall gains cannot be assigned entirely to explicit token feedback. In the new panel, ten contexts reject x1; a method that clamps x1 cannot improve their prefix through suffix editing.

Evidence: [completed outcome](gpu-attempt4/outcome.json), [repair anatomy](gpu-attempt4/repair-anatomy.json), [historical controlled study](../conditional-full-reference-precision-2026-09-22/results.md).

## Concrete implementation hypothesis

Use already-normalized first-pass states and retained first-pass scores. A compact coefficient network predicts a rank-r residual. With final head input h, coefficients z, up-projection U and frozen vocabulary head W:

`(h + z U^T) W^T = h W^T + z (W U)^T`.

Precompute W U after training, retain h W^T from the first pass, and compute only the correction. This is exact algebra for this parameterization, not an exact replacement for the trained teacher. Post-residual nonlinear normalization breaks this simple identity. BF16 regrouping can change predictions and needs a deployment check.

The CPU forward and gradient checks passed. At measured width 2560 and rank 256, residual vocabulary projection uses 10% of full-width projection arithmetic. It adds about 74.19 MiB of BF16 folded weights; retaining the 14 editable score rows needs about 4.06 MiB before temporary buffers. Folding costs about 199 GFLOPs once and must be refreshed after weight updates. These are resource calculations, not latency results.

The current runner retains token IDs but discards the first-pass score tensor. The replacement must retain scores explicitly. Editable states are rows 2 onward; original logits start at row 1, so editable reused logits start at their row 1. Preserve this alignment.

Evidence: [algebra checks](gpu-attempt4/readout-feasibility.json), [kernel audit](gpu-attempt4/kernel-cost-audit.json).

## Cost claims to avoid

The 0.473 ms vocabulary-kernel sum is separately instrumented eager evidence, not a wall-time lower bound. The 0.351 ms editor allowance assumes all observed gains survive, unchanged verification overhead, constant editor cost and equal-prompt savings. Pooled weighting instead yields 1.447 ms. Neither is a universal design budget. Recompute the economic model from matched new measurements; do not choose a weighting after observing which passes.

The present target verifier is graphed, but both draft networks are eager. Intermediate-state capture is not CUDA graph capture. Optimizing only the student would confound architectural savings with execution optimization.

Evidence: [cost scenarios](gpu-attempt4/replacement-cost-scenarios.json).

## Novelty ledger

| Established work | Overlap that must be acknowledged | Remaining question |
| --- | --- | --- |
| [xPress](https://arxiv.org/html/2608.02438v1) | Reused hidden states and logits, low-rank token-conditioned correction | Does a specified second-pass teacher improve a frozen-backbone editor beyond matched direct supervision? |
| [DEdit](https://arxiv.org/html/2609.38510v1) | Whole-proposal editing and a frozen-proposer separate-editor ablation | Can a cheaper reused-state model replace useful conditional recomputation? Our x1 teacher is not DEdit's future-token mechanism. |
| [LiLiCorr](https://arxiv.org/html/2608.20530v1) | Cheap candidate correction using draft information | Dense residual output can leave top-K; whether that matters requires candidate-coverage evidence. |
| [DiMS](https://aclanthology.org/2023.findings-acl.542/) and [CLLMs](https://proceedings.mlr.press/v235/kou24a.html) | Learning to replace iterative refinement with fewer steps | Generic refinement distillation is not our invention. |
| [DistillSpec](https://arxiv.org/abs/2310.08461) | Distillation for speculative acceptance | Does an intermediate conditional drafter teach anything useful beyond direct target alignment? |

No exhaustive novelty claim is established. Folding and original-proposal fallback are implementation choices, not standalone contributions. A successful controlled result must identify a learning and total-cost advantage over appropriate existing refinement methods.

## Training comparison correction

The provisional loss in experiments/second_pass_distillation.py halves all-valid target CE when eligible teacher rows exist, but uses full CE otherwise. KL is also normalized by eligible count. Thus loss strength and per-row teacher weight vary with batch composition. A local test confirmed approximately half the target-only gradient when teacher KL has zero gradient. Historical code is preserved; this objective must not launch unchanged as a clean teacher-information test.

Proposed correction: fixed all-valid CE plus lambda times the sum of eligible row KL divided by the number of valid rows. Use three matched arms: target-only, target-only with the same eligible-row emphasis, and target plus eligible teacher KL. The extra control tests whether teacher-correct example selection explains a gain. Correct teacher winners do not make all teacher probabilities harmless. If claiming useful alternative-token structure specifically, test an entropy-preserving non-target permutation in a later bounded diagnostic.

Freeze one input contract before learning. The prior whole-guess bidirectional student proposal and the x1-matched candidate are different hypotheses. Start with x1 for the specified teacher comparison; full-token feedback requires a separate input ablation. Fresh held-out prompts must exclude the repeatedly inspected development data.

Evidence: [objective audit](gpu-attempt4/objective-audit.json).

## Next bounded work

First prepare one untrained rank-256 coefficient network and compare direct versus folded scoring with identical weights and inputs. Include unchanged drafting and the full reference. Count input preparation, score retention, embedding lookup, network work, vocabulary addition and selection, packing, verification, cache work, memory, folding setup and graph setup. Use matched eager execution first; any captured comparison must optimize controls fairly.

For readout timing isolation, use identical downstream verifier inputs or demonstrate matching candidates. Nonzero corrections prevent a degenerate identity benchmark. Validate score errors and token differences separately. One context is enough for an initial cost falsification, not general runtime evidence.

Only a plausible complete cost warrants matched learning. Evaluate both edited-only and retained-union progress, teacher repairs, student-only repairs, damage, domains and prompt-level uncertainty. A weak result must not be rescued by pooling or selecting the two largest repairs. The teacher may be too limited, the student may not transfer, or direct training may already be equally effective. Those are distinct outcomes.

## Independent reviews

Three agents independently reviewed prior art, systems feasibility and learning design. They agreed on the narrower question, identified the loss confound and normalization/score-retention requirements, and rejected treating either FLOP ratios or the 0.35 ms scenario as measured feasibility. Their consolidated conclusions are incorporated above. No model was trained or GPU allocated in this follow-up.


## Measured compact prototype and positional edit limits

A subsequent bounded synthetic GPU probe completed on an A100 80GB PCIe. The candidate has 3,024,384 trainable parameters: two width256 attention layers, first-pass normalized states and x1 embedding, and a rank256 post-normalization residual. Local FP64 scores, all-parameter gradients, identity initialization, row alignment and folded-weight refresh checks passed. No model was fitted on research data.

| Execution | Direct coefficient network and scoring | Folded coefficient network and scoring |
| --- | ---: | ---: |
| Eager median | 1.088 ms | 0.669 ms |
| CUDA graph median | 0.602 ms | 0.206 ms |

These synchronized wall times cover the coefficient network, vocabulary score calculation and argmax over resident synthetic inputs. Thirty alternating paired repetitions were retained. Folding reduced the measured component by65.8% in graph mode, not whole-generation latency. One-time matrix preparation was0.971ms. Graph preparation was excluded. Three of14 BF16 winners differed, with maximum score difference0.015625: real-arithmetic equivalence does not establish deployment identity.

The probe excludes actual first-pass score retention/materialization, embedding lookup, copies, packing, verification and caches. Its base scores come from a14row projection, while production must retain the last14rows of the original15row projection. Earlier reference profiling used A100 SXM4, so do not insert this PCIe timing into the old budget as a matched result. The process-wide peak memory includes both graph pools and validation tensors, not an isolated per-arm footprint. Freeze and test the folded path directly as deployment arithmetic.

A separate exact recount of saved teacher prefixes found that a perfect selector limited to one positional replacement per block could recover7/29 gross gains; two replacements14/29; four18/29; eight26/29. This rules out those sparse positional-edit restrictions as ways of reproducing all observed teacher progress. It does not bound insertion/deletion algorithms, novel student continuations, general workloads or learnability. The candidate therefore permits parallel changes throughout the block, without knowing at inference which positions are wrong.

The synthetic probe was capped at$0.35 within the existing campaign limit after conservative lifetime reconciliation. The initial local result read failed because the launcher lacked torch; the same completed call was recovered with a torch-equipped interpreter, without another GPU submission. The error is preserved. Shutdown is verified, no temporary volumes were created, and no follow-up launched.

Evidence: [prototype checks](gpu-attempt4/prototype-checks.json), [edit-budget recount](gpu-attempt4/position-edit-budget.json), [GPU timings](compact-cost-probe/results.json), [interpretation and limitations](compact-cost-probe/analysis.json).

Next: integrate the folded candidate into the actual first-pass score-retention and dual-proposal path, measure complete costs against equally optimized controls, and then perform the fixed-CE teacher-supervision comparison on fresh held-out prompts if economics remain plausible. The component result justifies this integration question; it does not justify claiming teacher compression or an original technique.


## Controlled learning objective implemented

A new module, experiments/controlled_second_pass_distillation.py, implements the corrected comparison while preserving the legacy objective. The all-valid CE coefficient is fixed at1. Teacher KL and the eligible-target-CE control use the same all-valid denominator. Temperature1 is explicit. Teacher gradients are detached; eligibility is training-only.

At a valid position with student probabilities p, target label y, teacher probabilities q and correctness indicator m, the student-logit gradient is `[(1+lambda*m)p - one_hot(y) - lambda*m*q]/N`. Thus the teacher arm changes both the soft target and eligible-position weight. The eligible-target-CE arm controls the latter. A further permutation of non-target teacher probabilities can preserve target probability and entropy while removing token-specific alternative structure. This is an optional mechanism diagnostic, not evidence that the real teacher's structure is useful.

CPU checks verified the formula to maximum absolute error2.78e-17, absence of teacher gradients, exact full-CE behavior at zero eligibility, preserved CE gradient at zero KL gradient, and equivalent gradients under valid-count-weighted microbatch partitioning including unequal valid counts. These checks establish the intended intervention, not learning quality, convergence, or speedup.

Evidence: [controlled objective checks](gpu-attempt4/controlled-objective-checks.json). No new GPU work. The next empirical requirements remain integrated runtime and fresh held-out learning; neither is supplied by an objective proof.

## Integrated editor and the remaining proof obligation

The compact editor now runs inside a complete fixed-context verification round in experiments/compact_second_pass_decode.py. It retains the actual first-pass 15-row score tensor, edits its last 14 rows, preserves x1 and the original branch, and verifies the two proposals before updating the target cache. The editor has no private context cache. Folded weights are prepared outside the round for a frozen editor; callers must rebuild them if weights change.

Eight CPU rounds on two tiny native-model prefixes passed original-path parity, zero-residual identity, score alignment, independent causal scoring of both branches, native accepted tokens and bonus, and next-step cache scores. Both zero and nonzero residuals were exercised. These random tiny examples produced no acceptance improvement; they establish integration correctness in those cases, not learned quality or released-model GPU behavior. A cold code review correctly noted that none selected the revised suffix, leaving its physical cache gathering untested. An added deterministic verification-only fixture forced a correct revised branch and incorrect original suffix: all six revised positions were accepted and the following cache scores matched fresh causal evaluation. This fixture intentionally uses target answers to test verification, not as student input or evidence of learned repair. Repeated full decoding and released-GPU checks remain outstanding. [Test evidence](gpu-attempt4/compact-integration-checks.json).

The computational rationale needs a precise information distinction. With fixed output weights, x1 is a deterministic argmax of first-pass states. Given those full states, explicit x1 supplies no additional information in the information-theoretic sense. It makes an already-computed vocabulary decision available through a short computational path. Whether this helps a restricted small network is empirical. Moreover, the full teacher reprocesses committed target features that may not be recoverable from the student's first-pass states. There is no proof that the student inputs are sufficient to reproduce the teacher.

For fixed identical target branch scores, retaining both proposals and choosing the longer verified prefix cannot reduce accepted length relative to the original proposal: the candidate set still contains the original. This guarantee concerns selection, not runtime, and requires branch-equivalent target scoring. Finite-precision shape effects can change verifier decisions; native-output and cache checks remain necessary. More verified nodes also cost time. The observed 251 versus 222 positions therefore establishes available useful proposals on this development panel, not a free gain.

At a fixed context, if original round cost is C and progress is L, an editor adds complete cost dC and progress dL, the cost-per-position ratio is `(1+dC/C)/(1+dL/L)`. Five percent lower cost per position requires `dL/L >= (1+dC/C)/0.95 - 1`. Two percent added complete-round cost therefore requires at least 7.37 percent more progress under this model. Prompt-level averages and generation trajectories must be measured directly; inserting aggregate or cross-hardware numbers into this identity does not prove an end-to-end speedup.

A cold review found the architecture and corrected three-arm objective reasonable to test. It did not find proof of compressibility. The decisive learning comparison remains the same student trained with target CE alone, matched teacher-correct-row emphasis, or teacher probabilities, with identical data, initialization and exposure. An advantage from teacher probabilities would support their practical value under the recipe. It would not isolate x1's causal contribution; that requires a matched no-feedback teacher. An entropy-preserving alternative-token permutation would further distinguish meaningful alternative structure from generic soft supervision if the first result warrants it.

The primary sources were rechecked: [xPress](https://arxiv.org/html/2608.02438v1) already uses inexpensive additive logit refinement, and [DEdit](https://arxiv.org/html/2609.38510v1) already studies draft editing. The current work must establish an advantageous way to learn and deploy second-pass replacement; small size, folding and editing alone do not establish novelty.

No paid work occurred during this integration. The campaign ledger currently leaves approximately $0.14 unallocated; it cannot cover an integrated released-model run or the learning comparison. No additional spending was inferred from the request to continue research.

## Saved training data compatibility

The historical restore_example function sets final and normalized to the saved layer-3 midpoint as placeholders. That was sufficient for its former consumers, but those fields are not valid compact-editor inputs. Direct reuse would silently change the proposed experiment. experiments/compact_second_pass_bank.py now reconstructs the final state through the last two frozen layers and final normalization, rejecting cached-history inputs that require unavailable layer caches. The restored-example entry point also checks original draft-token agreement before returning scores.

Three tiny native CPU cases with context lengths 6, 17 and 31 reproduced final and normalized states exactly. Using the placeholder instead produced maximum element errors of approximately 2.20 to 2.70. One real bank was inspected: it contains BF16 committed target features and 32 by 16 by 2560 BF16 midpoint states, with no final first-pass states or teacher probabilities. This validates the reconstruction algorithm locally, not released-model GPU equivalence or all-bank integrity. [Evidence](gpu-attempt4/compact-bank-reconstruction.json).

Reading all metadata confirmed 384 training prompts, 12,288 training anchors, and 64 repeatedly exposed development prompts. Training can reuse these records without collecting new target continuations, after reconstructing correct student features and collecting the frozen second-pass teacher outputs. Fresh evaluation still requires an independently cleared holdout. The old development set cannot become fresh merely because the editor is new.

For all training anchors, storing base and teacher logits for 14 editable positions each would require 97.37 GiB at BF16, excluding metadata and overhead. Storing the 16 student states and 14 teacher states instead requires 1.758 GiB if both are BF16, or 2.578 GiB with FP32 teacher states. Preserve the chosen teacher precision; casting it for storage changes supervision. The same frozen output head can reconstruct distributions from states, but readout dtype and computation shape still require checks. Full-vocabulary scoring remains a training cost and temporary memory requirement. These are exact tensor-size calculations, not measured training time. [Storage calculation](gpu-attempt4/compact-training-storage.json).

This preparation removes a concrete input mismatch and identifies a practical storage route. It adds no learned repair result. Integrated released-model runtime, teacher extraction compatibility, learning exposure and held-out performance remain open; no additional GPU work or spending occurred.

## Breadth of useful teacher corrections

A prompt-level recount of the 48-context profiling panel found that seven of24 prompts contributed any retained-union gain. Two math prompts supplied22 of29 additional positions. Removing those two as a sensitivity analysis leaves7 extra positions over208 original positions, or3.37percent pooled progress. This is not a replacement primary estimate; it shows why reproducing the small panel's13.06percent should not define student success. [Concentration evidence](gpu-attempt4/teacher-gain-concentration.json).

The larger historical64-prompt,2048-block fixed-context comparison supports a broader teacher effect. The feedback teacher's retained union improved62 of64 prompts, adding1136 positions over11005 original positions. The largest two prompts contribute8.98percent of that gain. Excluding them leaves9.75percent pooled progress improvement. This reduces the concern that all observed teacher opportunity comes from two examples, while retaining the limitations of exposed development data and recorded-label rather than native-generation evaluation. The two panels use different contexts and verification measurements; do not treat their difference as a replication discrepancy.

The matched trained no-feedback teacher also adds950 positions. Equal-prompt gains are10.208percent with feedback and8.051percent without it, a2.157percentage-point difference. Feedback wins on42 prompts, loses on17 and ties on5. Both teachers repair227 of the same blocks;126 are repaired only by feedback and72 only by no-feedback. Neither teacher dominates every block. These numbers support a benefit from feedback-enabled training and inference in this matched recipe, not the claim that all second-pass benefit comes from token feedback.

The descriptive domain split further narrows the mechanism: equal-prompt feedback minus no-feedback gain is0.419percentage points on math and3.895points on chat. Math has16 winning versus12 losing prompts; chat has26 versus5. These are descriptive subgroup findings without a new significance claim or explanation of language structure. Do not select only chat as confirmatory evidence after observing this split.

The next student's primary practical question remains whether it can economically learn the useful teacher behavior. A teacher-supervision win over direct supervision would not itself show that token feedback caused the win. Any later specific claim about learning conditional feedback needs the no-feedback teacher control. Existing token-feedback and no-feedback teachers are already saved, making that distinction concrete without inventing a new architecture. [Larger-panel evidence](gpu-attempt4/teacher-feedback-support.json). No model calls or paid execution occurred.

## Matched complete cost comparison implemented

measure_compact_comparison in experiments/compact_second_pass_decode.py now compares original-only, teacher-revised-only, teacher-union and compact-union rounds. All draft networks execute eagerly and share the fixture's verifier execution mode. The timed region includes first-pass scores, actual score retention, x1 lookup, correction, proposal packing, target verification and cache compaction. Initial cache clones and fixture preparation are excluded consistently; folded-weight preparation is timed separately. This is diagnostic round accounting, not a fully optimized serving benchmark.

The runner rotates arm order, alternates direction, warms each arm, checks repeated output stability, records complete outputs and times, checks original-proposal preservation, and rejects parameter-version changes that could invalidate folded weights. The local test ran all four arms twice after warmup and passed. Tiny CPU timings are deliberately not reported as GPU predictions. A deployment must still enforce frozen checkpoint identities and perform released-weight fidelity checks; the local runner does not create a cloud allocation, stage assets, implement recovery or verify remote shutdown.

Different edited proposals produce different downstream verification work. Thus the complete-policy comparison cannot isolate the matrix-folding contribution by itself. The earlier synthetic paired-input test addresses component cost; complete-round testing addresses the combined policy. Neither may substitute for trained held-out generation. A later CUDA-graph comparison must optimize the full reference and unchanged drafter too; an eager failure would not prove all optimized implementations uneconomical.

The GPU bundle is not launch-ready: released-weight auditing and the bounded launch/recovery wrapper must be connected before requesting funding. No spending request or cloud execution occurred in this step. [Local runner checks](gpu-attempt4/compact-integration-checks.json).

## Repair retention and prompt economics

The GPU bundle was subsequently prepared and locally checked, as recorded in [its protocol](compact-integrated-cost/PROTOCOL.md); funding remains pending. While awaiting authorization, a saved-data analysis enumerated all128 ways to retain or lose the seven observed teacher repair blocks. This intentionally assumes each block's teacher gain is either wholly recovered or entirely lost. It does not model partial repairs, new student-only repairs, a deployable selector, or probabilities of learning each pattern.

At a hypothetical constant0.2ms editor cost, five patterns reach5percent equal-prompt modeled savings; every passing pattern recovers at least27 of29 extra positions. Even with zero editor cost, passing patterns recover at least26. At0.5ms, none passes, including full teacher retention. These are conditional results using the historical verification-cost differences, not measured student speed, a universal retention requirement, or a chance-of-success estimate.

Two patterns can recover the same14 positions yet yield approximately0.30percent versus1.73percent modeled savings at0.2ms. Aggregate recovery therefore does not determine equal-prompt economics. Preserve per-prompt costs and progress in the learning evaluation; do not promote a student solely because its pooled teacher-retention percentage looks high.

The calculation uses median timing repeats per context and pools costs and progress within each prompt before averaging prompt ratios, matching the frozen analysis. Its full-retention zero-editor case reproduces the earlier6.190023percent scenario within1e-10. The0.2ms value is an illustrative scenario, not the synthetic PCIe component timing transplanted into the SXM4 full-round measurements. [Evidence](gpu-attempt4/repair-retention-economics.json). No paid work occurred and no campaign cap was changed.

## Controlled editor learning implemented locally

After the completed integrated cost run, experiments/compact_second_pass_learning.py implements matched target-only, eligible-target emphasis and teacher-KL updates using the corrected loss. It keeps base CE fixed, normalizes accumulated gradients by actual valid editable rows, refreshes the differentiable folded output matrix for each update, and requires frozen features/head weights. AdamW applies ordinary updates without fixed-magnitude rescaling. The deployment evaluator requires a separate frozen copy and batch-one inputs, reports original/edited/union prefix progress, repairs, damage, fully-correct-block damage and unreachable x1 cases.

Synthetic CPU checks executed two real optimizer updates per arm. Effective-batch versus unequal-valid-count microbatch parameters matched within4.34e-19. Zero-initialized up projection initially produces no coefficient-network gradient, then nonzero coefficient gradients appear after its first update. Frozen inputs remained unchanged. Saving/reloading each checkpoint and recomputing folded weights reproduced evaluation predictions. These are optimizer-contract checks, not training results on research data. [Evidence](gpu-attempt4/compact-learning-checks.json).

A cold design review supports the three-arm screen and requires first-pass states, x1 and scores to come from the same reconstruction; the bank entry point currently rejects mismatched saved originals. Teacher scores are training-only. The candidate screen uses the384 historical training prompts and64 exposed development prompts with fixed checkpoint reporting; it cannot establish fresh generalization. Training and evaluation must preserve intended readout row shapes and BF16 deployment arithmetic. Full-vocabulary probabilities and training throughput still need actual GPU calibration before committing the remaining budget. No new paid run launched during this implementation.

## Feature extraction connected to learning

experiments/compact_second_pass_features.py connects historical midpoint reconstruction to the deployed teacher and controlled trainer. It derives x1 from the same reconstructed15-row vocabulary calculation used for the original draft, checks its embedding against the restored input, and stores first-pass normalized states, teacher14-row normalized suffix states, original and teacher winners, labels and masks. Teacher features and scores are detached. During materialization, each example uses its own15-row and14-row readouts before stacking; this avoids silently changing the vocabulary GEMM shape across examples. Saved winner checks fail before training if readout reconstruction changes token decisions. Full checkpoint identities and released-GPU numerical validation remain necessary.

The tiny-native CPU bridge test passed reconstructed student and teacher state equality, serialization/materialization, nonzero optimizer gradients in all three arms, and unchanged edited tokens when supervision labels change. Synthetic labels deliberately include teacher-correct positions to exercise both auxiliary losses; they are not evidence of target accuracy. [Evidence](gpu-attempt4/compact-feature-bridge-checks.json).

The proposed fixed exposure is three epochs over12288 training anchors,1152 batch32 updates per arm, reporting each fixed epoch on exposed development prompts. A GPU calibration must measure no-cache feature extraction, full-vocabulary materialization, forward/backward/update and deployment evaluation before treating this exposure as affordable. The small parameter count does not itself bound those costs. The existing$8.885 balance remains unchanged; no new paid run occurred in this bridge implementation.


## Full controlled learning comparison

Released-model calibration passed before this run: 192 extracted examples and all twelve updates reproduced the intended readouts and checkpoint predictions. Measured compute projected approximately sixteen GPU minutes for the full comparison, excluding staging, I/O and recovery.

The full comparison now uses all 384 historical training prompts and 64 exposed development prompts. Target-only supervision, eligible-target emphasis and teacher probability supervision receive the same initialization, three epochs, optimizer and training order. Epoch three is primary; no checkpoint is selected from development gains. Each epoch reports retained-union coverage, repairs and edited-prefix damage. Prompt-paired uncertainty describes development variability, not new-data generalization or training-seed robustness.

Teacher-KL beating both controls would support the value of this teacher distribution under the tested recipe. It would not show that the first guess supplies new information, that the module reproduces all second-pass computations, that the technique is novel, or that its generation speed improves. Failure would reject this fixed recipe rather than all possible compact editors. The existing complete-round overhead means positive coverage alone is insufficient.

The attempt has a $4 cap within approved funding, no retries, a 45-minute GPU maximum and separate bounded staging. Its [protocol](compact-full-training/protocol.json) and [execution record](compact-full-training/attempt1/execution.json) preserve the scope and live status. No result is assumed before completion.


## What the teacher supervision contrast identifies

For an eligible training position at temperature one, CE(y,p) + lambda KL(q||p) equals (1+lambda) CE((onehot(y)+lambda q)/(1+lambda),p) minus lambda times teacher entropy. The entropy term is constant with respect to student parameters. With lambda=0.5, the effective target assigns off-label mass (1-q_y)/3. The teacher therefore changes both confidence-dependent smoothing strength and which alternatives receive that mass. The eligible-CE control matches emphasis and total target coefficient, but does not separately isolate these two effects.

A positive teacher-KL result would establish benefit of this combined supervision under the tested conditions. It would not establish that alternative-token ranking caused the gain. If that distinction becomes consequential, a matched control can retain the correct-token probability and permute only the off-label probability identities, keeping eligibility, entropy and smoothing mass fixed. This is a possible follow-up, not an added arm or automatic spending commitment. A double-precision algebra and gradient check confirms the identity in [the saved check](compact-full-training/loss-interpretation-check.json).


## Teacher denominator audit

The historical per-block details exclude the known root, but their per-prompt summaries already include it. For all64prompts, original_progress equals the sum of one plus each block's original detail. The summary totals11005original positions, not8957; adding another root per block double-counts it. The earlier8.3969percent conversion was incorrect and is withdrawn. The historical teacher and new independently reconstructed teacher both show10.20799percent equal-prompt root-inclusive gain, with1136extra positions.

The structural recount remains valid:2048blocks include331wrong-first-guess blocks,183alreadyfull blocks and1534editable first rejections. The teacher repairs353blocks. These oracle categories are not deployable gates. [Audit](compact-training-completion/attempt2/denominator-audit.json). The withdrawn note is preserved separately.


## Added cost and whole round cost use different denominators

Across the four measured integrated-cost contexts, full-teacher union adds 8.404 to 9.002ms above original decoding, while untrained compact union adds 1.414 to 1.742ms. The compact policy removes 80.36 to 83.45 percent of that added cost, corresponding to 18.39 to 20.00 percent of the complete teacher-union round. This is a difference of matched complete-round medians, including packing and changed verification work; it must not be labeled an isolated editor-kernel speedup. Neither ratio establishes learned generation speed. [Exact breakdown](compact-integrated-cost/attempt1/added-cost-breakdown.json).


## Completed controlled student training

The [full comparison](compact-training-completion/attempt2/RESULTS.md) completed. Epoch-three retained-union gains were3.02percent for direct labels,3.70percent for eligible-row emphasis, and3.46percent for teacher probabilities. Neither paired teacher-probability contrast excluded zero. The probability-trained editor covered218of the teacher's1136extra positions;386total student extra positions include other gains and are not literal teacher retention. The full teacher baseline remains10.21percent on the matched root-inclusive metric. No generation speedup or novel contribution was established, and no automatic scale-up is justified. All evidence and operational failures are preserved; no jobs remain active.
