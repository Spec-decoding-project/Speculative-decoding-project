# Primary-source notes: token-only editing of DFlash proposals

Research date: 2026-11-26. Literature research and local code/report inspection;
no training, experiments or remote work. Architecture facts were checked against
[the decoder](../../../editor-experiment/autoregressive_editor/model.py),
[input construction](../../../editor-experiment/autoregressive_editor/data.py),
[the gate](../../../editor-experiment/refinement_research/gate.py), and
[native integration](../../../editor-experiment/refinement_research/native_speed_worker.py).
The generator has 40,750,336 parameters, a 151,936-token vocabulary and fourteen
sequential editable positions. The latest linear gate adds 265 parameters.
Measured results are in [the full-model report](../full-corpus-training-2026-11-17/REPORT.md)
and [the current selector report](CHEAP_SELECTOR.md).

## Main conclusion

There is no evidence here that training cannot improve this editor. There is evidence that **the present trained system has not improved acceptance enough to repay its runtime**. Distinguish three claims: (1) the tested checkpoint loses on time, (2) the fixed architecture imposes a latency cost, and (3) no future training/architecture could win. Only the first two follow from the stated project findings. The third does not.

A frozen generator with a learned COPY mixture has a specific limitation: at a fixed input state, gate training changes how strongly the proposal token competes with the existing generator distribution; it cannot teach the generator a new ranking among non-COPY replacements. It can still change subsequent visited states by changing earlier outputs. Full generator fine-tuning is a different experiment from fitting the gate.

## Verified primary-source facts

- **Original speculative decoding:** the target verifies candidate prefixes in parallel; rejection stops the retained prefix. The stochastic algorithm needs the actual proposal probabilities and the residual correction distribution, not merely token equality. Its latency analysis explicitly includes draft cost and hardware assumptions. Source: [Leviathan et al., ICML 2023, Algorithm 1 and §§3.1–3.3](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf).
- **DFlash:** predicts blocks in parallel using target context features. Its training weights early positions more heavily and randomizes anchors; its ablations support both choices. These are precedents for objective/data changes, not proof that our editor will improve. [Pinned original paper, §§3–4 and Appendix A](https://arxiv.org/html/2602.06036v1). Do not conflate this mechanism with newer revisions of the upstream project.
- **EAGLE-3:** removes feature-regression supervision, directly predicts tokens, and trains with its own intermediate outputs to simulate multiple drafting steps. It still **inputs target hidden features**, fusing low/middle/high layers; “direct token prediction” does not mean token-only input. Its ablations show improvements from the changed objective/training and feature fusion, and more training data becomes useful in that architecture. This is evidence against a general impossibility claim, not an experiment on this project. [Paper, §§3–4](https://arxiv.org/html/2503.01840v1); [official repository](https://github.com/SafeAILab/EAGLE).
- **DAgger:** collects states induced by the learner, asks the expert to label them, and aggregates data. Guarantees require the paper’s learning/reduction assumptions. [Ross et al., AISTATS 2011](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf). **Scheduled sampling** mixes generated and reference previous tokens during training; this motivates checking train/deployment mismatch but is not a universal fix or a speedup guarantee. [Bengio et al., 2015, author organization page](https://research.google/pubs/scheduled-sampling-for-sequence-prediction-with-recurrent-neural-networks/).
- **Parallel refinement precedent:** Mask-Predict trains a conditional masked model and revises uncertain positions in a fixed number of parallel iterations. Its results concern translation, not DFlash editing. [Ghazvininejad et al., EMNLP 2019](https://aclanthology.org/D19-1633/).
- **COPY precedent:** pointer-generator networks combine copying source tokens with generating vocabulary tokens. That architecture establishes a modeling option; it does not establish that a soft COPY mixture skips the expensive generator or improves latency. [See et al., ACL 2017](https://aclanthology.org/P17-1099/).

## Cost and objective reasoning (our deductions, not paper benchmark claims)

Let baseline cycle time be T, added editor/selection cost C, baseline committed tokens per cycle tau0, and edited committed tokens tau1. If verification and other cycle costs stay constant, the editor wins only if:

    (T + C) / tau1 < T / tau0
    equivalently tau1 / tau0 > 1 + C / T.

If verification cost changes, include that change in C. Count bonus tokens consistently. This is accounting, not a prediction; native end-to-end measurements remain decisive. More training can raise tau1 without changing inference work, but it does not by itself delete 14 sequential dependencies or a vocabulary projection. A larger backbone may increase both quality and C.

The exact useful statistic is a surviving prefix, not token accuracy averaged over the entire block. For acceptance events A1...Ak, expected committed tokens including one bonus are 1 + sum_k P(A1 and ... and Ak). No independence assumption is needed for this identity. A correction after an earlier rejection contributes nothing to that cycle’s retained prefix; an early harmful edit can destroy later useful tokens. Therefore early-position weighting, preservation of already-correct prefixes, and evaluation of free-running retained length are sensible.

The verified editor width is 256. Its dense vocabulary projection has
151936 * 256 = 38,895,616 weights and approximately 77.8 million multiply/add
FLOPs per token; fourteen projections are approximately 1.09 billion FLOPs.
This derived operation count is not a latency measurement: precision, caches,
kernel fusion, GPU occupancy, batch size, and overhead matter. Embedding lookup
reads selected rows; the dense output projection scores the entire vocabulary.

### Important teacher-forcing caveat

For a token-only autoregressive editor whose state is determined by the same input tokens, and whose teacher-forced previous outputs are the target tokens, inference and teacher forcing agree **until the first wrong output token**. That wrong token already terminates the acceptable prefix. Consequently, correcting errors on later off-target prefixes may improve full-block accuracy without improving acceptance. This is not a diagnosis that teacher forcing is harmless; inspect the actual state construction, labels, and full decoding loop.

EAGLE-3 differs: its self-predicted continuous intermediate inputs can differ from training features even while emitted token IDs are correct. Its training-time-test result cannot simply be transplanted as proof of the token-only editor’s failure mode. DAgger-style collection is most compelling if new editing policies shift block starts, contexts, or proposal distributions in the actual loop, or if the editor has a demonstrated mismatch on prefixes that can still be accepted. Labels must correspond to the visited state and the intended acceptance objective; blindly reusing a saved suffix after arbitrary changed prefixes is not automatically valid expert supervision.

## Five concrete alternatives

1. **Improve full token-only training and its objective.** Fine-tune the generator, not only the COPY gate; emphasize initial failures and preservation of correct prefixes; mix difficult repair examples with representative untouched blocks; select on held-out retained length and native time. Use deployment-loop data aggregation only where the mismatch described above is demonstrated. DFlash’s loss weighting supplies a precedent, and DAgger supplies a collection principle. This can improve usefulness at the same inference architecture cost. Risks: memorization, catastrophic overwrites, loss/metric mismatch, and little information to distinguish genuinely hard cases. This remains fully token-only at inference.

2. **Encode once and edit several positions in parallel.** A shared encoder reads accepted context IDs plus the full DFlash proposal; position heads emit COPY/replacement decisions simultaneously, possibly followed by one more parallel refinement round. Mask-Predict supplies a relevant architectural precedent. This reduces serial rounds from 14 to one or a few, but parallel replacements cannot immediately condition on each other. Jointly inconsistent edits may shorten prefixes, and a second pass must repay its own cost. This is token-only if its inputs remain token IDs.

3. **A sparse replacement interface with a real COPY fast path.** Predict COPY or choose among a small token candidate set; run a full-vocabulary generator only for selected positions if necessary. Candidate sets can come from the input tokens, a learned shortlist, or another token-only mechanism. DFlash logits/top-k candidates are extra inputs and violate a strict token-ID-only requirement. A shortlist can miss the correct replacement; generating the shortlist must itself be cheap. A gate evaluated after the full generator has already run cannot save that generator’s cost. Pointer-generator work motivates COPY, while actual skipping and candidate restriction are proposed engineering changes, not results established by that paper.

4. **Conditionally spend a small editing budget.** Train a cheap block selector to choose skip versus edit, or choose the first one to three positions worth considering. Train labels from net retained-token/time benefit, including harmful edits and selector overhead, rather than simply whether any token can be corrected. This attacks expected cost and follows the speculative-decoding cost equation; it is a proposed policy, not a cited proven method for this project. Risks: gains may be too unpredictable from token IDs, the selector may cost too much, or restricted depth may miss the first useful repair. Keep token-only inputs and include all routing costs in native timing.

5. **If token-only is relaxed: reuse richer representations or improve DFlash itself.** Feed available DFlash/target hidden features to a small parallel refinement head, or devote training/capacity to the existing parallel drafter instead of adding an independent sequential editor. DFlash and EAGLE-3 demonstrate that target features can support effective drafting in their own setups. They do not prove the best feature layer, reuse path, or speed here. Availability matters: accepted-context features may already exist, while target features of unverified proposals require verification and cannot be assumed free. This explicitly changes the user’s input constraint; it is not the default recommendation under that constraint.

## Claims these sources do NOT justify

- No theorem says a 40M token-only editor cannot learn useful repairs or that longer training must fail.
- More parameters/data, lower cross-entropy, higher all-position accuracy, and longer offline accepted prefixes each fail to establish native speedup by themselves.
- A frozen-gate result is not a fair test of every trainable generator or parallel editor architecture.
- Hidden features can supply useful computed information, but token-only prediction is not logically impossible; missing/truncated context and model capacity are separate issues.
- None of the five alternatives is proved faster on this project, hardware, dataset, or verifier. Compare against the unchanged DFlash baseline, using the same target semantics and output-length accounting.
- For stochastic decoding, modifying proposals requires correct probability accounting/verification. Greedy token equality alone does not prove preservation of a sampled target distribution.
