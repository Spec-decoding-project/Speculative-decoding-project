# The old teacher is a weak competitive compression target

**Do not launch the proposed full-proposal fine-tune on the old DFlash teacher yet.** Local implementation and data checks pass, but a newly aligned stronger-reference comparison exposes a much larger competitive gap than the old same-drafter comparisons showed. Preserve the implementation; it is not a trained model or a successful method.

## The learning question and its counterargument

The concrete proposed input change supplies all 15 guesses from the completed first draft to the full second-pass teacher, instead of supplying only its first guess and masking the rest. Target features still stop strictly before the draft anchor. The first guess remains unchanged in the output. Matched continued training with only the first guess would separate the input change from extra training and the teacher's pretrained weights.

This could let the second pass revise predictions in light of explicit choices throughout the block. It could instead learn to copy mistakes, overfit the small training set, or add no useful information beyond the existing first-pass representations and refiner. Visible later matches do not prove useful information, recognizability, or a learnable correction. It is not a novel teacher mechanism: the prior DEdit review already describes completed-proposal editing. Our intended contribution remains economical learned compression of useful second-pass computation.

The recent paired-score intervention also rules out treating input sensitivity as sufficient. Perfectly supplying the old teacher's alternative-input response did not make the current compact student competitive. A stronger teacher or a justified way of learning its useful original-input predictions is needed before another student fit.

## New matched-context comparison

All 256 contexts from 64 exposed responses align exactly on recorded labels and validity masks. Each proposal set has 64 nodes; counts include the known anchor. The initial drafter weights, proposal computations and selection rules differ. These are fixed-prefix coverage observations, not matched-latency measurements.

| Established proposal construction | Total matching positions |
|---|---:|
| Ordinary DFlash scores | 1,827 |
| Original DFlash plus old teacher chain, ordinary fill | 1,858 |
| Old teacher scores, original first guess | 1,886 |
| Old teacher scores, two first-guess conditions | 1,898 |
| Released xPress scores with ordinary expansion | 2,350 |
| xPress conditional expansion | 2,696 |

The two score-based teacher policies were a disclosed supplementary check after the chain comparison, so the conclusion does not rest only on forcing a complete teacher chain into the budget. They reuse established procedures; no allocator was optimized or proposed. An independent explicit-frontier implementation reproduces all 64 selected score-policy trees. Strong-reference endpoints are independently recounted from their saved trees, with all 256 label alignments verified.

Even the best checked old-teacher policy improves only eight of 64 response aggregates over shared xPress expansion and one over conditional expansion. At equal round cost it models 26.60% higher cost/progress than the shared xPress comparison and 49.02% higher than conditional expansion. Those percentages are scenario calculations, **not measured slowdowns**. Reaching 5% lower latency under the same uniform-ratio model would require rounds 24.96% cheaper than the former or 36.25% cheaper than the latter; no applicable timing proves that.

This does not identify why the small editor failed to learn. It identifies why solving that learning problem against this teacher would not, by itself, establish a competitive inference method. It is not a universal bound on a student that improves beyond its teacher or on other proposal policies. No end-to-end correctness or speedup is claimed from these saved references.

## Implementation and data preparation completed

`experiments/full_proposal_teacher.py` implements the two input conditions, shared weighted suffix loss, matched fitting, and evaluation using the supplied draft rather than target answers. It does not allocate cloud resources.

Ten tests passed in 39.82 seconds. They check:

- Exact control loss and complete parameter-gradient equality with the historical first-guess training path on eligible examples.
- An actual two-update control replay with exact resulting weights; full-proposal updates change the trained weights under equal supervision.
- No future target-answer or target-feature leakage, including perturbation checks.
- Native attention can use later supplied guesses; the first-guess control cannot respond to changes in masked later guesses.
- Frozen shared embedding/readout gradients and rejection of invalid input contracts.
- Inference keeps an incorrect first guess when it is supplied; it never replaces it with a reference answer.

All 448 metadata files were authenticated. Available training: 384 responses, 12,288 contexts, 165,351 valid editable token positions; 10,014 contexts have the original first guess correct. Development: 64 responses, 2,048 contexts, 27,580 editable positions. The proposed matched eligible-first schedule exactly reproduces the prior assignments and 384-update exposure. All full-block proposal IDs are available and valid. No data or weights were changed.

These checks validate a local implementation for the old teacher interface. They do not establish released-weight CUDA compatibility, a sufficient training dose, or compatibility with training through xPress's separate refinement head.

## Cost and decision

Historical matching full-teacher fits took 537–562 seconds each; two would imply about 18.7 minutes of fitting alone. A hypothetical bounded resource envelope totals $3.366: one A100-80GB for 45 minutes plus two shutdown minutes, up to four CPU cores and 64 GiB, separately budgeted staging/controller work, 3 GiB charged egress, storage and $0.60 extra reserve. Rates were checked against [Modal pricing](https://modal.com/pricing). These resource limits and the full collection/evaluation path have **not** been implemented or validated for the proposed run. This is not a launch-ready quote or a reservation.

Local free space was 6,483,664,896 bytes at the check. No files were deleted. No GPU work occurred and the authorized balance remains **$3.50**.

**Next implementation target:** a controlled full-proposal second-pass comparison starting from the already available stronger released drafter/refiner, with its existing cheap refinement retained as the control. This means testing whether a full additional pass earns a useful correction advantage before trying to compress it; it does not mean changing the goal to optimizing or merely compressing the existing xPress head. Its first-draft proposals must be collected from that checkpoint, not mislabeled old DFlash proposals. Its refiner's training gradient path, complete costs and remaining allowance need validation before a launch. No new student fit or GPU attempt is selected by this report.

Preserve the older single-first-guess xPress backbone screen: that untrained configuration helped only one case and lost to its stronger control. Completed-proposal input **with matched training** is a different, still-unproven teacher question. Avoid claiming that the new test will succeed or that a negative at a small dose closes the mechanism generally.

Evidence: `assessment.json`, `reference-rows.json`, `score-policy-check.json`, `score-policy-rows.json`, `schedule.json`, `sources.json`. Scripts: `analysis/prepare_full_proposal_teacher.py`, `analysis/check_full_proposal_teacher_reference.py`. No historical experiment was rerun or overwritten. Research goal active and unmet.
