# What the small editor would need to reproduce

## Decision-changing finding

The public DTA teacher's measured benefit cannot be assigned to one conveniently chosen redraft. Keeping just the boundary with the highest first-rejection score preserves11 of its26 extra matching positions (42.3%). On the same saved contexts, that single second-model proposal plus the original gives151positions, versus153 for the matched ordinary rerun. Equal-prompt difference is+0.30%, with a descriptive95% interval−5.96% to5.97%; the pooled total favors ordinary rerunning. There is no demonstrated single-query advantage under this rule.

| Available proposals, always retaining original | Matching positions |
|---|---:|
| Original only |140|
| First model, highest-scored boundary only |153|
| Second model, highest-scored boundary only |151|
| First model, all four selected boundaries |158|
| Second model, all four selected boundaries |166|

This is a post-hoc analysis of already saved predictions, with no rule tuning, new model calls or spending. The single-boundary rule is the original first-rejection score's highest-ranked choice, with the existing stable tie rule. It is one fixed policy, not an upper bound on learned selection or a new pruning proposal. Four-proposal numbers select the longest correct prefix with reference answers; they are fixed-context opportunity, not speedups.

The ten positions previously identified beyond both ordinary rerunning and unconditioned second-model output shrink to two with this one-boundary rule. Thus the compression target is not yet a proven superior single teacher query. A proposal to imitate only that query would discard most of the very evidence motivating compression.

## What the revealed prefix is doing

Across31 contexts where the original draft has an error, the true first-error boundary was among the four selected boundaries24times, but was highest-scored only14times. The target answers are used to describe these facts afterward, never to choose the saved proposals.

| Retained prefix relative to original first error | Chains per model | Ordinary rerun repairs | DTA repairs |
|---|---:|---:|---:|
| Stops before all correct guesses have been revealed |39|2|7|
| Preserves all correct guesses, remasks the first error |24|5|10|
| Already contains the first wrong guess |65|0|0|

The last row is a structural consequence: an immutable wrong prefix cannot be repaired by changing its suffix. Its zero is not evidence that either neural model lacks information. Similarly, zero prefix damage when preserving every previously correct token is guaranteed by the input/output contract, not learned perfect preservation.

In10 of the11 contexts where DTA improves over the original, preserving exactly the originally correct prefix attains its best saved improvement (ties allowed). That locates useful behavior: conditioning on the correct partial continuation often helps it choose the next missing token and sometimes a short following run. It does not establish which internal computation performs the repair, and the original scores do not reliably identify that partial continuation ahead of verification.

This identifies two requirements for compression: learn useful conditional completion, and retain enough of its value across uncertain prefix boundaries. Reproducing average teacher-token predictions while ignoring this distinction could look successful without preserving useful draft progress. These observations do not justify a particular student architecture or claim that boundary selection itself is our research contribution.

## Are the repairs simple output changes?

Of11 teacher-improved contexts,6 require changing only one original token to recover at least the teacher's accepted span. They account for10 of the26 additional positions. The other5 require two to four original-token changes. This is a retrospective output-edit calculation; discovering the right edit may still require substantial contextual computation.

Among the four contexts that add value beyond both controls, the required original-token changes are1,1,3 and4. Their extra value beyond those controls is1,2,3 and4positions. Consequently seven of the ten distinctive positions arise in the two cases requiring several adjacent corrections. A student trained only to fix one isolated token could miss much of this particular teacher's distinctive opportunity. This is a hypothesis about coverage of a restricted output operation, not a reason to scale an editor blindly.

## Implication for the research objective

Keep the goal of learning a cheaper replacement. Do not describe the task as copying a known superior single redraft: that has not been established. Nor has it been established that one learned proposal can combine the useful outcomes of the four teacher conditions.

Before choosing student supervision, a strong same-target comparison must establish which of these corrections remain worth preserving. Then explicitly choose whether the student approximates the conditional teacher separately for each prescribed prefix, or attempts to consolidate several conditional outcomes into one proposal. The first has a clear teacher function but retains multiple proposal costs. The second has a harder unproven consolidation requirement. Neither choice is justified by the present small panel alone.

No new fit, selector or architecture is proposed here. No stronger runtime claim follows. Full serving cost and comparison with a strong8B refinement model remain unresolved. Existing unallocated funds are0; no historical accounting hold was reclaimed. The goal remains active and unmet.

## Evidence

All47 source artifacts authenticated before analysis. `analysis.json` contains32 per-context rows, paired prompt estimates and11 minimal-edit examples. `boundary-strata.json` records the retrospective prefix categories. The executable analysis is `analysis/analyze_d2sd_compression_target.py`. Previously frozen GPU results are preserved unchanged; confidence intervals here use a separately recorded bootstrap seed and are descriptive of the exposed eight-prompt panel.
