# Full-backbone feedback supplies one additional repair, without demonstrated competitive headroom

All sixteen fixed development contexts from eight responses completed locally. The original process exited successfully after 54.2 minutes. No training, target generation, GPU execution or paid resources were used. Independent coverage reconstruction and forty recorded artifact hashes passed; model weights remained unchanged.

## What this asks about compression

This tests a candidate full-second-pass teacher, not a new tree method. The same released xPress backbone first drafts normally, then processes the same committed context with its own refined first guess visible. The comparison supplies that same guess to six additional iterations of the existing refiner while retaining first-pass backbone states. Both fix the anchor and first guess, start refinement from the same proposal, and use the same weights.

Thus the paired difference concerns changed backbone computation under identical refinement treatment. It does not compare separately trained teachers, and this released checkpoint was not established as trained for the partially visible second-pass input pattern.

## Complete result

| Proposal | Total matching prefix positions |
| --- | ---: |
| Initial backbone, before its normal refiner | 135 |
| Normal six-step refined draft | 155 |
| Six additional cheap refinement steps | 161 |
| Full backbone rerun plus the same additional refinement | 168 |
| Original refined draft retained alongside the full rerun | 169 |
| Ordinary candidate trees at each full-union node budget | 167 |
| Existing 64-node conditional control, reconstructed locally | 187 |

Counts include the known anchor. They measure fixed-context matching to the recorded target continuation, not new generated tokens or latency. Totals are descriptive; decision comparisons weight responses equally.

Against the cheap repeat, the full pass improves one context by eight positions, worsens one by one, and ties fourteen: +2.46% equal-response standalone progress. Retaining the original fallback removes this regression and yields +4.55% versus the cheap union. That entire incremental benefit comes from one response.

Against ordinary trees at the exact full-union node budgets, full-union progress is -3.11% with equal response weights: one context improves, three worsen, twelve tie. The slightly larger pooled total, 169 versus 167, does not override the different response-weighted result. Against the wider conditional control, no context improves and five worsen; all full-pass correct prefixes were already covered by that control. The wider control uses a different budget and extra computation, so this is an opportunity comparison, not a cost-matched speed result.

## What changed

In `ultrachat-12468/8`, the target continuation begins `  // Handle edge cases\n  if (num < 0) return false`. The cheap refinement chooses a different indentation token after the newline. The full pass fixes that token and extends matching from six to fourteen positions. The original raw backbone already ranks the correct indentation token first at that position; the correction is not simply discovery of a previously absent token. The ordinary matched tree already covers fourteen positions, and the conditional control covers fifteen.

In `ultrachat-12248/8`, the full pass instead shortens three matching positions to two. The original fallback retains three. No completely correct cheap proposal is damaged in this panel.

Across all valid editable positions, full versus cheap changes 47 token guesses: 42 wrong-to-wrong substitutions, three wrong-to-correct substitutions and two correct-to-wrong substitutions. Only one wrong-to-correct change repairs the first blocking error. The other two occur after an earlier remaining error. These are descriptive edit locations, not an explanation of how a student would learn them.

Eight cheap proposals are already completely correct; seven have an editable first error, and one is blocked by the locked wrong first guess. Therefore this small panel contains only seven contexts where this policy can extend the cheap prefix. It cannot estimate general rates of redrafting success.

Without the refiner, the raw second-pass readout repairs no prefix and loses five positions versus the raw first pass with the identical first guess clamped. The observed useful change therefore involves the refiner acting on changed scores/features. A separate post-hoc two-by-two substitution checks those pathways using saved tensors; see `../xpress-second-pass-channels/`.

## Decision and limits

Do not fund student training on this untrained teacher configuration from this evidence. It adds a real repair beyond repeated cheap refinement, but does not yet supply a convincing competitive compression target. This closes advancement of this configuration, not second-pass compression or trained full-pass editing generally. Lower student cost could matter even without greater coverage; this CPU study supplies no matched runtime evidence for such a tradeoff, and costs from unrelated proposal policies must not be attached to it.

The CPU BF16 no-cache computation differs from earlier native GPU execution at fourteen draft tokens across six contexts. Comparisons within this run are paired, but are not GPU equivalence checks. These eight responses have been repeatedly exposed during development. No fresh-prompt generalization, useful generation speedup, or novelty claim follows.

Evidence: `summary.json`, `verification.json`, `matched-contrasts.json`, `error-locations.json`, per-context token arrays and saved first/second-pass tensors. The primary protocol and runner hashes are unchanged. Preserve the earlier pre-measurement loading failure separately. Research remains active; the existing remaining allowance is $3.50.
