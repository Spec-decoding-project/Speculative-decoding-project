# One or two perfect token patches do not match the stronger control at equal cost

This exact saved-data calculation asks how demanding the correction target is. Start from each of256 published xPress draft chains on64 exposed responses. An oracle knows the target continuation and may replace at most k wrong tokens, leaving every other token unchanged. Compare its resulting prefix to the existing64-node conditional control on the identical recorded context. This is a privileged upper bound for that restricted editing policy, not a proposed algorithm.

| Maximum wrong tokens replaced per block | Total oracle matching positions | Equal-round-cost modeled savings versus conditional control | Uniform round-cost ratio needed for5% modeled savings |
| --- | ---: | ---: | ---: |
|0|2,060|-43.34%|at most0.663|
|1|2,382|-15.58%|at most0.822|
|2|2,583|-3.26%|at most0.920|
|3|2,770|5.40%|at most1.004|
|4|2,953|12.03%|at most1.080|

The conditional control covers2,696 positions. Savings scenarios average cost-per-progress ratios with responses weighted equally; they are not ratios of the table totals. Negative savings mean worse modeled cost per position. A chain has fewer candidate nodes than the control, so equal round cost is only an assumption. With two perfect patches, roughly8% cheaper rounds would be needed for the5% goal under the uniform-cost model. Actual costs and imperfect selection remain unmeasured.

The control extends the original chain in157 contexts. Matching that endpoint requires one changed token in65, two in25, and at least three in67. Thus the earlier observation that one edit often reproduces our weaker teacher's useful output does not establish that one edit reaches a stronger reference.

The bound is exact for this policy: replacing the first k wrong tokens reaches the next original wrong token, or the window end. Any selection leaving an earlier wrong token unchanged cannot do better. Independent enumeration of40,913 subsets, covering all one-, two- and three-error choices, reproduced the formula. Source recovery hashes and the control's coverage were independently checked; no neural computation occurred. Runtime3.2seconds; added spending$0.

This does not bound a learned editor that regenerates the suffix, changes more than k tokens, uses another window or proposal set, or is materially cheaper. It also does not prove that an oracle's corrections are learnable. The implication for compression is to measure complete corrected continuations against a strong control; do not design the target around a single attractive repaired token. The full sixteen-value curve, contexts and source identities are in results.json and rows.json.

No teacher/student fit or new tree method is selected by this structural bound. The next affordable teacher comparison must test useful continuation reconstruction, with data, feedback, training and complete cost separated. The research goal remains unmet.
