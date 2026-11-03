# Complete round cost of the compact editor

The integrated compact editor is substantially cheaper than the full second-pass union on the four tested contexts, but still adds cost over original drafting. It is untrained; this run establishes no learned improvement or generation speedup.

| Context | Original ms | Full second pass union ms | Compact union ms | Compact added cost over original |
| --- | ---: | ---: | ---: | ---: |
| Math anchor8 | 27.895 | 36.898 | 29.519 | 5.82% |
| Math anchor24 | 27.794 | 36.340 | 29.208 | 5.09% |
| Chat anchor8 | 27.801 | 36.670 | 29.543 | 6.27% |
| Chat anchor24 | 29.291 | 37.695 | 30.763 | 5.02% |

Values are median complete-round times from20 paired repetitions after warmup on an A10080GB PCIe. The four contexts come from two exposed development prompts. All draft networks run eagerly; the target verifier is graphed. These are matched hardware/execution comparisons, not a comparison with fully optimized serving systems. The compact union is18.4–20.0percent cheaper per round than the full-teacher union. Different proposed tokens can affect downstream work, so the difference does not isolate neural-network cost.

With the observed5.02–6.27percent additional round cost, a5percent lower time per accepted position requires10.55–11.86percent additional progress under the fixed-context cost model. That is a conditional break-even calculation, not a learning prediction. The untrained compact union did not increase progress in these contexts. The full teacher increased progress on one, and its original fallback prevented a regression on another. No quality inference should be drawn from random editor weights.

All16 branch/cache audits passed accepted-sequence, bonus, known-anchor and following-cache checks. Two rejected-suffix winner discrepancies were recorded and their score evidence recovered. The median one-time folded-weight preparation was1.597ms, reported separately from round cost. Fixture construction and initial cache cloning remain outside the timed region. No training or full-generation evaluation occurred.

Execution through recovery and cleanup took230.94seconds. All25 evidence files were recovered and checked; the app has zero active tasks and its owned temporary volume is deleted. The ledger retains the full$1.25 attempt allowance pending billing; this is not actual invoiced cost. After the user's$10 extension,$8.885 remains available under the$111.47 campaign cap.

The result supports proceeding to the controlled learning feasibility question rather than declaring success from a cheap component. Train the same editor with direct target supervision, matched teacher-correct-position emphasis, and teacher probabilities. Preserve prompt-level progress and cost accounting. A smaller loss or teacher agreement alone cannot establish usefulness; useful repairs must survive held-out regressions and complete overhead. Graph optimization remains available, but must include comparable controls and should not substitute for the missing learning evidence.

[Raw analysis](analysis.json) · [Verified outcome](verified-outcome.json) · [Recovered evidence](recovered/summary.json)
