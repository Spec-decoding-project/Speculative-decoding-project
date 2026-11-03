# Frozen readout feasibility: 24 verified witnesses

All 24 selected first-error corrections could be expressed by the existing learned output projection. All 24 witnesses survived the original BF16 GPU scoring operation, preserving the original prefix and correcting the tested token. No model weights changed.

The known correct token was supplied to the solver. These are oracle interventions, not learned predictions, an improved decoder, a speedup, or evidence of novelty.

## What was controlled

The original first-pass features, vocabulary readout, trained editor weights and folded output projection were frozen. All selected saved predictions reproduced exactly before search. The solver changed the coefficient vector at the known first wrong position. Earlier editable coefficients were set to zero to preserve the original correct prefix; later coefficients stayed as predicted. The full 14-row scoring shape and BF16 precision were retained.

The panel contains eight historical training-probe prompts and sixteen exposed development prompts, one known missed teacher repair each. All cases completed without replacement. Selection conditions limit the result to these diagnosed cases.

## Evidence

| Split | Selected cases | GPU-verified corrections | Median oracle / learned coefficient maximum magnitude |
|---|---:|---:|---:|
| Training probe | 8 | 8 | 6.27% |
| Development | 16 | 16 | 3.19% |

The largest ratio was 13.33%. Absolute oracle coefficient infinity norms ranged from0.1073to0.6959. After BF16 rounding, the correct token exceeded every competitor by1.0000to1.0625logit units. Thus these witnesses did not require coefficients with unusually large maximum magnitude relative to the learned row. This does not measure how difficult the vectors are to predict or how far they lie from normal network outputs.

The label-informed interventions added57positions across these selected contexts, versus128from the full teacher. That is neither full replacement of the second pass nor a serving result. The diagnostic targeted the first error only; it did not search for full teacher continuations.

## Decision

The learned output subspace does not make these24first-token corrections impossible. Widening this projection is therefore not supported as the first default response to these cases. The remaining questions concern producing the right coefficients from allowed inputs, including representation quality, the coefficient-producing network's capacity, optimization and training targets.

This does not establish that the entire frozen-size editor can jointly implement the individual witnesses, or that its inputs contain sufficient information to select them. Oracle prefix preservation also requires knowing where the original error is; no such inference-time detector was learned. Changing the full network, claiming a general capacity guarantee, or reporting these24successes as validation accuracy would be unsupported.

No follow-up fitting or generation was launched. Any next learning comparison must use training-only information to construct targets and keep fresh evaluation examples out of that process.

## Execution

All31artifacts15,199,998bytes recovered and hash-checked. App stopped with0active tasks, owned temporary volume deleted. Elapsed through cleanup3.85minutes. The retained conservative allowance is$0.85, below the$1.50cap; provider billing remains unsettled. No retries.

Remaining campaign allowance:$12.0351.

[Analysis](analysis.json) | [Cases](recovered/cases.json) | [Runtime](recovered/runtime.json) | [Execution](execution.json) | [Accounting](cost-reconciliation.json)
