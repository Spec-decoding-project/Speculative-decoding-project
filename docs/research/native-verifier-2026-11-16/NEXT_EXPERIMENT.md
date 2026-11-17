# Proposed next step: separate labels defined by the native verifier

Status: **rejected by the user on November 16, 2026; do not execute.** The user requires exact agreement with the original saved greedy tokens. [Accepted decision](../../adr/0001-preserve-exact-saved-target-tokens.md). The text below preserves the proposal for historical context; it is not an active recommendation or an authorized experiment. The original responses, failed banks and diagnostic artifacts remain immutable.

## Recommendation and reason

Keep the existing 50k responses as a source of committed contexts. Create a separately versioned small dataset whose corrected continuations are constructed for, and independently checked by, the actual native cached first-round verifier. This aligns the editor's supervised target with the path that will accept or reject its proposals. Do not claim exact agreement with the old token-by-token cached output or with all possible cache histories.

The alternative of training directly on the original saved continuations remains possible as a distillation experiment, but those continuations are not guaranteed to be perfectly accepted by the native verifier even when reproduced exactly. That distinction must be explicit in its training/evaluation goals. Continuing to require every saved token to match every replay shape preserves the current gate and leaves fitting blocked pending further numerical work.

## Reviewable first experiment, before a training bank

Scope: the original three fixtures plus the newly failing `ultrachat_200k:train_sft:37680`, anchor 24. One RTX A5000, at most four CPUs/32 GiB RAM, one allocation of at most 30 minutes including setup, at most 10 minutes and 120 target forward calls after model loading. No extra checkpoints, corpus-wide output or editor optimization.

For each fixture:

1. Authenticate the original response record and use its tokens only to construct the committed prefix. Compute the anchor and ordinary DFlash proposal through a fresh native round.
2. Retain an immutable prefix cache. Construct a new candidate one position at a time, using the same 16-position verification layout at every step. For candidate position i, read target verifier row i-1 after the already constructed prefix of the candidate. Future positions initially contain ordinary draft tokens; they must be causally invisible to the prediction being used.
3. Reset/crop to the identical prefix cache before every query and record positions/cache lengths. Check that changing future placeholder tokens cannot change earlier chosen logits/tokens. A failed causal or cache-invariance check stops the experiment.
4. Submit the completed candidate to the unchanged released native verification call as an explicit oracle diagnostic. Require the anchor plus all 15 proposals to match, repeated three times. Independently rebuild and compare labels after resetting model caches. Reproduction of the old saved continuation is measured and reported, not silently imposed or assumed.
5. Store a new label artifact with a new contract/schema identifier, generator/source/model fingerprints, original response hash and prefix hash. Record differences from original saved tokens. Never overwrite the source response or claim its suffix supplied the new labels.

This is a proposal for testing whether a coherent native-first-round label producer can be built, not a claim that the algorithm already works. Stop on any mismatch or budget exhaustion. The fixed verifier layout and future-token invariance must be demonstrated; successful self-consistency is not automatically full-generation equivalence.

## Subsequent gates if this experiment passes

Build a new token-only bank of 128 repairable plus 16 preservation blocks within the original candidate/time limits. Preserve source balancing, EOS-only initial selection, split/family checks, fixed first-proposal restrictions, and separate accounting for wrong-first cases. Add an independent checker for the new producer contract; the old exporter must not be tricked into treating regenerated labels as original saved labels.

Only after a validated bank and the necessary trainer stopping/ownership audits, fit the same 1/16/128 subsets with both autoregressive variants. Evaluate free-running suffix predictions, actual native acceptance, direct damage and eventual full round cost. Larger 384/3k/10k/50k learning-curve budgets remain a later decision. Full-generation and union-cache correctness remain separate from this first-round experiment.

## Choice required

- **Recommended:** approve the separate native-verifier label experiment above, keeping the original 50k responses intact as context sources.
- **Alternative:** use the original saved responses as distillation targets, explicitly permitting target/serving disagreement and measuring repair quality with native verification rather than saved-token agreement.
- **Alternative:** retain exact agreement with the original saved tokens as a prerequisite and continue bounded numerical diagnosis before any real fitting.

The decision changes the meaning of a correct training label. It is not a request for renewed SSH permission or permission to inspect data.
