# Tour checkpoint answers

Make a prediction before reading the corresponding answer. The point is to reconstruct the behavior, not memorize this page.

| Checkpoint | Answer and evidence |
| --- | --- |
| **1A — Entry point** | `dflash.cli:main` receives the command, dispatches to `_generate_transformers`, then calls `dflash.model.dflash_generate` after model loading and tokenization. |
| **1B — Block width** | A 16-position block has one target-chosen anchor and **15 proposals**. In `dflash_generate`, positions `[:, 1:]` receive the proposals. |
| **2A — Prefix acceptance** | **1**, because the second comparison fails. The third matching 6 cannot be retained after discarding its predecessor. The cumulative product in `dflash_generate` enforces this. |
| **2B — Correction** | The round retains `[a, 4]` and supplies target correction **5**, giving `[a, 4, 5]`; 5 is the next anchor. The proposed 9 and 6 are discarded. With no stopping or budget truncation, `start` advances by accepted proposals + 1. |
| **2C — Stop token** | No. The first stop token is included and later tokens are excluded. Look at `stop_indices`, the adjustment to `produced`, and the final output slice. |
| **3A — Context shape** | `[1, 10, 32]`. `extract_context_feature` concatenates two `[1, 10, 16]` arrays on the feature axis, not on the token axis. The draft projection then reduces the final dimension. |
| **3B — Causality** | No. A noncausal target verifier could inspect future candidate tokens when predicting an earlier proposal, invalidating the autoregressive comparison. Draft attention and target verification have different contracts. |
| **4A — Backend selection** | No. The Transformers path selects `cuda:0`; the `mlx` backend must be selected explicitly for the Apple Silicon implementation. The `local` dependency extra chooses platform packages but does not change CLI dispatch. |
| **4B — Cache rollback** | Future predictions would use the rejected candidate's states instead of the retained output history. A correct displayed prefix this round does not prove that the next round has the right internal state. MLX also handles supported recurrent states that need more than ordinary key/value trimming. |
| **4C — Positive temperature** | A target distribution can assign valid probability to tokens other than its argmax. Equality to the argmax would impose a greedy rule on a sampling request. Stochastic verification uses draft/target probabilities and a residual correction distribution to account for the proposal process. |
| **5A — Timing** | Baseline `1/0.04 = 25` tok/s, DFlash `1/0.02 = 50` tok/s, speedup `0.04/0.02 = 2`. This timing result says nothing about task accuracy, and by itself does not prove output-distribution correctness. |
| **5B — Weighting and accounting** | Printed mean = `(2 + 4)/2 = 3`; pooled-round mean = `(9*2 + 4)/10 = 2.2`. An untruncated greedy round with recorded length 4 accepted **3 draft proposals** and produced one target correction/bonus. These formulas describe this implementation's counters. |
| **5C — Change plan** | The secondary drafter runs after filling `block_output_ids[:, 1:]` and before target verification. It uses the accepted context plus the anchor and first proposal; it never sees future target answers. Preserve those two seed tokens, accepted-prefix state, and target/DFlash cache boundaries. A meaningful test compares complete greedy output IDs against independent target-only generation and checks rejection, EOS, and final partial blocks. The optional implementation applies after either DFlash proposal branch and rejects nonzero temperature. See [the experiment guide](SECONDARY_DRAFTER.md). |

Now change the example: let the proposals be `[8, 2, 3]` and aligned target predictions `[9, 2, 3]`. Explain why accepted draft proposals equal zero even though two values match. With enough budget and no stop token, explain why the recorded round length is nevertheless one. Then explain why changing only the suffix cannot repair the first rejected proposal.
