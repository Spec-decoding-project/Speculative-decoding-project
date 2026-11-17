# Single-row linear correction experiment

This diagnostic tests whether the target's verification-block linear operations can reproduce the original cached generator by processing each token separately. The prefill remains unchanged. All target linear layers, including the vocabulary head, use their original weights, dtype and forward method on contiguous `[1, 1, width]` inputs. Attention remains the original SDPA implementation. Methods and hooks are restored after every intervention, including exceptions.

The actual hash-verified released DFlash `spec_generate` still controls prefill, draft generation, cache positions and acceptance. Saved-candidate injection is an explicit diagnostic control; ordinary DFlash proposals are tested separately. No saved label is rewritten, no rejection bypassed and no editor trained.

`rowwise_session.py` authenticates the generator, original records and pinned model artifacts. It calls the original generator method with only its generation cap shortened from 2,048 to 16. Four first-block fixtures cover the new native failure, the old uncached failure and two controls.

The exact forward plan is:

| Work | Target forwards |
| --- | ---: |
| Four fixtures × three repeats × (16 original-generation calls + 2 native calls) | 216 |
| Independent cached trace through each focus position: 7 + 6 + 6 + 16 | 35 |
| Four fixtures × three repeats × two modified native calls | 24 |
| Ordinary DFlash control, default and modified | 4 |
| Total | 279 |

The inherited hard guard caps actual target forwards at 300 and synchronized target-forward time at 600 seconds at call boundaries. One A5000, four CPUs, 32 GiB host RAM and 30 minutes including setup. This is one correction hypothesis, not a backend/precision sweep. Existing failed artifacts must not be overwritten or retried without accounting for the original attempt.

Submit from the Nexus project root:

```sh
sbatch research/editor-feature-bank/replay_diagnostic/rowwise.sbatch \
  --output research/editor-feature-bank/UNUSED-OUTPUT-DIRECTORY
```

The script requires the original source tokens and default red regression to reproduce before testing the intervention. It checks every saved token in the modified native rounds and separately checks the first query projection against the cached reference. Traces localize remaining differences if the intervention fails. Exit 1 with `complete: true` and no `error` is a negative correction result, not a crashed experiment. Partial artifacts and `result.json` preserve evidence on exceptions.

Timing includes synchronized per-target-forward measurements and instrumented native-round wall time. The per-fixture summary uses the two non-traced repeats after one traced repeat. This small diagnostic is not an end-to-end speedup benchmark. Read the actual measurements and instrumentation limits before making a deployment claim.

CPU checks cover intervention scope, output ordering, unmodified weights, and method/hook restoration. They do not establish real-model GPU numerical equivalence:

```sh
python -m unittest discover -s editor-experiment/replay_diagnostic -p test_rowwise.py -v
```

A four-fixture pass alone is neither proof of agreement for every input/execution layout nor permission to train. A justified documented correction, complete bounded bank and independent checks remain required.
