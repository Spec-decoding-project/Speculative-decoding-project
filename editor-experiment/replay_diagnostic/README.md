# Fixed-fixture target replay diagnostic

This is diagnostic code, not a corrected collector. It uses the failing first block and two successful controls from the November 10 bank. It never trains, changes saved labels or marks a bank complete. The executed result is in [the report](../../docs/research/target-replay-2026-11-16/REPORT.md).

Submit from the Nexus project root after copying this directory alongside the existing `feature_bank/` helpers:

```sh
sbatch research/editor-feature-bank/replay_diagnostic/diagnose.sbatch \
  --output research/editor-feature-bank/NEW-UNUSED-DIRECTORY
```

The baseline checks pinned model/source hashes, authenticates source records and prompt tokenization, then repeats each original replay three times. It returns 1 if saved labels disagree; this expected red verdict must not be interpreted as an infrastructure crash. A fresh execution needs its own approved experiment budget. Do not blindly rerun the allocation.

The November 16 run added `--wait-for-probes`. After inspecting its repeated baseline and stating hypotheses, the agent copied the checked-in `probes.py` into the new output directory and created `probes.ready`. The session retained its frozen target in the same allocation, ran the probes and wrote `result.json`, then returned the baseline verdict. Without a marker it waits at most 20 minutes. SLURM enforces a 30-minute allocation; the probes additionally enforce a 10-minute combined forward budget at call boundaries.

`probes.py` receives the session globals and is not a standalone executable. It compares fixed-representation BF16/FP32 and one-row/16-row vocabulary projections, cached teacher forcing on identical saved prefixes, three original-greedy generation repeats, explicit attention masks and a one-row whole-model readout. It does not attempt full FP32 model execution or native DFlash verification. The absence of a corrective regression is deliberate: no correction has been established. The live three-fixture equality check is the regression seam that remains red.
