# Exact saved-token diagnostic

`exact_session.py` authenticates the original generator and target/draft artifacts, then calls the original `QwenBackend.generate` method. The only generation-configuration change is a cap of 16 new tokens instead of 2,048; the first full block is the scope of this diagnostic. Saved response files are not modified.

The four fixed first-block fixtures include both known failures and both earlier controls. Original generation, released native saved-candidate verification, and uncached replay each repeat three times per fixture. For the newest failure, a separate cached teacher-forced replay repeats three times through the failing position, keeping every earlier token fixed even if an earlier prediction differs. The ordinary DFlash candidate is a separate control and is not compared with saved-context labels beyond its first wrong token.

The saved-candidate verification explicitly overrides only the 15 tokens returned by the draft sampler. It then calls the actual hash-verified released `spec_generate`, including target prefill and block verification. This artificial candidate is a diagnostic, not editor output or evidence of repair quality.

After inspecting the baseline and stating hypotheses, supply `exact_probes.py` as `probes.py` inside the output directory, then create `probes.ready`. The process keeps both frozen models in the same allocation. The probes compare corresponding layer rows and final hidden states, evaluate the vocabulary head in FP32 on each path's fixed representation, and isolate first-layer query projection using an identical input vector alone versus inside the block matrix. Three repeated projection calls distinguish stable shape dependence from variation between identical calls. Disabling reduced BF16 reductions and transient FP32 projection are probes, not adopted corrections. A final whole-transformer-linear FP32 intervention is limited to the new fixture; it cannot establish an all-fixture correction.

Submit from the Nexus project root using an unused output directory:

```sh
sbatch research/editor-feature-bank/replay_diagnostic/exact.sbatch \
  --output research/editor-feature-bank/NEW-EXACT-DIAGNOSTIC \
  --wait-for-probes
```

Use the existing explicit diagnostic budget; this command is not permission to rerun an exhausted attempt. One A5000, four CPUs, 32 GiB host memory and 30 minutes including setup. Every top-level target forward, including internal `generate` forwards, is capped at 300. CUDA-synchronized target forward time is capped at 600 seconds at call boundaries. The SLURM limit bounds setup, drafter work, projection-only probes and the pause between stages. Target-forward time excludes these other costs; allocation elapsed time includes them.

The executed November 17 diagnostic used 251 baseline calls plus 43 trace calls, totaling 294. The optional FP32 intervention would require nine more, so the hard guard interrupted it at 300; that intervention has no usable result. The original estimate of 295 total calls was incorrect. The subsequent code now skips that optional probe if the remaining budget cannot accommodate it. No new GPU run was made for this change. See the [complete evidence report](../../docs/research/exact-replay-2026-11-17/REPORT.md) for the preserved baseline/traces, executed source version and limitations. Never increase the cap to conceal this planning error.

`baseline_exact` checks original source generation and native verification against the original labels. A false baseline returns exit 1 after writing `result.json`, even if a later probe changes an answer. The artifact must be checked for `error`/`traceback` before interpreting that exit as an intentional correctness verdict. `baseline.partial.json` and `traces.partial.json` retain progress on interruption. No bank is marked valid and no editor training occurs.

CPU tests check token-position/margin reporting and the hard target-forward cap:

```sh
python -m unittest discover -s editor-experiment/replay_diagnostic -p 'test_exact.py' -v
```
