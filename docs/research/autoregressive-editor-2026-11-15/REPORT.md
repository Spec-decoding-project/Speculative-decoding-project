# Token-only autoregressive implementations: synthetic checks only

The two requested input variants share a small causal decoder and readable training/evaluation runner. The context variant sees committed tokens plus the anchor and first DFlash proposal. The draft variant additionally sees the full original DFlash block before producing a revised suffix sequentially. Neither consumes target or drafter hidden activations.

The default model has **40,750,336 trainable parameters**, including its own tied token embedding/readout. Both variants have exactly the same parameterization and seed; only the conditioning configuration differs. This is a configurable architecture baseline, not a demonstrated best size.

## Local validation

Eleven tests passed on Python 3.12.13, Torch 2.15.0, pytest 9.2.0 and safetensors 0.8.0 on macOS CPU. Tests cover input distinction, teacher-forcing shifts, causal label exclusion, cached/full-forward agreement, per-round cache isolation, fixed-seed preservation, optimizer/checkpoint continuation, bank corruption and split leakage, wrong-first-proposal accounting, a real CLI train/reload cycle, and the completed-bank export gate with authenticated source-record matching.

Each input variant memorized two synthetic four-position blocks in 100 updates using a tiny width-16, one-layer model and a vocabulary of 20 tokens: teacher-forced CE <0.05 and free-running exact suffix accuracy 100%. This exercises the real model/update/generation path. It provides no evidence about fitting Qwen data, 40M-parameter optimization, held-out repair quality, native target acceptance or speedup.

The pilot adapter accepts only a complete independently checked 128-repair/16-preservation bank. A scalable token-only proposal collector, natural-distribution held-out bank and native replacement/union runtime integration remain outstanding. The new model/trainer is not presented as a completed 50k experiment.

## Nexus access and recovered evidence

VPN plus interactive authentication restored Nexus access. The response run metadata and failed-bank audit remain present. The class partition reported A5000 nodes in mixed allocation state; a successful allocation is not assumed from that snapshot. The user's queue was empty when inspected.

The previously pending [saved-bank integrity result](saved-bank-integrity.json) was retrieved: the smoke bank passed, the failed partial bank was rejected, and a corrupted tensor copy was rejected. This is an integrity result, not a repair of the original target replay mismatch. The mismatch remains unresolved and real-data editor training remains blocked.

No credentials or response corpora are stored in these artifacts. The original workspace and historical experiment results were preserved.

## Nexus synthetic GPU validation

SLURM job `8432471` completed successfully on one RTX A5000 in 1m21s, with one CPU and a 4 GiB host-memory allocation. Maximum host RSS was 1,504,496 KiB. The runtime was Python 3.12.12 and Torch 2.13.0+cu126, matching the cluster environment rather than the local newer Torch version.

Both tiny variants completed 100 synthetic updates. Context-only teacher-forced CE was 0.0005243143; draft-conditioned CE was 0.0005096544. Both achieved 100% free-running exact suffix match, and cached/full-sequence generation and checkpoint reload checks passed. [Machine-readable result](nexus-runtime-validation.json) contains source hashes, all verified against the committed implementation. These runs used width 16 and vocabulary 20, not the full 40,750,336-parameter default. Full-size memory and latency remain unmeasured.

The default runner now selects explicit nested 1/16/128 repair subsets plus up to 16 preservation blocks, recording the selected indices in checkpoints. The 11 local tests passed again after that addition. No real-data editor training or new target/draft inference job was launched.
