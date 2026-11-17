# One real block, two full-size editors, excluded-prompt evaluation

The user explicitly authorized this run after reviewing the project briefing:
fit both full-size variants on one real repairable DFlash block, evaluate test
prompts, then report before training on additional prompts. ADR 0003 supersedes
the earlier full-bank prerequisite for this milestone only.

## Fixed data selection

Reuse the four versioned invariant pilot prompts as ordered training-block
candidates. Capture actual 16-token verification inputs from native vLLM DFlash;
never inject saved future labels. Select the earliest repairable block in the
previously declared fixture order. Preserve the anchor and first proposal; train
only its 14-token suffix. At most four complete blocks per prompt are considered.

From authenticated original validation records, require EOS completion, at least
16 saved output tokens and at most 1,024 prompt tokens. Sort eligible IDs by
SHA256(ID) and select four per source, excluding exact normalized message-family
duplicates and overlap with the four training candidates. These eight prompts
form a small held-out validation probe, not an untouched final test benchmark.

Generate at most 64 reference tokens per prompt using invariant vLLM, and then
run genuine DFlash independently on the same prompts. Match emitted tokens and
full-vocabulary raw-logit hashes at identical prefixes. Check every captured
block's actual native acceptance against the reference and preserve all complete
blocks in the first-four-block budget, including wrong-first and preservation
cases. Record incomplete-window and context-limit exclusions. Any correctness
failure prevents training; no mismatch filtering or alternate backend sweep.

## Fixed training and tests

Use the existing model/Trainer from the two AR branches (context f5b50fa and
whole-draft 4565132). Same model code and seed zero; mode alone differs. Model:
151,936-token tied embedding/readout, width 256, two layers, four heads,
1,088 positions, exactly 40,750,336 editor-owned trainable parameters.

Each independent fit: AdamW 1e-3, FP32, no weight decay/dropout, microbatch one,
gradient clipping at 1. Evaluate every 25 updates. Stop at three consecutive checks
with exact free-running 14-token suffix and CE < 0.05, at 2,000 updates, or 600 training
seconds. Do not train on additional blocks even after passing. Target and drafter
are unloaded before training and are never optimizer members.

Check future-label exclusion, optimizer ownership, finite gradients, changed
weights, identical initial hashes, cached/full decoding and checkpoint reload.
Held-out metrics are computed only after checkpoint selection is complete;
regenerate the initial state afterward for an untrained held-out baseline.

Report exact suffix/token accuracy, repaired/damaged counts and original/edited/
union consecutive matching proposals after the anchor (maximum 15). Preserve raw
before/after tokens on Nexus. Measure FP32 cached editor latency with two warmups
and ten synchronized samples per block, including context prefill and 14 generated
tokens. Target/drafter are unloaded for this isolated measurement. Neither this
latency nor saved-context union coverage establishes full-generation speedup.

## Resource and integrity bounds

One RTX A5000 on tron61, four CPUs, 32 GiB, 30 minutes. Reuse the existing isolated
environment and checkpoints, with no downloads. Collection worker enforces a
shared 500-target-forward/120-target-second cap, including startup calls, across
the two engine subprocesses. Package freeze must match the completed pilot.
Original response files are read only. Outputs and checkpoints use a new job
directory; no previous results or either AR branch are overwritten.

Seventeen local tests passed before execution: eleven existing model/trainer/
data tests plus six checks of native capture alignment, acceptance, duplicates,
prompt-family separation, wrong-first retention and the memorization criterion.
