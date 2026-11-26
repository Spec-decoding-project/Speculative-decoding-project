# Cheap pre-editor selection: completed native experiment

**A cheap first-block-only rule gives a small native warm-cache win on GSM8K,
while UltraChat is effectively unchanged. It does not yet give a cold-cache win.**
The same earlier 3.6997 gate/editor is frozen throughout. The new component adds
no neural network: after its first edit, a host-side flag skips all later editing.

On 32 GSM8K development prompts, throughput rises **2.73%**, with a paired-prompt
bootstrap 95% interval of **+0.97% to +4.65%**. All three repeat ratios improve.
On 32 UltraChat prompts, the change is **−0.08%**, interval **−0.95% to +0.99%**.
The mixed-workload aggregate is +0.94%, but the per-source difference matters.
The original always-on editor still loses on both sources.

Every tested path returns the same final tokens and termination. This is a
bounded development result at batch one with a 64-token output cap and warmed
editor graphs. It is not fresh confirmation, a production-cache result, or a
claim about complete long responses. All 64 requests reached the cap; this run
adds no natural-EOS coverage.

## What was fitted, and why the selector is cheap

1. Keep Qwen3-8B, DFlash, the 40,750,336-parameter autoregressive generator and its
   earlier 265-parameter COPY/generate gate frozen. The editor reads the last 1,024 accepted context tokens (excluding the anchor)
   and the complete token proposal. It preserves the anchor/first proposal and
   generates 14 editable suffix tokens.
2. Generate baseline native trajectories on 96 training-origin fit prompts and
   32 separate training-origin calibration prompts, balanced by source/length.
   Run the editor in shadow mode to label repairs and damage, returning the
   original DFlash proposal to the target. No generated edit changes this collection.
3. Use only pre-editor signals: DFlash top-two score gaps, proposal repetition,
   previous committed-token count, and prompt/generated length. Vocabulary scores
   already exist in this pinned DFlash configuration. The uninitialized adaptive
   confidence buffer is never read. Neither hidden states nor future labels are
   inference inputs.
4. Fit one depth-three regression tree, minimum 64 rounds per leaf, shrinkage 32.
   There are 1,397 complete fit windows and 455 calibration windows; 620 incomplete
   terminal windows are excluded rather than assigned guessed labels. No editor
   optimizer update occurs. Calibration maximizes prefix gain minus a fixed
   0.45 tokens/edit over a preregistered threshold grid, including never editing.
5. Freeze threshold **0.5**. The resulting classification is exactly **edit only
   the first proposal of a response**. Its first-round leaf scores 1.0546875;
   every later-round leaf is below 0.5. Calibration selects 32 edits: 12 repairs,
   four damaged prefixes and 16 unchanged, for +46 net prefix tokens (+31.6 after
   the cost penalty). The fit first-round subset gives +135 prefix tokens.
6. Before any five-arm native comparison, structurally simplify the frozen tree
   to an equivalent first-edit counter. This removes later confidence reductions,
   GPU scalar reads, context copies and editor steps. No threshold/policy refit.
   Keep the general feature-based selector as a separate measured control.

The first-block rule was learned from training/calibration data, not chosen from
native evaluation outcomes. The general and compiled selectors edit exactly once
per request and have identical committed-token trajectories in every timed pair.

## Matched native measurements

One RTX A5000 on tron59; batch one; three rotated repeats per prompt; 32 prompts
per source, eight per source-specific length quartile. Evaluation prompt IDs and
families are disjoint from the preceding 128-prompt native benchmark and from this
selector's fit/calibration sets. They remain part of the previously inspected
project development split; the final test is untouched.

Pinned vLLM 0.34.0 / Torch 2.14.0+cu130; invariant BF16 target/drafter, TP1,
FlashAttention, eager target/drafter, synchronous scheduling, maximum context 4,096 tokens,
256-token chunked prefill, 2 GiB KV cache, temperature 0/seed 0, no prefix caching.
The frozen editor is FP32 with standard kernels captured into reusable CUDA graphs;
invariant target dispatch is restored before verification. All five arms use the
same GPU, requests and decoding configuration.

The timer includes native generation, editor work, feature extraction and all
editor token copies; it ends after the worker drains its GPU. Request setup/mode
RPCs and result serialization are outside timing. Average round time is the full
decode interval divided by native verification rounds. Committed tokens exclude
the one prefill anchor and clip the terminal block to actually returned output.
Unused last proposals, when computed, stay in the timed path. Cold capture is
measured separately; warm runs have zero graph misses.

Totals below include all **three repeats** (96 requests and 6,144 output tokens
per source/arm). These are real output commits, not offline draft-prefix scores.

| Source | Path | Output tokens/s | Decode ms/round | Output commits/round | Verification rounds | Editor calls |
|---|---|---:|---:|---:|---:|---:|
| UltraChat | Baseline DFlash | 38.372 | 69.036 | 2.71698 | 2,226 | 0 |
| UltraChat | Features, always skip | 37.678 | 70.353 | 2.71698 | 2,226 | 0 |
| UltraChat | Always-on editor | 34.840 | 77.680 | 2.76543 | 2,187 | 2,247 |
| UltraChat | General selector | 37.688 | 70.622 | 2.72801 | 2,217 | 96 |
| UltraChat | Cheap first-block selector | 38.339 | 69.372 | 2.72801 | 2,217 | 96 |
| GSM8K | Baseline DFlash | 64.865 | 69.128 | 4.63448 | 1,305 | 0 |
| GSM8K | Features, always skip | 63.715 | 70.442 | 4.63448 | 1,305 | 0 |
| GSM8K | Always-on editor | 60.658 | 77.732 | 4.85783 | 1,245 | 1,326 |
| GSM8K | General selector | 65.535 | 70.998 | 4.81146 | 1,257 | 96 |
| GSM8K | Cheap first-block selector | 66.633 | 69.767 | 4.81146 | 1,257 | 96 |

| Source | Cheap-selector throughput change | Three repeat speed ratios | Prompt-bootstrap 95% speed-ratio interval |
|---|---:|---|---|
| UltraChat | -0.084% | 0.999104, 0.999259, 0.999102 | 0.990457–1.009917 |
| GSM8K | +2.726% | 1.027163, 1.027206, 1.027410 | 1.009705–1.046535 |

The confidence/feature-only control costs about **1.316ms per complete round**,
losing 1.80% throughput overall despite never editing. Its overhead is material:
the general selector is −0.75% overall, while the equivalent compiled rule is
+0.94%. These are whole-path comparisons, not isolated CPU-tree execution timings.

Editor calls fall **3,573 → 192** across the timed workload (94.63% fewer than
always-on). One pass saves **19 of 1,177 baseline verification rounds**: UltraChat
742→739, GSM8K 435→419. Across UltraChat, four prompts need fewer rounds, three
need more and 25 are unchanged; for GSM8K, 14 need fewer, one more and 17 are
unchanged. The eight fully audited first-block prefix examples contain three
repairs, one damage and four unchanged; that small subset is diagnostic, not a
64-prompt estimate of prefix-repair frequency.

## Break-even economics

Use baseline whole-round time B, edited time E, and actual commits c0/c1.
Affordable extra cost is B×(c1/c0−1); extra commits required at measured cost are
c0×(E/B−1). Callback timing includes pending DFlash work and is not assigned
entirely to the editor. For first-only editing, these are costs amortized across
all decode rounds, not the cost of its single invocation.

| Source | Policy | Affordable added ms/round | Measured added ms/round | Extra commits observed | Extra commits needed |
|---|---|---:|---:|---:|---:|
| UltraChat | Always on | 1.231 | 8.644 | 0.04845 | 0.34019 |
| UltraChat | First block only | 0.280 | 0.336 | 0.01103 | 0.01322 |
| GSM8K | Always on | 3.331 | 8.604 | 0.22335 | 0.57683 |
| GSM8K | First block only | 2.640 | 0.639 | 0.17697 | 0.04286 |

GSM8K's selected acceptance gain buys 2.640ms/round, exceeding the measured
0.639ms addition. UltraChat buys 0.280ms but spends 0.336ms: essentially break-even,
slightly losing. This explains the source-specific result without attributing the
entire standalone-editor latency to every round.

## Graph setup and prompt length

With an empty editor graph cache, first-only editing captures one graph per
request: **64 captures, 6.475s total capture/warmup, about 101ms/request**.
Always-on editing captures 1,191 graphs, costing 119.198s. Cold first-only generation
takes 90.139s for the 64 requests versus 84.946s for one baseline pass: **5.76% lower
throughput**. Separately, cold first-only loses 5.23% on UltraChat and 6.64% on GSM8K.

Warm timings deliberately reuse shapes prepared before timing; no real-world hit
rate was measured. Because this rule edits only once per request, useful graph
reuse must occur between requests with compatible editor input shapes. The warm
win cannot be assumed for a stream of new shapes. Startup/model loading is
separately saved in the private manifests and is outside these generation timers.

Each length row has **eight prompts**, with source-specific quartiles. Reported
ranges are actual sampled input lengths, not output/draft lengths. There is no
long-context or long-output inference from this sample.

| Source | Quartile | Input-token range | n | Cheap-selector throughput change |
|---|---|---:|---:|---:|
| UltraChat | Q1 | 24–43 | 8 | +0.631% |
| UltraChat | Q2 | 63–102 | 8 | -0.462% |
| UltraChat | Q3 | 107–170 | 8 | -0.844% |
| UltraChat | Q4 | 232–576 | 8 | +0.616% |
| GSM8K | Q1 | 46–54 | 8 | +1.972% |
| GSM8K | Q2 | 56–66 | 8 | +3.352% |
| GSM8K | Q3 | 71–80 | 8 | +1.833% |
| GSM8K | Q4 | 82–120 | 8 | +3.611% |

All four GSM8K length groups improve in their point estimates, but n=8 intervals
are wide. UltraChat has mixed small changes. The JSON evidence includes each
length group's interval and all repeats; do not treat eight subgroups as separate
confirmed discoveries.

## One actual completed comparison

The first timed prompt is GSM8K, 75 input tokens, 64 returned tokens, repeat0.
All paths have identical final IDs and `length` termination. These are the first
chronological cases, not a selected favorable example. The initial edit damages
the draft prefix from three correct proposal tokens to two.

| Path | Total elapsed seconds | Verification rounds | Editor calls |
|---|---:|---:|---:|
| Baseline | 1.319680309 | 19 | 0 |
| Features, always skip | 1.343538139 | 19 | 0 |
| Always-on editor | 1.620142640 | 21 | 21 |
| General selector | 1.417842492 | 20 | 1 |
| Cheap first-block selector | 1.395407617 | 20 | 1 |

Raw records are `evaluation-00014.json` through `evaluation-00018.json`. This
prompt loses despite the average GSM8K win. The measured editor is the exact
earlier gate SHA256 `5ab048b622beffb10ab8d282027cb2510f5439e4621793d5a82e687e696d679b`,
not the broader-data gate. Generator SHA256:
`c79b8c59c1e2cf73a7adf7aa9738b1b273779d330fe32f8d2c4a15fd56b7e8d2`.
Frozen selector JSON SHA256:
`a2c69259bad52f4907be68c3301a09134a7a9d411889fab3e8d55bd26fe7d905`.

## Audit, artifacts and decision

- All 960 timed generations match outputs/termination across paths and repeats;
  1,576 evaluation/reference generations pass their corresponding equality checks.
- Independent audit checks 3,584 emitted-token positions and 3,072 aligned full
  target-logit hashes. General/compiled policies have identical trajectories.
- 18 local and 18 GPU-host checks pass, including split separation, damage labels,
  terminal overshoot, skip-before-editor behavior and classifier-equivalence proof.
- Target/drafter/original-generator file pins match the preceding independent audit.
  All 1,901 backed-up files match remote size/SHA256 manifests. Original data and
  original workspace edits are preserved. No final-test evaluation or editor fit.
- Evaluation job 8581045 completed successfully in 39m56s; CPU pin audit 8581059
  in 25s. Queue empty. Total GPU allocations including collection/failed startup:
  51m01s, below the 80-minute cap.

The initial launch 8580929 stopped on terminal-label bookkeeping after one
correctly generated request. The fixed collection 8580943 completed fitting and
reference controls, then stopped intentionally before five-arm evaluation to
compile the frozen rule. Source/logs from all attempts are retained. Evaluation
source is **e210c22**; its immutable archive SHA256 is
`84b72fbdb38587ad5f4b80e7fffb3c59716762016134d990d7f29b225600a9dd`.
The post-run independent auditor additionally reconstructs per-prompt outcomes.

**Decision:** retain the first-block selector as a promising experimental option,
especially for GSM8K under graph reuse. It substantially improves the editor's
cost trade-off, but the gain is modest, source-dependent and lost from a cold
cache. Do not restart larger training on this evidence. The next useful milestone
is fresh confirmation of this frozen rule on longer responses with measured cache
hit rates or cheaper shape reuse; target/drafter should remain fixed for comparison.
No follow-up experiment is running. Union/concurrency and fresh confirmation
remain open in the serving-fidelity investigation.

[Aggregate results](evidence/cheap-selector/summary.json),
[independent audit and anonymized per-prompt results](evidence/cheap-selector/audit.json),
[frozen selector](evidence/cheap-selector/selector.json),
[backup integrity](evidence/cheap-selector/backup-integrity.json).

Code: `editor-experiment/refinement_research/selector.py`, `selector_worker.py`,
`native_selector.py`, `native_speed_worker.py`, `selector_audit.py`, and
`test_selector.py`. Native timing and collection archives are immutable;
`native_selector.sbatch` documents the actual bounded continuation command.
Private raw outputs, model manifests, telemetry, logs, archives and hashes:
`~/.local/share/wayfinder-artifacts/selector-8581045/`, with collection under
`selector-8580943/` and the preserved failed attempt under `selector-8580929/`.

Tracked in [Can a cheap pre-editor selector recover native DFlash throughput?](https://github.com/Spec-decoding-project/Speculative-decoding-project/issues/27).
