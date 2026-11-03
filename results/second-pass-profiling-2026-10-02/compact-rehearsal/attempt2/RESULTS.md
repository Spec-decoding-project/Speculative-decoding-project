# Matched targeted rehearsal: fitting succeeds, transfer does not improve

## Main result

The same compact editor can jointly recover the full teacher prefix on all64selected repair examples while preserving all32selected correct drafts. Equal-budget representative continuation recovers the full teacher prefix on5of64. However, focused rehearsal gives no reliable advantage on separate development responses and both continued versions underperform the starting checkpoint. This is not a generation-speedup result.

| Editor | Development additional retained progress over original DFlash | Useful prefix repairs | Edited prefixes worse than original | Damaged fully correct originals |
|---|---:|---:|---:|---:|
| Starting editor |3.4607%|176|503|77|
| Ordinary continuation |2.5162%|150|573|74|
| Targeted continuation |2.5413%|157|638|87|

Primary targeted-minus-ordinary comparison: +0.0251percentage points, descriptive paired prompt95%interval[-0.4419,+0.5328]. Domain differences: GSM8K-0.0833pp; UltraChat+0.1334pp. Damage also rises. The advancement conditions do not pass.

Progress is the better prefix of original and revised proposals, including the root, averaged with equal prompt weight. Keeping the original branch prevents edited-prefix damage from reducing this coverage metric, but checking both branches costs time. These percentages are not tokens-per-second gains. No new runtime comparison was performed.

## What it resolves

Before focused practice, the editor recovered60of420teacher-extra positions on the96selected contexts. After practice it recovered420of420, completed all64teacher-improved prefixes, and damaged none of the32preservation cases. Ordinary continuation recovered70of420 and completed5teacher prefixes. This establishes simultaneous learnability for these selected examples within this architecture/objective; it contradicts an absolute inability to represent these corrections. It does not establish general capacity sufficiency, a universal training prescription, or useful transfer.

On128retention contexts from32other original training responses, retained-progress gains were25.61%starting,26.99%ordinary,18.39%targeted. This supports a fit-versus-retention tradeoff in this allocation, not isolated causation by repetition. Both continuation recipes reduced development gains relative to the saved starting checkpoint. Repetition, example concentration and changed gradient competition remain bundled. Learning-rate and optimizer alternatives were not tested.

Intermediate32/64development probes contain only256contexts, whereas final uses2048; do not compare their absolute figures as a single learning curve or choose a checkpoint from them retrospectively. All64development prompts are historically exposed. One initialization and one training order were tested.

## Execution and verification

Both arms restored the same FP32 checkpoint and full AdamW moments at1152updates. Same objective,128additional updates each,4096presentations; targeted96cases repeated42–43times, ordinary4096distinct cases. Original scientific plan differs from attempt1 only in the cap field. No GPU work or training occurred in attempt1.

Attempt2 setup timestamps show input hashing completed in18.7seconds, imports in3.2seconds, and the explicit commit in0.9seconds. The earlier timeout did not recur; these observations do not identify its cause. Full2272context starting-model comparison exactly reproduced the saved native outcomes before training.

256updates completed. Both saved optimizer states reach1280; independently checked BF16 deployment weights equal the cast FP32 training checkpoints. Independent scalar token recount reproduced all development metrics. All21recovered artifacts match their remote manifest hashes. Shutdown independently verified, zero tasks; controller verified the owned volume was deleted. Elapsed through cleanup:4.48minutes.

## Spending and next decision

Additional$50authorization recorded once, cumulative cap$171.47. This attempt's full$2.50allowance is retained pending billing; provider meter has no rows yet, so actual cost is unknown. Unallocated campaign balance$47.53506668. No follow-up launched.

Keep the original editor checkpoint as the better development reference. Do not promote this targeted recipe or merely repeat it with more updates. The next research question should address transfer while preserving correct predictions, using this demonstrated fitting ability as evidence—not claim that the compact architecture is impossible or that larger models/data necessarily solve it. Existing broad research direction stays active; this completed allocation test does not justify another paid fit by itself.
