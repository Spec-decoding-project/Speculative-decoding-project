# Ordinary original-drafter alternatives outperform this editor on saved coverage

Completed all 2,048 contexts from 64 reused development responses. Native BF16 projection reproduced every saved original token. An independently implemented frontier-list constructor reproduced all 16,384 tree results, including node counts. No training or new generation occurred.

## Main comparison

Every tree retains the original full chain. The following comparison keeps the original first guess fixed and gives each ordinary tree exactly the number of nodes used by that context's original-plus-editor tree.

| Proposal construction | Extra matching-prefix coverage over original DFlash |
| --- | ---: |
| Original plus frozen mixture editor | 3.46% |
| Ordinary tree using original top-two candidates | 9.61% |
| Ordinary tree using original top-four candidates | 12.17% |

The paired differences from the editor are +6.15 percentage points (response-bootstrap 95% interval +5.22 to +7.14) and +8.71 points (+7.63 to +9.78). Both ordinary references improve coverage more in both domains. The top-four matched tree beats the editor on 570 contexts and loses on 45, adding 887 positions the editor misses while missing 85 the editor covers.

These are **recorded-reference, fixed-context coverage results**, including the known root. They are not actual verified generation speedups. Equal node count does not establish equal runtime, and numerical behavior under tree verification remains unmeasured. The eight predeclared tree variants are all reported in the analysis; none is selected as a confirmatory winner on fresh data.

## Stronger-reference sensitivities

At the common 30-node maximum, fixed-first-guess top-two/top-four trees reach 11.22%/14.16%. Allowing the ordinary tree to change the first guess gives 12.76%/16.31% at matched actual node counts and 14.68%/18.60% at the common maximum. This benefit must not be attributed solely to candidate ranking: freeing the first guess changes the permitted search space.

## Does the full second pass still add something?

A post-hoc overlap analysis uses the same saved teacher predictions; no further compute was purchased. The full teacher's original-plus-teacher coverage is 10.21% on this development panel (the separate 10.92% result belongs to another panel).

Beyond the matched top-four fixed-first-guess tree, the teacher supplies 670 additional matching positions across 223 contexts. Hypothetically retaining both the tree and teacher gives another 5.06% coverage over that tree. The compact editor contributes only 85 positions across 45 contexts, or 0.63% further coverage. Beyond the stronger unrestricted 30-node tree, these figures fall to 4.41% for the teacher and 0.55% for the editor.

Those unions add uncharged nodes and computation. They are opportunity measurements, not a deployable matched-budget comparison. Of the teacher's 223 extensions beyond the matched tree, the first missing correct token is already in q0's top four in 177 cases; it is outside in 46. This points to both selection and prediction limitations, but is hindsight evidence, not a pre-verification signal or proof that a small network can fix either.

## Research decision

The earlier original-chain-only comparison was insufficient to justify the editor's practical value. A conventional use of existing candidate scores is substantially stronger on the same contexts and node budgets. This is established tree decoding, **not a newly invented technique**.

Do not start another editor fit based only on beating the original chain or matching more teacher probabilities. The editor now has a concrete stronger reference to beat. Full-second-pass complementarity remains, but its observed margin beyond that reference is modest before paying any cost. A complete-round cost/verification comparison of the ordinary tree versus the existing editor is the next missing practical check; preparation should reuse the existing verifier and compare identical prefixes without new training. Whether any subsequent training is justified depends on a measured cost advantage or useful corrections beyond this reference, not on claiming that extra capacity or epochs must help.

This does not prove every small editor or every second-pass replacement is inferior. It rejects a strong practical claim for the current recipe from its 3.46% coverage result alone. Research remains active; no new training recipe has been selected.

## Resources and reproducibility

The attempt closed in 3 minutes 19 seconds including upload, recovery and cleanup; the GPU worker took 141 seconds. All five artifacts (9,973,770 bytes) passed recovery hashes. App `ap-jrdhQQVR8i9HKxrQl5hpgG` is stopped with zero tasks; owned temporary volume `vo-NVUZfweUOv4JOLjIBxxEb4` is absent. The retained research archive was untouched.

The full $1 authorization is conservatively held as unsettled usage/storage/recovery allowance, not claimed as actual billing. $10.28506668 remains unallocated; no active reservations or jobs.

Evidence: [protocol](../PROTOCOL.md), [independent analysis](analysis.json), [post-hoc teacher overlap](teacher-complement.json), [execution and cleanup](execution.json), [recovery receipt](recovery.json), [GPU summary](recovered/summary.json).
