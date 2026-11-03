# Prompts for useful code tours

The most useful starting prompt specifies **what you want to learn, what you already know, a concrete execution path, and how the explanation must be checked**. There is no research-backed universally best wording. The templates below combine official CodeTour guidance, code-tour studies, and learning research with this repository's actual contracts. Their step counts and word budgets are practical design choices, not proven optima.

## What the research supports

| Evidence | How to use it here | Limit |
| --- | --- | --- |
| The [official CodeTour documentation](https://github.com/microsoft/codetour) defines JSON tours, source navigation, and ordered tours. | Store `.tour` files in `.tours/`, start with one primary tour, and link focused follow-ups. | The format itself does not guarantee good teaching. |
| GitHub's [code-tour authoring guide](https://github.com/github/awesome-copilot/blob/main/skills/code-tour/SKILL.md) emphasizes the learner's role, a narrative, and real source locations. | Give the generator a learner profile and a specific end goal; require source inspection and validation. | This is practitioner guidance, not a controlled learning comparison. |
| [Taylor and colleagues' CodeTour study, PPIG 2022](https://www.ppig.org/files/2022-PPIG-33rd-taylor.pdf), observed more focused navigation but had a small sample and raised the possibility of discouraging exploration. | Follow a task's call/data flow, then include one optional exploration outside the tour. | It does not establish that tours always improve understanding or speed. |
| A [2026 study of LLM-generated debugging tours](https://arxiv.org/abs/2607.26987) reported preferences for concise, useful explanations and unreliable LLM quality judgments. | Explain purpose and constraints instead of paraphrasing each line; verify locations and claims with code and tests. | A 26-developer Java-debugging study does not determine the best prompt for learning ML code. |
| [Karpicke and Blunt, 2011](https://pubmed.ncbi.nlm.nih.gov/21252317/) found benefits from retrieving and reconstructing information on conceptual-learning tasks. | Ask for a prediction or explanation before revealing an answer, followed by corrective feedback. | Applying this to code tours is a teaching-design inference; their study did not test these prompts. |
| [Cepeda and colleagues, 2006](https://pubmed.ncbi.nlm.nih.gov/16719566/) synthesized evidence on distributed practice and retention. | Revisit difficult checkpoints after a delay instead of reading all explanations repeatedly in one sitting. | The best spacing depends on the desired retention interval; no particular schedule is validated here. |

## Generate a tour for one learning objective

Paste this into an agent with access to the repository. Replace the bracketed fields. For this project, a strong objective is: “Trace one draft block and explain why more accepted proposals do not automatically mean a faster generator.”

```text
Create a CodeTour that teaches me [specific behavior or change I want to make].
I know [Python/tensors/transformers at a concrete level] and am unfamiliar with
[specific concepts]. I have about [15] minutes. By the end I should be able to
[observable task, such as trace one block and identify a safe insertion point].

Read the current implementation and relevant tests before writing. Trace one
concrete example from entry point through the important state changes to the
result. Separate code behavior, documented intent, and your inferences. State
what you could not verify. Do not invent functions, measurements, or capabilities.

Write a focused .tours/<descriptive-name>.tour JSON file using the official
CodeTour schema. Prefer 6-9 steps and split a larger topic into linked tours.
Use repository-relative file paths and unique pattern anchors OR verified
1-based line numbers; do not use both, since line takes precedence. Leave ref
unset when teaching the editable working tree. Preserve existing primary/link
ordering unless there is a clear reason to revise it.

For each step, explain why this code exists, its input/output or state change,
and the contract a future edit must preserve. Define jargon on first use.
Keep most descriptions under 100 words; spend more detail on the hard concept,
not boilerplate. Use a small token or tensor example where it helps.

Include 2-3 prediction or explain-back checkpoints. Put feedback in a later
step or a separate answer key so the question comes before the answer. End with
one transfer exercise and one optional exploration outside the tour's path.
Do not auto-run commands, download weights, or edit implementation files.

Validate JSON and every file/pattern/line/nextTour target against the current
checkout. Run scripts/validate_tours.py. Review each explanation against the
source and existing tests; distinguish static validation from model execution.
Report the files created, the exact checks run, and any unverified claims.
```

For this repository, add these constraints when relevant:

```text
Keep these distinctions explicit: target anchor vs draft proposals; accepted
draft proposals vs newly produced tokens including the correction/bonus;
original DFlash vs DFlash2; greedy vs stochastic decoding; PyTorch vs MLX vs
server inference; task accuracy vs target-output correctness vs speed.
This checkout adds an optional token-only secondary drafter to the upstream
baseline, with CPU tests and greedy output checks. Distinguish its pretrained
weights from a custom-trained refiner; no packed tree verifier is implemented.
Confirm these statements against the checkout.
```

## Use a tour as a tutor, not a lecture

```text
Tutor me using .tours/[filename].tour and the source files it references.
Ask one short diagnostic question to calibrate my knowledge, then teach one
step at a time. At checkpoints, ask for my prediction and wait for my answer.
Do not reveal the answer key first. Check my reasoning against the code, give
one focused correction or hint, and ask me to explain it back using a new
example. If I already understand a step, move on. If not, reduce the example.
At the end ask me to reconstruct the execution path from memory, identify
one invariant and one limitation, and write three questions to revisit tomorrow.
Keep track of my misconceptions. Do not modify code or launch model inference.
```

## Review or repair a tour after a code change

```text
Review .tours/[filename].tour against the current checkout and the diff from
[explicit known base commit]. First run scripts/validate_tours.py. Then verify
each behavior claim against its source and relevant tests. Check inputs,
outputs, indexing, tensor shapes, mutation/cache behavior, and control flow.

List concrete stale or misleading claims with source evidence. Distinguish
broken navigation from incorrect explanations. Update only affected steps,
their answer-key entries, and links; keep the learning objective coherent.
Do not add implementation changes. Re-run validation. Report static checks
separately from any tests actually executed; do not score quality just by how
confident or polished the prose sounds.
```

## Learn before making an experiment change

```text
I want to [specific experiment change]. Before implementing it, create a short
tour from the current entry point to the owning functions and relevant tests.
Use a before/after token example to explain the intended behavior. Identify
which invariants must stay true and which assumptions intentionally change.
For a new conditioning rule, check information visibility and shared candidates.
For branch selection, include target-output equality, cache state, positions,
stop tokens, and output budgets. For a metric, include its denominator and
prompt grouping. End with a test plan that could falsify the change. Leave the
implementation untouched until I finish the tour.
```

Judge success by whether you can predict a new example and locate the right code to change. A longer tour or a fluent explanation is not evidence that you learned more.
