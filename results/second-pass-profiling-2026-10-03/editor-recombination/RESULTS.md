# Can rearranging existing editor guesses rescue useful progress?

Completed locally for $0. Independent bit-mask recount passed for all 16,384 model/context rows (eight models, 2,048 contexts each). Source artifacts were authenticated against recovery manifests.

We compared retaining the two complete drafts with an oracle allowed one switch, one edited span, or arbitrary token-by-token choices between those same drafts. The last is an unattainable selection oracle: it uses the recorded answer. It cannot invent tokens neither draft proposed.

On the six models evaluated on the previously fresh 64-response panel, arbitrary recombination yields only 4.00–4.39% modeled time savings at **zero additional cost**. Their ordinary two-branch figures are 2.62–2.78%. The development mixture editor reaches 4.93%; the pure teacher-imitation editor reaches 3.83%. These are reciprocal progress calculations on saved contexts, not measured generation latency.

Some response-bootstrap upper intervals exceed 5%. Therefore this is not a population-wide impossibility result. It does establish that the observed point estimates leave no overhead margin for the 5% objective, even with perfect choices. A selected pruned tree need not contain the entire oracle family; exhaustive family sizes are not lower bounds on practical tree cost.

Decision: do not fund recombination of these fixed argmax proposals as the next 5% rescue. Better useful token predictions remain necessary on this evidence. This does not bound new models, richer candidate sets, or smaller worthwhile improvements, and does not establish why editor learning transfers poorly.

Before more training, compare the editor with a conventional tree of the original drafter's runner-up predictions. This checks whether the learned corrections offer opportunity beyond alternatives already available without an editor. Candidate trees are established practice, not our novelty: [Medusa](https://arxiv.org/html/2401.10774v3).

Evidence: [results](results.json), [independent verification](verification.json), [context rows](rows.json). The six previously fresh models and the two development models remain separate; no best-model selection or new final-test claim.
