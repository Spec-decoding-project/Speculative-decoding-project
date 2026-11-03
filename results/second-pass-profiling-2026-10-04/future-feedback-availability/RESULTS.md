# Later correct guesses exist at some stable errors, but are not a proven repair signal

Recounted all256 saved xPress contexts from64 development responses, using the eager six-step proposals that match the existing stability diagnostic. Authenticated512 source records. No model execution or spending; calculation took3.4seconds.

Of256 contexts,61 are complete and195 have a first error. That error is already fixed by the causal refinement map in180 contexts; more identical refinement cannot change it. The existing source-level causal certificate establishes this for the frozen head, not for other models.

Among those180 stable errors:

- 82 have at least one later token matching the recorded continuation.
- 35 have a matching token immediately after the error.
- 37 have a later run of at least two matching tokens.
- 50 have a later matching token containing an alphabetic character.
- 98 have no later match; 22 errors occur at the first future position.

Across the61 responses with a stable error, the mean within-response fraction with any later match is45.36%. Alphabetic characters are only a crude description; neither they nor longer runs establish semantic information useful for repair. Common syntax, repeated token types and positional coincidence remain explanations. Cases without later matches also are not proof that all later information is useless.

This leaves a specific input distinction worth assessing: the current causal refiner cannot use later token choices to revise a stable earlier error, whereas a trained full-block editor can receive them. Our just-completed full-backbone screen revealed only the first guessed token; it was not a test of feeding the completed proposal. Availability of later matches is a necessary-looking opportunity check, not proof that the editor can recognize their reliability or derive the correct earlier token.

Bidirectional proposal editing is already central to [DEdit](https://arxiv.org/html/2609.38510), so this input distinction is not our invention. Its published trained mechanism supports considering a full-proposal teacher as groundwork for compression, not claiming novelty for repeating it. No ready-to-use DEdit checkpoint was verified in the bounded availability check. A new teacher fit would need matched no-feedback training and a costed learning budget; this count alone does not authorize selecting an architecture or promise that a small-data reproduction will work.

The proposal policy also needs enough improvement against strong alternatives. See ../second-pass-edit-requirement/RESULTS.md for a reference-label edit-count ceiling. All results remain exploratory development evidence; native/eager proposal differences are retained per context.
