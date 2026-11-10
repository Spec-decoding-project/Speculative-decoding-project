# Speculative decoding research

Vocabulary for deciding whether a compact editor improves draft proposals enough to justify further experiments.

## Language

**Anchor**:
The next token already chosen by the target at the start of a speculative block.

**First proposal**:
The first unverified token guessed after the anchor. The compact editor preserves this token.

**Editable suffix**:
The proposed tokens after the first proposal that the editor may revise.

**Repairable block**:
A diagnostic example whose first proposal matches the target continuation but whose later proposals contain an error. This label describes a training example, not information available to select blocks during generation.

**Preservation block**:
A diagnostic example whose original proposals already match the target continuation, used to measure damage introduced by editing.

**Replacement**:
Verification of the candidate with the edited suffix in place of the original suffix.

**Union verification**:
Verification of both the original and edited candidate branches, retaining a valid continuation with the original available as fallback.

**Saved-context coverage**:
The consecutive matching positions in a proposal against a recorded target continuation at a fixed context. It is not a measurement of generation speed.

**Prefix damage**:
An edit that shortens the consecutive correct prefix relative to the original proposal.

**Scale decision**:
An evidence-backed decision to proceed with, revise, or stop a larger training and inference experiment. A justified negative result is a valid outcome.
