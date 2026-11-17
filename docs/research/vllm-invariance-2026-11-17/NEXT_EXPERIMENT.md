# Next experiment: supported invariant contract, then a versioned data pilot

The current diagnostic gives a concrete policy choice. Batch-invariant vLLM
agrees with itself on the four tested first blocks, including the enabled-chunking
control. It consistently selects `Employee` (16984) where the old generator
selected `employee` (12946). The old generator reproduces its saved choice; no
original generation error has been established.

**User decision, November 17:** adopt invariant vLLM and the small separately
versioned pilot, after supported-configuration validation. Original response
files remain immutable. This supersedes the old-label requirement for the new
versioned experiment; it does not declare the old generator incorrect.

1. **Adopt invariant vLLM as the canonical target.** First repeat the four-fixture
   validation with chunked prefill enabled in every cell, changing only the
   scheduling token budget (256 versus 64) to exercise prefill chunk boundaries.
   Repeat each genuine greedy/DFlash pair three times, verify raw logit hashes,
   batch-order/companion changes and rejection continuation, using the pinned
   weights, environment and documented tie behavior. This removes the unsupported
   configuration warning from the principal test. Preserve the old outputs as
   historical evidence.
2. Once that supported configuration passes, make a **separate, versioned pilot**
   from the four original prompts. Generate their actual greedy continuations
   under the agreed canonical engine, preserving prompt IDs and exact token IDs.
   Independently verify the new continuations. Record both old/new differences
   and complete engine/data provenance. This is new data from a new target
   definition, not a correction claiming the original generator was wrong.
3. Set an explicit pilot length/call budget before allocating. This proposal does
   not authorize an automatic 50k regeneration, overwrite, feature-bank pass or
   editor fit. Scaling response generation needs a measured throughput/storage
   budget. The exact 128-repair/16-preservation bank and prompt-split gates still
   precede training.

The user chose the first option. Do not substitute new outputs into the old bank
or silently filter mismatches.

The completed experiment's allocation and forward budget is closed. A new
bounded task implements the user's authorized validation and four-prompt pilot.
