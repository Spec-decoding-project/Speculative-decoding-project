# First wider-pool attempt failed in the optional token-rank diagnostic

No completed-panel measurement was recovered. The code tried to gather a score for a padded label (`-100`) while explaining the correct token's original vocabulary rank. CUDA rejected the out-of-range index. The preparation checks had tested candidate nesting but omitted padded-label gather behavior; this was a preparation error.

The local audit identified 1,092 out-of-vocabulary labels in 2,048 contexts, all explicitly marked invalid. None of the valid labels is out of range. The repaired diagnostic gathers a harmless index at invalid positions and emits null ranks there; scores and candidate choices for real positions are unchanged. Actual tensor gather and rank computations passed 300 synthetic cases, including ties and invalid labels. All saved label masks were checked. See ../label-mask-checks.json.

The failed run had passed scoring operations before reaching this diagnostic, but incomplete in-memory results were not recovered; no exact number of completed score checks or candidate improvements is claimed. Only preflight, staged paths, and the terminal failure record were recovered. The repair also saves every eight contexts rather than waiting for a complete prompt.

App `ap-CDwMItXEIjkhUAmfjTGl0V` is stopped with zero tasks. Owned volume `vo-4vIRTIcsJYveqUvAtPuHEK` is absent. Recovery authenticated three files / 969 bytes. Approximately 55 seconds elapsed through cleanup. The full $1 cap is retained as an unsettled allowance, not reported usage. The independent repaired attempt has separate files, identifiers, and accounting; no original evidence is overwritten.
