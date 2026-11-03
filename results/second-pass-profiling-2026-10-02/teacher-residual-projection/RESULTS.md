# Can the compact output basis reconstruct the teacher's changes?

This CPU audit finds a large hidden-state reconstruction difference at the same output rank. It does not establish better tokens, learned corrections, or a speedup.

We use 64 hash-selected training prompts and all 64 development prompts, with two hash-selected anchors per prompt. Selection did not inspect labels or error outcomes. All 128 feature files and the saved teacher-KL checkpoint are hash-verified. Training and development prompt sets are disjoint. There are 1,692 valid editable training rows and 1,763 development rows.

For each row, subtract the original final normalized state from the full teacher's final state. Project that difference onto a fixed basis with freely chosen coefficients obtained from the teacher itself. The editor normally has to predict those coefficients; this diagnostic bypasses that difficult task.

| Output basis | Dimensions | Training squared change captured | Development squared change captured |
|---|---:|---:|---:|
| Existing probability-trained editor | 256 | 9.78% | 9.75% |
| Random orthogonal control | 256 | 10.01% | 10.05% |
| Approximate PCA fitted to training residuals only | 256 | 80.37% | 53.45% |
| Training mean direction, per-row oracle magnitude | 1 | 22.13% | 21.20% |

Copying one fixed training-mean vector explains only about 10.47% of training and 7.67% of development squared change. That differs from projecting onto its direction with a teacher-informed magnitude for each example.

The pattern also appears on the 70 sampled development positions that extend the teacher's matching prefix beyond the original draft: the learned basis captures 10.48% of their squared change and the training-fitted PCA basis captures 52.05%. This outcome-defined subset is small, correlated within prompts, and descriptive. It is not the full development teacher repair panel.

## Interpretation

The teacher's changes are more compressible in Euclidean hidden-state space than the existing learned basis captures. This makes a direct teacher-residual training target concrete enough to investigate, without immediately widening the editor. It does not show that the current basis is bad for its actual token-prediction objective. The earlier 24 oracle witnesses showed that selected token corrections are feasible in that very basis despite poor reconstruction of the teacher state.

Hidden-state reconstruction can waste effort on changes irrelevant to token ordering. With shared vocabulary readout W, what matters is the resulting scores and winner margins, not unweighted residual norm. A low residual norm alone supplies no token guarantee near ties. Projected teacher states are still teacher-informed oracle predictions; even good token results would not prove the editor can learn their coefficients.

The next discriminating check is therefore vocabulary readout of these projected states, with original and full-teacher controls, before another fit. A local output-head artifact has matching recorded Qwen3-4B revision and source-shard identity. Its actual bytes must be hash-verified before use. CPU arithmetic must be distinguished from saved BF16 CUDA outputs; any mismatch must be reported, not silently treated as native equivalence. No GPU run or new training has been selected.

## Prior art and scope

Feature alignment with a reused classifier is established: [Chen et al., SimKD, CVPR 2022](https://arxiv.org/html/2203.14001v1) trains a student through feature alignment and reuses the teacher classifier. Its vision experiments support feasibility of that strategy elsewhere, not this editor's acceptance or runtime. [Hsu et al., 2022](https://arxiv.org/abs/2203.10163) provide a counterpoint: their comparison favors logits as an efficient knowledge source and highlights feature dimension. Neither establishes that hidden supervision is superior here. Do not claim PCA or hidden-state distillation as our novelty.

The implemented projector checks full learned rank, orthonormality, orthogonality of reconstruction errors, and Pythagorean energy decomposition. PCA uses a randomized rank-256 approximation fitted only on training rows, not a certified best rank-256 result. The basis parameters and source hashes are preserved with the results for subsequent readout checks. No existing model was overwritten. Additional cloud cost: $0.

Reproduce with `analysis/audit_teacher_residual_projection.py`; detailed sampling, checks, source identities and results are in `results.json`.
