# M4-V1: frozen layer-5 validity-repair experiment

The user approved this single bounded follow-up on 2026-09-30. This protocol,
its machine configuration and exact split/CV IDs are committed before any
new real-data representation fitting or probe training.

## Scope and hypothesis

Use only the existing RAVDESS WavLM cache `d99979b610d68bef`, layer 5, and
statement identity. Layer 5 is the M3 task anchor, not a layer selected for
being close to the M4 validity cutoff.

Hypothesis: a fixed 64-dimensional, training-only PCA bottleneck can stabilize
LEACE sufficiently to suppress held-out statement decodability while
preserving emotion and unrelated information.

This is a validity study, not M4A reliance, M4B harmful-under-shift learning,
or M5 replication. No new encoder, corpus, extraction, dimensionality search,
alternative layer, or outcome-guided seed/threshold change is authorized.
Existing M3 and M4 results remain immutable, including their failed gates.

## Frozen data roles

Keep all four recordings in each `(speaker, emotion, intensity)` cell
together: two statements and two repetitions. Allocate the 360 cells as:

| Role | Cells | Utterances | Permitted fitting |
|---|---|---|---|
| A | 180 | 720 | Representation scaler, PCA, LEACE |
| B | 90 | 360 | Diagnostic probes and their inner-CV scalers |
| C | 90 | 360 | None: one-time final evaluation and uncertainty |

The metadata-only allocation uses seed `20260930`. Independently permute
speakers within each gender, assign offsets 0/1/2/3 cyclically, and permute
the 15 emotion/intensity cell types once. Allocate using the repeating role
cycle `A,A,B,C` shifted by each speaker's offset. Counterbalancing preserves
the exact emotion, statement, gender and intensity marginals. Every speaker
occurs in all roles. A speaker supplies 7 or 8 cells to A and 3 or 4 to B/C.

Persisted files:

- [Role IDs](configs/experiments/m4_validity_v1_roles.parquet):
  1,440 utterance/cell/role records.
- [B-only CV IDs](configs/experiments/m4_validity_v1_cv.parquet):
  1,800 records, one validation-fold assignment per B utterance and attribute.

Probe CV uses three deterministic stratified group folds within B. The group
is the complete matched cell, not an individual recording. All label spaces
must occur on both sides of every fold. These folds are shared across arms
and across the two statement probe families. Each B validation fold has
120 utterances, leaving 240 for inner training. Freeze the actual IDs, not
just the seed; verify their hashes at every run.

These are new validity partitions, not replacements for M1/M3's reference
or OOD splits. Sharing actor identities is intentional for the five-attribute
audit; do not call this unseen-speaker evaluation. The corpus has already
been explored in M3/M4, so this is not a fresh independent replication.

## Four fixed representation arms

Fit an initial StandardScaler on A only. All matrices use float64.

1. **original:** original standardized 768-dimensional vectors.
2. **leace:** full-dimensional LEACE fitted on standardized A and A statement
   labels, using the existing tested implementation.
3. **pca64:** PCA with exactly 64 components, full SVD, no whitening, fitted
   on standardized A only.
4. **pca64_leace:** LEACE fitted on the 64-dimensional A PCA scores and A
   statement labels. Do not add another scaler before fitting this eraser.

Keep the existing LEACE covariance/rank tolerances unchanged. Both fitted
erasers must satisfy their mean, true-rank and training covariance invariants.
Record the retained covariance spectrum/rank and the full pipeline's
dimensions. PCA discards 704 dimensions; this is not falsely described as
only a rank-one change to the original representation.

The PCA-only arm isolates compression. If compression itself meets the
absence screen, that is not evidence of an incremental LEACE repair.

## Probe fitting: B only

For every arm, train the existing standardized L2/LBFGS linear probe for
emotion, speaker, statement, gender and intensity. Preserve the M3 C grid
`[0.01, 0.1, 1]`, macro-F1 selection, seed `20260930`, tolerance `1e-4`,
maximum 2,000 iterations, and smaller-C tie break.

For statement only, additionally fit an RBF-SVM with C in `[0.1, 1, 10]`,
`gamma = 1 / input_dimension`, tolerance `1e-3` and at most 20,000 iterations.
Its feature StandardScaler is inside the same B-only grouped CV. Do not tune
gamma. Probability calibration is disabled; preserve decision scores rather
than inventing calibrated probabilities.

Select C separately for each arm/probe/attribute using B's frozen folds.
All probe preprocessing is fitted separately inside each inner training fold.
Refit each selected probe on B only. No eraser or PCA is refitted on B.

Finish and freeze **all 24 final models** before any C prediction or metric
is produced. This is 72 C-candidate records, each with three inner-fold
scores. Do not inspect C results and then alter any fitting choice.

## Evaluation and uncertainty

Evaluate C exactly once per final model: 24 metric rows and 8,640 prediction
records, with IDs, label order, fitted-model provenance and either logistic
probabilities or the SVM decision score. Recompute macro-F1 over the full
training label space and accuracy from these predictions. Preserve majority
baselines fitted using B labels, and the uniform-chance accuracy reference.

For statement only, obtain one-sided 95% upper bounds for macro-F1 and
accuracy by percentile cluster bootstrap:

- clusters are C's 24 speaker IDs;
- sample 24 clusters with replacement per draw, keeping every selected
  speaker's C utterances together;
- use 10,000 draws and NumPy SeedSequence `[20260930, 4201]`;
- use the identical sampled cluster counts for all four arms and both probes;
- use percentile 0.95 with NumPy's `linear` quantile rule;
- never refit models or choose hyperparameters in a bootstrap draw;
- persist the cluster counts and all 80,000 model/draw score records.

These are individual, conditional upper bounds for the specified fitted
probes on this explored corpus, not proof of zero mutual information,
simultaneous universal bounds over all possible classifiers, or uncertainty
over new encoders/datasets. Speaker-identity macro-F1 is not bootstrapped by
resampling away its own classes; other collateral results are descriptive.

## Unchanged validity thresholds and stronger precision check

The primary candidate is **pca64_leace**, compared with **pca64** for the
incremental effect. Require all of the following:

1. Both original and PCA-only statement probes are competent: before-erasure
   macro-F1 at least 0.60 for both the linear probe and RBF-SVM.
2. Both candidate statement probes lose at least 0.10 macro-F1 relative to
   their paired PCA-only probe.
3. Candidate macro-F1 and accuracy are each at most **0.55 for both probes**.
   A nonlinear recovery is a veto, not a reason to select the linear result.
4. Each corresponding one-sided 95% upper bound is at most **0.55**. A
   passing point estimate with an upper bound above the ceiling is
   inconclusive, not a pass.
5. Preserve collateral limits: emotion macro-F1 loss at most 0.10, and at
   most one other unrelated attribute losing more than 0.20. Check all three
   comparisons: original to PCA-only, PCA-only to PCA+LEACE, and original
   to PCA+LEACE. Do not hide damage caused by compression.

Keep every arm and all probe outcomes. Distinguish residual-information
failure, collateral failure, insufficient probe competence/precision, and
compression-only attribution. A compression-only absence result is not an
incremental LEACE success. These operational screens do not retroactively
change M4's failed conditions or certify absence beyond the tested probes.

## Stop, artifacts and deadline

After this one experiment, stop for review regardless of outcome. A validity
pass alone does not run or authorize fixed-head causal comparisons, random
control sweeps, M4B, speaker erasure, M5, or another dataset. The original
leave-one-statement-out contrast remains unidentifiable.

Save a new immutable result directory with protocol/config, role and CV IDs,
provenance and input hashes, A-fitted transforms, B-fitted probe parameters,
all C predictions, CV scores, bootstrap counts/scores, collateral tables,
decision JSON, figures, summary and output checksums. Verify old M3/M4
artifacts remain byte-identical.

If this bounded attempt fails or is inconclusive, follow the approved
October 5 cutoff: freeze M4 as failure analysis and consolidate the paper
for October 12 rather than searching additional methods or dimensions.
This execution must not edit the paper repository.
