# Frozen M4 protocol: WavLM / RAVDESS

This protocol is committed before any real-data M4 intervention is fitted.
The user has authorized M4 after reviewing M3. No M3 result or paper file may
change. No additional encoder, dataset, or extraction is authorized.

## Inputs and layer selection

Use the validated cache `d99979b610d68bef` and the unchanged persisted M1
partitions. Bind M3 to result commit
`8160bc8733cdbf2f03792ddee8cc4a6b1089dcff` and verify all M3 output hashes.

Choose the repetition-reference macro-F1 maximum and the corresponding mean
shift-gap maximum using M3 only; break ties by lower layer. The per-target
union is fixed as follows, not the Cartesian product with all selected layers:

| Target | Layer | Anchor | M3 reference macro-F1 | M3 corresponding mean gap |
|---|---|---|---|---|
| Statement | 5 | Task | 0.9051025240187633 | 0.07535856505377903 |
| Statement | 12 | Lexical shift | 0.7908730735852958 | 0.20376218204580704 |
| Speaker | 1 | Speaker shift | 0.8331845243140246 | 0.20831989699940387 |
| Speaker | 5 | Task | 0.9051025240187633 | 0.1485356278410141 |

Methods are MP and LEACE. Run statement conditions first. No layer or method
may be replaced in response to its result.

## Identifiability and preserved splits

Both lexical directions have a **single statement label in training**. A
supervised statement contrast cannot be fitted without consuming the held-out
statement. These eight layer/method/direction conditions are explicitly
`blocked_unidentifiable`; they are not rank-zero erasure successes, unchanged
task predictions, or evidence of no harmful reliance. Do not fit an eraser on
the global repetition split and transfer it into these tests: it would have
seen some of their test utterances.

For speaker shifts, the emotion training and test sets have disjoint speaker
labels. A 20-class training-speaker probe must never be scored against four
unseen identities. Use the independent diagnostic audit below instead.

Core emotion evaluations retain exactly the M3 reference, both lexical
directions, and all six speaker folds. Auxiliary diagnostic folds do not
change these assignments. Missing/nonidentifiable conditions remain visible.

## Geometry and reusable erasers

Operate in the original emotion head's **training-only standardized feature
space**. The fixed-head intervention is between that unchanged scaler and
the unchanged logistic classifier. Use float64 throughout erasure fitting.

For centered training rows `Xc`, let `C = Xc.T Xc / n` and let `S` be the
cross-covariance with centered one-hot confounder labels. Preserve the training
mean `mu`. Transform rows by
`T(X) = X - ((X - mu) right) left.T`.

- **MP:** `Q = orth(S)`; `left = right = Q`. This is orthogonal mean-contrast
  projection for binary or multiclass confounders.
- **LEACE:** eigendecompose C on its positive numerical support. Let
  `W = C^(-1/2)`, `Wplus = C^(1/2)`, `U = orth(W S)`.
  Set `left = Wplus U`, `right = W U`. This is the affine minimum-training-MSE
  linear covariance eraser, not an orthogonal approximation.
- Covariance support uses relative tolerance `1e-10`. Concept rank uses
  `max(1e-12, largest_singular_value * 1e-10)`.
- Reject single-class concepts, zero identified rank, nonfinite data and
  material negative covariance eigenvalues. Never silently return identity.
- Check `right.T left = I`, training cross-covariance removal, preserved mean
  and idempotence with relative tolerance `1e-7`. Record true removed rank,
  training/test RMS and mean-L2 changes, and low-rank parameters.

## Original heads and independent erasure validation

Refit each original emotion pipeline at its **already selected M3 C**, using
the same solver, tolerance, seed and outer training IDs. Verify its held-out
predictions reproduce M3. Hash its fitted parameters before and after M4A.
Do not call fit on this object again.

Training-set covariance removal alone is not evidence of held-out erasure.
For all five attributes, train **fresh diagnostic probes entirely within the
outer test pool**, using three deterministic stratified cross-fitting folds
per attribute. The eraser and original head never fit on this pool. A
diagnostic probe's training and evaluation IDs are disjoint, and its scaler
and inner CV see only its own training fold.

For each original diagnostic training fold, select C using the existing M3
three-fold grid `[0.01, 0.1, 1]`; freeze that C for paired transformed probes.
Refit only the diagnostic probe/scaler on transformed diagnostic training rows.
Pool the out-of-fold predictions to compute macro-F1 and accuracy, preserving
all predictions and fold assignments. Apply this identical diagnostic recipe
to the target attribute under every random control as a specificity check.

These probes are **not** the emotion head used for M4A or M4B. Evaluation
labels never fit or tune the eraser or either task head. This also checks
whether erasure transfers to unseen speaker identities without treating
unseen IDs as known training classes. It is a distinct diagnostic estimate,
not an additional M3 audit result.

## Matched random controls

Freeze **20** draws per targeted condition; do not increase this count after
seeing results. Seeds are derived from SHA-256 of the base seed `20260930`,
condition identifier and draw index, and each concrete seed is recorded.

Use a random Gaussian frame in the training covariance's positive support.
For covariance eigenvalues `lambda`, orthonormalize
`diag(lambda^power) G`. Select power by bisection on `[-4, 4]`, at most 80
steps, to match targeted **training RMS perturbation**, relative error at
most `1e-5`. This uses only training geometry, never emotion outcomes.

For MP use this frame as both factors. For LEACE use the same whitening/
unwhitening construction as the target. In either case the expected squared
training displacement is `trace(Q.T C Q)`. Thus controls genuinely remove
the same number of dimensions; multiplying a projector by an arbitrary
strength and falsely calling it rank removal is forbidden.

Reject unmatched controls or a control indistinguishable from the targeted
operator (relative factor-product difference at most `0.001`). Do not
resample based on task effects. Record every draw, power, rank, RMS error,
mean-L2 change and target-probe outcome. These are covariance/RMS-conditioned
random subspaces, **not** unconditional isotropic controls. Higher moments
and each utterance's perturbation are not exactly matched; report this limit.

## Predeclared validity and scaling gates

These are conservative operational screening thresholds, not significance
tests. Primary target labels are balanced in the diagnostic pools, so `1/K`
is the uniform-chance accuracy and population macro-F1 reference.

A condition is valid only if all of the following hold:

1. Before-erasure target macro-F1 is at least chance + 0.10.
2. Target macro-F1 drops by at least 0.10, and after-erasure macro-F1 and
   accuracy are both at most chance + 0.05.
3. Fresh emotion-probe macro-F1 loses no more than 0.10.
4. At most one other unrelated attribute loses more than 0.20 macro-F1.
   Gender is structurally nested in speaker identity: report its loss, but
   do not count it as an unrelated collapse for speaker interventions.
5. All 20 controls pass true-rank, nonidentity and RMS-matching checks.

Complete the predeclared statement diagnostic block (both methods at both
selected layers). Do not run speaker conditions unless at least one statement
reference condition is valid. Retain failed conditions rather than changing
their layers, thresholds, erasers or controls. M4B is blocked for invalid
conditions and for OOD conditions lacking a valid matching reference.

Validity screening uses predeclared evaluation diagnostics. Report every
screened condition; do not present the surviving subset as an unbiased
selection procedure or use evaluation scores to tune fitting choices.

## M4A effect and interpretation

Use `effect = intervened score - original score`.
The signed reliance contrast is targeted effect minus mean random effect.
Report macro-F1 and accuracy, all 20 effects, mean, sample SD, and empirical
2.5%/97.5% quantiles (not a confidence interval over datasets).

Only interpret a valid condition with a target-specific comparison: median
random-control target-probe F1 must exceed targeted target-probe F1 by 0.05.
An effect outside the empirical control range with absolute excess at least
0.01 is exploratory evidence of excess fixed-head sensitivity. An effect
inside that range is consistent with no detected excess effect, not a proof
of equivalence or absence of reliance. Preserve both signs.

## M4B: separately retrained heads

Only after the validity block, fit a **new** emotion pipeline on transformed
outer training data. Keep the corresponding M3 C fixed; refit its scaler on
erased training data only. Do not reuse the audit probe or modify the M4A head.
The already fitted outer-training eraser remains identical.

Record original/erased reference and OOD macro-F1 and accuracy. Use
`delta_ID = erased_ID - original_ID`,
`delta_OOD = erased_OOD - original_OOD`,
`gap = reference - OOD`,
`delta_gap = erased_gap - original_gap` (negative means a reduced gap).
Do not compute a speaker aggregate from a selected subset of folds: all six
must have valid erasers and a valid matching reference.

An exploratory harmful-shortcut pattern requires OOD gain at least 0.02,
gap reduction at least 0.02, and reference loss no larger than 0.05.
Gap reduction caused only by worsening reference performance is not support.
No inferential claim or replication authorization follows automatically.

## Artifacts and stopping

Save protocol/config, M3-only selected-layer values, provenance and all input
hashes in a new immutable M4 result directory. Save eligibility/status rows
for **all 40** planned target conditions, fixed-head results, collateral
matrix, control distributions, targeted validation, M4B rows or explicit
blocked statuses, all held-out predictions, probe fold/C records, transform
parameters, figures, five-part summary, metric verification and checksums.

Fail loudly on engineering/numerical errors. Scientific gate failures produce
an explicit gated-stop/partial status, never a success-shaped zero effect.
Update experiment STATUS/RUNS only. Preserve the requested five-part human
handoff, including null/contradictory findings and a replication recommendation.
