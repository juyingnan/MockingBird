# M4-V1: bounded validity-repair result

**The approved experiment completed. Decision: FAIL_RESIDUAL_DECODABILITY.**

All four predeclared arms ran at layer 5. PCA-64 improved the conditioning
and reduced residual statement decodability relative to full-dimensional
LEACE, but neither the linear nor nonlinear probe met the unchanged 0.55
ceiling. No additional dimension, layer, seed, method or downstream experiment
was tried.

## Primary candidate: PCA-64 + LEACE

Final evaluation used C only, after all 24 B-fitted probe models were frozen.
U95 is the one-sided 95% actor-cluster percentile upper bound.

| Statement probe | Macro-F1 | Accuracy | F1 U95 | Accuracy U95 |
|---|---|---|---|---|
| Linear logistic regression | 0.5829 | 0.5833 | 0.6292 | 0.6306 |
| RBF-SVM | 0.6663 | 0.6667 | 0.7078 | 0.7083 |

Both point estimates already exceed 0.55; this is not merely an inconclusive
confidence-bound result. The nonlinear probe recovers still more information,
but it is not the sole veto: the linear probe also fails.

## All four arms, same frozen split

| Representation | Statement linear F1 | Statement RBF-SVM F1 |
|---|---|---|
| Original standardized representation | 0.9917 | 0.9806 |
| Full-dimensional LEACE | 0.6580 | 0.7722 |
| PCA-64 only | 0.9944 | 0.9917 |
| PCA-64 + LEACE | 0.5829 | 0.6663 |

PCA alone did not erase statement identity, so the observed reduction cannot
be credited to compression alone. The proposed combination improved on
full-space LEACE in this comparison, but did not establish information
absence within the specified probe/tolerance contract.

The A-fitted covariance's retained positive-support condition number
decreased from approximately `5.37e7` (719 supported directions in the
768-dimensional arm) to `85.33` (64 directions). Both erasers removed rank 1
and satisfied training covariance invariants to numerical precision.
Improved conditioning, as implemented here, was insufficient to validate
held-out erasure. This does not establish a universal cause of erasure failure.

## Collateral information

All three predeclared collateral comparisons passed. Relative to the
original representation, PCA-64 + LEACE changed fresh linear-probe macro-F1:

| Attribute | Change |
|---|---|
| Emotion | -0.0353 |
| Speaker | -0.0015 |
| Gender | 0.0000 |
| Intensity | -0.0321 |

PCA alone cost 0.0411 emotion macro-F1; adding LEACE after PCA changed it
by +0.0057. Compression damage was therefore measured, not hidden by using
only a PCA reference.

These are new diagnostic probes trained on B and evaluated on held-out
cells in C, not the original M3/M4 emotion heads. Their absolute scores are
not directly comparable to the M3 repetition reference or its OOD gaps.

## Leakage controls and uncertainty

- A/B/C contain 720/360/360 utterances in 180/90/90 disjoint
  `(speaker, emotion, intensity)` cells. Both statements and repetitions stay
  together. All speakers and audited label classes occur in each role.
- A alone fits representation scaling, PCA and erasers. B alone fits/tunes
  probes using the committed cell-grouped CV. C does not fit any model.
- All 24 models and representation parameters were frozen before C scoring
  and retained identical parameter hashes afterward.
- The statement bounds use 10,000 shared actor-cluster bootstrap draws,
  yielding 80,000 saved model/draw records. No model was refitted for a draw.
- Actor identities overlap across roles by design; this is not an
  unseen-speaker test or fresh independent corpus replication.
- The bounds concern these fitted probe procedures on a previously explored
  corpus, not all possible decoders or zero mutual information.

## Recommendation

Follow the approved stopping rule: **consolidate the paper as an
encoding/generalization audit with an out-of-sample erasure-validity failure
analysis.** Preserve reliance and harmful-use hypotheses as unresolved.

Do not relax thresholds or start another dimension/layer/seed/method search.
M4A, M4B and M5 were not launched. Any further intervention or redesigned
lexical-shift study needs a separately reviewed protocol. The original
single-statement training identifiability problem remains unresolved.

## Provenance and verification

- Run: `m4-validity-v1-wavlm-ravdess-20260930`.
- Branch: `research2027/m4-validity-v1`.
- Protocol/split commit: `d9fdf6575da150d0c8ce0a0ebe7070a5207137d4`.
- Scientific code commit: `db0a29638d9e7b2def50a9acd5b5431e29c82540`.
- Seed `20260930`; existing cache `d99979b610d68bef`; about 10 seconds on CPU.
- All 70 regression tests, dependency checks and frozen-input preflight pass.
- Verified 24 metric rows, 8,640 predictions, 72 CV candidates and every
  bootstrap score/bound. Independent confusion-matrix and binary-bootstrap
  calculations reproduced the published values to floating-point precision.
- All 164 recorded input hashes and 49 output payload hashes match. Original
  M3/M4 bundles and summaries are unchanged. No paper edits were authored.

Start with the [generated report and figures](results/m4-validity-v1-wavlm-ravdess-20260930/summary.md).
Also see the [decision](results/m4-validity-v1-wavlm-ravdess-20260930/decision.json),
[metrics](results/m4-validity-v1-wavlm-ravdess-20260930/metrics.csv),
[verification](results/m4-validity-v1-wavlm-ravdess-20260930/verification.json),
[provenance](results/m4-validity-v1-wavlm-ravdess-20260930/provenance.json),
[frozen protocol](M4_VALIDITY_V1_PROTOCOL.md) and [run index](RUNS.md).

Exact launch from `research2027`:

```powershell
.\scripts\stage2_validity.ps1 -OutputDir "results\m4-validity-v1-wavlm-ravdess-20260930"
```

The completed directory is immutable; this command must not overwrite it.
