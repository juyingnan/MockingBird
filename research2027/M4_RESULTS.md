### 1. Statement identity

- **Encoded?** Yes, from the unchanged M3 audit.
- **Successfully removed?** Partially, but **0/4** conditions passed the frozen
  validity gate. Fresh held-out statement macro-F1 fell substantially but
  remained above the predeclared `chance + 0.05 = 0.55` ceiling.
- **Used by the fixed emotion head?** Not established. The original heads were
  unchanged; targeted emotion macro-F1 changes were small, but incomplete
  target removal prevents a validated reliance or non-reliance conclusion.
- **Harmful under lexical shift?** Not identifiable with the preserved splits.
  Each directional training set has only one statement label. Fitting its
  statement contrast using the other statement would use held-out data.
- **Important collateral effects?** Limited: no unrelated collapse. The
  largest absolute non-target audit change was 0.0114 macro-F1 (intensity).

All four rows below use the repetition-reference partition. Diagnostic
macro-F1 comes from fresh three-fold probes inside the eraser-unseen
evaluation pool, not from reusing the training-set confounder classifier.
Task effects use the separate, fixed M3 emotion heads.

| Layer | Eraser | Statement F1 before | Statement F1 after | Fixed-head emotion F1 change | Mean random-control change | Target minus random |
|---|---|---|---|---|---|---|
| 5 | MP | 0.9986 | 0.6124 | +0.0033 | -0.0091 | +0.0124 |
| 5 | LEACE | 0.9986 | 0.6069 | +0.0001 | -0.0161 | +0.0162 |
| 12 | MP | 1.0000 | 0.5833 | -0.0036 | -0.0402 | +0.0366 |
| 12 | LEACE | 1.0000 | 0.5806 | -0.0005 | -0.0206 | +0.0201 |

These are retained **diagnostic** effects, not validated reliance estimates.
All four erasers removed rank 1. All 80 controls matched true removed rank
and training RMS perturbation; the largest relative RMS mismatch was
`9.13e-6`, within the frozen `1e-5` tolerance.

### 2. Speaker identity

- **Encoded?** Yes, from M3.
- **Successfully removed?** Not tested.
- **Used by the fixed emotion head?** Not tested.
- **Harmful under speaker shift?** Not tested.
- **Important collateral effects?** Not measured.

The statement validity gate blocked all 28 planned speaker conditions. This
is an enforced stop, not a negative speaker result. Selected layers remain
1 and 5; they were not replaced.

### 3. Headline result

**Substantial partial statement-information removal produced little direct
emotion-head change and little collateral damage, but residual held-out
decodability prevented a validated causal interpretation.**

M4 reached its mandated scientific stop, not completion of the causal chain.
No M4B emotion head was trained. Nothing here establishes "encoded but not
used" or harmful reliance under shift.

### 4. Null / contradictory findings

- Training cross-covariance was erased to numerical precision: the largest
  relative residual was `8.25e-15`. Nevertheless, fresh held-out probes
  recovered statement labels at 0.5806-0.6124 macro-F1. Training-set
  decorrelation therefore did not certify held-out erasure.
- Matched random removals generally harmed emotion more than the targeted
  removals. This contrast is preserved, but cannot override the failed
  target-validity gate.
- Control matching was defined on **training** RMS. Held-out perturbations
  were not exactly matched: for layer-5 LEACE the target test RMS was 4.6844,
  while random test RMS ranged from 3.8133 to 7.5699. This is an additional
  limitation for interpreting effect differences, not a reason to tune or
  resample controls after seeing outcomes.
- Eight lexical OOD conditions are `blocked_unidentifiable`; 28 speaker
  conditions are `blocked_statement_gate`; four statement reference
  conditions are `invalid`. All 40 appear in the inventory.
- All 40 M4B rows contain blocked reasons, not fabricated zero effects.

### 5. Replication recommendation

**Intervention debugging**, not more encoders or CREMA-D.

The confirmed issue is failure to transfer training-defined covariance
erasure into near-chance held-out target decodability, rather than a failed
rank check, unchanged-head check, split/ID check, or training-RMS control
match. Review this residual-information problem and design an independently
approved, leakage-free lexical contrast before attempting the missing
causal chain. Do not relax this run's thresholds or overwrite its results.

Engineering checks passed: 54 M0-M4 tests, CLI validation, dependency checks,
four original-head integrity checks and every prediction/metric check.
The run saved 208 metric rows, 149,760 held-out predictions, 360 diagnostic
probe fits, 84 transformations, 80 random draws and five PNG/PDF figure pairs.
Independent confusion-matrix calculations reproduced all 208 scores to
floating-point precision. All 33 input and 125 output hashes were checked.

- Run: `m4-wavlm-ravdess-20260930`, about 64 seconds on CPU.
- Branch: `research2027/m4-wavlm-ravdess`.
- Protocol commit: `0017c9cbb4d04763130958df386cf0345e3d8302`.
- Implementation/run commit: `6d2fec6c526d0996b0b22a15236300162d6e9e3f`.
- Seed: `20260930`; cache: `d99979b610d68bef`.
- [Frozen protocol](M4_PROTOCOL.md);
  [provenance and versions](results/m4-wavlm-ravdess-20260930/provenance.json);
  [machine summary](results/m4-wavlm-ravdess-20260930/summary.json);
  [condition inventory](results/m4-wavlm-ravdess-20260930/conditions.csv);
  [verification](results/m4-wavlm-ravdess-20260930/verification.json);
  [artifact index](RUNS.md).

Exact launch from `research2027`:

```powershell
.\scripts\stage2_intervene.ps1 -OutputDir "results\m4-wavlm-ravdess-20260930"
```

The result directory is immutable. M3 results, legacy directories and the
paper repository are unchanged. No new encoder, dataset or extraction was run.
