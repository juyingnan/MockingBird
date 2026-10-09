# Scientific run index

Raw manifests and encoder caches remain in ignored `artifacts/`. Versioned
`results/` contains interpretable experiment outputs, not audio or embeddings.

| Experiment ID | Date (UTC) | Milestone | Code commit | Seed | State |
|---|---|---|---|---|---|
| [m3-wavlm-ravdess-20260930](results/m3-wavlm-ravdess-20260930/summary.md) | 2026-09-30 | M3A + M3B | `8a9e18fed6095b73ad7c29c9a0889c5a8a38802b` | 20260930 | Complete; PARTIAL / REVIEW NEEDED gate |
| [m4-wavlm-ravdess-20260930](results/m4-wavlm-ravdess-20260930/summary.md) | 2026-09-30 | M4A statement diagnostics; later stages gated | `6d2fec6c526d0996b0b22a15236300162d6e9e3f` | 20260930 | Gated stop; causal chain incomplete |
| [m4-validity-v1-wavlm-ravdess-20260930](results/m4-validity-v1-wavlm-ravdess-20260930/summary.md) | 2026-09-30 | Bounded M4 validity repair only | `db0a29638d9e7b2def50a9acd5b5431e29c82540` | 20260930 | Complete; FAIL_RESIDUAL_DECODABILITY |
| [das-statement-wavlm-ravdess-20261008](results/das-statement-wavlm-ravdess-20261008/summary.md) | 2026-10-09 | Exploratory post-freeze DAS validity only | `f3e4b22d36dd8f4974610b5d851059b2855f2032` | 20261008 / 20261009 | Complete; FAIL_RESIDUAL_DECODABILITY; local publication handoff |

## m3-wavlm-ravdess-20260930

- Branch: `research2027/m3-wavlm-ravdess`.
- Scope: all 13 WavLM Base Plus hidden states on the existing RAVDESS bundle.
- Cache fingerprint: `d99979b610d68bef`; unchanged throughout the run.
- Start/end: `2026-09-30T07:04:43.830876+00:00` /
  `2026-09-30T07:10:02.920664+00:00`; exit code 0.
- Configuration was committed before fitting: common C grid
  `[0.01, 0.1, 1]`, three-fold training-only selection, speaker-grouped
  validation for speaker shifts, train-only standardization and seed `20260930`.
- [Provenance](results/m3-wavlm-ravdess-20260930/provenance.json) records the
  exact command, package versions, model revision and every input SHA-256.
- [Probe table](results/m3-wavlm-ravdess-20260930/probes.csv): 65 rows.
- [Generalization table](results/m3-wavlm-ravdess-20260930/generalization.csv):
  117 rows, including both lexical directions and all six speaker folds.
- [CV candidates](results/m3-wavlm-ravdess-20260930/cv_results.csv): 507 rows
  for 169 distinct final fits, with every inner-fold score.
- [Held-out predictions](results/m3-wavlm-ravdess-20260930/predictions.parquet):
  93,600 records with labels, probabilities, class order and majority predictions.
- Figures: [encoding PNG](results/m3-wavlm-ravdess-20260930/encoding_heatmap.png),
  [encoding PDF](results/m3-wavlm-ravdess-20260930/encoding_heatmap.pdf),
  [generalization PNG](results/m3-wavlm-ravdess-20260930/emotion_generalization.png),
  [generalization PDF](results/m3-wavlm-ravdess-20260930/emotion_generalization.pdf).
- [Verification](results/m3-wavlm-ravdess-20260930/verification.json):
  all saved scores and held-out IDs checked; maximum score discrepancy zero.
  [Output checksums](results/m3-wavlm-ravdess-20260930/output_sha256.json)
  preserve payload integrity across machines. Git attributes retain exact
  result bytes and the recorded configuration's CRLF line endings.
- [Completion marker](results/m3-wavlm-ravdess-20260930/status.json) and
  [per-layer progress](results/m3-wavlm-ravdess-20260930/progress.jsonl).

Exact launch from `research2027`:

```powershell
.\scripts\stage1_probe.ps1 -OutputDir "results\m3-wavlm-ravdess-20260930"
```

Validation: `.\scripts\smoke_test.ps1` (34 tests and CLI help),
`.\.venv\Scripts\python.exe -m pip check`, and
`.\scripts\stage1_probe.ps1 -ValidateOnly` passed. The output directory is
immutable to the runner; choose a new directory for any explicitly requested
replay. No encoder extraction, erasure, new model or extra dataset was run.

Review [M3_RESULTS.md](M3_RESULTS.md) before M4. These are encoding and
baseline-generalization results only; no reliance or harmful-use claim is made.

## m4-wavlm-ravdess-20260930

- Branch: `research2027/m4-wavlm-ravdess`.
- Protocol frozen at `0017c9cbb4d04763130958df386cf0345e3d8302` before
  real-data intervention fitting; code/run commit
  `6d2fec6c526d0996b0b22a15236300162d6e9e3f`.
- Cache `d99979b610d68bef`, original M1 partitions and all M3 results unchanged.
- M3-only selection: statement 5/12; speaker 1/5. MP and LEACE; 20 fixed
  random draws per tested condition; seed `20260930`.
- Four statement reference conditions ran. None met the frozen held-out
  target-erasure criterion. Eight lexical OOD conditions were unidentifiable;
  28 speaker conditions were gated. M4B trained zero heads.
- Runtime about 64 seconds on CPU; exit code 0 means the prescribed gates
  and artifact verification completed, not that the causal stage succeeded.
- [Frozen config](results/m4-wavlm-ravdess-20260930/config.json),
  [protocol](results/m4-wavlm-ravdess-20260930/protocol.md),
  [selected layers](results/m4-wavlm-ravdess-20260930/selected_layers.json),
  [provenance](results/m4-wavlm-ravdess-20260930/provenance.json).
- [Condition inventory](results/m4-wavlm-ravdess-20260930/conditions.csv):
  all 40 planned conditions with explicit status/reason.
- [Target validation](results/m4-wavlm-ravdess-20260930/targeted_validation.csv);
  [collateral damage](results/m4-wavlm-ravdess-20260930/collateral_damage.csv);
  [80 random controls](results/m4-wavlm-ravdess-20260930/random_controls.csv);
  [fixed-head effects](results/m4-wavlm-ravdess-20260930/fixed_head_reliance.csv);
  [blocked M4B rows](results/m4-wavlm-ravdess-20260930/retrained_heads.csv).
- [208 metric rows](results/m4-wavlm-ravdess-20260930/metrics.csv) and
  [149,760 held-out predictions](results/m4-wavlm-ravdess-20260930/predictions/).
- [Diagnostic folds](results/m4-wavlm-ravdess-20260930/diagnostic_folds/),
  [360 probe-fit records](results/m4-wavlm-ravdess-20260930/diagnostic_probe_fits.csv),
  [outer fitting IDs](results/m4-wavlm-ravdess-20260930/training_ids.parquet),
  [head integrity](results/m4-wavlm-ravdess-20260930/head_integrity.csv),
  [saved heads](results/m4-wavlm-ravdess-20260930/heads/),
  [84 saved transforms](results/m4-wavlm-ravdess-20260930/transforms/).
- Figures (PNG with corresponding PDF): [target decodability](results/m4-wavlm-ravdess-20260930/target_decodability.png),
  [fixed head and controls](results/m4-wavlm-ravdess-20260930/fixed_head_controls.png),
  [collateral matrix](results/m4-wavlm-ravdess-20260930/collateral_damage.png),
  [M4B gate notice](results/m4-wavlm-ravdess-20260930/retraining_performance.png),
  [OOD-gap gate notice](results/m4-wavlm-ravdess-20260930/ood_gap_change.png).
- [Metric/ID/transform verification](results/m4-wavlm-ravdess-20260930/verification.json),
  [completion/gate marker](results/m4-wavlm-ravdess-20260930/status.json),
  [machine summary](results/m4-wavlm-ravdess-20260930/summary.json),
  [125 output checksums](results/m4-wavlm-ravdess-20260930/output_sha256.json).

Exact launch from `research2027`:

```powershell
.\scripts\stage2_intervene.ps1 -OutputDir "results\m4-wavlm-ravdess-20260930"
```

Validation: 54 M0-M4 tests, CLI checks and dependency checks pass. Independent
confusion-matrix calculations reproduce all 208 metric rows. Original
emotion heads are unchanged; all controls match rank and training RMS.
Held-out residual target decodability still fails the scientific validity
gate. Read [M4_RESULTS.md](M4_RESULTS.md) before approving new work.

## m4-validity-v1-wavlm-ravdess-20260930

- Branch: `research2027/m4-validity-v1`.
- Protocol and exact split/CV IDs frozen at
  `d9fdf6575da150d0c8ce0a0ebe7070a5207137d4`; scientific code commit
  `db0a29638d9e7b2def50a9acd5b5431e29c82540`.
- Existing cache `d99979b610d68bef`, layer 5 only. Four arms:
  original, LEACE, PCA-64, PCA-64 + LEACE.
- A/B/C are 720/360/360 utterances in disjoint complete matched cells.
  All 24 diagnostic probes are frozen before C evaluation.
- The candidate's linear/RBF statement F1 is 0.5829/0.6663; upper bounds
  0.6292/0.7078. Both fail the frozen 0.55 ceiling despite passing
  collateral checks. No follow-on causal or replication experiment ran.
- [Protocol](results/m4-validity-v1-wavlm-ravdess-20260930/protocol.md),
  [config](results/m4-validity-v1-wavlm-ravdess-20260930/config.json),
  [roles](results/m4-validity-v1-wavlm-ravdess-20260930/roles.parquet),
  [B-only CV](results/m4-validity-v1-wavlm-ravdess-20260930/probe_cv.parquet),
  [provenance](results/m4-validity-v1-wavlm-ravdess-20260930/provenance.json).
- [24 metric rows](results/m4-validity-v1-wavlm-ravdess-20260930/metrics.csv),
  [8,640 predictions](results/m4-validity-v1-wavlm-ravdess-20260930/predictions.parquet),
  [72 CV candidates](results/m4-validity-v1-wavlm-ravdess-20260930/cv_results.csv),
  [collateral comparisons](results/m4-validity-v1-wavlm-ravdess-20260930/collateral.csv).
- [80,000 bootstrap scores](results/m4-validity-v1-wavlm-ravdess-20260930/bootstrap_scores.parquet)
  and [10,000 common actor-cluster draws](results/m4-validity-v1-wavlm-ravdess-20260930/bootstrap_clusters.npz).
- [Model fits](results/m4-validity-v1-wavlm-ravdess-20260930/model_fits.csv),
  [model parameters](results/m4-validity-v1-wavlm-ravdess-20260930/models/),
  [frozen-before-C marker](results/m4-validity-v1-wavlm-ravdess-20260930/fit_complete.json),
  [model integrity](results/m4-validity-v1-wavlm-ravdess-20260930/model_integrity.csv),
  [phase times](results/m4-validity-v1-wavlm-ravdess-20260930/phase_times.json).
- [Representation parameters](results/m4-validity-v1-wavlm-ravdess-20260930/representation_parameters.npz)
  and [training covariance diagnostics](results/m4-validity-v1-wavlm-ravdess-20260930/transform_diagnostics.csv).
  SVM snapshots store support-row indices and hashes, not copied embeddings.
- Figures, each with a PDF counterpart:
  [statement validity and upper bounds](results/m4-validity-v1-wavlm-ravdess-20260930/statement_validity.png),
  [attribute information](results/m4-validity-v1-wavlm-ravdess-20260930/attribute_information.png).
- [Decision](results/m4-validity-v1-wavlm-ravdess-20260930/decision.json),
  [verification](results/m4-validity-v1-wavlm-ravdess-20260930/verification.json),
  [completion marker](results/m4-validity-v1-wavlm-ravdess-20260930/status.json),
  [checksums](results/m4-validity-v1-wavlm-ravdess-20260930/output_sha256.json).

Exact launch from `research2027`:

```powershell
.\scripts\stage2_validity.ps1 -OutputDir "results\m4-validity-v1-wavlm-ravdess-20260930"
```

Runtime about 10 seconds on CPU; exit code 0. All 70 regression tests and
preflight/dependency checks pass. Independent confusion-matrix and binary
cluster-bootstrap calculations reproduce every published score/bound.
All 164 input hashes and 49 output payload hashes match. M3/M4 remain
unchanged. Read [M4_VALIDITY_V1_RESULTS.md](M4_VALIDITY_V1_RESULTS.md) for the
failure interpretation and approved stopping recommendation.
