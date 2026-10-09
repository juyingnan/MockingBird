# Status

## M0 — Scaffold

**Complete.**

- Added the isolated `mockingbird2027` package scaffold and setuptools
  configuration.
- Added PowerShell environment setup and project-local ignore rules.
- Added the minimal CLI and synthetic smoke test.
- No model extraction or experiments were implemented.
- Runtime validated with Python 3.11.9 in the project virtual environment.
- Environment setup, package import, CLI behavior, and lightweight tests pass.

## M1 — Dataset manifests and split metadata

**Complete; real-corpus runtime validation passes.**

- Added strict RAVDESS audio-only speech filename parsing and WAV-header metadata
  extraction.
- Added manifest validation for identifiers, paths, required labels, durations,
  sample rates, complete-corpus counts, label distributions, and per-speaker
  emotion balance.
- Added deterministic repetition, bidirectional lexical, bidirectional gender,
  and gender-balanced speaker-group folds with coverage and leakage checks.
- Added Parquet manifest and split outputs plus a compact JSON manifest report.
- Added a `manifest ravdess` CLI command and synthetic tests; CREMA-D remains
  deferred because it would broaden M1.
- Synthetic manifest and split validation tests pass in the project virtual
  environment.
- Added ignored machine-local `configs\paths.yaml` with `paths.ravdess_root`
  pointing to the supplied corpus; the example configuration remains generic.
- Generated `artifacts\manifests\ravdess.parquet`,
  `artifacts\manifests\ravdess_splits.parquet`, and
  `artifacts\reports\ravdess_manifest_report.json` from the real corpus.
- Real validation passed: 1,440 utterances, 24 speakers, no duplicate paths or
  IDs, no missing labels, deterministic split metadata, complete split coverage,
  and no split leakage.
- Counts: neutral 96; each of calm, happy, sad, angry, fearful, disgust, and
  surprised 192; statements 1/2 each 720; repetitions 1/2 each 720; female/male
  each 720.
- No neural models, extraction, or experiments were implemented.

## Scaffold correction

**Complete.**

- Added repository-level agent guidance protecting all legacy directories and
  directing NAACL 2027 work to `research2027/`.
- Added the planned `configs/` directory hierarchy and
  `configs/paths.example.yaml`.
- Kept the detailed project instructions in `research2027/AGENTS.md` unchanged.
- No research functionality or legacy experiment code was modified.
- Scaffold integrity checks and the existing lightweight Python tests pass.

## M2 — Frozen WavLM extraction

**Complete; real-data smoke and full extraction validation pass.**

- Implemented `microsoft/wavlm-base-plus` loading with its matching processor,
  model evaluation mode, frozen parameters, and inference-mode execution.
- Added complete-utterance mono preprocessing, polyphase resampling to 16 kHz,
  processor attention masks, feature-frame mask conversion, and float32 masked
  mean pooling for every hidden state.
- Added deterministic utterance ordering and immutable fingerprinted float16
  caches containing `ids.npy`, every `layer_XX.npy`, and `metadata.json`.
- Cache validation checks identity fingerprints, completion status, utterance
  order, layer count, shapes, dtype, and finite values. Valid caches are reused
  before loading the model.
- Added a smoke-gated CLI and `scripts\stage1_extract.ps1`; uncached full
  extraction cannot begin until the approximately 16-utterance smoke cache
  passes validation.
- Added targeted synthetic tests for resampling, padding-aware pooling, frozen
  all-layer extraction, cache reuse, ordering, NaN rejection, and smoke gating.

### Execution results

- **Interpreter:** Python 3.11.9 through the project-local
  `.venv\Scripts\python.exe`.
- **Environment/tests:** Setup passed. The final M0-M2 suite passed 11 tests in
  3.01 seconds; focused regression checks also passed after correcting module CLI
  dispatch and using the model's feature extractor rather than requiring a
  tokenizer.
- **Smoke extraction:** Passed for the first 16 deterministically ordered
  utterances. Cache fingerprint `707d40ee06e41906`; 13 layers, each shaped
  `(16, 768)`, stored as finite float16 values. Ordering, feature-vector-mask
  pooling metadata, and reuse without model loading were validated.
- **Full extraction:** Completed successfully exactly once on CPU. Cache
  fingerprint `d99979b610d68bef`; 1,440 utterances and 13 layers, each shaped
  `(1440, 768)`, stored as finite float16 values. Ordering and reuse without
  model loading were validated.
- **Full cache:**
  `artifacts\embeddings\ravdess\microsoft_wavlm-base-plus\d99979b610d68bef`.
- **Launch:** PID 26672 (launcher PID 14424), batch size 4, smoke size 16,
  device `auto`. Command:
  `.venv\Scripts\python.exe -m mockingbird2027 extract wavlm artifacts\manifests\ravdess.parquet --cache-root artifacts\embeddings --batch-size 4 --smoke-size 16 --device auto`.
- **Log:** `artifacts\logs\wavlm-full-d99979b610d68bef.log` records the command,
  PIDs, completion metadata, captured output, and exit code 0.
- **Warnings:** SciPy skipped an unrecognized non-data WAV chunk; PyTorch warned
  that mixed attention-mask types are deprecated. Neither warning caused a
  validation failure.

## M3 — Probe map and controlled OOD baseline

**Complete; gate is PARTIAL / REVIEW NEEDED. M4 has not started.**

- Reused and validated the transferred M1/M2 bundle without audio access or
  encoder inference: exact cache `d99979b610d68bef`, manifest identity,
  1,440 utterances, 13 x 768 finite float16 states, and all 12 split groups.
- Added training-only standardized logistic probes with a common small C grid,
  deterministic inner CV, speaker-grouped tuning for speaker shifts, and
  explicit failure on non-convergence.
- Added complete result tables, saved predictions, CV scores, provenance,
  table-driven figures, prediction/metric verification and review-only gate
  reporting. Results cannot overwrite an existing run.
- The pre-run [M3 protocol](M3_RESULTS.md) records seeds, selection conventions,
  baseline/gap definitions and interpretation limits.
- Validation: all 34 M0-M3 tests and CLI smoke checks pass; `pip check` passes.
  The real-input `stage1_probe.ps1 -ValidateOnly` check also passes.
- Completed run `m3-wavlm-ravdess-20260930` from code commit
  `8a9e18fed6095b73ad7c29c9a0889c5a8a38802b`, seed `20260930`, in about
  319 seconds on CPU. All 13 layers finished with exit code 0.
- Saved 65 audit rows, 117 generalization rows, 507 CV-candidate rows,
  93,600 held-out prediction records, PNG/PDF figures and complete provenance.
  Every saved score was recomputed against persisted test IDs; all input and
  output checksums match.
- Best emotion reference: layer 5, macro-F1 0.9051. Speaker encoding peaks
  at layer 4 (0.9958); statement encoding first reaches 1.0000 at layer 3.
- At layer 5, lexical macro-F1 is 0.8156 / 0.8439 (gaps +0.0895 / +0.0612);
  speaker-fold mean is 0.7566 (gap +0.1485; fold range 0.6891-0.8088).
  Speaker-shift mean peaks at layer 6 (0.7666). Both lexical directions and
  the speaker mean have positive gaps at every layer.
- The directional gate screens have supporting descriptive evidence, but no
  numerical meaningful-effect threshold was predeclared. Human review remains
  required; these results do not establish reliance or harmful use.
- Full results, caveats and commands: [M3_RESULTS.md](M3_RESULTS.md).
  Artifact index: [RUNS.md](RUNS.md). No known missing M3 outputs or runtime
  failures remain.

## M4 — Reliance and harmful-under-shift experiments

**Statement diagnostic block completed; scientific gate stopped expansion.
The M4 causal chain is NOT complete.**

- Frozen protocol commit: `0017c9cbb4d04763130958df386cf0345e3d8302`.
  [Protocol](M4_PROTOCOL.md) and
  [configuration](configs/experiments/m4_wavlm_ravdess.json) predate all
  real-data interventions.
- M3-only selected layers: statement 5/12, speaker 1/5. M3 inputs and all
  result checksums have been revalidated and remain unchanged.
- Eight lexical OOD conditions are unidentifiable: each original training
  split contains only one statement label. They remain explicit blocked
  conditions; no test-data eraser or alternate split is substituted.
- Added reusable float64 MP/LEACE erasers, true-rank/RMS-matched random
  controls, fresh cross-fitted held-out attribute diagnostics, immutable
  fixed-head checks, separately gated retraining, saved predictions,
  verification, and table-derived figures.
- All 54 M0-M4 tests, CLI smoke checks, `pip check`, and
  `stage2_intervene.ps1 -ValidateOnly` pass.
- Run `m4-wavlm-ravdess-20260930` used implementation commit
  `6d2fec6c526d0996b0b22a15236300162d6e9e3f`, seed `20260930`, and completed
  its gated schedule in about 64 seconds on CPU, exit code 0.
- MP/LEACE at statement layers 5/12 removed rank 1 and training
  cross-covariance to numerical precision. Fresh held-out statement F1
  remained 0.6124 / 0.6069 / 0.5833 / 0.5806, above the frozen 0.55 ceiling.
  Thus all four conditions failed target-removal validity despite substantial
  partial removal and small collateral effects.
- All 80 random controls passed true-rank/training-RMS matching. Original
  emotion heads stayed fixed and reproduced M3; targeted task changes ranged
  from -0.0036 to +0.0033 macro-F1. These are not validated reliance claims.
- The gate correctly blocked all 28 speaker conditions and every M4B fit.
  The eight lexical OOD conditions remain unidentifiable. All 40 planned
  conditions retain explicit statuses/reasons; none was hidden or imputed.
- Saved 208 metric rows, 149,760 held-out predictions, 360 diagnostic fits,
  84 transforms, figures, provenance and checksums. Independent confusion
  matrices reproduced all scores; all 33 input and 125 output hashes match.
- Full five-part handoff and limitations: [M4_RESULTS.md](M4_RESULTS.md).
  No engineering/runtime failure remains, but intervention validity and the
  lexical identifiability constraint scientifically block M4 completion.

## M4-V1 — Approved bounded validity-repair follow-up

**Complete. Decision: FAIL_RESIDUAL_DECODABILITY. No downstream expansion.**

- Approved scope: layer 5, statement identity, and four arms only:
  original, full-dimensional LEACE, PCA-64, and PCA-64 + LEACE.
- Protocol/split commit: `d9fdf6575da150d0c8ce0a0ebe7070a5207137d4`.
  [Protocol](M4_VALIDITY_V1_PROTOCOL.md) and
  [configuration](configs/experiments/m4_validity_v1.json) predate fitting.
- Exact independent matched-cell roles: A 720 utterances for representation
  fitting, B 360 for probe fitting/tuning, and C 360 for final evaluation.
  Frozen B-only CV contains 1,800 assignments and keeps entire cells together.
- All 24 linear/nonlinear diagnostic models are frozen before C scoring.
  Statement precision uses 10,000 paired actor-cluster bootstrap draws.
  The 0.55 ceilings and original collateral limits are not relaxed.
- All 70 regression tests, CLI checks, dependency checks and frozen-input
  preflight pass. M3/M4 inputs and result bundles remain unchanged.
- Run `m4-validity-v1-wavlm-ravdess-20260930` used code commit
  `db0a29638d9e7b2def50a9acd5b5431e29c82540`, completed all four arms and
  24 models in about 10 seconds on CPU, and exited 0.
- Primary PCA-64 + LEACE statement results on C: linear F1 0.5829,
  accuracy 0.5833, upper bounds 0.6292/0.6306; RBF-SVM F1 0.6663,
  accuracy 0.6667, upper bounds 0.7078/0.7083. Both point estimates and
  upper bounds exceed the unchanged 0.55 ceiling.
- Full-dimensional LEACE has linear/RBF F1 0.6580/0.7722; PCA-only
  retains F1 0.9944/0.9917. The bottleneck helped but did not validate
  held-out erasure. Its positive-support covariance condition is 85.33
  versus about 5.37e7 in the full-dimensional training estimate.
- All collateral checks pass; total emotion diagnostic F1 change is
  -0.0353 versus the new split's original-representation probe.
- Saved and verified 24 metric rows, 8,640 predictions, 72 CV candidates,
  80,000 bootstrap score rows, model parameters, figures and hashes.
  Independent calculations reproduce all metrics/bounds; all 164 input
  and 49 output payload hashes match.
- Full interpretation and boundaries:
  [M4_VALIDITY_V1_RESULTS.md](M4_VALIDITY_V1_RESULTS.md).

## Next

Stop the approved bounded experiment here. Consolidate the encoding/
generalization audit and M4 validity-failure analysis for the October 12
deadline rather than trying additional dimensions, layers, seeds or methods.
Reliance and harmful use remain unresolved. New M4A/M4B, speaker erasure,
M5 or another dataset require separately reviewed approval. Preserve all
existing results; no paper edits are authorized by this execution.

## DAS exploratory follow-up (authorized 2026-10-08)

**Complete. FAIL_RESIDUAL_DECODABILITY; no downstream or rescue experiment.**

- New, explicitly authorized post-freeze experiment; no changes to the paper
  or M3/M4/M4-V1 bundles. Upstream parent: `24607d4`.
- [Design and stop rules](DAS_FOLLOWUP_PLAN.md) distinguish readout-mediated
  matched-pair DAS from native model interpretation and independent erasure.
  Fixed layer 5/cache; 14 A-only dimension/seed candidates, mean ablation,
  fresh B diagnostics, arbitrary-mapping and limited random-subspace controls.
- Persisted A-train/A-val/B/C metadata has 480/240/360/360 utterances with
  complete cells kept together. The staged CLI prevents C use before the
  candidate and all B models are frozen and committed.
- Code/protocol commit: `f3e4b22d36dd8f4974610b5d851059b2855f2032`.
  All 44 targeted synthetic/shared-helper tests, dependency checks and
  immutable-input checks pass.
- The predeclared 14-candidate A-only search selected rank 32, seed
  20261009, checkpoint 200. A-val interchange accuracy is 1.0, but fresh
  A-val statement F1 remains 0.9542 linear / 0.9040 RBF. These are
  development results, not the locked result or evidence of erasure.
- Pre-C freeze commit: `166baadb38aafc6562db73cb832097013c6372b9`.
  All 30 independent B diagnostic models were frozen and committed before
  C was opened exactly once.
- Locked-C statement macro-F1/accuracy: linear 0.9305/0.9306,
  RBF 0.8971/0.8972. Corresponding upper bounds are
  0.9592/0.9592 and 0.9290/0.9290; all exceed 0.55. Both F1 reductions
  also fail the 0.10 minimum. Collateral limits pass.
- Three rank-matched random controls miss the intended RMS tolerance;
  they remain explicitly unmatched, not stronger causal controls.
- Recorded phases: fitting/selection/B 290.77 s; C/verification/plots
  4.03 s, CPU with one numerical thread. All 30 metrics, 10,800 predictions
  and 100,000 bootstrap scores recompute with error zero; 212 input hashes
  checked. The original cached representations and bundles are unchanged.
- Published to the authoritative upstream on
  [research2027/das-statement-subspace](https://github.com/juyingnan/MockingBird/tree/research2027/das-statement-subspace).
  Scientific result commit: `62b3e97a4f6178671235c7261c4df6262cb5eb42`.
  The frozen run summary records the earlier fork-permission failure;
  that publishing limitation was resolved through authorized upstream
  access. No scientific payload was changed or C evaluation repeated.
- [Compact result and reproduction commands](results/das-statement-wavlm-ravdess-20261008/summary.md).
  Stop here. This is successful readout steering without validated erasure,
  not an explanation of LEACE's failure or evidence of emotion reliance.
