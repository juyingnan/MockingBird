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

**Implementation and synthetic checks ready; scientific run pending.**

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

## Next

Run the full M3 command from `M3_RESULTS.md` after the implementation is
committed. Review M3A/M3B before any M4 work. The paper repository remains
unchanged; no M3 scientific result has been claimed yet.
