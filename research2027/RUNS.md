# Scientific run index

Raw manifests and encoder caches remain in ignored `artifacts/`. Versioned
`results/` contains interpretable experiment outputs, not audio or embeddings.

| Experiment ID | Date (UTC) | Milestone | Code commit | Seed | State |
|---|---|---|---|---|---|
| [m3-wavlm-ravdess-20260930](results/m3-wavlm-ravdess-20260930/summary.md) | 2026-09-30 | M3A + M3B | `8a9e18fed6095b73ad7c29c9a0889c5a8a38802b` | 20260930 | Complete; PARTIAL / REVIEW NEEDED gate |

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
