# M3 results

Experiment: `m3-wavlm-ravdess-20260930`.
Code: `8a9e18fed6095b73ad7c29c9a0889c5a8a38802b` on `research2027/m3-wavlm-ravdess`.
Cache: `d99979b610d68bef`.
Python: `3.11.9`; seed: `20260930`.

Exact invocation from `research2027`:

```powershell
.\.venv\Scripts\python.exe -m mockingbird2027 probe --config "configs\experiments\m3_wavlm_ravdess.json" --output-dir "results\m3-wavlm-ravdess-20260930"
```

Package versions, complete configuration and input SHA-256 hashes: [provenance.json](provenance.json).
C is selected by 3-fold training-only macro-F1; smaller C wins exact ties. Speaker-shift inner folds group speakers; other inner folds are stratified. Every scaler is fitted within its training fold. Final fits use all outer training data. The repetition reference reuses the identical emotion-audit fit and predictions.

## M3A: encoding only

Cells are macro-F1 / accuracy. Layer 0 is the initial hidden state; layers 1-12 are transformer states.

| Layer | Emotion | Speaker | Statement | Gender | Intensity |
|---|---|---|---|---|---|
| 0 | 0.7018 / 0.7042 | 0.9584 / 0.9583 | 0.9694 / 0.9694 | 0.9958 / 0.9958 | 0.7865 / 0.7889 |
| 1 | 0.8332 / 0.8347 | 0.9772 / 0.9778 | 0.9889 / 0.9889 | 1.0000 / 1.0000 | 0.8160 / 0.8181 |
| 2 | 0.8462 / 0.8500 | 0.9778 / 0.9778 | 0.9944 / 0.9944 | 1.0000 / 1.0000 | 0.8241 / 0.8250 |
| 3 | 0.8565 / 0.8625 | 0.9847 / 0.9847 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.8197 / 0.8208 |
| 4 | 0.8804 / 0.8847 | 0.9958 / 0.9958 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.8191 / 0.8194 |
| 5 | 0.9051 / 0.9083 | 0.9903 / 0.9903 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.8153 / 0.8167 |
| 6 | 0.8970 / 0.8972 | 0.9917 / 0.9917 | 1.0000 / 1.0000 | 0.9986 / 0.9986 | 0.8207 / 0.8222 |
| 7 | 0.8783 / 0.8778 | 0.9805 / 0.9806 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.8099 / 0.8111 |
| 8 | 0.8417 / 0.8417 | 0.9735 / 0.9736 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.8029 / 0.8042 |
| 9 | 0.8314 / 0.8319 | 0.9734 / 0.9736 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.8048 / 0.8069 |
| 10 | 0.8168 / 0.8181 | 0.9624 / 0.9625 | 0.9986 / 0.9986 | 0.9972 / 0.9972 | 0.8066 / 0.8083 |
| 11 | 0.8103 / 0.8167 | 0.9723 / 0.9722 | 0.9986 / 0.9986 | 1.0000 / 1.0000 | 0.8020 / 0.8042 |
| 12 | 0.7909 / 0.7917 | 0.9513 / 0.9514 | 1.0000 / 1.0000 | 0.9986 / 0.9986 | 0.7898 / 0.7917 |

| Attribute | Majority F1 / accuracy | Uniform chance accuracy | Best F1 layer | Weakest F1 layer |
|---|---|---|---|---|
| emotion | 0.0294 / 0.1333 | 0.1250 | 5 (0.9051) | 0 (0.7018) |
| speaker | 0.0033 / 0.0417 | 0.0417 | 4 (0.9958) | 12 (0.9513) |
| statement | 0.3333 / 0.5000 | 0.5000 | 3 (1.0000) | 0 (0.9694) |
| gender | 0.3333 / 0.5000 | 0.5000 | 1 (1.0000) | 0 (0.9958) |
| intensity | 0.3478 / 0.5333 | 0.5000 | 2 (0.8241) | 0 (0.7865) |

Uniform chance is an accuracy reference, not a macro-F1 estimate.
![Encoding heatmap](encoding_heatmap.png)

## M3B: controlled emotion generalization

Cells are macro-F1 / accuracy. Speaker values are unweighted means over equal-size folds.
Gaps below use macro-F1: repetition reference minus shifted score (positive means worse under shift).

| Layer | Repetition | Lexical 1 to 2 | Lexical 2 to 1 | Speaker mean | Gap L1 to 2 | Gap L2 to 1 | Gap speaker |
|---|---|---|---|---|---|---|---|
| 0 | 0.7018 / 0.7042 | 0.6535 / 0.6556 | 0.6708 / 0.6722 | 0.5285 / 0.5437 | +0.0483 | +0.0309 | +0.1733 |
| 1 | 0.8332 / 0.8347 | 0.7699 / 0.7722 | 0.8019 / 0.8014 | 0.6249 / 0.6361 | +0.0633 | +0.0313 | +0.2083 |
| 2 | 0.8462 / 0.8500 | 0.7514 / 0.7542 | 0.8111 / 0.8167 | 0.6579 / 0.6681 | +0.0948 | +0.0350 | +0.1883 |
| 3 | 0.8565 / 0.8625 | 0.7576 / 0.7667 | 0.8137 / 0.8208 | 0.6905 / 0.7014 | +0.0989 | +0.0428 | +0.1660 |
| 4 | 0.8804 / 0.8847 | 0.8186 / 0.8278 | 0.8453 / 0.8500 | 0.7345 / 0.7417 | +0.0618 | +0.0351 | +0.1459 |
| 5 | 0.9051 / 0.9083 | 0.8156 / 0.8264 | 0.8439 / 0.8556 | 0.7566 / 0.7639 | +0.0895 | +0.0612 | +0.1485 |
| 6 | 0.8970 / 0.8972 | 0.8192 / 0.8264 | 0.8359 / 0.8361 | 0.7666 / 0.7687 | +0.0778 | +0.0611 | +0.1304 |
| 7 | 0.8783 / 0.8778 | 0.8173 / 0.8264 | 0.8316 / 0.8319 | 0.7591 / 0.7618 | +0.0610 | +0.0467 | +0.1192 |
| 8 | 0.8417 / 0.8417 | 0.7436 / 0.7444 | 0.8097 / 0.8083 | 0.7340 / 0.7375 | +0.0981 | +0.0320 | +0.1077 |
| 9 | 0.8314 / 0.8319 | 0.7406 / 0.7486 | 0.6996 / 0.7000 | 0.7021 / 0.7062 | +0.0908 | +0.1317 | +0.1293 |
| 10 | 0.8168 / 0.8181 | 0.6748 / 0.6833 | 0.7126 / 0.7347 | 0.6783 / 0.6833 | +0.1420 | +0.1042 | +0.1385 |
| 11 | 0.8103 / 0.8167 | 0.6062 / 0.6278 | 0.7226 / 0.7250 | 0.6606 / 0.6660 | +0.2042 | +0.0877 | +0.1497 |
| 12 | 0.7909 / 0.7917 | 0.5420 / 0.5542 | 0.6323 / 0.6514 | 0.6473 / 0.6549 | +0.2489 | +0.1586 | +0.1436 |

### Full speaker-fold distribution

| Layer | Fold 0 F1 / accuracy | Fold 1 F1 / accuracy | Fold 2 F1 / accuracy | Fold 3 F1 / accuracy | Fold 4 F1 / accuracy | Fold 5 F1 / accuracy |
|---|---|---|---|---|---|---|
| 0 | 0.5650 / 0.5708 | 0.5005 / 0.5333 | 0.4614 / 0.4708 | 0.5566 / 0.5875 | 0.5183 / 0.5167 | 0.5690 / 0.5833 |
| 1 | 0.6228 / 0.6292 | 0.5946 / 0.6042 | 0.6178 / 0.6375 | 0.7307 / 0.7292 | 0.5382 / 0.5500 | 0.6452 / 0.6667 |
| 2 | 0.6600 / 0.6792 | 0.6257 / 0.6208 | 0.6833 / 0.7042 | 0.6957 / 0.7083 | 0.6470 / 0.6458 | 0.6356 / 0.6500 |
| 3 | 0.6822 / 0.7000 | 0.6521 / 0.6625 | 0.7566 / 0.7583 | 0.7301 / 0.7417 | 0.6656 / 0.6625 | 0.6565 / 0.6833 |
| 4 | 0.7248 / 0.7375 | 0.6711 / 0.6792 | 0.7559 / 0.7667 | 0.8278 / 0.8333 | 0.7247 / 0.7167 | 0.7025 / 0.7167 |
| 5 | 0.7633 / 0.7708 | 0.6891 / 0.6958 | 0.7608 / 0.7750 | 0.8088 / 0.8167 | 0.8001 / 0.8000 | 0.7172 / 0.7250 |
| 6 | 0.7413 / 0.7500 | 0.7555 / 0.7542 | 0.7637 / 0.7708 | 0.7841 / 0.7833 | 0.7983 / 0.7917 | 0.7567 / 0.7625 |
| 7 | 0.7274 / 0.7333 | 0.7618 / 0.7625 | 0.7566 / 0.7625 | 0.7885 / 0.7875 | 0.7484 / 0.7500 | 0.7721 / 0.7750 |
| 8 | 0.7122 / 0.7167 | 0.7050 / 0.7083 | 0.7747 / 0.7833 | 0.7602 / 0.7625 | 0.7493 / 0.7458 | 0.7027 / 0.7083 |
| 9 | 0.6582 / 0.6667 | 0.7069 / 0.7208 | 0.7292 / 0.7417 | 0.7373 / 0.7375 | 0.7117 / 0.6958 | 0.6690 / 0.6750 |
| 10 | 0.6763 / 0.6833 | 0.7182 / 0.7292 | 0.6576 / 0.6667 | 0.7124 / 0.7250 | 0.6778 / 0.6583 | 0.6275 / 0.6375 |
| 11 | 0.6503 / 0.6583 | 0.6568 / 0.6667 | 0.6763 / 0.6833 | 0.6642 / 0.6792 | 0.6595 / 0.6375 | 0.6567 / 0.6708 |
| 12 | 0.6587 / 0.6708 | 0.6184 / 0.6250 | 0.6186 / 0.6375 | 0.6680 / 0.6792 | 0.6750 / 0.6542 | 0.6451 / 0.6625 |

![Generalization and gaps](emotion_generalization.png)

## Gate

**PARTIAL / REVIEW NEEDED**

Encoding above majority and a corresponding positive shift gap are descriptive screens, not meaningful-effect or significance tests. The plan specifies no numerical gate thresholds; inspect the magnitudes and full fold distribution before human approval. M4 remains blocked.

M3 tests encoding and baseline generalization, not reliance or harmful use.

## Interpretation limits and remaining review

- No intervention, erasure, new encoder, or additional corpus was run.
- One predeclared seed; no statistical-significance claim or bootstrap confidence interval.
- Fold standard deviation/range is descriptive, not a confidence interval; training folds overlap.
- Speaker fits use 1,200 training utterances versus 720 for the reference and lexical fits. The requested gap is not a matched-training-size causal estimate.
- Neutral occurs only at normal intensity; intensity is not independent of emotion.
- Layer maxima are descriptive test-set summaries, not unbiased layer-selection estimates.
- Human scientific review and explicit M4 approval remain required.

Complete metrics: [probes.csv](probes.csv), [generalization.csv](generalization.csv). Training-only selection: [cv_results.csv](cv_results.csv). Held-out predictions, class order, probabilities and majority predictions: [predictions.parquet](predictions.parquet). Exact ID/metric recomputation: [verification.json](verification.json). Machine-readable gate and fold summaries: [summary.json](summary.json).
