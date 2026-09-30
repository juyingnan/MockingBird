### 1. Statement identity

- Encoded? **Yes**, from the unchanged M3 audit.
- Successfully removed? 0/4 tested conditions passed target-removal screening; 0 passed the complete validity gate.
- Used by fixed emotion head? **Not established**; no validated condition supports this claim.
- Harmful under lexical shift? **Unidentifiable in this protocol**: each lexical training split has only one statement label. No held-out utterance was used to fit an eraser.
- Important collateral effects? See the paired five-attribute diagnostics below; diagnostic probes are separate from the task heads.

| Layer | Method | Split | Target F1 before | Target F1 after | Emotion probe F1 loss | Valid? |
|---|---|---|---|---|---|---|
| 5 | mp | repetition_shift_1_to_2 / 0 | 0.9986 | 0.6124 | +0.0056 | invalid |
| 5 | leace | repetition_shift_1_to_2 / 0 | 0.9986 | 0.6069 | +0.0048 | invalid |
| 12 | mp | repetition_shift_1_to_2 / 0 | 1.0000 | 0.5833 | -0.0047 | invalid |
| 12 | leace | repetition_shift_1_to_2 / 0 | 1.0000 | 0.5806 | -0.0072 | invalid |

### 2. Speaker identity

- Encoded? **Yes**, from the unchanged M3 audit.
- Successfully removed? **Not tested**: blocked by the statement-validation gate.
- Used by fixed emotion head? **Not established**; no validated condition supports this claim.
- Harmful under speaker shift? 0 reference/fold conditions completed retraining; blocked conditions are not null effects.
- Important collateral effects? Not measured; no speaker intervention was run.

### 3. Headline result

Statement information is encoded, but no statement intervention passed the predeclared removal/collateral/control validity gate. Fixed-head changes are diagnostics, not evidence of validated reliance. Harmful-under-shift inference is unavailable; speaker expansion and M4B were gated.

### 4. Null / contradictory findings

- Every planned condition is retained in [conditions.csv](conditions.csv); blocked conditions are not encoded as zeros or evidence of no effect.
- Removal of training cross-covariance does not guarantee removal of held-out linear decodability. Fresh diagnostic probes assess this separately.
- The eight lexical OOD target conditions cannot identify a supervised contrast from their original training data. No alternate split or leaked eraser was substituted.
- Random controls are conditioned on rank and training RMS, not per-utterance displacement or all higher moments. Their empirical quantiles are not dataset confidence intervals.
- Validity-gated results are exploratory; one seed, selected layers and a selected valid subset cannot establish universal non-reliance or causal harmfulness.

### 5. Replication recommendation

**Intervention debugging.** Resolve failed intervention validity and the train-only lexical identifiability constraint in a newly approved protocol before any encoder/dataset replication. Do not adjust the frozen primary results.

No new encoder or CREMA-D run was started; the paper repository and M3 artifacts are unchanged.

Artifacts: [provenance](provenance.json), [frozen protocol](protocol.md), [selected layers](selected_layers.json), [target validation](targeted_validation.csv), [collateral matrix](collateral_damage.csv), [random controls](random_controls.csv), [fixed-head reliance](fixed_head_reliance.csv), [retrained heads](retrained_heads.csv), [all metrics](metrics.csv), [predictions](predictions/), [verification](verification.json), [checksums](output_sha256.json).
