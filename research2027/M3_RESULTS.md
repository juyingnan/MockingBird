# M3: WavLM / RAVDESS

**Scientific run pending.** This revision records the protocol before fitting
the real cached representations. M4 remains blocked on human review.

## Predeclared protocol

- Use only cache `d99979b610d68bef`: 1,440 utterances, 13 hidden states,
  768 dimensions, float16 storage converted to float64 for the linear fits.
- Reuse the original manifest and persisted M1 splits. Require their validated
  identity, report consistency, ID ordering, complete coverage and no leakage.
  Do not relocate paths inside the manifest, regenerate splits or read audio.
- M3A: emotion, speaker, statement, gender and intensity probes on
  repetition 1 to repetition 2, for every layer.
- M3B: the identical repetition-reference emotion fit, both lexical
  directions and all six persisted speaker folds, for every layer.
- StandardScaler plus L2 logistic regression with the LBFGS solver.
  Use the same `C = [0.01, 0.1, 1]` grid throughout; seed `20260930`,
  maximum 2,000 iterations and tolerance `1e-4`.
- Select C by three-fold training-only macro-F1, with speaker-grouped inner
  folds for speaker-shift runs and stratified inner folds otherwise.
  Fit the scaler separately within each inner fold; refit on outer training
  data only. Exact ties prefer smaller C. Test outcomes never select C.
- Report macro-F1 over the full label space, accuracy, training-majority
  macro-F1/accuracy and uniform-chance accuracy. Chance accuracy is not
  presented as expected macro-F1.
- Save every held-out prediction, probability vector and class order, along
  with all C candidates and inner-fold scores. Recompute all published scores
  from saved predictions and check their IDs against persisted test partitions.
- Define gaps as repetition-reference performance minus shifted performance.
  Keep both lexical directions and all speaker folds; do not hide negative gaps.

The handoff did not specify a CV selection rule. Three-fold training-only
selection was chosen before the run; no test-guided protocol changes are allowed.
The paper plan also provides no numerical definition of a "meaningful" effect.
The automated gate therefore reports descriptive evidence for human review,
not an invented significance/effect-size threshold or permission to start M4.

## Commands

Run from `research2027` using the project environment:

```powershell
.\scripts\setup.ps1
.\scripts\smoke_test.ps1
.\scripts\stage1_probe.ps1 -ValidateOnly
.\scripts\stage1_probe.ps1 -OutputDir "results\m3-wavlm-ravdess-20260930"
```

The run requires committed code/configuration and a new output directory.
The [configuration](configs/experiments/m3_wavlm_ravdess.json) pins the cache
fingerprint and SHA-256 of the transferred split artifact. Raw audio, cached
embeddings and machine-local paths must not enter Git.

## Interpretation limits

M3 measures encoding and baseline emotion generalization, not reliance or
harmful use. Speaker-shift training uses 1,200 utterances versus 720 in the
reference/lexical splits; the requested gaps are not matched-size causal
estimates. Neutral occurs only at normal intensity. Fold variation is not a
confidence interval, and descriptive best layers are not unbiased estimates
of a layer-selection procedure.
