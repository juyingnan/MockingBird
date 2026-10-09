# Exploratory DAS statement-subspace result

**Decision: FAIL_RESIDUAL_DECODABILITY. Stop; no reliance experiment.**

This is a new post-freeze experiment, not an update to the paper's evidence.
The paper, ledger, cache, and M3/M4/M4-V1 bundles remain unchanged.

## A. Method

Implemented feature-level, **readout-mediated DAS/DII** on frozen WavLM
Base Plus layer 5, cache `d99979b610d68bef`. Matched base/source recordings
share actor, emotion, intensity, and repetition; the sentence changes.
Both swap directions are included.

An orthonormal 768 x k basis Q is optimized using source-label cross-entropy
through a frozen A-train-only 768 -> 128 tanh -> 2 statement MLP:

```text
DII(b, s) = b + ((s - b) Q) Q-transpose
mean_ablation(x) = x - ((x - mean_Atrain) Q) Q-transpose
```

This is an actual source/base interchange objective, not an objective that
merely makes a probe incorrect. It is nevertheless an adaptation: the speech
cache has no native statement-output head, so the optimized behavior belongs
to a surrogate readout, not WavLM's original behavior or the emotion head.
See the sourced [pre-fit design](protocol.md).

## B. Selection, before C

- Reused cell-disjoint A/B/C = 720/360/360; subdivided A into 480 A-train
  and 240 A-val utterances without splitting matched cells.
- Fixed layer 5, mean ablation, learning rate 0.01, 200 DAS steps.
  Evaluated k = 1, 4, 8, 16, 32, 64, 128 with seeds 20261008 and 20261009:
  **14 candidates**, no adaptive expansion.
- Each candidate checkpoint minimizes A-val interchange cross-entropy.
  Eligible candidates require A-val interchange accuracy >= 0.90 and
  collateral limits. Selection minimizes max(linear, RBF) A-val statement
  F1, then mean F1, smaller k, and lower seed.
- Selected **k = 32, seed 20261009, checkpoint 200**. No refitting after
  selection. A-val interchange accuracy was 1.0000; independent residual
  A-val F1 was already 0.9542 linear / 0.9040 RBF.
- Every candidate and all 30 B diagnostic models were persisted and
  committed before C. See [selection](candidate_selection.csv),
  [candidate freeze](candidate_frozen.json), and [model freeze](fit_complete.json).

The arbitrary cell-mapping control reached 0.8271 training accuracy but
0.5208 A-val accuracy, versus 1.0000 for real-task A-val interchange.
This is an A-only expressivity stress test, not CausalGym's unrelated-token
selectivity metric and not a causal-faithfulness certificate.

## C. Locked statement result

All rows below are on the same locked C, with fresh B-trained diagnostics.
U95 is the individual one-sided 95% actor-cluster percentile upper bound,
using 10,000 shared draws and no refitting.

| Probe / arm | Macro-F1 | Accuracy | F1 U95 | Accuracy U95 |
|---|---:|---:|---:|---:|
| Linear, original | 0.9917 | 0.9917 | 1.0000 | 1.0000 |
| Linear, DAS mean ablation | **0.9305** | **0.9306** | **0.9592** | **0.9592** |
| RBF-SVM, original | 0.9806 | 0.9806 | 0.9973 | 0.9973 |
| RBF-SVM, DAS mean ablation | **0.8971** | **0.8972** | **0.9290** | **0.9290** |

Both point scores and all upper bounds exceed the unchanged **0.55** ceiling.
F1 reductions are 0.0612 / 0.0835, also below the required 0.10. Original
probes are competent; this is not a borderline precision failure.
[Exact scores](target_validity.csv) and [decision](decision.json) are retained.

## D. Collateral and random controls

These are independently retrained diagnostic probes, not the original
fixed M3 emotion classifier. Small differences are descriptive.

| Attribute | Original F1 | Ablated F1 | F1 change | Original accuracy | Ablated accuracy |
|---|---:|---:|---:|---:|---:|
| Emotion | 0.5955 | 0.6054 | +0.0098 | 0.6278 | 0.6389 |
| Speaker | 0.8793 | 0.8667 | -0.0127 | 0.8778 | 0.8667 |
| Gender | 0.9972 | 1.0000 | +0.0028 | 0.9972 | 1.0000 |
| Intensity | 0.7427 | 0.7397 | -0.0029 | 0.7444 | 0.7417 |

All declared collateral limits pass. This candidate did not fail by broadly
destroying the measured representation; it failed to remove the target.

Three rank-32 random controls were chosen from the fixed 32-draw pool by
closest A-train RMS, not by any outcome. They remain **RMS-unmatched**:
relative errors 0.1642, 0.1642, 0.1745 exceed the declared 0.05 tolerance.
Their statement F1/accuracy equal the original values under both probes.
They are descriptive rank-matched controls, not valid magnitude-matched
evidence of reliance. No further control search was run.
[Collateral](collateral.csv); [complete random pool](random_control_pool.csv).

## E. Interpretation

**Failed residual decodability**, despite successful surrogate-readout
interchange. Steering a frozen readout with a source component and erasing
information for independently trained classifiers are different objectives.
The result is compatible with readout-specific steering, but does not prove
that as a unique mechanism.

It does not establish an intrinsic 32-dimensional statement representation,
exclude other higher-dimensional subspaces, explain why LEACE struggled,
or establish reliance/non-reliance/harmful use. A-val selection and the
already-explored corpus limit confirmatory interpretation. Do not compare
absolute scores across M3, M4 and this follow-up as one shared protocol.

## F. Recommendation

Treat this as a **failed eraser with a useful methodological negative result**,
not a successful repair or an implementation failure. Increasing capacity to
128 dimensions did not produce a compelling removal candidate in A
development, and the fixed selected candidate failed clearly on C.
Do not escalate dimension or optimize against C.

A separately approved study could distinguish behavioral interchange
sufficiency from information-removal necessity with a different, predeclared
design. This result alone does not justify that new experiment, an explanation
of LEACE, or promotion into the current manuscript.

## G. Reproducibility

- Branch: `research2027/das-statement-subspace`.
- Parent experiment commit: `24607d4bf5bb739546b945156f36a26b5c0d1c03`.
- Code/protocol commit: `f3e4b22d36dd8f4974610b5d851059b2855f2032`.
- Pre-C freeze / evaluation-launch commit:
  `166baadb38aafc6562db73cb832097013c6372b9`.
- Experiment ID: `das-statement-wavlm-ravdess-20261008`.
- Recorded fitting/selection/B-model runtime: **290.77 s**.
  C evaluation, recomputation and plots: **4.03 s**.
  Total recorded scientific phases: **294.79 s**, excluding setup/research.
- CPU, one PyTorch/BLAS numerical thread, float64; **no GPU**, audio
  extraction, fixed-emotion-head reliance test, or post-C rescue search.
- Python 3.11.9; PyTorch 2.14.1. Exact numerical dependencies are pinned
  in [requirements](../../requirements-das.txt); all versions and hashes
  are in [provenance](provenance.json).
- 44 targeted synthetic/shared-helper tests pass. Independent recomputation
  verifies 30 metric rows, 10,800 predictions and 100,000 bootstrap scores
  with maximum error **0**; 212 immutable input hashes checked.
  [Verification](verification.json).
- Local commits persist all scientific artifacts. Publication to the
  requested fork was rejected by write permissions; a Git bundle handoff
  preserves the branch without bypassing those permissions.

Verify the saved run without rescoring C, from `research2027`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c requirements-das.txt -e ".[dev,probes,das]"
.\.venv\Scripts\python.exe -m mockingbird2027 das verify --output-dir results\das-statement-wavlm-ravdess-20261008
```

For an exact, explicitly requested replay, use a new directory; never
overwrite this run or change the frozen settings:

```powershell
.\.venv\Scripts\python.exe -m mockingbird2027 das prepare
.\.venv\Scripts\python.exe -m pytest tests\test_das.py tests\test_das_pipeline.py tests\test_validity_splits.py tests\test_validity_statistics.py tests\test_validity_pipeline.py tests\test_m4_erasure.py tests\test_smoke.py -q
.\.venv\Scripts\python.exe -m mockingbird2027 das fit --output-dir results\das-statement-wavlm-ravdess-20261008-replay
git add -- results\das-statement-wavlm-ravdess-20261008-replay
git commit -m "Freeze exact replay candidate and B models before C"
.\.venv\Scripts\python.exe -m mockingbird2027 das evaluate --output-dir results\das-statement-wavlm-ravdess-20261008-replay
.\.venv\Scripts\python.exe -m mockingbird2027 das verify --output-dir results\das-statement-wavlm-ravdess-20261008-replay
```

The staged fit recreates local frozen sklearn objects under ignored
`artifacts/das_fits/`. They may contain B support vectors and are not published;
the result bundle retains parameter summaries and immutable B indices.
The `evaluate` command rejects missing/uncommitted model freezes, changed
artifacts, or a second C opening.

Figures: [A-only selection](candidate_selection.png) and
[locked validity and collateral](locked_validity.png).
