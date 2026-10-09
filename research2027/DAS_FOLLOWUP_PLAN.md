# Exploratory statement-subspace discovery with DAS

**Authorization:** 2026-10-08; new post-freeze exploration only.
**Parent:** upstream `24607d4bf5bb739546b945156f36a26b5c0d1c03`.
**Branch:** `research2027/das-statement-subspace` on `chenyueg/MockingBird`.
**Run:** `das-statement-wavlm-ravdess-20261008`.
The paper, ledger, cache and M3/M4/M4-V1 bundles must remain unchanged.
No result below is assumed in advance.

## 1. What the references actually establish

- [CausalGym](https://arxiv.org/abs/2402.12560), sections 4.1-4.2:
  distributed interchange intervention (DII) replaces the base activation's
  component in a subspace with the corresponding component of a source
  activation. DAS learns that subspace by minimizing **cross-entropy on the
  counterfactual/source output**, with the underlying model fixed.
  Log odds-ratio is an evaluation statistic, not its training loss.
- [Original DAS](https://arxiv.org/abs/2303.02536), definitions 3-4 and
  section 3.5: learn an orthogonal change of basis; interchange blocks in
  the rotated space; rotate back. Several dimensions per variable are
  allowed. Both high-level and low-level models stay fixed during alignment.
  This extends the interchange/causal-abstraction framework of
  [Geiger et al. (2021)](https://arxiv.org/abs/2106.02997).
- CausalGym benchmarks **one-dimensional** DII for binary linguistic tasks.
  Its [driver](https://github.com/aryamanarora/causalgym/blob/0f3129ff3b6c5c8264892f30a25be25150ae9179/das.py)
  configures pyvene's low-rank intervention with rank 1. This is a
  configuration, not an intrinsic restriction of DAS. Its
  [loss](https://github.com/aryamanarora/causalgym/blob/0f3129ff3b6c5c8264892f30a25be25150ae9179/eval.py)
  is cross-entropy; do not substitute ordinary probe fitting or adversarial
  probe fooling and call that DAS.
- [Boundless DAS](https://arxiv.org/abs/2305.08809), section 3.2, learns
  subspace **boundary indices** using temperature-annealed soft masks,
  then discretizes them. It is not simply a multi-layer DAS configuration.
  It is relevant but adds boundary/temperature/discretization choices.
  This small first study uses transparent fixed-k candidates instead.
- CausalGym section 5.2 compares real-task effects with effects on arbitrary
  output-token mappings; section 5.3 warns that expressive DAS can manipulate
  outputs without identifying a faithful mechanism. High interchange
  accuracy is not by itself evidence of information removal or selectivity.

Implementation review: CausalGym commit
`0f3129ff3b6c5c8264892f30a25be25150ae9179` uses pyvene; its repository has
no declared license, so no source is vendored. Apache-2.0
[pyvene's implementation](https://github.com/stanfordnlp/pyvene/blob/9e333904dcf9e597ca76170010d17f4d4580de8d/pyvene/models/interventions.py)
supports `LowRankRotatedSpaceIntervention` and Boundless masks.
Rather than install its transformer/tokenizer/dataset/hook stack for an
already pooled feature matrix, reuse **PyTorch's tested orthogonal
parametrization**, as those implementations do, with a small feature-level
adapter implementing the published DII equation. Test equivalence to a
full rotated-space swap. Record PyTorch and adapter versions.

## 2. Exact mapping and its limitation

Use frozen WavLM Base Plus **layer 5**, fixed from the existing task anchor;
cache `d99979b610d68bef`. No new inference, layer search, or PCA.
For standardized row vectors b and s and Q of shape 768 x k with
Q-transpose Q = I:

```text
DII_Q(b, s) = b + ((s - b) Q) Q-transpose
loss(Q) = mean cross_entropy(frozen_statement_readout(DII_Q(b, s)), z_s)
mean_ablation_Q(x) = x - ((x - mean_Atrain) Q) Q-transpose
```

Optimize the alignment Q, **not** the readout or emotion classifier.
The output target is the source sentence label, not the opposite of the
probe's prediction. Use both directions of each matched pair.

WavLM's pooled cache has no native sentence-classification output. Therefore
fit a separate nonlinear statement readout on **A-train only**, then freeze
it. The high-level model simply sets the statement variable to the source
statement while retaining the matched nuisance context; its output is z_s.
This is **readout-mediated DII/DAS**, not native language-model continuation
as in CausalGym, and not evidence about the existing emotion head.

A single affine binary readout depends on one weight contrast, so discovering
a high-dimensional subspace from that objective would be underdetermined
in an especially obvious way. Use a fixed 768 -> 128 tanh -> 2 MLP instead.
Even this readout can use only a subset of the encoded information. A high-k
alignment is not an estimate of the true intrinsic statement dimension.
Independent probes after ablation are the primary scientific test.

## 3. Frozen data roles and pairing

Reuse the exact committed M4-V1 A/B/C IDs and B-only grouped CV:
720/360/360 utterances; entire `(actor, emotion, intensity)` cells stay
together, including both statements and repetitions. These roles share
actors, and the corpus/C have already been used in M4-V1. This is a new
locked evaluation **for this algorithm**, not a fresh independent dataset.
No C features, predictions or metrics enter any fitting/selection decision.

Split A into **A-train 480 / A-val 240**, metadata-only. For each of the 15
emotion/intensity types, select four A cells (two actors per gender) for
A-val, preferring actors assigned fewer validation cells with seeded
tie-breaking. Leave eight cells per type in A-train. Require all actor and
attribute labels in both parts; fail rather than redraw if validation fails.
Persist every utterance's role/cell and its hash before scientific fitting.

Within each role, pair exactly `(actor, emotion, intensity, repetition)`,
changing statement 1 <-> 2. This yields 480 directed A-train swaps and
240 directed A-val swaps. These are matched recordings, not literal audio
counterfactuals: residual acoustic variation is not magically controlled.

## 4. Bounded fitting and A-only selection

All settings are in [the machine config](configs/experiments/das_statement.json).
Fit StandardScaler on A-train only, then freeze it.

1. Fit the statement MLP on A-train for 200 full-batch Adam steps,
   learning rate 0.001, weight decay 0.0001, seed 20261008. Every 10 steps
   evaluate ordinary A-val cross-entropy; retain its minimum (earlier step
   on ties). Require A-val accuracy >= 0.90 before DAS; otherwise stop as
   an inadequate behavioral readout without opening C.
2. Optimize orthonormal Q for k in **[1, 4, 8, 16, 32, 64, 128]** and
   seeds **[20261008, 20261009]**: 14 candidates, not an adaptive expansion.
   Full-batch Adam, learning rate 0.01, 200 steps, no weight decay.
   PyTorch Householder orthogonal parametrization, no dynamic trivialization,
   float64 CPU. Retain the checkpoint with minimum A-val interchange
   cross-entropy among steps 0, 10, ..., 200; earlier step wins ties.
3. For each retained candidate, fit *fresh* diagnostics on A-train after
   mean ablation, choosing probe C by grouped A-train CV, and score A-val.
   Never use the behavioral MLP itself as the independent removal probe.
   Original-feature diagnostics supply the paired collateral baseline.
4. Deterministic selection tiers: prefer candidates with A-val interchange
   accuracy >= 0.90 **and** collateral limits satisfied. Within the eligible
   set minimize the **maximum** linear/RBF statement macro-F1, then their
   mean, then smaller k, then lower seed. No A-val 0.55 crossing is required.
   If no candidate is eligible, retain a clearly labeled diagnostic
   candidate by the same recovery ranking with eligibility flags retained;
   do not report it as a successful discovery. Report every candidate.
5. Do not refit the chosen transform on A, B, or C. Persist its exact
   scaler, mean, Q, readout hash, k, seed, checkpoint and selection rule.

The seven geometrically spaced k values cover rank one through 100+ dimensions
without exhaustive search. The two seeds are selection candidates, not an
uncertainty estimate over optimizations. A-val is reused for development;
neither its best score nor its control difference is confirmatory evidence.

## 5. Controls and ablation policy

**Mean replacement is the only C-facing DAS ablation.** In the A-train-centered
standardized space, zero replacement and training-mean replacement coincide
up to floating-point error. Do not manufacture a redundant zero-ablation
comparison or choose an ablation after C.

Before C, record A-train/A-val interchange accuracy (IIA), source-label
cross-entropy, and no-op/full-source-swap baselines. Full-source swap is a
readout-capability check, not selective removal.

At the selected k, train one **arbitrary mapping control** with the same
budget: randomly flip the intended source labels in exactly half of the
complete cells independently in A-train and A-val; preserve both swap
directions/repetitions. This control probes memorization/expressivity while
using the same frozen readout. Report its training and A-val IIA alongside
real-task IIA. It is not CausalGym's dog/give token-control task: a binary
statement head cannot supply those unrelated vocabulary outputs. Do not
call the difference a certified causal selectivity score.

For ablation, draw 32 Gaussian orthonormal rank-k bases using fixed seeds.
Choose the three closest to the learned transform's **A-train RMS
perturbation**, without outcome/label scoring. Do not rescale projectors.
Record all pool RMS values, selection and exact mismatch. Call a control
RMS-matched only if relative error <= 0.05; otherwise explicitly call it
rank-only/unmatched. No repeated rescue draws or elaborate M4 matching
pipeline. These controls are descriptive, not fixed-emotion-head experiments.

## 6. Freeze B models before one locked C evaluation

Final arms: original, DAS mean ablation, and three prespecified random
subspace mean ablations. For each, B alone fits the existing five linear
diagnostics and one statement RBF-SVM: **30 models**.
Use M4-V1's grids, scaling inside CV, frozen grouped folds, convergence
errors, and per-arm selection. No Q/dimension/ablation choices use B.

Persist `candidate_frozen.json` before B fitting, and `fit_complete.json`
with all 30 model hashes before C. Commit the frozen fit bundle before
running the separate `evaluate` command. A persistent exclusive C-open
marker forbids repeat evaluations/reselection; an interrupted/failed
evaluation must remain explicit. Verification uses saved predictions,
not another C scoring pass.

The unchanged primary validity screen compares original with DAS mean:
both pre-ablation statement probes F1 >= 0.60; each F1 reduction >= 0.10;
both final F1 **and** accuracy <= 0.55; each one-sided 95% upper bound
<= 0.55. Emotion F1 loss <= 0.10; at most one of speaker/gender/intensity
loses > 0.20. Report all four non-target attributes before/after and
both metrics, not only pass/fail.

Use 10,000 shared actor-cluster bootstrap draws on C, exactly M4-V1's
seed sequence `[20260930, 4201]` and percentile-0.95 linear quantile rule;
no refitting. These are individual bounds conditional on fitted models
and the observed actors, not simultaneous all-decoder guarantees.

## 7. Engineering, artifacts and stop

Reuse existing manifest/role validation, grouped probe fitting, affine
removal, metric, parameter-hash and bootstrap helpers. Add tests for
cell/pair leakage, deterministic candidate selection, orthogonality,
DII/full-rotation equivalence, mean ablation, frozen-readout parameters,
C protection, and independent recomputation/tamper rejection.

Persist config, exact split/pair IDs, all candidate/training records,
subspace/control diagnostics, frozen transforms/model parameter summaries,
predictions, target/collateral tables, bootstrap draws/scores, decision,
input/output hashes, code/parent commits, dependency versions and measured
phase runtimes. Locally cached fitted sklearn objects may contain B support
vectors: retain them under ignored `artifacts/`, not the published result
bundle. Two figures are generated from persisted tables after evaluation.

On failure, preserve residual recovery and stop: no C-facing rescue search.
On pass, report **validity under this operational probe screen only** and
stop before reliance. Neither outcome identifies why LEACE failed, proves
zero mutual information, or establishes harmful generalization.

Planned commands from `research2027`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c requirements-das.txt -e ".[dev,probes,das]"
.\.venv\Scripts\python.exe -m mockingbird2027 das prepare
.\.venv\Scripts\python.exe -m pytest tests\test_das.py tests\test_das_pipeline.py -q
.\.venv\Scripts\python.exe -m mockingbird2027 das fit --output-dir results\das-statement-wavlm-ravdess-20261008
# Commit the frozen fit bundle before opening C.
.\.venv\Scripts\python.exe -m mockingbird2027 das evaluate --output-dir results\das-statement-wavlm-ravdess-20261008
.\.venv\Scripts\python.exe -m mockingbird2027 das verify --output-dir results\das-statement-wavlm-ravdess-20261008
```
