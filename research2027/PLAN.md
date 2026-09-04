# MockingBird 2027 / NAACL 2027 Research Execution Plan

## 0. Project Charter

### Working title

**Encoded ≠ Used: Auditing Non-Target Information in Speech Representations under Distribution Shift**

Alternative conservative title:

**From Decodability to Reliance: Auditing Speech Representations for Emotion Recognition under Controlled Distribution Shifts**

### Target venue

NAACL 2027 main conference via the October 2026 ARR cycle.

### Central research question

When a pretrained speech representation encodes information about lexical content, speaker identity, gender, or other non-target attributes, does a downstream emotion classifier actually rely on that information?

More specifically:

1. Which task-relevant and non-target attributes are linearly decodable from different layers of pretrained speech encoders?
2. Does high decodability imply that an emotion classifier relies on the corresponding information?
3. Does targeted removal of non-target information change emotion performance under the distribution shift associated with that attribute?
4. Are the relationships between **encoding → reliance → generalization failure** consistent across models and datasets?

### Scientific distinction

The paper must distinguish three concepts:

**Encoded**

A probe can recover attribute \(z\) from representation \(h_l\).

**Used**

Intervening on the representation to remove \(z\) changes the behavior of an already-trained emotion classifier beyond an appropriately matched control intervention.

**Harmful under shift**

Removing \(z\), followed by training an emotion classifier on the cleaned representation, reduces the relevant OOD generalization gap.

These are NOT interchangeable.

The paper's intended headline is:

> Information being decodable from a representation is not sufficient evidence that the downstream system relies on it, and reliance itself does not always imply that the information harms generalization.

Do not change this framing unless experimental evidence forces us to.

---

# 1. Scope Control

## Core target attributes

Use:

1. **Lexical content / statement**
2. **Speaker identity**

Use as secondary analyses:

3. Gender
4. Emotion intensity

Do not describe gender as inherently "spurious."

Use the terms:

- non-target attribute
- nuisance factor when appropriate
- distribution-shift factor
- shortcut only when experimental evidence supports shortcut behavior

Speaker and gender are nested variables; speaker removal may necessarily affect gender information. Always measure such collateral effects.

---

# 2. What NOT to Do

These are explicit anti-goals.

Do NOT:

- rebuild the old MFCC CNN as the main model;
- claim SOTA emotion-recognition accuracy;
- use t-SNE as primary evidence;
- treat probe accuracy as proof of model reliance;
- fine-tune large speech encoders during the initial experiments;
- run every model × every layer × every attribute × every intervention from the beginning;
- use large model variants until the core hypothesis survives cheaper experiments;
- use overlapping slicing in the first-stage experiments;
- reuse legacy train/test code without validating its semantics;
- let an LLM inspect thousands of raw log lines;
- repeatedly recompute encoder embeddings;
- let an agent browse literature during engineering milestones;
- let an agent spawn subagents unless explicitly requested;
- modify legacy MockingBird code in place.

The old code is historical provenance and a source of dataset semantics, not the architecture for the new implementation.

---

# 3. Research Strategy: Cheap-to-Expensive Cascade

The project uses a strict cascade.

A more expensive stage runs only if the previous stage produces evidence worth pursuing.

## Tier A — Development model

**microsoft/wavlm-base-plus**

Use this model for all initial engineering and hypothesis testing.

Why:

- pretrained speech encoder;
- easy hidden-state access;
- moderate model size;
- suitable for layer-wise analysis;
- avoids wasting compute on several models before the analysis pipeline is trustworthy.

## Tier B — Replication models

After WavLM results pass the research gate:

1. **HuBERT Base**
2. **Whisper Small**

These give a useful contrast in pretraining objective and architecture without immediately moving to large checkpoints.

## Tier C — Domain-specific comparator

Only after the core pipeline works:

**emotion2vec / emotion2vec+**

This is scientifically valuable because it allows:

> Does emotion-oriented representation learning change the degree to which non-target information is encoded or relied upon?

But do not let integration difficulties with FunASR/model-specific code block the main paper.

## Tier D — Do not run initially

- WavLM Large
- HuBERT Large
- Whisper Medium/Large
- large audio-language models
- end-to-end encoder fine-tuning

These are stretch experiments, not prerequisites.

---

# 4. Dataset Strategy

## Dataset 1: RAVDESS

Purpose:

**controlled laboratory**

Metadata:

- emotion
- intensity
- statement
- repetition
- actor
- gender derived from actor ID

Use complete utterances.

Do NOT slice utterances in the initial experiments.

RAVDESS is useful specifically because controlled factors can be manipulated independently enough to study representation behavior.

## Dataset 2: CREMA-D

Purpose:

**replication on a larger independent corpus**

Use the existing MockingBird code only to understand historical label conventions.

Reimplement parsing and validation from raw metadata.

## Optional Dataset 3

IEMOCAP or another suitable corpus may be added only after the two-dataset core result exists.

It must not delay the October ARR submission.

## Optional cross-corpus test

If time permits, map RAVDESS and CREMA-D to the compatible six-emotion subset and test cross-corpus transfer.

Treat this as a compound real-world shift, not as a controlled shift.

---

# 5. New Repository Structure

Keep legacy code unchanged.

Create:

```text
MockingBird/
├── classification/             # legacy: untouched
├── clustering/                 # legacy: untouched
├── visualization/              # legacy: untouched
├── utils/                      # legacy: untouched
│
└── research2027/
    ├── README.md
    ├── PLAN.md
    ├── AGENTS.md
    ├── STATUS.md
    ├── pyproject.toml
    ├── .gitignore
    │
    ├── configs/
    │   ├── paths.example.yaml
    │   ├── datasets/
    │   │   ├── ravdess.yaml
    │   │   └── cremad.yaml
    │   ├── models/
    │   │   ├── wavlm_base_plus.yaml
    │   │   ├── hubert_base.yaml
    │   │   └── whisper_small.yaml
    │   └── experiments/
    │       ├── stage1_wavlm_ravdess.yaml
    │       ├── stage2_intervention.yaml
    │       └── stage3_replication.yaml
    │
    ├── src/
    │   └── mockingbird2027/
    │       ├── data/
    │       │   ├── manifest.py
    │       │   ├── ravdess.py
    │       │   ├── cremad.py
    │       │   └── splits.py
    │       ├── models/
    │       │   ├── registry.py
    │       │   ├── extract.py
    │       │   └── pooling.py
    │       ├── analysis/
    │       │   ├── probes.py
    │       │   ├── erasure.py
    │       │   ├── controls.py
    │       │   ├── metrics.py
    │       │   └── stats.py
    │       ├── figures/
    │       │   └── paper_figures.py
    │       └── cli.py
    │
    ├── scripts/
    │   ├── setup.ps1
    │   ├── smoke_test.ps1
    │   ├── stage1_extract.ps1
    │   ├── stage1_probe.ps1
    │   ├── stage2_intervene.ps1
    │   └── stage3_replicate.ps1
    │
    └── tests/
```

PowerShell is the primary shell.

Do not require Bash.

---

# 6. Reproducibility Architecture

## 6.1 Manifest first

Never let the model pipeline infer metadata from ad hoc file order.

Create one manifest row per utterance.

Minimum columns:

```text
utterance_id
dataset
path
emotion
speaker_id
gender
statement_id
repetition
intensity
duration_seconds
sample_rate_original
```

Validate:

- unique `utterance_id`;
- unique file path;
- no missing required labels;
- expected dataset counts;
- emotion counts;
- speaker counts;
- per-speaker emotion balance;
- statement counts;
- gender counts.

Output:

```text
artifacts/manifests/ravdess.parquet
artifacts/manifests/cremad.parquet
artifacts/reports/ravdess_manifest_report.json
```

Fail loudly if validation fails.

## 6.2 Standardized audio preprocessing

Unless the model requires otherwise:

- mono;
- 16 kHz;
- preserve complete utterance;
- no augmentation;
- no normalization beyond what the corresponding pretrained processor expects.

Store preprocessing metadata.

## 6.3 Immutable embedding cache

Encoder inference is expensive; probes are cheap.

Therefore encoder outputs must be computed exactly once per:

```text
dataset
model
model revision
preprocessing configuration
pooling configuration
```

Save every encoder layer.

Use utterance-level mean pooling over valid non-padding frames as the primary representation.

Example:

```text
artifacts/embeddings/
  ravdess/
    microsoft_wavlm-base-plus/
      <fingerprint>/
        metadata.json
        ids.npy
        layer_00.npy
        layer_01.npy
        ...
        layer_12.npy
```

Storage may use float16.

Convert to float32 for covariance / erasure calculations.

Every cache must record:

- model ID;
- model/config revision if available;
- transformers version;
- torch version;
- preprocessing parameters;
- number of utterances;
- hidden size;
- layer count;
- manifest hash.

Never silently overwrite an existing cache with a different fingerprint.

---

# 7. Experimental Split Design

There are TWO different concepts of split.

Do not mix them.

## 7.1 Attribute-audit split

Purpose:

Determine whether an attribute is encoded.

For RAVDESS, the repetition split is ideal because both sides contain:

- all speakers;
- both statements;
- all emotions;
- both genders.

For example:

```text
train = repetition 1
test  = repetition 2
```

Use this split for probes predicting:

- emotion;
- statement;
- speaker;
- gender;
- intensity.

Important:

A speaker classifier cannot be evaluated on completely unseen speaker IDs.

Therefore do NOT measure speaker decodability using a held-out-speaker split.

## 7.2 Distribution-shift splits

Purpose:

Measure emotion generalization.

### Repetition shift

Train repetition 1 → test repetition 2.

Serves as near-IID controlled reference.

### Lexical shift

Train statement 1 → test statement 2.

Also report reverse direction:

statement 2 → statement 1.

### Speaker shift

Use deterministic speaker-group folds.

All utterances from a speaker must belong entirely to either train or test.

Use multiple held-out speaker folds rather than relying on one arbitrary actor cutoff.

Persist the folds to disk so every model uses identical splits.

### Gender shift

Secondary analysis:

male → female  
female → male

Do not make this the headline experiment.

---

# 8. Stage 1 — Cheapest Scientific Test

## Goal

Determine whether the basic research hypothesis exists before expanding scope.

Dataset:

**RAVDESS**

Model:

**WavLM Base Plus**

Encoder:

**frozen**

### Step 1A — Extract all layer representations

Smoke test first on a small subset.

Checks:

- correct utterance count;
- finite embeddings;
- stable ordering;
- sensible dimensions;
- no gradient tracking;
- model in eval mode;
- padding excluded from pooling.

Then extract full RAVDESS once.

### Step 1B — Layer-wise probes

Train linear probes for every layer.

Targets:

```text
emotion
statement
speaker
gender
intensity
```

Primary classifier:

regularized logistic regression.

Do not hyperparameter-search aggressively.

Primary metric:

**macro-F1**

Also record:

- accuracy;
- majority baseline;
- chance level where meaningful.

Fit preprocessing only on training data.

For every layer produce:

```text
model
dataset
layer
attribute
train_split
test_split
macro_f1
accuracy
baseline
```

### Required first figure

A layer × attribute heatmap.

This is the modern replacement for the old t-SNE-first analysis.

### Step 1C — Baseline emotion generalization

For every layer train a linear emotion head and measure:

- repetition shift;
- statement shift;
- speaker shift;
- optional gender shift.

Define:

```text
OOD_gap = reference_performance - shifted_performance
```

Do not assume the final layer is optimal.

### Stage 1 research gate

Continue only if all of the following are true:

1. emotion is meaningfully decodable;
2. at least one non-target attribute is clearly above its baseline;
3. at least one controlled shift creates a meaningful emotion-performance gap;
4. results are not explained by a manifest/split bug.

If this fails:

STOP expansion.

Diagnose before running another model.

---

# 9. Stage 2 — Encoded vs. Used

This is the main scientific contribution.

Run on WavLM/RAVDESS first.

Do NOT initially run every factor at every layer.

## Select layers

Select a small set based on Stage 1:

1. layer with strongest emotion representation;
2. layer with strongest lexical/speaker decodability;
3. one intermediate layer;
4. final layer if not already selected.

The selection rule must be defined programmatically and recorded.

Do not hand-pick favorable layers after seeing intervention outcomes.

## Primary interventions

Implement:

1. **Mean Projection (MP)**
2. **LEACE**

Do not make INLP a core method.

## Primary factors

Run lexical-content removal first.

Then speaker removal.

Why lexical first:

- binary;
- tightly controlled;
- low-dimensional;
- easier to validate;
- cleanest causal sanity check.

---

# 10. Two Different Intervention Questions

These must be reported separately.

## 10.1 Fixed-head intervention: "Was the trained classifier using it?"

Procedure:

1. Train emotion classifier on original representation \(h\).
2. Fit eraser using training data only.
3. Apply targeted erasure to test representation.
4. Evaluate the unchanged emotion classifier.

This measures behavioral sensitivity of the existing classifier.

But erasure itself perturbs the representation.

Therefore targeted intervention alone is NOT sufficient evidence.

### Mandatory control

Construct matched random-removal controls.

The random intervention must match targeted intervention as closely as practical in:

- removed dimensionality/rank;
- representation perturbation scale.

Run multiple random draws.

Report:

```text
targeted task delta
mean random-control task delta
control confidence interval
targeted-minus-control delta
```

Define an operational reliance score:

```text
Reliance(attribute) =
    task_change(targeted erasure)
    - expected task_change(matched random erasure)
```

Sign convention must be documented clearly.

The paper should not interpret raw task degradation without the random control.

## 10.2 Retrain-after-erasure: "Would a representation without this information generalize better?"

Procedure:

1. Fit eraser on training representation only.
2. Erase train representation.
3. Erase test representation using the same learned transformation.
4. Retrain the emotion classifier on erased training representations.
5. Compare ID and OOD performance.

This does NOT measure the existing classifier's reliance.

It measures whether the target representation information is beneficial, irrelevant, or harmful to learning a robust emotion classifier.

Possible golden finding:

```text
original:
high ID performance
large speaker-shift gap

speaker-erased:
slightly lower ID performance
higher unseen-speaker performance
smaller OOD gap
```

Interpretation:

Speaker information was useful in-distribution but harmful to generalization.

---

# 11. Erasure Validation

Every targeted erasure must be checked.

After removing attribute Z:

### Required check A

Probe Z again.

Confirm the intended information was reduced.

### Required check B

Probe all OTHER attributes again.

Example after speaker removal:

```text
speaker
gender
statement
intensity
emotion
```

This forms a **collateral-damage matrix**.

This matters especially because:

```text
speaker identity -> gender
```

is structurally correlated.

Do not claim a surgical speaker intervention if gender and emotion information were simultaneously destroyed.

### Required check C

Representation distortion.

Record at least:

- dimensionality/rank removed;
- mean L2 perturbation;
- cosine similarity before/after if appropriate.

---

# 12. Stage 3 — Model Replication

Only after a clear WavLM finding.

Run the exact same cached-embedding pipeline on:

1. HuBERT Base
2. Whisper Small

Do NOT redesign experiments per model.

Use identical:

- manifests;
- split files;
- metrics;
- statistical procedures;
- layer-selection rules.

First generate probe maps.

Then run interventions only for the already-established primary factors.

If the WavLM finding does not replicate:

Do not hide it.

Ask whether reliance is representation-objective-specific.

A consistent difference between models may itself be the stronger paper.

---

# 13. Stage 4 — Dataset Replication

Run the successful core experiment on CREMA-D.

The first CREMA-D goal is NOT to reproduce every RAVDESS split.

Priority:

1. speaker information;
2. speaker-shift emotion performance;
3. speaker intervention;
4. gender as secondary analysis;
5. lexical content if the metadata supports an appropriately controlled experiment.

Do not force RAVDESS's exact factorial design onto CREMA-D.

The paper can explicitly distinguish:

```text
RAVDESS = controlled discovery
CREMA-D = independent replication
```

---

# 14. Stage 5 — High-Value Optional Experiments

Only run these if the core paper already exists.

Priority order:

### 5A. emotion2vec

High scientific value.

Question:

Does an emotion-specialized encoder encode or rely on less non-target information than general speech encoders?

### 5B. Small nonlinear probe after LEACE

Purpose:

Test whether linearly erased information remains recoverable nonlinearly.

Do this only on a few selected layers/factors.

Do not turn the project into a nonlinear concept-erasure paper.

### 5C. Cross-corpus emotion transfer

Use only compatible emotion classes.

### 5D. Randomly initialized encoder baseline

Useful methodological control if cheap enough.

### 5E. Fine-tuned encoder

Only if reviewers would otherwise be able to dismiss the paper as a frozen-representation analysis.

Start with one model and one dataset.

### Explicit stretch-only direction: Sparse autoencoders

Do not build the main paper around SAEs.

Recent work already applies SAEs across speech encoder layers and performs concept removal/steering.

An SAE comparison can be added later, but it is not required for the central contribution.

---

# 15. Statistics

Prefer evaluation rigor over more models.

## Speaker shift

Use multiple fixed group folds.

Report distribution across folds.

## Deterministic directional shifts

For:

- sentence 1 → 2;
- sentence 2 → 1;
- male → female;
- female → male;

report both directions separately.

Do not average away asymmetry before inspecting it.

## Confidence intervals

Use bootstrap confidence intervals over held-out utterances where appropriate.

Do statistical processing from saved predictions.

Never rerun encoder inference for bootstrap.

## Random interventions

Use several random matched-removal draws.

These are CPU-cheap once embeddings are cached.

---

# 16. Minimum Paper-Quality Evidence

A main-paper submission should ideally contain:

### Models

At least 3 meaningfully different pretrained speech encoders.

### Datasets

At least 2 corpora.

### Attributes

At minimum:

- lexical content;
- speaker identity.

### Required methodological controls

- layer-wise linear probes;
- controlled OOD evaluation;
- MP and/or LEACE;
- matched random-removal intervention;
- before/after nuisance probes;
- collateral-damage analysis;
- confidence intervals / group-fold variation.

The number of experiments is less important than whether the causal interpretation is defensible.

---

# 17. Expected Paper Figures

The code should generate these directly from result tables.

## Figure 1 — Method overview

```text
audio
  ↓
frozen speech encoder
  ↓
layer representations
  ├── task probe: emotion
  ├── attribute probes
  │      speaker / lexical / gender / intensity
  ↓
targeted concept removal
  ↓
fixed-head behavior + retrained-head OOD behavior
```

## Figure 2 — Layer-wise information map

Rows:

attributes

Columns:

layers

Values:

normalized probe performance.

## Figure 3 — Generalization by layer

Emotion performance across:

- repetition;
- lexical;
- speaker shifts.

## Figure 4 — Decodability vs reliance

For attribute/layer/model points:

```text
x = non-target decodability
y = targeted intervention effect beyond random control
```

This directly tests:

**Does encoded imply used?**

## Figure 5 — Intervention and OOD gap

Before vs after erasure.

## Table 1

Datasets and controlled factors.

## Table 2

Model-level representation and shift summary.

## Table 3

Targeted erasure vs matched random controls.

## Appendix

- every layer;
- full fold results;
- collateral-damage matrices;
- hyperparameters;
- additional directions.

---

# 18. Result File Contract

Every experimental command must produce structured output.

No result may live only in stdout.

Use Parquet or CSV.

Example schema:

```text
experiment_id
timestamp
git_commit
dataset
model
model_revision
layer
split_name
fold
task
attribute_removed
intervention
control_id
metric
value
```

Predictions should also be optionally persisted:

```text
utterance_id
gold_label
predicted_label
probabilities
experiment_id
```

Figures must be generated from these tables.

Do not manually transcribe numbers into plots.

---

# 19. Compute / Credit Optimization Rules

These rules are mandatory.

## Rule 1 — Sol writes infrastructure; Python runs science

Never use the coding agent to "reason through" every experiment result while it runs.

Agent tasks:

- implement pipeline;
- fix bugs;
- write tests;
- implement analysis methods;
- create deterministic scripts.

Python tasks:

- embedding extraction;
- probes;
- erasures;
- metrics;
- statistics;
- plots.

## Rule 2 — Never pay twice for encoder inference

Cache all hidden representations.

Changing:

- probe;
- regularization;
- split;
- intervention;
- metrics;
- plots;

must NOT require another encoder forward pass.

## Rule 3 — Smoke test everything

Before full inference:

```text
16 utterances
1 model
2 layers if possible
1 tiny probe
```

Only after passing invariants should full extraction run.

## Rule 4 — Freeze foundation models

Initial paper uses frozen encoders.

This makes the majority of the experiment CPU analysis after one GPU extraction pass.

## Rule 5 — One development model

Do not extract HuBERT or Whisper until WavLM passes Stage 1.

## Rule 6 — No combinatorial sweep

First intervention:

```text
RAVDESS
WavLM
statement
selected layers only
```

Then expand.

## Rule 7 — Agent does not watch long jobs

Agent launches a deterministic command.

The experiment writes:

```text
results
logs
exit code / completion marker
```

The next agent invocation should inspect the compact result summary, not replay the job.

## Rule 8 — Persist context in Git

The agent should never require a long conversational recap.

Maintain:

```text
PLAN.md
AGENTS.md
STATUS.md
```

`STATUS.md` must contain:

```text
Current milestone
Completed work
Files changed
Commands that passed
Known failures
Important scientific observations
Next exact action
```

## Rule 9 — No automatic web browsing

Engineering agent must not search papers, GitHub issues, or tutorials unless:

1. local API/docs are insufficient;
2. an actual incompatibility blocks progress.

If web access becomes necessary, explain the exact missing fact before using it.

## Rule 10 — No subagents

Do not spawn subagents unless the user explicitly requests them.

## Rule 11 — Compact responses

Agent final messages should report only:

```text
what changed
what passed
what failed
next command
```

Detailed context belongs in repo files.

---

# 20. `AGENTS.md` Content

Create this file verbatim or very close to it:

```text
# MockingBird 2027 Agent Rules

This repository contains a NAACL 2027 research project.

Before modifying code:
1. Read PLAN.md.
2. Read STATUS.md.
3. Work only on the currently requested milestone.

Do not broaden the research scope.

Do not modify legacy MockingBird experiment files unless specifically requested.
All new work belongs under research2027/.

Optimize for reproducibility and minimal repeated compute.

Requirements:
- PowerShell-first workflow.
- Frozen pretrained encoders unless explicitly authorized otherwise.
- Cache hidden-state embeddings.
- Never recompute valid cached embeddings unnecessarily.
- Fit all preprocessing and erasure transforms on training data only.
- Prevent speaker/data leakage.
- Save results as structured files, not only stdout.
- Write synthetic/unit tests for split and intervention logic.
- Smoke-test before full dataset execution.
- Do not use web search unless an actual blocker requires it.
- Do not spawn subagents.
- Do not run expensive model sweeps without an explicit milestone.
- Do not silently change scientific definitions or evaluation metrics.

At the end of each milestone:
1. run targeted tests;
2. update STATUS.md;
3. give a concise summary;
4. stop.

If scientific correctness conflicts with convenience, choose correctness.
If unsure whether an operation could cause leakage, stop and flag it.
```

---

# 21. Agent Milestones

Do NOT give the agent "implement the entire paper."

Use one milestone per invocation.

---

## M0 — Scaffold

### Prompt

```text
Execute only milestone M0 from research2027/PLAN.md.

Create the new research2027 project structure while leaving all legacy MockingBird files untouched.

Implement:
- package scaffold
- pyproject.toml
- .gitignore
- PowerShell environment setup
- PLAN.md / AGENTS.md / STATUS.md
- minimal CLI skeleton
- synthetic smoke test

Do not implement model extraction or experiments yet.

Avoid web browsing and subagents.

Run only lightweight tests.

Update STATUS.md and stop when M0 is complete.
```

### Acceptance criteria

- environment installs;
- package imports;
- CLI help works;
- tests pass;
- legacy code unchanged.

---

## M1 — Dataset manifests

### Prompt

```text
Read PLAN.md, AGENTS.md, and STATUS.md.

Execute only milestone M1: dataset manifests and validated split metadata.

Implement RAVDESS first.
Implement CREMA-D parsing only if doing so does not materially broaden the milestone.

Do not reuse legacy file ordering as a split mechanism.
Derive metadata explicitly from filenames / official metadata.

Add validation tests and generate a compact manifest report.

Do not implement neural models.

Update STATUS.md and stop.
```

### Human/scientific checkpoint

Inspect counts and label distributions before proceeding.

---

## M2 — WavLM extraction

### Prompt

```text
Execute only milestone M2.

Implement frozen microsoft/wavlm-base-plus hidden-state extraction for the validated RAVDESS manifest.

Requirements:
- processor-correct 16 kHz preprocessing
- attention-mask-aware temporal pooling
- all encoder layers
- eval mode
- no gradients
- deterministic utterance ordering
- immutable fingerprinted cache
- metadata.json
- NaN/shape/order validation

First run a tiny smoke test.
Only if it passes, run full RAVDESS extraction.

Do not implement HuBERT, Whisper, interventions, or paper figures.

Update STATUS.md and stop.
```

---

## M3 — Probe map and OOD baseline

### Prompt

```text
Execute only milestone M3.

Use the cached WavLM/RAVDESS representations. Do not rerun encoder inference.

Implement:
1. attribute-audit split;
2. layer-wise regularized linear probes for emotion, statement, speaker, gender, intensity;
3. emotion generalization evaluation under repetition, statement, and speaker shifts;
4. structured result files;
5. layer-wise heatmap and generalization plot.

Use macro-F1 as the primary metric.

Add leakage and split-invariant tests.

Do not implement concept erasure yet.

At completion, write a compact Stage 1 scientific summary into STATUS.md:
- best emotion layers
- strongest non-target layers
- OOD gaps
- whether the Stage 1 research gate passes

Stop.
```

### This is the first major GO / NO-GO point.

Do not spend credits expanding models until these results have been reviewed.

---

## M4 — Targeted interventions

### Prompt

```text
Execute only milestone M4.

Use cached WavLM/RAVDESS representations.

Implement MP and LEACE interventions for statement first, then speaker only if statement intervention passes validation.

Use the predefined layer-selection rule.

For each intervention:
- fit on training data only;
- verify target attribute decodability after removal;
- measure collateral effect on other attributes;
- evaluate fixed-head emotion behavior;
- evaluate retrain-after-erasure emotion behavior;
- compare against matched random-removal controls;
- save representation-distortion statistics.

Do not add new models or datasets.

Add unit tests using synthetic representations where the target concept is known.

Update STATUS.md with the resulting encoded-vs-used evidence and stop.
```

---

## M5 — Model replication

### Prompt

```text
Execute only milestone M5.

Replicate the validated pipeline on HuBERT Base and Whisper Small.

Reuse exactly the same:
- manifest
- split files
- metrics
- layer selection policy
- intervention definitions
- statistical evaluation

Do not alter WavLM methodology to make model results agree.

Cache every model once.

Update structured results and STATUS.md.

Stop.
```

---

## M6 — CREMA-D replication

### Prompt

```text
Execute only milestone M6.

Apply the already validated method to CREMA-D.

Focus on:
- representation audit;
- speaker decodability;
- speaker-shift emotion performance;
- successful targeted intervention from RAVDESS;
- secondary gender analysis if appropriate.

Do not force unsupported RAVDESS factors onto CREMA-D.

Generate dataset-replication tables and update STATUS.md.

Stop.
```

---

## M7 — Paper artifact generation

### Prompt

```text
Execute only milestone M7.

Do not run new model inference.

Using saved structured results:
- compute final statistics;
- generate paper-quality figures;
- generate compact LaTeX-ready tables;
- export a machine-readable summary of every reported number;
- identify missing evidence required for each intended paper claim.

Do not invent or interpolate missing results.

For every proposed paper claim, classify it as:
SUPPORTED
PARTIALLY SUPPORTED
UNSUPPORTED

Update STATUS.md and stop.
```

---

# 22. Calendar / Scope Freeze

Target:

## First phase

Get one complete result:

```text
RAVDESS
+
WavLM
+
layer-wise audit
+
controlled OOD gap
+
one validated intervention
```

This is more important than having four encoders.

## Second phase

Replicate across models.

## Third phase

Replicate across CREMA-D.

## Final phase

Paper writing and only reviewer-critical missing experiments.

Near the submission deadline:

**freeze feature development.**

Do not respond to anxiety by adding another model.

---

# 23. Scientific Decision Tree

After Stage 1:

### Case A

Non-target information is decodable and OOD performance drops.

→ Run intervention.

### Case B

Non-target information is decodable but no OOD gap exists.

→ Interesting for encoded-vs-generalization distinction, but not enough for central claim.

Test another controlled factor before adding models.

### Case C

OOD gap exists but nuisance information is weakly decodable.

→ Investigate pooling/layers/split correctness before expanding.

### Case D

Neither exists.

→ Stop. Do not burn compute reproducing a null result across four encoders.

---

After intervention:

### Case E

Strong decodability + targeted erasure ≈ random-control effect.

Headline:

**Encoded ≠ Used.**

Potentially excellent result.

### Case F

Erasure hurts ID and OOD similarly.

Information is actually used but may be task-relevant rather than spurious.

Do not call it a shortcut.

### Case G

Erasure slightly hurts ID but improves OOD.

Headline:

**Non-target information can provide in-distribution utility while harming generalization.**

Potential strongest result.

### Case H

Erasure destroys several unrelated attributes.

Intervention is not sufficiently surgical.

Do not draw causal conclusions.

Improve intervention validation before continuing.

### Case I

Different encoders behave differently.

Do NOT call it failed replication automatically.

Ask whether pretraining objective predicts nuisance reliance.

This may become a stronger paper.

---

# 24. Resume / Job-Search Deliverable

The engineering should ultimately support a bullet approximately like:

> Developed a reproducible representation-auditing framework for pretrained speech models, combining layer-wise probing, controlled distribution shifts, causal representation interventions, and multi-dataset robustness evaluation to distinguish encoded from behaviorally relied-upon information.

The project should visibly demonstrate:

- PyTorch;
- pretrained foundation models;
- Hugging Face;
- representation learning;
- probing;
- causal/interventional evaluation;
- robustness/OOD evaluation;
- scalable cached experiment infrastructure;
- statistical analysis;
- research reproducibility.

This employment value is a project constraint, not merely a paper-writing afterthought.

---

# 25. Final Principle

The expensive resource is not GPU time.

The expensive resource is uncontrolled research iteration.

Therefore:

> **Precompute once. Analyze many times. Gate expansion. Persist every decision. Use the coding agent only to build or repair deterministic machinery.**

The goal is not to maximize the number of experiments.

The goal is to produce the smallest defensible body of evidence that can answer:

> **What is encoded, what is actually used, and what causes failure under distribution shift?**