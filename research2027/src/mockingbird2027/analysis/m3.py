"""The bounded M3 WavLM/RAVDESS audit, operating only on existing caches."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq
from sklearn.dummy import DummyClassifier

from ..data.manifest import (
    ManifestRow,
    ManifestValidationError,
    manifest_report,
    read_parquet_manifest,
    write_json,
    write_parquet,
)
from ..data.ravdess import validate_ravdess_manifest
from ..data.splits import SplitAssignment, split_report, validate_split_assignments
from ..models.extract import CacheValidationError, manifest_hash, validate_cache
from .probes import ProbeConfig, classification_metrics, fit_probe


EXPECTED_FINGERPRINT = "d99979b610d68bef"
AUDIT = "attribute_audit_repetition_1_to_2"
REFERENCE = "repetition_shift_1_to_2"
LEXICAL = ("lexical_shift_1_to_2", "lexical_shift_2_to_1")
ATTRIBUTES = {
    "emotion": "emotion",
    "speaker": "speaker_id",
    "statement": "statement_id",
    "gender": "gender",
    "intensity": "intensity",
}


@dataclass
class M3Inputs:
    rows: list[ManifestRow]
    partitions: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]
    cache: Path
    metadata: dict[str, Any]
    hashes: dict[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def persisted_partitions(
    rows: Sequence[ManifestRow],
    assignments: Sequence[SplitAssignment],
    *,
    speaker_folds: int = 6,
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    """Index persisted assignments by ID, never by Parquet row position."""
    validate_split_assignments(rows, assignments)
    expected = {
        (name, 0)
        for name in (
            AUDIT, REFERENCE, *LEXICAL,
            "gender_shift_male_to_female", "gender_shift_female_to_male",
        )
    } | {("speaker_shift", fold) for fold in range(speaker_folds)}
    if {(item.split_name, item.fold) for item in assignments} != expected:
        raise ManifestValidationError("missing or unexpected persisted M1 split groups")
    test_counts = Counter(
        item.utterance_id
        for item in assignments
        if item.split_name == "speaker_shift" and item.role == "test"
    )
    if test_counts != Counter({row.utterance_id: 1 for row in rows}):
        raise ManifestValidationError("each utterance must be held out in exactly one speaker fold")
    index = {row.utterance_id: position for position, row in enumerate(rows)}
    partitions: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for name, fold in sorted(expected):
        role_indices = {
            role: np.array(
                sorted(
                    index[item.utterance_id]
                    for item in assignments
                    if (item.split_name, item.fold, item.role) == (name, fold, role)
                ),
                dtype=np.int64,
            )
            for role in ("train", "test")
        }
        partitions[(name, fold)] = (role_indices["train"], role_indices["test"])
    for audit, reference in zip(partitions[(AUDIT, 0)], partitions[(REFERENCE, 0)]):
        if not np.array_equal(audit, reference):
            raise ManifestValidationError("audit and repetition reference assignments differ")
    return partitions


def load_m3_inputs(config: dict[str, Any]) -> M3Inputs:
    if config["expected_fingerprint"] != EXPECTED_FINGERPRINT:
        raise CacheValidationError("M3 requires the explicitly authorized cache fingerprint")
    rows = sorted(
        read_parquet_manifest(Path(config["manifest"])), key=lambda row: row.utterance_id
    )
    validate_ravdess_manifest(rows)
    splits_path = Path(config["splits"])
    if sha256_file(splits_path) != config["expected_splits_sha256"]:
        raise ManifestValidationError("persisted split artifact differs from the validated M1 bundle")
    assignments = [SplitAssignment(**record) for record in pq.read_table(splits_path).to_pylist()]
    partitions = persisted_partitions(rows, assignments)
    report = manifest_report(rows)
    report["splits"] = split_report(assignments)
    if report != json.loads(Path(config["manifest_report"]).read_text(encoding="utf-8")):
        raise ManifestValidationError("saved manifest report does not match manifests and splits")

    cache = Path(config["cache"])
    result = validate_cache(
        cache, [row.utterance_id for row in rows], expected_fingerprint=EXPECTED_FINGERPRINT
    )
    if (result.utterance_count, result.layer_count, result.hidden_size) != (1440, 13, 768):
        raise CacheValidationError("M3 requires the complete 1440 x 13 x 768 WavLM cache")
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    if metadata["manifest_hash"] != manifest_hash(rows) or metadata["extraction_mode"] != "full":
        raise CacheValidationError("cache does not identify this complete manifest")
    paths = [
        Path(config[name]) for name in ("manifest", "splits", "manifest_report")
    ] + [cache / "metadata.json", cache / "ids.npy"] + [
        cache / f"layer_{layer:02d}.npy" for layer in range(result.layer_count)
    ]
    hashes = {str(path): sha256_file(path) for path in paths}
    return M3Inputs(rows, partitions, cache, metadata, hashes)


def git_provenance() -> dict[str, str]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], check=True, capture_output=True, text=True
        ).stdout.strip()

    root = git("rev-parse", "--show-toplevel")
    dirty = git(
        "-C", root, "status", "--porcelain", "--untracked-files=all",
        "--", "research2027", ":!research2027/results",
    )
    if dirty:
        raise RuntimeError("commit research2027 implementation/configuration before a scientific run")
    return {"git_commit": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current")}


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        raise ValueError(f"cannot write an empty result table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def evaluate_task(
    inputs: M3Inputs,
    features: np.ndarray,
    layer: int,
    split_name: str,
    fold: int,
    attribute: str,
    probe_config: ProbeConfig,
    context: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    train, test = inputs.partitions[(split_name, fold)]
    labels = np.array([str(getattr(row, ATTRIBUTES[attribute])) for row in inputs.rows])
    classes = np.unique(labels[train])
    if not np.array_equal(classes, np.unique(labels[test])):
        raise ManifestValidationError("outer train/test label spaces must agree for M3")
    groups = (
        np.array([inputs.rows[index].speaker_id for index in train])
        if split_name == "speaker_shift" else None
    )
    search = fit_probe(features[train], labels[train], probe_config, groups=groups)
    predicted = search.predict(features[test])
    probabilities = search.predict_proba(features[test])
    baseline = DummyClassifier(strategy="most_frequent").fit(features[train], labels[train])
    baseline_predictions = baseline.predict(features[test])
    scores = classification_metrics(labels[test], predicted, classes)
    baseline_scores = classification_metrics(labels[test], baseline_predictions, classes)
    fit_id = f"layer_{layer:02d}:{split_name}:fold_{fold}:{attribute}"
    task = "encoding" if split_name == AUDIT else "emotion_generalization"
    record = {
        **context,
        "model_fit_id": fit_id,
        "task": task,
        "layer": layer,
        "attribute": attribute,
        "split_name": split_name,
        "fold": fold,
        "train_split": f"{split_name}:fold_{fold}:train",
        "test_split": f"{split_name}:fold_{fold}:test",
        "n_train": len(train),
        "n_test": len(test),
        "classes": json.dumps(classes.tolist()),
        "selected_c": float(search.best_params_["classifier__C"]),
        "cv_macro_f1": float(search.best_score_),
        "cv_strategy": "stratified_group_speaker" if groups is not None else "stratified",
        "n_iter": int(search.best_estimator_.named_steps["classifier"].n_iter_.max()),
        **scores,
        "majority_macro_f1": baseline_scores["macro_f1"],
        "majority_accuracy": baseline_scores["accuracy"],
        "uniform_chance_accuracy": 1.0 / len(classes),
        "normalized_macro_f1": (
            (scores["macro_f1"] - baseline_scores["macro_f1"])
            / (1.0 - baseline_scores["macro_f1"])
        ),
    }
    predictions = [
        {
            "experiment_id": context["experiment_id"],
            "model_fit_id": fit_id,
            "task": task,
            "layer": layer,
            "attribute": attribute,
            "split_name": split_name,
            "fold": fold,
            "utterance_id": inputs.rows[index].utterance_id,
            "gold_label": str(labels[index]),
            "predicted_label": str(predicted[position]),
            "majority_predicted_label": str(baseline_predictions[position]),
            "classes": search.classes_.tolist(),
            "probabilities": probabilities[position].tolist(),
        }
        for position, index in enumerate(test)
    ]
    candidates = [
        {
            "experiment_id": context["experiment_id"],
            "model_fit_id": fit_id,
            "layer": layer,
            "attribute": attribute,
            "split_name": split_name,
            "fold": fold,
            "c": c,
            "selected": index == search.best_index_,
            "mean_macro_f1": float(search.cv_results_["mean_test_score"][index]),
            "std_macro_f1": float(search.cv_results_["std_test_score"][index]),
            **{
                f"cv_fold_{inner}_macro_f1": float(
                    search.cv_results_[f"split{inner}_test_score"][index]
                )
                for inner in range(probe_config.cv_folds)
            },
        }
        for index, c in enumerate(probe_config.c_grid)
    ]
    return record, predictions, candidates


def summarize_results(output_dir: Path) -> dict[str, Any]:
    probes = read_csv(output_dir / "probes.csv")
    generalization = read_csv(output_dir / "generalization.csv")
    encoding = {}
    for attribute in ATTRIBUTES:
        records = [record for record in probes if record["attribute"] == attribute]
        ordered = sorted(records, key=lambda item: (-float(item["macro_f1"]), int(item["layer"])))
        best, worst = ordered[0], ordered[-1]
        encoding[attribute] = {
            "best_layer": int(best["layer"]),
            "best_macro_f1": float(best["macro_f1"]),
            "best_accuracy": float(best["accuracy"]),
            "worst_layer": int(worst["layer"]),
            "worst_macro_f1": float(worst["macro_f1"]),
            "majority_macro_f1": float(best["majority_macro_f1"]),
            "majority_accuracy": float(best["majority_accuracy"]),
            "above_majority_layers": [
                int(record["layer"]) for record in records
                if float(record["macro_f1"]) > float(record["majority_macro_f1"])
            ],
        }
    layers = sorted({int(record["layer"]) for record in generalization})
    shifts = {}
    for name in (*LEXICAL, "speaker_shift"):
        records = [record for record in generalization if record["split_name"] == name]
        per_layer = []
        for layer in layers:
            group = [record for record in records if int(record["layer"]) == layer]
            f1 = [float(record["macro_f1"]) for record in group]
            accuracy = [float(record["accuracy"]) for record in group]
            gaps = [float(record["macro_f1_gap"]) for record in group]
            per_layer.append({
                "layer": layer,
                "macro_f1_mean": float(np.mean(f1)),
                "macro_f1_min": min(f1),
                "macro_f1_max": max(f1),
                "macro_f1_fold_std": float(np.std(f1, ddof=1)) if len(f1) > 1 else None,
                "accuracy_mean": float(np.mean(accuracy)),
                "macro_f1_gap_mean": float(np.mean(gaps)),
                "accuracy_gap_mean": float(np.mean([float(row["accuracy_gap"]) for row in group])),
                "folds": [int(record["fold"]) for record in group],
                "fold_macro_f1": f1,
                "fold_accuracy": accuracy,
                "fold_macro_f1_gap": gaps,
            })
        shifts[name] = per_layer
    corresponding = {
        "statement": bool(encoding["statement"]["above_majority_layers"]) and any(
            item["macro_f1_gap_mean"] > 0 for name in LEXICAL for item in shifts[name]
        ),
        "speaker": bool(encoding["speaker"]["above_majority_layers"]) and any(
            item["macro_f1_gap_mean"] > 0 for item in shifts["speaker_shift"]
        ),
    }
    signs_present = bool(encoding["emotion"]["above_majority_layers"]) and any(corresponding.values())
    return {
        "encoding": encoding,
        "shifts": shifts,
        "gate": {
            "decision": "PARTIAL / REVIEW NEEDED" if signs_present else "FAIL / DIAGNOSE",
            "corresponding_positive_gap_and_above_majority": corresponding,
            "reason": (
                "Encoding above majority and a corresponding positive shift gap are descriptive "
                "screens, not meaningful-effect or significance tests. The plan specifies no "
                "numerical gate thresholds; inspect the magnitudes and full fold distribution "
                "before human approval. M4 remains blocked."
                if signs_present else
                "At least one required directional screen is absent: emotion above majority, "
                "or primary-confounder encoding with a corresponding positive shift gap. "
                "Diagnose rather than expand."
            ),
            "claim_boundary": "M3 tests encoding and baseline generalization, not reliance or harmful use.",
        },
    }


def verify_result_tables(output_dir: Path, inputs: M3Inputs) -> dict[str, Any]:
    """Recompute every reported score from saved, ID-checked held-out predictions."""
    probes = read_csv(output_dir / "probes.csv")
    generalization = read_csv(output_dir / "generalization.csv")
    candidates = read_csv(output_dir / "cv_results.csv")
    predictions = pq.read_table(output_dir / "predictions.parquet").to_pylist()
    layers = range(inputs.metadata["layer_count"])

    def key(row: dict[str, Any]) -> tuple[int, str, str, int]:
        return int(row["layer"]), row["attribute"], row["split_name"], int(row["fold"])

    expected_probes = {(layer, attribute, AUDIT, 0) for layer in layers for attribute in ATTRIBUTES}
    expected_generalization = {
        (layer, "emotion", name, fold)
        for layer in layers for name, fold in inputs.partitions
        if name in (REFERENCE, *LEXICAL, "speaker_shift")
    }
    for table, expected in ((probes, expected_probes), (generalization, expected_generalization)):
        if len(table) != len(expected) or {key(row) for row in table} != expected:
            raise ValueError("result table has missing, duplicate, or unexpected layer/task rows")
    grouped: dict[tuple[int, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[key(prediction)].append(prediction)
    if set(grouped) != expected_probes | expected_generalization:
        raise ValueError("saved prediction task coverage differs from result tables")
    rows_by_id = {row.utterance_id: row for row in inputs.rows}
    references = {int(row["layer"]): row for row in generalization if row["split_name"] == REFERENCE}
    metric_errors = []
    probability_errors = []
    for record in probes + generalization:
        group = grouped[key(record)]
        train, test = inputs.partitions[(record["split_name"], int(record["fold"]))]
        expected_ids = Counter({inputs.rows[index].utterance_id: 1 for index in test})
        if Counter(item["utterance_id"] for item in group) != expected_ids:
            raise ValueError("saved predictions do not match the persisted held-out IDs")
        if (int(record["n_train"]), int(record["n_test"])) != (len(train), len(test)):
            raise ValueError("recorded split sizes differ from persisted splits")
        attribute = ATTRIBUTES[record["attribute"]]
        train_labels = np.array([str(getattr(inputs.rows[index], attribute)) for index in train])
        classes = np.unique(train_labels)
        if json.loads(record["classes"]) != classes.tolist():
            raise ValueError("result label order differs from training classes")
        gold = np.array([str(getattr(rows_by_id[item["utterance_id"]], attribute)) for item in group])
        if not np.array_equal(gold, [item["gold_label"] for item in group]):
            raise ValueError("prediction labels do not match manifest IDs")
        predicted = np.array([item["predicted_label"] for item in group])
        probabilities = np.array([item["probabilities"] for item in group])
        if any(item["classes"] != classes.tolist() for item in group):
            raise ValueError("saved probability columns do not match training classes")
        if probabilities.shape != (len(test), len(classes)) or not np.isfinite(probabilities).all():
            raise ValueError("invalid saved probability shape or values")
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("saved probabilities lie outside [0, 1]")
        if not np.array_equal(classes[probabilities.argmax(axis=1)], predicted):
            raise ValueError("saved predictions disagree with probability argmax")
        probability_errors.extend(abs(probabilities.sum(axis=1) - 1.0).tolist())
        scores = classification_metrics(gold, predicted, classes)
        label_counts = Counter(train_labels)
        majority = min(classes, key=lambda label: (-label_counts[label], label))
        baseline_predictions = np.full(len(test), majority)
        if not np.array_equal(baseline_predictions, [item["majority_predicted_label"] for item in group]):
            raise ValueError("saved majority baseline differs from training-label baseline")
        baseline = classification_metrics(gold, baseline_predictions, classes)
        expected_scores = {
            **scores, "majority_macro_f1": baseline["macro_f1"],
            "majority_accuracy": baseline["accuracy"], "uniform_chance_accuracy": 1 / len(classes),
            "normalized_macro_f1": (scores["macro_f1"] - baseline["macro_f1"]) / (1 - baseline["macro_f1"]),
        }
        if record["split_name"] != AUDIT:
            reference = references[int(record["layer"])]
            for metric in ("macro_f1", "accuracy"):
                expected_scores[f"reference_{metric}"] = float(reference[metric])
                expected_scores[f"{metric}_gap"] = float(reference[metric]) - scores[metric]
        metric_errors.extend(abs(float(record[name]) - value) for name, value in expected_scores.items())
    if not np.isfinite(metric_errors).all() or max(metric_errors) > 1e-12 or max(probability_errors) > 1e-12:
        raise ValueError("saved metrics or probability normalization failed recomputation")
    config = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))["config"]["probe"]
    fit_ids = {row["model_fit_id"] for row in probes + generalization}
    expected_candidates = {(fit_id, c) for fit_id in fit_ids for c in config["c_grid"]}
    if len(candidates) != len(expected_candidates) or {
        (row["model_fit_id"], float(row["c"])) for row in candidates
    } != expected_candidates:
        raise ValueError("CV candidate table does not cover the predeclared grid for every fit")
    return {
        "status": "pass",
        "probe_rows": len(probes), "generalization_rows": len(generalization),
        "prediction_rows": len(predictions), "distinct_model_fits": len(fit_ids),
        "cv_candidate_rows": len(candidates),
        "maximum_metric_recomputation_error": max(metric_errors),
        "maximum_probability_sum_error": max(probability_errors),
        "held_out_ids_and_gold_labels_match_persisted_inputs": True,
    }


def write_summary(output_dir: Path, summary: dict[str, Any], provenance: dict[str, Any]) -> None:
    probes = read_csv(output_dir / "probes.csv")
    generalization = read_csv(output_dir / "generalization.csv")
    layers = sorted({int(row["layer"]) for row in probes})
    lines = [
        "# M3 results", "", f"Experiment: `{provenance['experiment_id']}`.",
        f"Code: `{provenance['git_commit']}` on `{provenance['git_branch']}`.",
        f"Cache: `{provenance['cache_metadata']['fingerprint']}`.",
        f"Python: `{provenance['python_version']}`; seed: `{provenance['config']['probe']['seed']}`.",
        "", "Exact invocation from `research2027`:", "", "```powershell",
        provenance["command"], "```", "",
        "Package versions, complete configuration and input SHA-256 hashes: [provenance.json](provenance.json).",
        f"C is selected by {provenance['config']['probe']['cv_folds']}-fold training-only macro-F1; smaller C wins exact ties. "
        "Speaker-shift inner folds group speakers; other inner folds are stratified. "
        "Every scaler is fitted within its training fold. Final fits use all outer training data. "
        "The repetition reference reuses the identical emotion-audit fit and predictions.",
        "", "## M3A: encoding only", "",
        "Cells are macro-F1 / accuracy. Layer 0 is the initial hidden state; layers 1-12 are transformer states.",
        "", "| Layer | Emotion | Speaker | Statement | Gender | Intensity |",
        "|---|---|---|---|---|---|",
    ]
    lookup = {(int(row["layer"]), row["attribute"]): row for row in probes}
    for layer in layers:
        cells = [
            f"{float(lookup[(layer, attribute)]['macro_f1']):.4f} / "
            f"{float(lookup[(layer, attribute)]['accuracy']):.4f}"
            for attribute in ATTRIBUTES
        ]
        lines.append(f"| {layer} | " + " | ".join(cells) + " |")
    lines += [
        "", "| Attribute | Majority F1 / accuracy | Uniform chance accuracy | Best F1 layer | Weakest F1 layer |",
        "|---|---|---|---|---|",
    ]
    for attribute, item in summary["encoding"].items():
        row = lookup[(layers[0], attribute)]
        lines.append(
            f"| {attribute} | {item['majority_macro_f1']:.4f} / {item['majority_accuracy']:.4f} | "
            f"{float(row['uniform_chance_accuracy']):.4f} | "
            f"{item['best_layer']} ({item['best_macro_f1']:.4f}) | "
            f"{item['worst_layer']} ({item['worst_macro_f1']:.4f}) |"
        )
    lines += [
        "", "Uniform chance is an accuracy reference, not a macro-F1 estimate.",
        "![Encoding heatmap](encoding_heatmap.png)", "",
        "## M3B: controlled emotion generalization", "",
        "Cells are macro-F1 / accuracy. Speaker values are unweighted means over equal-size folds.",
        "Gaps below use macro-F1: repetition reference minus shifted score (positive means worse under shift).",
        "", "| Layer | Repetition | Lexical 1 to 2 | Lexical 2 to 1 | Speaker mean | Gap L1 to 2 | Gap L2 to 1 | Gap speaker |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for layer in layers:
        reference = next(row for row in generalization if int(row["layer"]) == layer and row["split_name"] == REFERENCE)
        items = [next(item for item in summary["shifts"][name] if item["layer"] == layer) for name in (*LEXICAL, "speaker_shift")]
        cells = [f"{float(reference['macro_f1']):.4f} / {float(reference['accuracy']):.4f}"]
        cells += [f"{item['macro_f1_mean']:.4f} / {item['accuracy_mean']:.4f}" for item in items]
        cells += [f"{item['macro_f1_gap_mean']:+.4f}" for item in items]
        lines.append(f"| {layer} | " + " | ".join(cells) + " |")
    folds = summary["shifts"]["speaker_shift"][0]["folds"]
    lines += [
        "", "### Full speaker-fold distribution", "",
        "| Layer | " + " | ".join(f"Fold {fold} F1 / accuracy" for fold in folds) + " |",
        "|---|" + "---|" * len(folds),
    ]
    for item in summary["shifts"]["speaker_shift"]:
        cells = [f"{f1:.4f} / {accuracy:.4f}" for f1, accuracy in zip(item["fold_macro_f1"], item["fold_accuracy"])]
        lines.append(f"| {item['layer']} | " + " | ".join(cells) + " |")
    lines += [
        "", "![Generalization and gaps](emotion_generalization.png)", "",
        "## Gate", "", f"**{summary['gate']['decision']}**", "",
        summary["gate"]["reason"], "", summary["gate"]["claim_boundary"], "",
        "## Interpretation limits and remaining review", "",
        "- No intervention, erasure, new encoder, or additional corpus was run.",
        "- One predeclared seed; no statistical-significance claim or bootstrap confidence interval.",
        "- Fold standard deviation/range is descriptive, not a confidence interval; training folds overlap.",
        "- Speaker fits use 1,200 training utterances versus 720 for the reference and lexical fits. "
        "The requested gap is not a matched-training-size causal estimate.",
        "- Neutral occurs only at normal intensity; intensity is not independent of emotion.",
        "- Layer maxima are descriptive test-set summaries, not unbiased layer-selection estimates.",
        "- Human scientific review and explicit M4 approval remain required.", "",
        "Complete metrics: [probes.csv](probes.csv), [generalization.csv](generalization.csv). "
        "Training-only selection: [cv_results.csv](cv_results.csv). "
        "Held-out predictions, class order, probabilities and majority predictions: "
        "[predictions.parquet](predictions.parquet). "
        "Exact ID/metric recomputation: [verification.json](verification.json). "
        "Machine-readable gate and fold summaries: [summary.json](summary.json).",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_m3(
    config_path: Path, output_dir: Path | None, *, validate_only: bool = False
) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = dict(config["probe"])
    settings["c_grid"] = tuple(settings["c_grid"])
    probe_config = ProbeConfig(**settings)
    inputs = load_m3_inputs(config)
    if validate_only:
        print(f"M3 inputs validated: {len(inputs.rows)} utterances, 13 layers, 768 dimensions; no fitting.")
        return
    if output_dir is None:
        raise ValueError("M3 requires a new --output-dir (or --validate-only)")
    root = Path.cwd().resolve()
    config_name = str(config_path.resolve().relative_to(root))
    output_name = str(output_dir.resolve().relative_to(root))
    provenance = {
        "schema_version": 1,
        "experiment_id": output_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **git_provenance(),
        "python_version": platform.python_version(),
        "compute": {"system": platform.system(), "architecture": platform.machine(), "device": "cpu", "fit_threads": 1},
        "packages": {
            name: version(name)
            for name in ("numpy", "scipy", "scikit-learn", "matplotlib", "pyarrow", "threadpoolctl", "torch", "transformers")
        },
        "config": config,
        "config_sha256": sha256_file(config_path),
        "cache_metadata": inputs.metadata,
        "input_sha256": inputs.hashes,
        "command": (
            f'.\\.venv\\Scripts\\python.exe -m mockingbird2027 probe '
            f'--config "{config_name}" --output-dir "{output_name}"'
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(provenance, output_dir / "provenance.json")
    context = {
        key: provenance[key] for key in ("experiment_id", "timestamp_utc", "git_commit")
    }
    context.update({
        "dataset": "ravdess", "model": inputs.metadata["model_id"],
        "model_revision": inputs.metadata["resolved_revision"],
        "cache_fingerprint": inputs.metadata["fingerprint"],
        "seed": probe_config.seed,
    })
    probes: list[dict[str, Any]] = []
    generalization: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    status = {"status": "running", "completed_layers": 0, "expected_layers": inputs.metadata["layer_count"]}
    write_json(status, output_dir / "status.json")
    complete = False
    try:
        for layer in range(inputs.metadata["layer_count"]):
            features = np.load(inputs.cache / f"layer_{layer:02d}.npy", allow_pickle=False).astype(np.float64)
            for attribute in ATTRIBUTES:
                record, task_predictions, cv = evaluate_task(
                    inputs, features, layer, AUDIT, 0, attribute, probe_config, context
                )
                probes.append(record)
                predictions.extend(task_predictions)
                candidates.extend(cv)
                if attribute == "emotion":
                    reference = {
                        **record, "task": "emotion_generalization", "split_name": REFERENCE,
                        "train_split": f"{REFERENCE}:fold_0:train",
                        "test_split": f"{REFERENCE}:fold_0:test",
                        "reference_macro_f1": record["macro_f1"], "reference_accuracy": record["accuracy"],
                        "macro_f1_gap": 0.0, "accuracy_gap": 0.0,
                    }
                    generalization.append(reference)
                    predictions.extend(
                        {**item, "task": "emotion_generalization", "split_name": REFERENCE}
                        for item in task_predictions
                    )
            for name, fold in sorted(inputs.partitions):
                if name not in (*LEXICAL, "speaker_shift"):
                    continue
                record, task_predictions, cv = evaluate_task(
                    inputs, features, layer, name, fold, "emotion", probe_config, context
                )
                generalization.append({
                    **record,
                    "reference_macro_f1": reference["macro_f1"],
                    "reference_accuracy": reference["accuracy"],
                    "macro_f1_gap": reference["macro_f1"] - record["macro_f1"],
                    "accuracy_gap": reference["accuracy"] - record["accuracy"],
                })
                predictions.extend(task_predictions)
                candidates.extend(cv)
            write_csv(probes, output_dir / "probes.csv")
            write_csv(generalization, output_dir / "generalization.csv")
            write_csv(candidates, output_dir / "cv_results.csv")
            write_parquet(predictions, output_dir / "predictions.parquet")
            status["completed_layers"] = layer + 1
            write_json(status, output_dir / "status.json")
            with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"layer": layer, "timestamp_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
            print(f"M3 layer {layer:02d} complete; {len(probes)} audit / {len(generalization)} generalization rows.", flush=True)

        for name, digest in inputs.hashes.items():
            if sha256_file(Path(name)) != digest:
                raise RuntimeError(f"scientific input changed during M3: {Path(name).name}")
        if sha256_file(config_path) != provenance["config_sha256"]:
            raise RuntimeError("M3 configuration changed during the run")
        write_json(verify_result_tables(output_dir, inputs), output_dir / "verification.json")
        from ..figures.m3 import generate_figures

        generate_figures(output_dir)
        summary = summarize_results(output_dir)
        write_json(summary, output_dir / "summary.json")
        write_summary(output_dir, summary, provenance)
        write_json(
            {path.name: sha256_file(path) for path in sorted(output_dir.iterdir()) if path.name != "status.json"},
            output_dir / "output_sha256.json",
        )
        complete = True
    finally:
        status.update({
            "status": "complete" if complete else "failed",
            "exit_code": 0 if complete else 1,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        write_json(status, output_dir / "status.json")
    print(f"M3 complete: {output_dir}; gate: {summary['gate']['decision']}. M4 not started.")
