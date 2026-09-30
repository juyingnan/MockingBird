"""Prediction-based verification and explicitly gated M4 reporting."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ..data.manifest import write_json
from .erasure import AffineRemoval
from .m3 import ATTRIBUTES, REFERENCE, M3Inputs, read_csv
from .m4 import condition_plan, parameters_hash, validity_decision
from .probes import classification_metrics


def metric_value(row: dict[str, Any], name: str) -> float:
    value = float(row[name])
    if not np.isfinite(value):
        raise ValueError(f"nonfinite published value: {name}")
    return value


def verify_m4(output_dir: Path, inputs: M3Inputs) -> dict[str, Any]:
    provenance = json.loads((output_dir / "provenance.json").read_text())
    config = provenance["config"]
    conditions = read_csv(output_dir / "conditions.csv")
    expected = condition_plan(inputs, config)
    if len(conditions) != len(expected) or {row["condition_id"] for row in conditions} != {row["condition_id"] for row in expected}:
        raise ValueError("M4 condition inventory does not match the frozen protocol")
    condition_map = {row["condition_id"]: row for row in conditions}
    metrics = read_csv(output_dir / "metrics.csv")
    metric_map = {row["evaluation_id"]: row for row in metrics}
    if len(metric_map) != len(metrics):
        raise ValueError("duplicate M4 evaluation IDs")
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((output_dir / "predictions").glob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            predictions[row["evaluation_id"]].append(row)
    if set(predictions) != set(metric_map):
        raise ValueError("M4 saved predictions do not exactly cover published evaluations")
    manifest = {row.utterance_id: row for row in inputs.rows}
    errors = []
    probability_errors = []
    fold_maps = {}
    for path in sorted((output_dir / "diagnostic_folds").glob("*.parquet")):
        records = pq.read_table(path).to_pylist()
        grouped: dict[str, dict[str, int]] = defaultdict(dict)
        for row in records:
            attribute, identity = row["attribute"], row["utterance_id"]
            if identity in grouped[attribute]:
                raise ValueError("duplicated diagnostic held-out ID")
            grouped[attribute][identity] = row["held_out_fold"]
        fold_maps[path.stem] = grouped
    for record in metrics:
        condition = condition_map[record["condition_id"]]
        train, test = inputs.partitions[(condition["split_name"], int(condition["fold"]))]
        group = predictions[record["evaluation_id"]]
        expected_ids = Counter({inputs.rows[index].utterance_id: 1 for index in test})
        if Counter(item["utterance_id"] for item in group) != expected_ids or int(record["n_test"]) != len(test):
            raise ValueError("published M4 evaluation has wrong held-out IDs or count")
        attribute = ATTRIBUTES[record["attribute"]]
        gold = np.array([str(getattr(manifest[item["utterance_id"]], attribute)) for item in group])
        if not np.array_equal(gold, [item["gold_label"] for item in group]):
            raise ValueError("M4 prediction gold labels disagree with the manifest")
        classes = np.array(json.loads(record["classes"]))
        predicted = np.array([item["predicted_label"] for item in group])
        if any(item["classes"] != classes.tolist() for item in group):
            raise ValueError("M4 probability class order is inconsistent")
        values = np.array([item["probabilities"] for item in group])
        if values.shape != (len(test), len(classes)) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("invalid M4 probabilities")
        probability_errors.extend(abs(values.sum(axis=1) - 1).tolist())
        if not np.array_equal(classes[values.argmax(axis=1)], predicted):
            raise ValueError("M4 predicted labels differ from saved probabilities")
        scores = classification_metrics(gold, predicted, classes)
        errors.extend(abs(metric_value(record, metric) - scores[metric]) for metric in scores)
        if record["purpose"] == "audit":
            folds = fold_maps[record["condition_id"]][record["attribute"]]
            if set(folds) != set(expected_ids) or set(folds.values()) != set(range(config["audit"]["folds"])):
                raise ValueError("diagnostic folds do not partition the outer held-out pool")
            if any(item["diagnostic_fold"] != folds[item["utterance_id"]] for item in group):
                raise ValueError("saved audit predictions disagree with diagnostic held-out folds")
        elif any(item["diagnostic_fold"] != -1 for item in group):
            raise ValueError("task predictions must not be conflated with diagnostic probe folds")

    fits = read_csv(output_dir / "diagnostic_probe_fits.csv")
    original_c = {}
    for fit in fits:
        condition = condition_map[fit["condition_id"]]
        _, test = inputs.partitions[(condition["split_name"], int(condition["fold"]))]
        folds = fold_maps[fit["condition_id"]][fit["attribute"]]
        train_ids = [
            inputs.rows[index].utterance_id for index in test
            if folds[inputs.rows[index].utterance_id] != int(fit["diagnostic_fold"])
        ]
        if len(train_ids) != int(fit["n_train"]) or hashlib.sha256(json.dumps(train_ids).encode("ascii")).hexdigest() != fit["train_ids_sha256"]:
            raise ValueError("diagnostic probe training IDs overlap evaluation or differ from frozen folds")
        key = (fit["condition_id"], fit["attribute"], fit["diagnostic_fold"])
        if fit["variant"] == "original":
            candidates = json.loads(fit["original_cv_scores"])
            for candidate in candidates:
                errors.append(abs(candidate["mean_macro_f1"] - float(np.mean(candidate["fold_macro_f1"]))))
            chosen = min(candidates, key=lambda item: (-item["mean_macro_f1"], item["c"]))
            if float(fit["selected_c"]) != chosen["c"]:
                raise ValueError("diagnostic C was not selected by original training-only CV")
            original_c[key] = chosen["c"]
        elif float(fit["selected_c"]) != original_c[key]:
            raise ValueError("transformed diagnostic C differs from the frozen original C")

    training = pq.read_table(output_dir / "training_ids.parquet").to_pylist()
    fitted_ids: dict[str, list[str]] = defaultdict(list)
    for row in training:
        fitted_ids[row["condition_id"]].append(row["utterance_id"])
    integrity = read_csv(output_dir / "head_integrity.csv")
    validations = read_csv(output_dir / "targeted_validation.csv")
    validation_map = {row["condition_id"]: row for row in validations}
    controls = read_csv(output_dir / "random_controls.csv")
    controls_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for control in controls:
        controls_by_condition[control["condition_id"]].append(control)
    transform_count = 0
    for check in integrity:
        identity = check["condition_id"]
        condition = condition_map[identity]
        train, test = inputs.partitions[(condition["split_name"], int(condition["fold"]))]
        if Counter(fitted_ids[identity]) != Counter({inputs.rows[index].utterance_id: 1 for index in train}):
            raise ValueError("eraser/head fitting IDs differ from the original outer training set")
        if check["before_sha256"] != check["after_sha256"] or check["unchanged"] != "True" or check["m3_predictions_reproduced"] != "True":
            raise ValueError("original M4A emotion head did not remain fixed")
        draws = controls_by_condition[identity]
        if len(draws) != config["controls"]["draws"] or {int(row["draw"]) for row in draws} != set(range(config["controls"]["draws"])):
            raise ValueError("control distribution differs from the predeclared fixed draw count")
        with np.load(output_dir / "heads" / f"{identity}_original.npz", allow_pickle=False) as saved_head:
            if parameters_hash({name: saved_head[name] for name in saved_head.files}) != check["before_sha256"]:
                raise ValueError("persisted original-head parameters do not match the fixed-head hash")
            fixed_records = [row for row in metrics if row["condition_id"] == identity and row["purpose"] == "fixed_head"]
            if any(row["head_hash"] != check["before_sha256"] for row in fixed_records):
                raise ValueError("M4A evaluations used different original emotion heads")
            raw = np.load(inputs.cache / f"layer_{int(condition['layer']):02d}.npy", allow_pickle=False).astype(np.float64)
            train_features = (raw[train] - saved_head["mean"]) / saved_head["scale"]
            test_features = (raw[test] - saved_head["mean"]) / saved_head["scale"]
        valid_row = validation_map[identity]
        concept = np.array([str(getattr(inputs.rows[index], ATTRIBUTES[condition["target"]])) for index in train])
        _, inverse = np.unique(concept, return_inverse=True)
        one_hot = np.eye(len(np.unique(concept)))[inverse]
        cross = (train_features - train_features.mean(axis=0)).T @ (one_hot - one_hot.mean(axis=0)) / len(train)
        expected_variants = ["targeted"] + [
            item["variant"] for item in controls_by_condition[identity] if item["status"] == "matched"
        ]
        for variant in expected_variants:
            with np.load(output_dir / "transforms" / f"{identity}_{variant}.npz", allow_pickle=False) as factors:
                removal = AffineRemoval(factors["mean"], factors["left"], factors["right"], condition["method"])
            if removal.removed_rank != int(valid_row["removed_rank"]):
                raise ValueError("random/targeted removed dimensions differ")
            np.testing.assert_allclose(removal.mean, train_features.mean(axis=0), atol=1e-12, rtol=0)
            perturbation = removal.perturbation(train_features)
            if variant == "targeted":
                residual = np.linalg.norm(cross - removal.left @ (removal.right.T @ cross)) / np.linalg.norm(cross)
                if residual > config["numerics"]["invariant_tolerance"]:
                    raise ValueError("saved targeted eraser fails training covariance removal")
                errors.append(abs(residual - float(valid_row["training_cross_covariance_residual"])))
                errors.append(abs(perturbation["rms"] - float(valid_row["train_rms"])))
                errors.append(abs(removal.perturbation(test_features)["rms"] - float(valid_row["test_rms"])))
            else:
                control = next(item for item in controls_by_condition[identity] if item["variant"] == variant)
                relative_error = abs(perturbation["rms"] / float(valid_row["train_rms"]) - 1)
                if relative_error > config["controls"]["relative_rms_tolerance"]:
                    raise ValueError("saved random control fails RMS matching")
                errors.append(abs(perturbation["rms"] - float(control["train_rms"])))
            transform_count += 1
        audit_original = {
            name: metric_map[f"{identity}:audit:original:{name}"] for name in ATTRIBUTES
        }
        audit_erased = {
            name: metric_map[f"{identity}:audit:targeted:{name}"] for name in ATTRIBUTES
        }
        scores_original = {name: {metric: float(row[metric]) for metric in ("macro_f1", "accuracy")} for name, row in audit_original.items()}
        scores_erased = {name: {metric: float(row[metric]) for metric in ("macro_f1", "accuracy")} for name, row in audit_erased.items()}
        target_class_count = len(json.loads(audit_original[condition["target"]]["classes"]))
        decision = validity_decision(
            scores_original, scores_erased, condition["target"], target_class_count, config,
            sum(item["status"] == "matched" for item in controls_by_condition[identity]) == config["controls"]["draws"],
        )
        if str(decision["valid"]) != valid_row["valid"] or (condition["status"] == "valid") != decision["valid"]:
            raise ValueError("recorded validity gate differs from saved diagnostic predictions")

    for row in read_csv(output_dir / "collateral_damage.csv"):
        before, after = metric_map[row["before_evaluation_id"]], metric_map[row["after_evaluation_id"]]
        for metric in ("macro_f1", "accuracy"):
            errors.extend([
                abs(float(row[f"before_{metric}"]) - float(before[metric])),
                abs(float(row[f"after_{metric}"]) - float(after[metric])),
                abs(float(row[f"delta_{metric}"]) - (float(after[metric]) - float(before[metric]))),
            ])
    for control in controls:
        if control["status"] != "matched":
            continue
        score = metric_map[control["fixed_evaluation_id"]]
        before = metric_map[f"{control['condition_id']}:fixed_head:original:emotion"]
        audit_score = metric_map[control["audit_evaluation_id"]]
        for metric in ("macro_f1", "accuracy"):
            errors.extend([
                abs(float(control[metric]) - float(score[metric])),
                abs(float(control[f"{metric}_effect"]) - (float(score[metric]) - float(before[metric]))),
                abs(float(control[f"target_audit_{metric}"]) - float(audit_score[metric])),
            ])
    for row in read_csv(output_dir / "fixed_head_reliance.csv"):
        metric = row["metric"]
        before, after = metric_map[row["original_evaluation_id"]], metric_map[row["targeted_evaluation_id"]]
        targeted_effect = float(after[metric]) - float(before[metric])
        errors.append(abs(float(row["targeted_effect"]) - targeted_effect))
        matched = [control for control in controls_by_condition[row["condition_id"]] if control["status"] == "matched"]
        if int(row["matched_draws"]) != len(matched):
            raise ValueError("reliance control count does not match the saved distribution")
        if len(matched) == config["controls"]["draws"]:
            effects = np.array([float(control[f"{metric}_effect"]) for control in matched])
            low, high = np.quantile(effects, config["reliance"]["control_quantiles"])
            for name, expected_value in (
                ("random_mean_effect", effects.mean()), ("random_sd_effect", effects.std(ddof=1)),
                ("random_q025", low), ("random_q975", high),
                ("targeted_minus_random", targeted_effect - effects.mean()),
            ):
                errors.append(abs(float(row[name]) - expected_value))
    retraining = read_csv(output_dir / "retrained_heads.csv")
    if len(retraining) != len(conditions) or {row["condition_id"] for row in retraining} != set(condition_map):
        raise ValueError("M4B must preserve every completed or blocked planned condition")
    for row in retraining:
        if row["status"] != "complete":
            continue
        if condition_map[row["condition_id"]]["status"] != "valid":
            raise ValueError("M4B was run on an invalid eraser")
        after = metric_map[row["evaluation_id"]]
        before = metric_map[f"{row['condition_id']}:fixed_head:original:emotion"]
        for metric in ("macro_f1", "accuracy"):
            errors.extend([
                abs(float(row[f"erased_{metric}"]) - float(after[metric])),
                abs(float(row[f"original_{metric}"]) - float(before[metric])),
                abs(float(row[f"delta_{metric}"]) - (float(after[metric]) - float(before[metric]))),
                abs(float(row[f"original_{metric}_gap"]) - (float(row[f"original_reference_{metric}"]) - float(before[metric]))),
                abs(float(row[f"erased_{metric}_gap"]) - (float(row[f"erased_reference_{metric}"]) - float(after[metric]))),
                abs(float(row[f"delta_{metric}_gap"]) - (float(row[f"erased_{metric}_gap"]) - float(row[f"original_{metric}_gap"]))),
            ])
    if not errors or not np.isfinite(errors).all() or max(errors) > 1e-9 or max(probability_errors) > 1e-12:
        raise ValueError("M4 metric/transform/probability recomputation failed")
    return {
        "status": "pass", "planned_conditions": len(conditions), "metric_rows": len(metrics),
        "prediction_rows": sum(len(group) for group in predictions.values()),
        "diagnostic_probe_fits": len(fits), "verified_transforms": transform_count,
        "unchanged_original_heads": len(integrity), "maximum_recomputation_error": max(errors),
        "maximum_probability_sum_error": max(probability_errors),
        "all_eraser_training_ids_match_outer_train": True,
        "all_diagnostic_training_ids_exclude_their_evaluation_fold": True,
        "all_m4b_conditions_validated_before_retraining": True,
    }


def summarize_m4(output_dir: Path) -> dict[str, Any]:
    conditions = read_csv(output_dir / "conditions.csv")
    validation = {row["condition_id"]: row for row in read_csv(output_dir / "targeted_validation.csv")}
    fixed = {row["condition_id"]: row for row in read_csv(output_dir / "fixed_head_reliance.csv") if row["metric"] == "macro_f1"}
    retrained = {row["condition_id"]: row for row in read_csv(output_dir / "retrained_heads.csv")}
    provenance = json.loads((output_dir / "provenance.json").read_text())
    targets = {}
    for target in ("statement", "speaker"):
        group = [condition for condition in conditions if condition["target"] == target]
        observations = []
        for condition in group:
            identity = condition["condition_id"]
            observation = {
                **condition, "m4b_status": retrained[identity]["status"],
                "m4b_reason": retrained[identity]["reason"],
            }
            if identity in validation:
                row = validation[identity]
                observation.update({
                    "target_before_macro_f1": float(row["target_before_macro_f1"]),
                    "target_after_macro_f1": float(row["target_after_macro_f1"]),
                    "target_after_accuracy": float(row["target_after_accuracy"]),
                    "target_removal_valid": row["target_removal_valid"] == "True",
                    "collateral_valid": row["collateral_valid"] == "True",
                    "emotion_probe_f1_loss": float(row["emotion_probe_f1_loss"]),
                    "removed_rank": int(row["removed_rank"]),
                    "fixed_head_interpretation": fixed[identity]["interpretation"],
                    "fixed_head_effect": float(fixed[identity]["targeted_effect"]),
                    "targeted_minus_random": (
                        float(fixed[identity]["targeted_minus_random"])
                        if fixed[identity].get("targeted_minus_random") else None
                    ),
                })
            observations.append(observation)
        targets[target] = {
            "encoded_from_m3": True, "selected_layers": provenance["config"]["selected_layers"][target],
            "status_counts": dict(Counter(row["status"] for row in group)),
            "valid_conditions": sum(row["status"] == "valid" for row in group),
            "m4b_completed": sum(retrained[row["condition_id"]]["status"] == "complete" for row in group),
            "conditions": observations,
        }
    no_valid_statement = targets["statement"]["valid_conditions"] == 0
    speaker_comparisons = []
    for layer in provenance["config"]["selected_layers"]["speaker"]:
        for method in provenance["config"]["methods"]:
            group = [
                row for row in conditions if row["target"] == "speaker"
                and int(row["layer"]) == layer and row["method"] == method
            ]
            reference = next(row for row in group if row["split_name"] == REFERENCE)
            folds = [row for row in group if row["split_name"] == "speaker_shift"]
            complete = all(retrained[row["condition_id"]]["status"] == "complete" for row in group)
            comparison: dict[str, Any] = {
                "layer": layer, "method": method,
                "status": "complete" if complete else "blocked_incomplete_valid_fold_set",
                "required_folds": len(folds),
            }
            if complete:
                reference_scores = retrained[reference["condition_id"]]
                fold_scores = [retrained[row["condition_id"]] for row in folds]
                for metric in ("macro_f1", "accuracy"):
                    original_ood = float(np.mean([float(row[f"original_{metric}"]) for row in fold_scores]))
                    erased_ood = float(np.mean([float(row[f"erased_{metric}"]) for row in fold_scores]))
                    original_id = float(reference_scores[f"original_{metric}"])
                    erased_id = float(reference_scores[f"erased_{metric}"])
                    comparison[metric] = {
                        "original_reference": original_id, "erased_reference": erased_id,
                        "delta_reference": erased_id - original_id,
                        "original_ood": original_ood, "erased_ood": erased_ood,
                        "delta_ood": erased_ood - original_ood,
                        "original_gap": original_id - original_ood,
                        "erased_gap": erased_id - erased_ood,
                        "delta_gap": (erased_id - erased_ood) - (original_id - original_ood),
                        "original_fold_scores": [float(row[f"original_{metric}"]) for row in fold_scores],
                        "erased_fold_scores": [float(row[f"erased_{metric}"]) for row in fold_scores],
                    }
                f1 = comparison["macro_f1"]
                policy = provenance["config"]["m4b"]
                comparison["exploratory_harmful_shortcut_pattern"] = (
                    f1["delta_reference"] >= -policy["maximum_modest_id_loss"]
                    and f1["delta_ood"] >= policy["minimum_ood_gain"]
                    and f1["delta_gap"] <= -policy["minimum_gap_reduction"]
                )
            speaker_comparisons.append(comparison)
    complete_chain = any(row["status"] == "complete" for row in speaker_comparisons)
    recommendation = "replication across encoders" if complete_chain else "intervention debugging"
    return {
        "experiment_id": provenance["experiment_id"], "targets": targets,
        "milestone_state": "gated_stop" if no_valid_statement else "partial_identifiability_and_validity_gates",
        "headline": (
            "Statement information is encoded, but no statement intervention passed the "
            "predeclared removal/collateral/control validity gate. Fixed-head changes are "
            "diagnostics, not evidence of validated reliance. Harmful-under-shift inference "
            "is unavailable; speaker expansion and M4B were gated."
            if no_valid_statement else
            "Some intervention conditions passed operational validation; interpret only their "
            "saved targeted-minus-control fixed-head effects. Lexical harmful-under-shift "
            "effects remain unidentifiable under the preserved single-statement training splits."
        ),
        "null_and_contradictory_conditions": [
            {"condition_id": row["condition_id"], "status": row["status"], "reason": row["reason"]}
            for row in conditions if row["status"] != "valid"
        ],
        "speaker_retraining_comparisons": speaker_comparisons,
        "replication_recommendation": recommendation,
        "recommendation_reason": (
            "Review the complete validated WavLM speaker chain, including null or adverse "
            "effects, before authorizing encoder replication. No replication was launched."
            if complete_chain else
            "Resolve failed intervention validity and the train-only lexical identifiability "
            "constraint in a newly approved protocol before any encoder/dataset replication. "
            "Do not adjust the frozen primary results."
        ),
        "claim_boundary": "Encoding, fixed-head sensitivity, and harmful-under-shift learning effects remain distinct.",
    }


def write_m4_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = []
    for number, target, title in ((1, "statement", "Statement identity"), (2, "speaker", "Speaker identity")):
        item = summary["targets"][target]
        attempted = [row for row in item["conditions"] if "target_before_macro_f1" in row]
        valid_count = item["valid_conditions"]
        successful_target = sum(row["target_removal_valid"] for row in attempted)
        lines += [
            f"### {number}. {title}", "",
            "- Encoded? **Yes**, from the unchanged M3 audit.",
            (
                f"- Successfully removed? {successful_target}/{len(attempted)} tested conditions "
                f"passed target-removal screening; {valid_count} passed the complete validity gate."
                if attempted else
                "- Successfully removed? **Not tested**: blocked by the statement-validation gate."
            ),
            (
                "- Used by fixed emotion head? Interpret only validated, target-specific control "
                "comparisons in the fixed-head table; effect signs and invalid conditions are retained."
                if valid_count else
                "- Used by fixed emotion head? **Not established**; no validated condition supports this claim."
            ),
            (
                "- Harmful under lexical shift? **Unidentifiable in this protocol**: each lexical "
                "training split has only one statement label. No held-out utterance was used to fit an eraser."
                if target == "statement" else
                f"- Harmful under speaker shift? {item['m4b_completed']} reference/fold conditions "
                "completed retraining; blocked conditions are not null effects."
            ),
            (
                "- Important collateral effects? See the paired five-attribute diagnostics below; "
                "diagnostic probes are separate from the task heads."
                if attempted else
                "- Important collateral effects? Not measured; no speaker intervention was run."
            ),
        ]
        if attempted:
            lines += [
                "", "| Layer | Method | Split | Target F1 before | Target F1 after | Emotion probe F1 loss | Valid? |",
                "|---|---|---|---|---|---|---|",
            ]
            for row in attempted:
                lines.append(
                    f"| {row['layer']} | {row['method']} | {row['split_name']} / {row['fold']} | "
                    f"{row['target_before_macro_f1']:.4f} | {row['target_after_macro_f1']:.4f} | "
                    f"{row['emotion_probe_f1_loss']:+.4f} | {row['status']} |"
                )
        lines += [""]
    lines += [
        "### 3. Headline result", "", summary["headline"], "",
        "### 4. Null / contradictory findings", "",
        "- Every planned condition is retained in [conditions.csv](conditions.csv); blocked "
        "conditions are not encoded as zeros or evidence of no effect.",
        "- Removal of training cross-covariance does not guarantee removal of held-out "
        "linear decodability. Fresh diagnostic probes assess this separately.",
        "- The eight lexical OOD target conditions cannot identify a supervised contrast "
        "from their original training data. No alternate split or leaked eraser was substituted.",
        "- Random controls are conditioned on rank and training RMS, not per-utterance "
        "displacement or all higher moments. Their empirical quantiles are not dataset confidence intervals.",
        "- Validity-gated results are exploratory; one seed, selected layers and a selected "
        "valid subset cannot establish universal non-reliance or causal harmfulness.",
        "", "### 5. Replication recommendation", "",
        f"**{summary['replication_recommendation'].capitalize()}.** {summary['recommendation_reason']}",
        "", "No new encoder or CREMA-D run was started; the paper repository and M3 artifacts are unchanged.",
        "", "Artifacts: [provenance](provenance.json), [frozen protocol](protocol.md), "
        "[selected layers](selected_layers.json), [target validation](targeted_validation.csv), "
        "[collateral matrix](collateral_damage.csv), [random controls](random_controls.csv), "
        "[fixed-head reliance](fixed_head_reliance.csv), [retrained heads](retrained_heads.csv), "
        "[all metrics](metrics.csv), [predictions](predictions/), [verification](verification.json), "
        "[checksums](output_sha256.json).",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_m4(output_dir: Path, inputs: M3Inputs) -> None:
    verification = verify_m4(output_dir, inputs)
    write_json(verification, output_dir / "verification.json")
    summary = summarize_m4(output_dir)
    write_json(summary, output_dir / "summary.json")
    write_m4_summary(output_dir, summary)
    from ..figures.m4 import generate_m4_figures

    generate_m4_figures(output_dir)
