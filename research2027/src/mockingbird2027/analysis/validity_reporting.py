"""Verification and reporting of the frozen validity-only study."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ..data.manifest import write_json
from .m3 import ATTRIBUTES, M3Inputs, read_csv, sha256_file, write_csv
from .m4 import parameters_hash
from .probes import classification_metrics
from .validity import ids_hash
from .validity_splits import frozen_probe_cv, role_indices
from .validity_statistics import bootstrap_scores, cluster_bootstrap_counts, validity_decision


def typed_metrics(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    numeric = (
        "selected_c", "gamma", "macro_f1", "accuracy", "majority_macro_f1",
        "majority_accuracy", "uniform_chance_accuracy", "macro_f1_upper_95", "accuracy_upper_95",
    )
    for row in read_csv(path):
        item: dict[str, Any] = dict(row)
        for name in numeric:
            item[name] = float(row[name]) if row[name] else None
        item["n_test"] = int(row["n_test"])
        item["dimension"] = int(row["dimension"])
        output.append(item)
    return output


def verify_validity(output_dir: Path, inputs: M3Inputs) -> dict[str, Any]:
    provenance = json.loads((output_dir / "provenance.json").read_text())
    config = provenance["config"]
    roles = pq.read_table(output_dir / "roles.parquet").to_pylist()
    cv_records = pq.read_table(output_dir / "probe_cv.parquet").to_pylist()
    indices = role_indices(inputs.rows, roles)
    cv = frozen_probe_cv(inputs.rows, roles, cv_records, config["linear_probe"]["cv_folds"])
    if sha256_file(output_dir / "roles.parquet") != config["roles_sha256"] or sha256_file(output_dir / "probe_cv.parquet") != config["cv_sha256"]:
        raise ValueError("run split/CV copies differ from the frozen protocol")
    metrics = typed_metrics(output_dir / "metrics.csv")
    expected_keys = {
        (arm, "linear", attribute) for arm in config["arms"] for attribute in config["attributes"]
    } | {(arm, "rbf_svm", "statement") for arm in config["arms"]}
    if len(metrics) != len(expected_keys) or {
        (row["arm"], row["probe"], row["attribute"]) for row in metrics
    } != expected_keys:
        raise ValueError("missing, duplicated, or unexpected final validity models")
    fit_rows = read_csv(output_dir / "model_fits.csv")
    fit_map = {row["model_id"]: row for row in fit_rows}
    if len(fit_map) != len(metrics):
        raise ValueError("frozen model inventory is incomplete")
    metric_map = {row["model_id"]: row for row in metrics}
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(output_dir / "predictions.parquet").to_pylist():
        predictions[row["model_id"]].append(row)
    if set(predictions) != set(metric_map):
        raise ValueError("prediction coverage differs from the frozen model inventory")
    before_C = json.loads((output_dir / "fit_complete.json").read_text())
    phases = json.loads((output_dir / "phase_times.json").read_text())
    if (
        before_C["model_count"] != len(metrics)
        or datetime.fromisoformat(phases["all_models_frozen_utc"]) > datetime.fromisoformat(phases["C_evaluation_started_utc"])
        or datetime.fromisoformat(before_C["timestamp_utc"]) > datetime.fromisoformat(phases["C_evaluation_started_utc"])
    ):
        raise ValueError("C was evaluated before all models were frozen")
    b_ids = [inputs.rows[index].utterance_id for index in indices["B"]]
    c_ids = [inputs.rows[index].utterance_id for index in indices["C"]]
    by_id = {row.utterance_id: row for row in inputs.rows}
    errors = []
    probability_errors = []
    for metric in metrics:
        identity = metric["model_id"]
        fit = fit_map[identity]
        if (
            fit["fit_role"] != "B" or int(fit["n_fit"]) != len(b_ids)
            or fit["fit_ids_sha256"] != ids_hash(b_ids) or metric["evaluation_role"] != "C"
            or metric["n_test"] != len(c_ids)
            or datetime.fromisoformat(fit["fit_completed_utc"]) > datetime.fromisoformat(phases["C_evaluation_started_utc"])
        ):
            raise ValueError("model-fitting roles, IDs or phase boundary are inconsistent")
        expected_dimension = inputs.metadata["hidden_size"] if metric["arm"] in ("original", "leace") else config["pca"]["components"]
        if metric["dimension"] != expected_dimension or int(fit["dimension"]) != expected_dimension:
            raise ValueError("representation dimension differs from the frozen arm")
        with np.load(output_dir / "models" / f"{identity}.npz", allow_pickle=False) as parameters:
            if parameters_hash({name: parameters[name] for name in parameters.files}) != fit["frozen_parameter_hash"]:
                raise ValueError("saved model parameters differ from their pre-evaluation hash")
            if metric["probe"] == "rbf_svm":
                support = parameters["support_indices"]
                if support.dtype.kind not in "iu" or np.any(support < 0) or np.any(support >= len(b_ids)):
                    raise ValueError("SVM support vectors must reference B only")
                if "support_vectors" in parameters.files:
                    raise ValueError("support-vector embeddings must not be published")
                errors.append(abs(float(parameters["gamma"][0]) - 1.0 / expected_dimension))
        if fit["frozen_parameter_hash"] != metric["frozen_parameter_hash"] or before_C["model_hashes"][identity] != fit["frozen_parameter_hash"]:
            raise ValueError("scoring did not use the frozen fitted model")
        group = predictions[identity]
        if Counter(row["utterance_id"] for row in group) != Counter({identity: 1 for identity in c_ids}):
            raise ValueError("prediction IDs must cover C only and exactly once")
        attribute = ATTRIBUTES[metric["attribute"]]
        classes = np.array(json.loads(metric["classes"]))
        labels_B = np.array([str(getattr(inputs.rows[index], attribute)) for index in indices["B"]])
        if not np.array_equal(classes, np.unique(labels_B)):
            raise ValueError("scored label space differs from B training classes")
        gold = np.array([str(getattr(by_id[row["utterance_id"]], attribute)) for row in group])
        if not np.array_equal(gold, [row["gold_label"] for row in group]):
            raise ValueError("saved gold labels differ from manifest IDs")
        if any(row["speaker_id"] != by_id[row["utterance_id"]].speaker_id or row["classes"] != classes.tolist() for row in group):
            raise ValueError("prediction speakers or class ordering differ from the manifest/model")
        predicted = np.array([row["predicted_label"] for row in group])
        scores = classification_metrics(gold, predicted, classes)
        labels, counts = np.unique(labels_B, return_counts=True)
        majority = np.full(len(group), labels[counts.argmax()])
        if not np.array_equal(majority, [row["majority_predicted_label"] for row in group]):
            raise ValueError("majority baseline did not come from B labels")
        baseline = classification_metrics(gold, majority, classes)
        for name in ("macro_f1", "accuracy"):
            errors.append(abs(metric[name] - scores[name]))
            errors.append(abs(metric[f"majority_{name}"] - baseline[name]))
        errors.append(abs(metric["uniform_chance_accuracy"] - 1 / len(classes)))
        if metric["probe"] == "linear":
            probabilities = np.array([row["probabilities"] for row in group])
            if probabilities.shape != (len(group), len(classes)) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
                raise ValueError("invalid linear-probe probabilities")
            if not np.array_equal(classes[probabilities.argmax(axis=1)], predicted):
                raise ValueError("linear labels disagree with saved probabilities")
            if any(row["score_kind"] != "probabilities" or row["decision_score"] is not None for row in group):
                raise ValueError("linear score representation is inconsistent")
            probability_errors.extend(abs(probabilities.sum(axis=1) - 1).tolist())
        else:
            if any(row["score_kind"] != "decision_function" or row["probabilities"] is not None for row in group):
                raise ValueError("uncalibrated SVM scores must not be published as probabilities")
            decisions = np.array([row["decision_score"] for row in group])
            if not np.isfinite(decisions).all():
                raise ValueError("nonfinite SVM decision scores")
            if np.any(predicted[decisions > 0] != classes[1]) or np.any(predicted[decisions < 0] != classes[0]):
                raise ValueError("SVM labels disagree with nonzero decision-score signs")

    cv_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(output_dir / "cv_results.csv"):
        cv_rows[row["model_id"]].append(row)
    if set(cv_rows) != set(metric_map):
        raise ValueError("missing CV candidate distributions")
    for identity, candidates in cv_rows.items():
        metric = metric_map[identity]
        grid = config["linear_probe"]["c_grid"] if metric["probe"] == "linear" else config["rbf_probe"]["c_grid"]
        if len(candidates) != len(grid) or {float(row["c"]) for row in candidates} != set(grid):
            raise ValueError("probe search differs from the fixed C grid")
        if sum(row["selected"] == "True" for row in candidates) != 1:
            raise ValueError("CV must select exactly one candidate")
        for candidate in candidates:
            values = [float(candidate[f"cv_fold_{fold}_macro_f1"]) for fold in range(len(cv[metric["attribute"]]))]
            errors.extend([
                abs(float(candidate["mean_macro_f1"]) - float(np.mean(values))),
                abs(float(candidate["std_macro_f1"]) - float(np.std(values))),
            ])
        selected = min(candidates, key=lambda row: (-float(row["mean_macro_f1"]), float(row["c"])))
        if selected["selected"] != "True" or float(selected["c"]) != metric["selected_c"] or float(fit_map[identity]["selected_c"]) != metric["selected_c"]:
            raise ValueError("model C does not follow the training-only CV scores/tie rule")
        errors.append(abs(float(selected["mean_macro_f1"]) - float(fit_map[identity]["cv_macro_f1"])))
    integrity = read_csv(output_dir / "model_integrity.csv")
    if len(integrity) != len(metrics) or {row["model_id"] for row in integrity} != set(metric_map):
        raise ValueError("model-integrity checks are incomplete")
    if any(row["unchanged"] != "True" or row["before_hash"] != row["after_hash"] or row["before_hash"] != fit_map[row["model_id"]]["frozen_parameter_hash"] for row in integrity):
        raise ValueError("a model was modified after the fitting phase")
    with np.load(output_dir / "representation_parameters.npz", allow_pickle=False) as parameters:
        if parameters_hash({name: parameters[name] for name in parameters.files}) != before_C["preprocessing_hash"]:
            raise ValueError("A-fitted representation parameters were modified")
        features_A = np.load(inputs.cache / f"layer_{config['layer']:02d}.npy", allow_pickle=False)[indices["A"]].astype(np.float64)
        np.testing.assert_allclose(parameters["scaler_mean"], features_A.mean(axis=0), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(parameters["pca_components"] @ parameters["pca_components"].T, np.eye(config["pca"]["components"]), atol=1e-7, rtol=0)
        for prefix in ("full", "reduced"):
            rank = parameters[f"{prefix}_left"].shape[1]
            np.testing.assert_allclose(parameters[f"{prefix}_right"].T @ parameters[f"{prefix}_left"], np.eye(rank), atol=config["numerics"]["invariant_tolerance"], rtol=0)
    for row in read_csv(output_dir / "transform_diagnostics.csv"):
        a_ids = [inputs.rows[index].utterance_id for index in indices["A"]]
        if row["fit_role"] != "A" or int(row["n_fit"]) != len(a_ids) or row["fit_ids_sha256"] != ids_hash(a_ids):
            raise ValueError("eraser/scaler/PCA fitting did not reference A only")
        if float(row["training_cross_covariance_residual"]) > config["numerics"]["invariant_tolerance"]:
            raise ValueError("training erasure invariants failed")

    speakers_C = np.array([inputs.rows[index].speaker_id for index in indices["C"]])
    clusters, expected_counts = cluster_bootstrap_counts(speakers_C, config["bootstrap"])
    with np.load(output_dir / "bootstrap_clusters.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["speakers"], clusters)
        np.testing.assert_array_equal(saved["counts"], expected_counts)
    bootstrap: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(output_dir / "bootstrap_scores.parquet").to_pylist():
        bootstrap[row["model_id"]].append(row)
    expected_bootstrap_ids = {row["model_id"] for row in metrics if row["attribute"] == "statement"}
    if set(bootstrap) != expected_bootstrap_ids:
        raise ValueError("bootstrap must cover statement only, for both probes in all four arms")
    for identity, records in bootstrap.items():
        records.sort(key=lambda row: row["draw"])
        if [row["draw"] for row in records] != list(range(config["bootstrap"]["draws"])):
            raise ValueError("missing or duplicated bootstrap draws")
        group = predictions[identity]
        classes = np.array(json.loads(metric_map[identity]["classes"]))
        f1, accuracy, sizes = bootstrap_scores(
            np.array([row["gold_label"] for row in group]),
            np.array([row["predicted_label"] for row in group]),
            np.array([row["speaker_id"] for row in group]),
            classes, clusters, expected_counts,
        )
        np.testing.assert_allclose(f1, [row["macro_f1"] for row in records], atol=1e-12, rtol=0)
        np.testing.assert_allclose(accuracy, [row["accuracy"] for row in records], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(sizes, [row["n_examples"] for row in records])
        for name, values in (("macro_f1", f1), ("accuracy", accuracy)):
            expected_upper = float(np.quantile(values, config["bootstrap"]["quantile"], method=config["bootstrap"]["quantile_method"]))
            errors.append(abs(metric_map[identity][f"{name}_upper_95"] - expected_upper))
    decision = validity_decision(metrics, config)
    if decision != json.loads((output_dir / "decision.json").read_text()):
        raise ValueError("published decision does not follow the frozen thresholds and saved scores")
    if not np.isfinite(errors).all() or max(errors) > 1e-12 or max(probability_errors) > 1e-12:
        raise ValueError("validity metric/CI recomputation failed")
    return {
        "status": "pass", "metric_rows": len(metrics),
        "prediction_rows": sum(len(rows) for rows in predictions.values()),
        "cv_candidate_rows": sum(len(rows) for rows in cv_rows.values()),
        "bootstrap_score_rows": sum(len(rows) for rows in bootstrap.values()),
        "maximum_score_or_bound_recomputation_error": max(errors),
        "maximum_probability_sum_error": max(probability_errors),
        "all_24_models_frozen_before_C": True, "all_grouped_roles_and_CV_verified": True,
        "all_post_evaluation_model_hashes_unchanged": True,
        "all_bootstrap_counts_and_scores_recomputed": True,
        "gate_recomputed_without_threshold_changes": True,
    }


def write_validity_report(output_dir: Path, metrics: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    provenance = json.loads((output_dir / "provenance.json").read_text())
    config = provenance["config"]
    lines = [
        "# M4-V1: bounded erasure-validity result", "",
        f"**Decision: {decision['decision']}**", "",
        f"Run `{provenance['experiment_id']}`; code `{provenance['git_commit']}`; "
        f"protocol `{provenance['protocol_commit']}`.", "",
        f"Layer {config['layer']}, existing WavLM/RAVDESS cache, "
        f"{config['pca']['components']} PCA components, seed {config['seed']}. "
        "A/B/C contain 720/360/360 utterances in disjoint matched cells. "
        "All 24 probe models were fitted/tuned on B and frozen before C scoring. "
        "Scaler, PCA and both erasers were fitted on A only.", "",
        "## Statement information on locked C", "",
        "U95 is a one-sided 95% speaker-cluster percentile upper bound, conditional "
        "on these fitted probes and this previously explored corpus. It is not "
        "a universal information-absence certificate.", "",
        "| Arm | Probe | Macro-F1 | Accuracy | F1 U95 | Accuracy U95 |",
        "|---|---|---|---|---|---|",
    ]
    for row in metrics:
        if row["attribute"] == "statement":
            lines.append(
                f"| {row['arm']} | {row['probe']} | {row['macro_f1']:.4f} | "
                f"{row['accuracy']:.4f} | {row['macro_f1_upper_95']:.4f} | {row['accuracy_upper_95']:.4f} |"
            )
    lines += [
        "", "The point and upper-bound ceilings are unchanged at **0.55** for both "
        "probe families and both metrics. The primary candidate is PCA-64 + LEACE; "
        "PCA-only is its incremental reference, not an omitted control.", "",
        "## Collateral information", "",
        "| Comparison | Emotion F1 change | Speaker | Statement | Gender | Intensity | Pass? |",
        "|---|---|---|---|---|---|---|",
    ]
    collateral_rows = []
    lookup = {(row["arm"], row["probe"], row["attribute"]): row for row in metrics}
    for comparison in decision["collateral"]:
        delta = comparison["delta_macro_f1"]
        lines.append(
            f"| {comparison['before_arm']} to {comparison['after_arm']} | "
            + " | ".join(f"{delta[name]:+.4f}" for name in config["attributes"])
            + f" | {comparison['pass']} |"
        )
        for attribute in config["attributes"]:
            before = lookup[(comparison["before_arm"], "linear", attribute)]
            after = lookup[(comparison["after_arm"], "linear", attribute)]
            collateral_rows.append({
                "before_arm": comparison["before_arm"], "after_arm": comparison["after_arm"],
                "attribute": attribute, "before_macro_f1": before["macro_f1"], "after_macro_f1": after["macro_f1"],
                "delta_macro_f1": delta[attribute], "before_accuracy": before["accuracy"],
                "after_accuracy": after["accuracy"], "delta_accuracy": after["accuracy"] - before["accuracy"],
            })
    write_csv(collateral_rows, output_dir / "collateral.csv")
    lines += [
        "", "## Decision and next action", "", decision["next_action"], "",
        decision["claim_boundary"], "",
        "- Both tested probe families must pass; nonlinear recovery is a veto.",
        "- Point estimates below the ceiling with upper bounds above it are inconclusive.",
        "- Compression damage is checked against the original representation as well as PCA-only.",
        "- If PCA alone suppresses the target, that does not establish an incremental LEACE repair.",
        "- No new dimension, layer, seed, method, eraser, causal task head or replication was run.",
        "- These are new validity partitions, not M3 OOD results or unseen-speaker evaluation. "
        "Actor identities overlap across roles, while complete matched cells and utterances do not.",
        "- No literal zero-information, reliance, harmful-use or M5 claim follows from this study.",
        "", "## Provenance and verification", "",
        "[Frozen protocol](protocol.md), [configuration](config.json), [roles](roles.parquet), "
        "[B-only grouped CV](probe_cv.parquet), [provenance](provenance.json), "
        "[all metrics](metrics.csv), [predictions](predictions.parquet), "
        "[CV candidates](cv_results.csv), [bootstrap scores](bootstrap_scores.parquet), "
        "[cluster counts](bootstrap_clusters.npz), [model fits](model_fits.csv), "
        "[integrity](model_integrity.csv), [verification](verification.json), "
        "[decision](decision.json), [checksums](output_sha256.json).",
        "", "Model snapshots reference SVM support rows by B indices and hashes; "
        "cached utterance embeddings are not copied into the result bundle.",
        "", "![Statement validity](statement_validity.png)", "",
        "![Attribute information](attribute_information.png)",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_validity(output_dir: Path, inputs: M3Inputs) -> None:
    verification = verify_validity(output_dir, inputs)
    write_json(verification, output_dir / "verification.json")
    metrics = typed_metrics(output_dir / "metrics.csv")
    decision = json.loads((output_dir / "decision.json").read_text())
    write_validity_report(output_dir, metrics, decision)
    from ..figures.validity import generate_validity_figures

    generate_validity_figures(output_dir)
