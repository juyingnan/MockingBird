"""Predeclared clustered precision bounds and validity-only decisions."""

from __future__ import annotations

from typing import Any

import numpy as np


def cluster_bootstrap_counts(speakers: np.ndarray, settings: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    clusters = np.unique(speakers)
    if len(clusters) < 2 or settings["draws"] < 2:
        raise ValueError("cluster bootstrap requires multiple speakers and draws")
    generator = np.random.default_rng(np.random.SeedSequence(settings["seed_sequence"]))
    samples = generator.integers(0, len(clusters), size=(settings["draws"], len(clusters)))
    counts = np.zeros((settings["draws"], len(clusters)), dtype=np.int64)
    np.add.at(counts, (np.arange(settings["draws"])[:, None], samples), 1)
    return clusters, counts


def bootstrap_scores(
    gold: np.ndarray, predicted: np.ndarray, speakers: np.ndarray,
    classes: np.ndarray, clusters: np.ndarray, counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (len(gold) == len(predicted) == len(speakers)) or len(gold) == 0:
        raise ValueError("bootstrap observations must be nonempty and aligned")
    if counts.ndim != 2 or counts.shape[1] != len(clusters) or len(counts) == 0:
        raise ValueError("bootstrap cluster-count matrix has the wrong shape")
    if counts.dtype.kind not in "iu" or np.any(counts < 0) or not np.all(counts.sum(axis=1) == len(clusters)):
        raise ValueError("bootstrap draws must resample exactly the declared number of clusters")
    if set(speakers) != set(clusters) or not set(gold).issubset(set(classes)) or not set(predicted).issubset(set(classes)):
        raise ValueError("bootstrap labels or clusters differ from the declared evaluation space")
    label_index = {label: index for index, label in enumerate(classes)}
    cluster_index = {speaker: index for index, speaker in enumerate(clusters)}
    matrices = np.zeros((len(clusters), len(classes), len(classes)), dtype=np.int64)
    for truth, prediction, speaker in zip(gold, predicted, speakers):
        matrices[cluster_index[speaker], label_index[truth], label_index[prediction]] += 1
    draws = (counts @ matrices.reshape(len(clusters), -1)).reshape(-1, len(classes), len(classes))
    true_positive = np.diagonal(draws, axis1=1, axis2=2)
    denominators = draws.sum(axis=1) + draws.sum(axis=2)
    f1 = np.divide(
        2 * true_positive, denominators,
        out=np.zeros_like(true_positive, dtype=np.float64), where=denominators != 0,
    ).mean(axis=1)
    sizes = draws.sum(axis=(1, 2))
    if np.any(sizes == 0):
        raise ValueError("bootstrap produced an empty evaluation draw")
    accuracy = true_positive.sum(axis=1) / sizes
    return f1, accuracy, sizes


def validity_decision(metrics: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    lookup = {(row["arm"], row["probe"], row["attribute"]): row for row in metrics}
    expected = {
        (arm, "linear", attribute)
        for arm in config["arms"] for attribute in config["attributes"]
    } | {(arm, "rbf_svm", "statement") for arm in config["arms"]}
    if len(metrics) != len(expected) or set(lookup) != expected:
        raise ValueError("validity decision requires every predeclared arm/probe/attribute")
    for row in metrics:
        fields = ["macro_f1", "accuracy"]
        if row["attribute"] == "statement":
            fields += ["macro_f1_upper_95", "accuracy_upper_95"]
        if any(not np.isfinite(row[name]) or not 0 <= row[name] <= 1 for name in fields):
            raise ValueError("validity decision requires finite scores and bounds in [0, 1]")
    policy = config["validity"]
    primary = config["primary_arm"]
    reference = config["incremental_reference"]
    probe_checks = {}
    for probe in ("linear", "rbf_svm"):
        original = lookup[("original", probe, "statement")]
        before = lookup[(reference, probe, "statement")]
        after = lookup[(primary, probe, "statement")]
        probe_checks[probe] = {
            "original_competent": original["macro_f1"] >= policy["before_min_macro_f1"],
            "pca_only_competent": before["macro_f1"] >= policy["before_min_macro_f1"],
            "target_f1_drop": before["macro_f1"] - after["macro_f1"],
            "drop_pass": before["macro_f1"] - after["macro_f1"] >= policy["minimum_target_f1_drop"],
            "macro_f1": after["macro_f1"], "accuracy": after["accuracy"],
            "macro_f1_upper_95": after["macro_f1_upper_95"],
            "accuracy_upper_95": after["accuracy_upper_95"],
            "point_pass": (
                after["macro_f1"] <= policy["after_max_macro_f1"]
                and after["accuracy"] <= policy["after_max_accuracy"]
            ),
            "precision_pass": (
                after["macro_f1_upper_95"] <= policy["upper_confidence_ceiling"]
                and after["accuracy_upper_95"] <= policy["upper_confidence_ceiling"]
            ),
        }
    collateral = []
    for before_arm, after_arm in policy["collateral_comparisons"]:
        deltas = {
            attribute: (
                lookup[(after_arm, "linear", attribute)]["macro_f1"]
                - lookup[(before_arm, "linear", attribute)]["macro_f1"]
            )
            for attribute in config["attributes"]
        }
        losses = [
            name for name in ("speaker", "gender", "intensity")
            if -deltas[name] > policy["large_unrelated_f1_drop"]
        ]
        collateral.append({
            "before_arm": before_arm, "after_arm": after_arm,
            "delta_macro_f1": deltas, "large_unrelated_losses": losses,
            "pass": (
                -deltas["emotion"] <= policy["maximum_emotion_f1_drop"]
                and len(losses) <= policy["maximum_large_unrelated_losses"]
            ),
        })
    pca_alone_absence = all(
        lookup[(reference, probe, "statement")][metric] <= ceiling
        for probe in ("linear", "rbf_svm")
        for metric, ceiling in (
            ("macro_f1", policy["after_max_macro_f1"]), ("accuracy", policy["after_max_accuracy"]),
            ("macro_f1_upper_95", policy["upper_confidence_ceiling"]),
            ("accuracy_upper_95", policy["upper_confidence_ceiling"]),
        )
    )
    if not all(item["point_pass"] for item in probe_checks.values()):
        decision = "FAIL_RESIDUAL_DECODABILITY"
    elif not all(item["pass"] for item in collateral):
        decision = "FAIL_COLLATERAL_DAMAGE"
    elif not all(item["original_competent"] for item in probe_checks.values()):
        decision = "INCONCLUSIVE_PROBE_COMPETENCE"
    elif pca_alone_absence:
        decision = "COMPRESSION_ONLY_NOT_LEACE_VALIDATION"
    elif not all(item["pca_only_competent"] and item["drop_pass"] for item in probe_checks.values()):
        decision = "FAIL_INCREMENTAL_TARGET_VALIDITY"
    elif not all(item["precision_pass"] for item in probe_checks.values()):
        decision = "INCONCLUSIVE_PRECISION"
    else:
        decision = "PASS_VALIDITY_ONLY"
    return {
        "decision": decision,
        "primary_arm": primary,
        "probe_checks": probe_checks,
        "collateral": collateral,
        "pca_alone_meets_absence_screen": pca_alone_absence,
        "nonlinear_veto": probe_checks["linear"]["point_pass"] and not probe_checks["rbf_svm"]["point_pass"],
        "next_action": (
            "Human review of this bounded validity result; no automatic M4A/M4B or M5."
            if decision == "PASS_VALIDITY_ONLY" else
            "Preserve this bounded result and consolidate M4 failure/limitation analysis; no additional dimension, layer, seed or method search."
        ),
        "claim_boundary": (
            "Conditional decodability evidence for the specified probes on an explored corpus; "
            "not zero mutual information, original-M3-head reliance, harmful use, or unseen-speaker replication."
        ),
    }
