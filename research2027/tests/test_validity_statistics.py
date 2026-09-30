import copy
import json
from pathlib import Path

import numpy as np
import pytest

from mockingbird2027.analysis.probes import classification_metrics
from mockingbird2027.analysis.validity_statistics import (
    bootstrap_scores, cluster_bootstrap_counts, validity_decision,
)


def test_cluster_bootstrap_matches_explicit_repeated_observations():
    speakers = np.repeat(["a", "b", "c", "d"], 4)
    gold = np.tile(["1", "1", "2", "2"], 4)
    predicted = np.array(["1", "1", "2", "1", "1", "2", "2", "2", "2", "1", "2", "2", "1", "1", "1", "2"])
    settings = {"draws": 27, "seed_sequence": [20260930, 4201]}
    clusters, counts = cluster_bootstrap_counts(speakers, settings)
    again, repeated = cluster_bootstrap_counts(speakers, settings)
    np.testing.assert_array_equal(clusters, again)
    np.testing.assert_array_equal(counts, repeated)
    assert np.all(counts.sum(axis=1) == 4)
    f1, accuracy, sizes = bootstrap_scores(gold, predicted, speakers, np.array(["1", "2"]), clusters, counts)
    for draw, weights in enumerate(counts):
        selected = np.concatenate([
            np.tile(np.flatnonzero(speakers == speaker), int(weight))
            for speaker, weight in zip(clusters, weights)
        ])
        expected = classification_metrics(gold[selected], predicted[selected], np.array(["1", "2"]))
        assert f1[draw] == pytest.approx(expected["macro_f1"])
        assert accuracy[draw] == pytest.approx(expected["accuracy"])
        assert sizes[draw] == len(selected)


def test_bootstrap_preserves_balanced_chance_and_rejects_invalid_weights():
    speakers = np.repeat(["a", "b"], 4)
    gold = np.tile(["1", "1", "2", "2"], 2)
    predicted = np.full(len(gold), "1")
    clusters, counts = cluster_bootstrap_counts(speakers, {"draws": 20, "seed_sequence": [3]})
    f1, accuracy, _ = bootstrap_scores(gold, predicted, speakers, np.array(["1", "2"]), clusters, counts)
    np.testing.assert_allclose(f1, 1 / 3)
    np.testing.assert_allclose(accuracy, 0.5)
    with pytest.raises(ValueError, match="resample exactly"):
        bootstrap_scores(gold, predicted, speakers, np.array(["1", "2"]), clusters, counts + 1)


@pytest.fixture
def gate_inputs():
    config = json.loads((Path(__file__).parents[1] / "configs" / "experiments" / "m4_validity_v1.json").read_text())
    metrics = []
    for arm in config["arms"]:
        for probe, attribute in [("linear", name) for name in config["attributes"]] + [("rbf_svm", "statement")]:
            value = 0.9 if attribute == "statement" else 0.8
            if arm == "pca64_leace" and attribute == "statement":
                value = 0.5
            metrics.append({
                "arm": arm, "probe": probe, "attribute": attribute,
                "macro_f1": value, "accuracy": value,
                "macro_f1_upper_95": min(1, value + 0.02) if attribute == "statement" else None,
                "accuracy_upper_95": min(1, value + 0.02) if attribute == "statement" else None,
            })
    return config, metrics


def target_row(metrics, probe="linear", attribute="statement", arm="pca64_leace"):
    return next(row for row in metrics if (row["arm"], row["probe"], row["attribute"]) == (arm, probe, attribute))


def test_validity_requires_both_probe_families_and_preserves_the_ceiling(gate_inputs):
    config, metrics = gate_inputs
    assert validity_decision(metrics, config)["decision"] == "PASS_VALIDITY_ONLY"
    target_row(metrics, "rbf_svm")["macro_f1"] = 0.550001
    result = validity_decision(metrics, config)
    assert result["decision"] == "FAIL_RESIDUAL_DECODABILITY"
    assert result["nonlinear_veto"]


def test_point_pass_with_wide_upper_bound_is_inconclusive(gate_inputs):
    config, metrics = gate_inputs
    target_row(metrics)["accuracy_upper_95"] = 0.550001
    assert validity_decision(metrics, config)["decision"] == "INCONCLUSIVE_PRECISION"
    target_row(metrics)["accuracy_upper_95"] = 0.55
    assert validity_decision(metrics, config)["decision"] == "PASS_VALIDITY_ONLY"


def test_compression_damage_cannot_be_hidden_behind_a_pca_only_reference(gate_inputs):
    config, metrics = gate_inputs
    target_row(metrics, attribute="emotion", arm="pca64")["macro_f1"] = 0.69
    target_row(metrics, attribute="emotion")["macro_f1"] = 0.69
    result = validity_decision(metrics, config)
    assert result["decision"] == "FAIL_COLLATERAL_DAMAGE"
    assert not result["collateral"][0]["pass"]
    assert not result["collateral"][2]["pass"]


def test_pca_alone_absence_is_not_credited_to_leace(gate_inputs):
    config, metrics = gate_inputs
    for probe in ("linear", "rbf_svm"):
        row = target_row(metrics, probe, arm="pca64")
        row.update({"macro_f1": 0.5, "accuracy": 0.5, "macro_f1_upper_95": 0.53, "accuracy_upper_95": 0.53})
    result = validity_decision(metrics, config)
    assert result["decision"] == "COMPRESSION_ONLY_NOT_LEACE_VALIDATION"
    assert result["pca_alone_meets_absence_screen"]


def test_missing_or_nonfinite_scores_fail_loudly(gate_inputs):
    config, metrics = gate_inputs
    with pytest.raises(ValueError, match="every predeclared"):
        validity_decision(metrics[:-1], config)
    invalid = copy.deepcopy(metrics)
    invalid[0]["macro_f1"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validity_decision(invalid, config)
