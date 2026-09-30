import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mockingbird2027.analysis import validity
from mockingbird2027.analysis.erasure import LEACE
from mockingbird2027.analysis.m3 import ATTRIBUTES, M3Inputs, read_csv, sha256_file, write_csv
from mockingbird2027.analysis.probes import ProbeConfig
from mockingbird2027.analysis.validity_reporting import verify_validity
from mockingbird2027.analysis.validity_splits import build_roles, build_probe_cv, frozen_probe_cv, role_indices
from mockingbird2027.cli import main
from mockingbird2027.data.manifest import write_json, write_parquet
from test_m3 import _rows


@pytest.fixture
def synthetic_study(tmp_path, monkeypatch):
    config_path = Path(__file__).parents[1] / "configs" / "experiments" / "m4_validity_v1.json"
    config = json.loads(config_path.read_text())
    monkeypatch.chdir(tmp_path)
    rows = _rows(complete=True)
    roles = build_roles(rows, config["seed"])
    cv_records = build_probe_cv(rows, roles, ProbeConfig())
    config["pca"]["components"] = 3
    config["bootstrap"]["draws"] = 31
    config.update({
        "protocol_file": "protocol.md", "roles_file": "roles.parquet",
        "cv_file": "cv.parquet", "cache_fingerprint": "synthetic",
    })
    write_parquet(roles, Path(config["roles_file"]))
    write_parquet(cv_records, Path(config["cv_file"]))
    config["roles_sha256"] = sha256_file(Path(config["roles_file"]))
    config["cv_sha256"] = sha256_file(Path(config["cv_file"]))
    Path("protocol.md").write_text("Synthetic validity-only test protocol.\n")
    features = np.random.default_rng(73).normal(size=(len(rows), 12))
    features[:, 0] += np.array([row.statement_id == 2 for row in rows]) * 3
    features[:, 1] += np.array([row.gender == "male" for row in rows]) * 2
    features[:, 2] += np.array([row.intensity == "strong" for row in rows])
    cache = Path("cache")
    cache.mkdir()
    np.save(cache / "layer_05.npy", features.astype(np.float16), allow_pickle=False)
    inputs = M3Inputs(rows, {}, cache, {
        "fingerprint": "synthetic", "hidden_size": 12,
        "model_id": "synthetic", "resolved_revision": "synthetic",
    }, {str(cache / "layer_05.npy"): sha256_file(cache / "layer_05.npy")})
    source = {
        "roles": roles, "cv_records": cv_records,
        "cv": frozen_probe_cv(rows, roles, cv_records, 3),
        "indices": role_indices(rows, roles), "input_sha256": inputs.hashes,
    }
    path = Path("config.json")
    write_json(config, path)
    return config, inputs, source, path


def test_full_validity_run_enforces_fitting_roles_and_freezes_models_before_C(synthetic_study, monkeypatch):
    config, inputs, source, path = synthetic_study
    monkeypatch.setattr(validity, "load_validity_inputs", lambda path: (config, inputs, source))
    monkeypatch.setattr(validity, "git_provenance", lambda: {"git_commit": "a" * 40, "git_branch": "synthetic"})
    raw = np.load(inputs.cache / "layer_05.npy", allow_pickle=False).astype(np.float64)
    observed = {"A": 0, "B": 0}
    fit_arms = validity.fit_representation_arms
    active_arms = []

    def check_A(features, labels, settings):
        observed["A"] += 1
        np.testing.assert_array_equal(features, raw[source["indices"]["A"]])
        np.testing.assert_array_equal(labels, [str(inputs.rows[index].statement_id) for index in source["indices"]["A"]])
        result = fit_arms(features, labels, settings)
        active_arms.append(result)
        return result

    fit_probe = validity.fit_validity_probe

    def check_B(features, labels, groups, probe, settings, cv):
        observed["B"] += 1
        b = source["indices"]["B"]
        expected = active_arms[0].transform(raw[b])
        assert any(np.array_equal(features, matrix) for matrix in expected.values())
        assert len(features) == 360
        assert set(groups) == {
            f"{inputs.rows[index].speaker_id}|{inputs.rows[index].emotion}|{inputs.rows[index].intensity}"
            for index in b
        }
        assert any(
            np.array_equal(labels, [str(getattr(inputs.rows[index], field)) for index in b])
            for field in ATTRIBUTES.values()
        )
        result = fit_probe(features, labels, groups, probe, settings, cv)
        if probe == "rbf_svm":
            assert result.best_estimator_.named_steps["classifier"].gamma == 1 / features.shape[1]
            assert not hasattr(result.best_estimator_.named_steps["classifier"], "predict_proba")
        return result

    evaluate_C = validity.ValidityRunner.evaluate_C

    def frozen_evaluation(runner):
        assert observed == {"A": 1, "B": 24}

        def forbidden(*args, **kwargs):
            raise AssertionError("no representation or probe may fit after C scoring begins")

        with monkeypatch.context() as guard:
            for estimator in (PCA, StandardScaler, LEACE, LogisticRegression, SVC):
                guard.setattr(estimator, "fit", forbidden)
            return evaluate_C(runner)

    monkeypatch.setattr(validity, "fit_representation_arms", check_A)
    monkeypatch.setattr(validity, "fit_validity_probe", check_B)
    monkeypatch.setattr(validity.ValidityRunner, "evaluate_C", frozen_evaluation)
    assert main(["validity", "--config", str(path), "--validate-only"]) == 0
    assert observed == {"A": 0, "B": 0}
    output = Path("results") / "synthetic-validity"
    assert main(["validity", "--config", str(path), "--output-dir", str(output)]) == 0
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "complete"
    assert status["downstream_experiments_started"] is False
    verification = json.loads((output / "verification.json").read_text())
    assert verification["metric_rows"] == 24
    assert verification["prediction_rows"] == 8640
    assert verification["cv_candidate_rows"] == 72
    assert verification["bootstrap_score_rows"] == 8 * 31
    assert verification["maximum_score_or_bound_recomputation_error"] < 1e-12
    for filename, digest in json.loads((output / "output_sha256.json").read_text()).items():
        assert sha256_file(output / filename) == digest
    predictions = pq.read_table(output / "predictions.parquet").to_pylist()
    svm = [row for row in predictions if "rbf_svm" in row["model_id"]]
    assert len(svm) == 1440
    assert all(row["probabilities"] is None and row["decision_score"] is not None for row in svm)
    for model_path in (output / "models").glob("*rbf_svm*.npz"):
        with np.load(model_path, allow_pickle=False) as parameters:
            assert "support_indices" in parameters.files
            assert "support_vectors_sha256" in parameters.files
            assert "support_vectors" not in parameters.files
    assert (output / "statement_validity.png").read_bytes().startswith(b"\x89PNG")
    with pytest.raises(FileExistsError):
        validity.run_validity(path, output)
    metrics = read_csv(output / "metrics.csv")
    metrics[0]["macro_f1"] = "0.123456789"
    write_csv(metrics, output / "metrics.csv")
    with pytest.raises(ValueError, match="recomputation|decision"):
        verify_validity(output, inputs)


def test_C_evaluation_rejects_unfitted_and_partial_model_sets(synthetic_study):
    config, inputs, source, _ = synthetic_study
    runner = validity.ValidityRunner(config, inputs, source, Path("unused"))
    with pytest.raises(RuntimeError, match="all models"):
        runner.evaluate_C()


def test_changed_frozen_protocol_is_rejected_before_fitting(tmp_path, monkeypatch):
    original = json.loads((Path(__file__).parents[1] / "configs" / "experiments" / "m4_validity_v1.json").read_text())
    altered = copy.deepcopy(original)
    altered["pca"]["components"] = 128
    monkeypatch.chdir(tmp_path)
    write_json(altered, Path("config.json"))
    monkeypatch.setattr(validity.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(original)))
    with pytest.raises(ValueError, match="pre-fitting commit"):
        validity.load_validity_inputs(Path("config.json"))


def test_numerical_failure_is_not_a_scientific_gate_result(synthetic_study, monkeypatch):
    config, inputs, source, path = synthetic_study
    monkeypatch.setattr(validity, "load_validity_inputs", lambda path: (config, inputs, source))
    monkeypatch.setattr(validity, "git_provenance", lambda: {"git_commit": "a" * 40, "git_branch": "synthetic"})

    def fail(*args, **kwargs):
        raise ValueError("synthetic fitting failure")

    monkeypatch.setattr(validity.ValidityRunner, "fit_all", fail)
    output = Path("results") / "failed"
    with pytest.raises(ValueError, match="synthetic fitting failure"):
        validity.run_validity(path, output)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed" and status["exit_code"] == 1
    assert not (output / "decision.json").exists()
