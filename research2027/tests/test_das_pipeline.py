import copy
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mockingbird2027.analysis import das_experiment as experiment
from mockingbird2027.analysis.das import build_discovery_split
from mockingbird2027.analysis.m3 import sha256_file
from mockingbird2027.analysis.probes import ProbeConfig
from mockingbird2027.analysis.validity_splits import build_probe_cv, build_roles, frozen_probe_cv
from mockingbird2027.cli import main
from mockingbird2027.data.manifest import write_json, write_parquet
from test_m3 import _rows


@pytest.fixture
def synthetic_das(tmp_path, monkeypatch):
    config = json.loads((Path(__file__).parents[1] / "configs" / "experiments" / "das_statement.json").read_text())
    config["readout"].update({"hidden_size": 8, "steps": 6, "evaluation_interval": 2, "minimum_validation_accuracy": 0})
    config["das"].update({"dimensions": [1, 3], "seeds": [17], "steps": 6, "evaluation_interval": 2})
    config["linear_probe"]["c_grid"] = [0.1]
    config["rbf_probe"]["c_grid"] = [1.0]
    config["random_controls"].update({"pool_size": 4, "count": 1})
    config["bootstrap"]["draws"] = 31
    config.update({
        "cache": "cache", "protocol_file": "protocol.md",
        "split_file": "split.json", "cv_file": "cv.parquet",
    })
    monkeypatch.chdir(tmp_path)
    rows = _rows(complete=True)
    roles = build_roles(rows, config["role_seed"])
    split = build_discovery_split(rows, roles, config["seed"])
    cv_records = build_probe_cv(rows, roles, ProbeConfig())
    config["cv_sha256"] = ""
    write_parquet(cv_records, Path("cv.parquet"))
    write_json({"seed": config["seed"], "assignments": split}, Path("split.json"))
    Path("protocol.md").write_text("Synthetic test only; not scientific evidence.\n")
    cache = Path("cache")
    cache.mkdir()
    values = np.random.default_rng(21).normal(size=(len(rows), 12))
    values[:, 0] += 4 * np.array([row.statement_id - 1.5 for row in rows])
    values[:, 1] += np.array([row.gender == "female" for row in rows])
    np.save(cache / "layer_05.npy", values.astype(np.float16))
    config_path = Path("config.json")
    write_json(config, config_path)
    hashes = {str(cache / "layer_05.npy"): sha256_file(cache / "layer_05.npy")}
    inputs = experiment.DASInputs(
        rows, config, {"hidden_size": 12, "fingerprint": "synthetic"}, split,
        frozen_probe_cv(rows, roles, cv_records, 3), hashes,
    )
    monkeypatch.setattr(experiment, "git_provenance", lambda: {"git_commit": "a" * 40, "git_branch": "synthetic"})
    monkeypatch.setattr(experiment, "require_committed_freeze", lambda output: None)
    monkeypatch.setattr(experiment, "scientific_hashes", lambda path, config: {
        str(path): sha256_file(path), "protocol.md": sha256_file(Path("protocol.md")),
    })
    return config_path, inputs


def test_feature_access_rejects_C_and_B_until_their_declared_freezes(synthetic_das):
    _, inputs = synthetic_das
    root = Path("guard")
    root.mkdir()
    access = experiment.FeatureAccess(inputs, root)
    assert access.read("A_train").shape == (480, 12)
    with pytest.raises(RuntimeError, match="C is locked"):
        access.read("C")
    with pytest.raises(RuntimeError, match="candidate is frozen"):
        access.read("B")
    write_json({"frozen": True}, root / "candidate_frozen.json")
    assert access.read("B").shape == (360, 12)
    write_json({
        "model_hashes": {"first": "abc"}, "model_count": 1, "expected_model_count": 18,
        "transform_hash": "xyz", "C_access_count": 0, "training_artifact_sha256": {},
    }, root / "fit_complete.json")
    with pytest.raises(RuntimeError, match="complete, unchanged"):
        access.open_C({"first": "abc"}, "xyz", "a" * 40)
    assert not (root / "C_opened.json").exists()
    frozen = experiment.read_json(root / "fit_complete.json")
    frozen["expected_model_count"] = 1
    write_json(frozen, root / "fit_complete.json")
    access.open_C({"first": "abc"}, "xyz", "a" * 40)
    assert access.read("C").shape == (360, 12)
    with pytest.raises(RuntimeError, match="second C read"):
        access.read("C")
    with pytest.raises(RuntimeError, match="no fitting-role access"):
        access.read("A_train")


def test_full_staged_run_never_uses_C_for_fitting_and_recomputes_saved_predictions(synthetic_das, monkeypatch):
    config_path, inputs = synthetic_das
    output = Path("results") / "synthetic-das"
    reads = []
    original_read = experiment.FeatureAccess.read
    fitting = {"active": True}

    def monitored_read(access, role):
        reads.append(role)
        if fitting["active"] and role == "C":
            raise AssertionError("scientific C must not be accessed during fitting")
        return original_read(access, role)

    monkeypatch.setattr(experiment.FeatureAccess, "read", monitored_read)
    experiment.fit_study(config_path, output, inputs)
    assert reads == ["A_train", "A_val", "B"]
    assert not (output / "C_opened.json").exists()
    frozen = experiment.read_json(output / "fit_complete.json")
    assert frozen["C_access_count"] == 0 and frozen["model_count"] == 18
    assert experiment.read_json(output / "candidate_frozen.json")["selected"]["selected"] is True
    assert experiment.read_json(output / "status.json")["status"] == "frozen_awaiting_C"
    fitting["active"] = False

    def forbidden(*args, **kwargs):
        raise AssertionError("no model fitting is permitted during locked C evaluation")

    with monkeypatch.context() as guard:
        for estimator in (LogisticRegression, StandardScaler, SVC):
            guard.setattr(estimator, "fit", forbidden)
        experiment.evaluate_study(config_path, output, inputs)
    assert reads == ["A_train", "A_val", "B", "C"]
    result = experiment.verify_study(output, inputs)
    assert result["metric_rows"] == 18
    assert result["prediction_rows"] == 18 * 360
    assert result["bootstrap_score_rows"] == 6 * 31
    assert result["maximum_metric_or_bound_recomputation_error"] < 1e-12
    assert experiment.read_json(output / "status.json")["status"] == "complete"
    assert (output / "candidate_selection.png").read_bytes().startswith(b"\x89PNG")
    assert (output / "locked_validity.png").read_bytes().startswith(b"\x89PNG")
    with pytest.raises(RuntimeError, match="already opened"):
        experiment.evaluate_study(config_path, output, inputs)
    with pytest.raises(FileExistsError):
        experiment.fit_study(config_path, output, inputs)

    stored = experiment.read_json(output / "metrics.json")
    corrupted = copy.deepcopy(stored)
    corrupted[0]["macro_f1"] = 0.123456789
    write_json(corrupted, output / "metrics.json")
    with pytest.raises(ValueError, match="recomputation"):
        experiment.verify_study(output, inputs, check_outputs=False)
    write_json(stored, output / "metrics.json")
    predictions = pq.read_table(output / "predictions.parquet").to_pylist()
    predictions[0]["utterance_id"] = inputs.role_rows("A_train")[0].utterance_id
    write_parquet(predictions, output / "predictions.parquet")
    with pytest.raises(ValueError, match="cover C"):
        experiment.verify_study(output, inputs, check_outputs=False)


def test_changed_frozen_artifact_blocks_C_before_any_read(synthetic_das):
    config_path, inputs = synthetic_das
    output = Path("results") / "synthetic-tamper"
    experiment.fit_study(config_path, output, inputs)
    with (output / "candidate_selection.csv").open("a") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="immutable artifact"):
        experiment.evaluate_study(config_path, output, inputs)
    assert not (output / "C_opened.json").exists()


def test_cli_exposes_distinct_prepare_fit_evaluate_verify_phases(monkeypatch):
    calls = []
    monkeypatch.setattr(experiment, "run_das", lambda phase, config, output: calls.append((phase, config, output)))
    assert main(["das", "prepare"]) == 0
    assert main(["das", "verify", "--output-dir", "results\\example"]) == 0
    assert [row[0] for row in calls] == ["prepare", "verify"]
