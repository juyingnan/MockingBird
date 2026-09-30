import copy
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from mockingbird2027.analysis import m4
from mockingbird2027.analysis.m3 import (
    ATTRIBUTES, LEXICAL, REFERENCE, M3Inputs, persisted_partitions,
    read_csv, sha256_file, write_csv,
)
from mockingbird2027.analysis.m4_reporting import verify_m4
from mockingbird2027.analysis.probes import ProbeConfig, fit_fixed_probe
from mockingbird2027.cli import main
from mockingbird2027.data.manifest import ManifestRow, write_json, write_parquet
from mockingbird2027.data.splits import build_ravdess_splits


@pytest.fixture
def protocol():
    path = Path(__file__).parents[1] / "configs" / "experiments" / "m4_wavlm_ravdess.json"
    return json.loads(path.read_text())


def test_layer_selection_is_m3_only_and_ties_prefer_lower_index():
    records = []
    for layer, reference, lexical_gap, speaker_gap in (
        (1, 0.8, 0.1, 0.3), (5, 0.9, 0.1, 0.2), (12, 0.9, 0.4, 0.3)
    ):
        records.append({"layer": layer, "split_name": REFERENCE, "macro_f1": reference})
        for name in LEXICAL:
            records.append({"layer": layer, "split_name": name, "macro_f1_gap": lexical_gap})
        for fold in range(6):
            records.append({"layer": layer, "split_name": "speaker_shift", "fold": fold, "macro_f1_gap": speaker_gap})
    selection = m4.selected_layers(records)
    assert selection["task_anchor"] == 5
    assert selection["selected_layers"] == {"statement": [5, 12], "speaker": [1, 5]}


def test_validity_screen_requires_removal_collateral_and_matched_controls(protocol):
    original = {name: {"macro_f1": 0.9, "accuracy": 0.9} for name in ATTRIBUTES}
    erased = copy.deepcopy(original)
    erased["statement"] = {"macro_f1": 0.5, "accuracy": 0.5}
    assert m4.validity_decision(original, erased, "statement", 2, protocol, True)["valid"]
    assert not m4.validity_decision(original, erased, "statement", 2, protocol, False)["valid"]
    erased["statement"] = {"macro_f1": 0.8, "accuracy": 0.8}
    assert not m4.validity_decision(original, erased, "statement", 2, protocol, True)["target_removal_valid"]
    erased["statement"] = {"macro_f1": 0.5, "accuracy": 0.5}
    erased["emotion"] = {"macro_f1": 0.7, "accuracy": 0.7}
    assert not m4.validity_decision(original, erased, "statement", 2, protocol, True)["collateral_valid"]
    erased = copy.deepcopy(original)
    erased["speaker"] = {"macro_f1": 0.25, "accuracy": 0.25}
    erased["gender"] = {"macro_f1": 0.5, "accuracy": 0.5}
    assert m4.validity_decision(original, erased, "speaker", 4, protocol, True)["valid"]
    erased["statement"] = {"macro_f1": 0.5, "accuracy": 0.5}
    erased["intensity"] = {"macro_f1": 0.5, "accuracy": 0.5}
    assert not m4.validity_decision(original, erased, "speaker", 4, protocol, True)["collateral_valid"]


@pytest.fixture
def synthetic_m4(tmp_path, monkeypatch, protocol):
    monkeypatch.chdir(tmp_path)
    rows = []
    for speaker in range(1, 5):
        for emotion in ("happy", "sad"):
            for statement in (1, 2):
                for repetition in (1, 2):
                    for intensity in ("normal", "strong"):
                        identity = f"synthetic:{speaker}:{emotion}:{statement}:{repetition}:{intensity}"
                        rows.append(ManifestRow(
                            identity, "ravdess", identity + ".wav", emotion,
                            f"actor_{speaker:02d}", "male" if speaker % 2 else "female",
                            statement, repetition, intensity, 1.0, 16000,
                        ))
    rows.sort(key=lambda row: row.utterance_id)
    assignments = build_ravdess_splits(rows, speaker_folds=2)
    partitions = persisted_partitions(rows, assignments, speaker_folds=2)
    encoded = []
    for attribute in ATTRIBUTES.values():
        labels = np.array([str(getattr(row, attribute)) for row in rows])
        _, inverse = np.unique(labels, return_inverse=True)
        encoded.append(np.eye(len(np.unique(labels)))[inverse])
    generator = np.random.default_rng(32)
    features = np.column_stack(encoded)
    features += generator.normal(scale=0.25, size=features.shape)
    cache = Path("cache")
    cache.mkdir()
    np.save(cache / "layer_00.npy", features.astype(np.float16), allow_pickle=False)
    features = np.load(cache / "layer_00.npy", allow_pickle=False).astype(np.float64)
    inputs = M3Inputs(
        rows, partitions, cache,
        {"fingerprint": "synthetic", "resolved_revision": "test"},
        {str(cache / "layer_00.npy"): sha256_file(cache / "layer_00.npy")},
    )
    config = copy.deepcopy(protocol)
    config.update({
        "m3_run": "m3", "selected_layers": {"statement": [0], "speaker": [0]},
        "methods": ["mp"], "protocol_file": "protocol.md",
    })
    config["controls"]["draws"] = 2
    probe_config = ProbeConfig(c_grid=(0.1,))
    m3_run = Path("m3")
    m3_run.mkdir()
    generalization = []
    predictions = []
    emotion = np.array([row.emotion for row in rows])
    for (name, fold), (train, test) in partitions.items():
        if name not in (REFERENCE, *LEXICAL, "speaker_shift"):
            continue
        head = fit_fixed_probe(features[train], emotion[train], probe_config, c=0.1)
        generalization.append({"layer": 0, "split_name": name, "fold": fold, "selected_c": 0.1})
        predicted = head.predict(features[test])
        probabilities = head.predict_proba(features[test])
        for position, index in enumerate(test):
            predictions.append({
                "layer": 0, "split_name": name, "fold": fold, "task": "emotion_generalization",
                "utterance_id": rows[index].utterance_id, "gold_label": rows[index].emotion,
                "predicted_label": str(predicted[position]), "probabilities": probabilities[position].tolist(),
            })
    write_csv(generalization, m3_run / "generalization.csv")
    write_parquet(predictions, m3_run / "predictions.parquet")
    config_path = Path("config.json")
    write_json(config, config_path)
    Path("protocol.md").write_text("Synthetic test protocol.\n")
    source = {
        "selection": {"selected_layers": config["selected_layers"]},
        "m3_provenance": {"config": {"probe": {
            "c_grid": [0.1], "cv_folds": 3, "seed": 20260930,
            "max_iter": 2000, "tol": 1e-4,
        }}},
        "input_sha256": inputs.hashes,
    }
    return config, inputs, source, config_path


def test_lexical_conditions_are_explicitly_unidentifiable_without_fitting(synthetic_m4):
    config, inputs, _, _ = synthetic_m4
    conditions = m4.condition_plan(inputs, config)
    lexical = [row for row in conditions if row["split_name"] in LEXICAL]
    assert len(lexical) == 2
    assert all(row["status"] == "blocked_unidentifiable" and row["training_target_classes"] == 1 for row in lexical)
    speaker = [row for row in conditions if row["target"] == "speaker"]
    assert len(speaker) == 3
    assert all(row["status"] == "pending" for row in speaker)


def test_fresh_probes_only_see_the_disjoint_diagnostic_pool(synthetic_m4, monkeypatch):
    config, inputs, source, _ = synthetic_m4
    condition = m4.condition_plan(inputs, config)[0]
    outer_train, test = inputs.partitions[(REFERENCE, 0)]
    results = m4.ConditionResults(condition, inputs)
    features = np.column_stack((test, np.sin(test), np.cos(test))).astype(float)
    observed = []
    original_fit = m4.fit_probe

    def inspect_fit(values, labels, probe_config):
        observed.append(set(values[:, 0].astype(int)))
        assert set(values[:, 0].astype(int)).issubset(set(test))
        assert set(values[:, 0].astype(int)).isdisjoint(set(outer_train))
        return original_fit(values, labels, probe_config)

    monkeypatch.setattr(m4, "fit_probe", inspect_fit)
    audit = m4.FreshAudit(features, test, results, ProbeConfig(c_grid=(0.1,)), ["statement"])
    assert len(observed) == 3
    for fold, (train, held_out) in enumerate(audit.folds["statement"]):
        assert observed[fold] == set(test[train])
        assert observed[fold].isdisjoint(set(test[held_out]))
    assert len(results.predictions) == len(test)


def test_end_to_end_gates_saved_metrics_and_immutable_run(synthetic_m4, monkeypatch):
    config, inputs, source, config_path = synthetic_m4
    monkeypatch.setattr(m4, "load_m4_inputs", lambda path: (config, inputs, source))
    monkeypatch.setattr(m4, "git_provenance", lambda: {"git_commit": "a" * 40, "git_branch": "synthetic"})
    output = Path("results") / "m4-synthetic"
    assert main(["intervene", "--config", str(config_path), "--validate-only"]) == 0
    assert not output.exists()
    assert main(["intervene", "--config", str(config_path), "--output-dir", str(output)]) == 0
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "completed_gated" and status["exit_code"] == 0
    verification = json.loads((output / "verification.json").read_text())
    assert verification["status"] == "pass"
    assert verification["maximum_recomputation_error"] < 1e-9
    assert verification["unchanged_original_heads"] >= 1
    conditions = read_csv(output / "conditions.csv")
    assert len(conditions) == 6
    retrained = read_csv(output / "retrained_heads.csv")
    assert len(retrained) == 6
    assert all(
        next(item for item in retrained if item["condition_id"] == row["condition_id"])["status"] == "blocked"
        for row in conditions if row["status"] != "valid"
    )
    for filename, digest in json.loads((output / "output_sha256.json").read_text()).items():
        assert sha256_file(output / filename) == digest
    for figure in ("target_decodability.png", "fixed_head_controls.png", "collateral_damage.png", "retraining_performance.png", "ood_gap_change.png"):
        assert (output / figure).read_bytes().startswith(b"\x89PNG")
    with pytest.raises(FileExistsError):
        m4.run_m4(config_path, output)
    metrics = read_csv(output / "metrics.csv")
    metrics[0]["macro_f1"] = "-1"
    write_csv(metrics, output / "metrics.csv")
    with pytest.raises(ValueError, match="recomputation"):
        verify_m4(output, inputs)


def test_no_valid_statement_blocks_speaker_and_m4b_without_zero_effects(synthetic_m4, monkeypatch):
    config, inputs, source, _ = synthetic_m4
    output = Path("results") / "gating-only"
    output.mkdir(parents=True)
    runner = m4.M4Runner(config, inputs, source, output)
    invoked = []

    def invalid_statement(condition):
        assert condition["target"] == "statement"
        invoked.append(condition["condition_id"])
        condition.update({"status": "invalid", "reason": "synthetic failed removal"})

    monkeypatch.setattr(runner, "run_condition", invalid_statement)
    runner.run()
    assert len(invoked) == 1
    assert all(row["status"] == "blocked_statement_gate" for row in runner.conditions if row["target"] == "speaker")
    assert all(row["status"] == "blocked" for row in runner.retraining)
    assert all("erased_macro_f1" not in row for row in runner.retraining)


def test_m4b_trains_a_new_head_without_mutating_original(synthetic_m4):
    config, inputs, source, _ = synthetic_m4
    output = Path("results") / "separate-head"
    output.mkdir(parents=True)
    runner = m4.M4Runner(config, inputs, source, output)
    condition = runner.conditions[0]
    runner.run_condition(condition)
    state = runner.states[condition["condition_id"]]
    original_digest = m4.model_hash(state.original_head)
    # Exercise the retraining machinery independently of this synthetic validity outcome.
    state.valid = True
    condition["status"] = "valid"
    runner.retrain()
    assert m4.model_hash(state.original_head) == original_digest
    row = next(row for row in runner.retraining if row["condition_id"] == condition["condition_id"])
    assert row["status"] == "complete"
    assert (output / "heads" / f"{condition['condition_id']}_retrained.npz").exists()
    assert row["delta_macro_f1_gap"] == 0
    assert any(item["purpose"] == "retrained_head" for item in state.results.metrics)


def test_numerical_run_failure_is_not_a_completed_scientific_result(synthetic_m4, monkeypatch):
    config, inputs, source, config_path = synthetic_m4
    monkeypatch.setattr(m4, "load_m4_inputs", lambda path: (config, inputs, source))
    monkeypatch.setattr(m4, "git_provenance", lambda: {"git_commit": "a" * 40, "git_branch": "synthetic"})

    def fail(self, condition):
        raise ValueError("synthetic numerical failure")

    monkeypatch.setattr(m4.M4Runner, "run_condition", fail)
    output = Path("results") / "failed"
    with pytest.raises(ValueError, match="synthetic numerical"):
        m4.run_m4(config_path, output)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed" and status["exit_code"] == 1
    assert not (output / "summary.json").exists()
