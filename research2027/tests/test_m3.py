import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from mockingbird2027.analysis import m3
from mockingbird2027.analysis.probes import (
    ProbeConfig,
    classification_metrics,
    fit_probe,
    inner_splits,
)
from mockingbird2027.cli import main
from mockingbird2027.data.manifest import (
    ManifestRow,
    ManifestValidationError,
    manifest_report,
    write_json,
    write_parquet,
)
from mockingbird2027.data.splits import build_ravdess_splits, split_report
from mockingbird2027.models import extract


def _rows(complete=False):
    emotions = {
        1: "neutral", 2: "calm", 3: "happy", 4: "sad",
        5: "angry", 6: "fearful", 7: "disgust", 8: "surprised",
    } if complete else {3: "happy", 4: "sad"}
    rows = []
    for actor in range(1, 25 if complete else 5):
        for emotion_code, emotion in emotions.items():
            for intensity_code in ((1,) if emotion == "neutral" else (1, 2)):
                for statement in (1, 2):
                    for repetition in (1, 2):
                        stem = f"03-01-{emotion_code:02d}-{intensity_code:02d}-{statement:02d}-{repetition:02d}-{actor:02d}"
                        rows.append(ManifestRow(
                            utterance_id=f"ravdess:{stem}", dataset="ravdess",
                            path=str(Path("unavailable_audio") / f"{stem}.wav"),
                            emotion=emotion, speaker_id=f"actor_{actor:02d}",
                            gender="male" if actor % 2 else "female",
                            statement_id=statement, repetition=repetition,
                            intensity="normal" if intensity_code == 1 else "strong",
                            duration_seconds=1.0, sample_rate_original=48000,
                        ))
    return sorted(rows, key=lambda row: row.utterance_id)


@pytest.fixture
def complete_bundle(tmp_path, monkeypatch):
    rows = _rows(complete=True)
    assignments = build_ravdess_splits(rows)
    manifest = tmp_path / "manifest.parquet"
    splits = tmp_path / "splits.parquet"
    report_path = tmp_path / "report.json"
    write_parquet(list(reversed(rows)), manifest)
    write_parquet(assignments, splits)
    report = manifest_report(rows)
    report["splits"] = split_report(assignments)
    write_json(report, report_path)
    identity = extract._cache_identity(rows, "main", "synthetic-revision", "full")
    fingerprint = extract._fingerprint(identity)
    monkeypatch.setattr(m3, "EXPECTED_FINGERPRINT", fingerprint)
    cache = tmp_path / fingerprint
    cache.mkdir()
    metadata = {
        **identity, "fingerprint": fingerprint, "status": "complete",
        "utterance_count": 1440, "layer_count": 13, "hidden_size": 768,
    }
    write_json(metadata, cache / "metadata.json")
    np.save(cache / "ids.npy", np.array([row.utterance_id for row in rows]), allow_pickle=False)
    features = np.zeros((1440, 768), dtype=np.float16)
    features[:, 0] = np.arange(1440)
    for layer in range(13):
        np.save(cache / f"layer_{layer:02d}.npy", features, allow_pickle=False)
    return {
        "manifest": str(manifest), "splits": str(splits), "manifest_report": str(report_path),
        "cache": str(cache), "expected_fingerprint": fingerprint,
        "expected_splits_sha256": m3.sha256_file(splits),
    }


def test_input_loading_preserves_cache_id_mapping_without_audio_or_inference(complete_bundle, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("M3 must not load audio or an encoder")

    monkeypatch.setattr(extract, "load_wavlm_runtime", forbidden)
    monkeypatch.setattr(extract, "load_audio_16khz", forbidden)
    inputs = m3.load_m3_inputs(complete_bundle)
    ids = np.load(inputs.cache / "ids.npy", allow_pickle=False).tolist()
    assert ids == [row.utterance_id for row in inputs.rows]
    assert len(inputs.partitions) == 12
    features = np.load(inputs.cache / "layer_00.npy", allow_pickle=False)
    for name, fold in inputs.partitions:
        train, test = inputs.partitions[(name, fold)]
        assert not set(train) & set(test)
        assert sorted(np.concatenate((train, test)).tolist()) == list(range(1440))
        np.testing.assert_array_equal(features[train, 0], train)
    assert inputs.hashes[complete_bundle["splits"]] == complete_bundle["expected_splits_sha256"]


@pytest.mark.parametrize("corruption", ["fingerprint", "order", "nonfinite", "splits", "report", "manifest"])
def test_input_loading_rejects_changed_scientific_artifacts(complete_bundle, corruption):
    config = complete_bundle
    cache = Path(config["cache"])
    if corruption == "fingerprint":
        config["expected_fingerprint"] = "another-cache"
    elif corruption == "order":
        ids = np.load(cache / "ids.npy", allow_pickle=False)
        np.save(cache / "ids.npy", ids[::-1], allow_pickle=False)
    elif corruption == "nonfinite":
        array = np.load(cache / "layer_00.npy", allow_pickle=False)
        array[0, 0] = np.nan
        np.save(cache / "layer_00.npy", array, allow_pickle=False)
    elif corruption == "splits":
        records = pq.read_table(config["splits"]).to_pylist()
        write_parquet(list(reversed(records)), Path(config["splits"]))
    elif corruption == "report":
        report = json.loads(Path(config["manifest_report"]).read_text())
        report["utterance_count"] = 1
        write_json(report, Path(config["manifest_report"]))
    elif corruption == "manifest":
        records = pq.read_table(config["manifest"]).to_pylist()
        records[0]["path"] = "changed_manifest_identity.wav"
        write_parquet(records, Path(config["manifest"]))
    with pytest.raises((ManifestValidationError, extract.CacheValidationError)):
        m3.load_m3_inputs(config)


def test_split_reuse_is_order_independent_and_keeps_audit_label_spaces():
    rows = _rows()
    assignments = build_ravdess_splits(rows, speaker_folds=2)
    original = m3.persisted_partitions(rows, assignments, speaker_folds=2)
    reordered = m3.persisted_partitions(rows, list(reversed(assignments)), speaker_folds=2)
    for key in original:
        for first, second in zip(original[key], reordered[key]):
            np.testing.assert_array_equal(first, second)
    train, test = original[(m3.AUDIT, 0)]
    assert {rows[index].repetition for index in train} == {1}
    assert {rows[index].repetition for index in test} == {2}
    for attribute in m3.ATTRIBUTES.values():
        assert {getattr(rows[index], attribute) for index in train} == {
            getattr(rows[index], attribute) for index in test
        }


@pytest.mark.parametrize("corruption", ["missing_group", "speaker_leak", "repeated_fold", "duplicate"])
def test_persisted_partitions_reject_leakage_and_incomplete_folds(corruption):
    rows = _rows()
    assignments = build_ravdess_splits(rows, speaker_folds=2)
    if corruption == "missing_group":
        assignments = [item for item in assignments if item.split_name != m3.LEXICAL[0]]
    elif corruption == "speaker_leak":
        index = next(index for index, item in enumerate(assignments) if item.split_name == "speaker_shift")
        assignments[index] = replace(assignments[index], role="train" if assignments[index].role == "test" else "test")
    elif corruption == "repeated_fold":
        roles = {item.utterance_id: item.role for item in assignments if (item.split_name, item.fold) == ("speaker_shift", 0)}
        assignments = [
            replace(item, role=roles[item.utterance_id])
            if (item.split_name, item.fold) == ("speaker_shift", 1) else item
            for item in assignments
        ]
    elif corruption == "duplicate":
        assignments.append(assignments[0])
    with pytest.raises(ManifestValidationError):
        m3.persisted_partitions(rows, assignments, speaker_folds=2)


def test_standardization_is_train_only_in_every_inner_fold_and_refit(monkeypatch):
    features = np.arange(48, dtype=float).reshape(24, 2)
    labels = np.array(["a", "b"] * 12)
    calls = []
    original_fit = StandardScaler.fit

    def record_fit(self, values, *args, **kwargs):
        calls.append(values.copy())
        return original_fit(self, values, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record_fit)
    config = ProbeConfig(c_grid=(0.01, 0.1))
    search = fit_probe(features, labels, config)
    assert len(calls) == 7
    for index, values in enumerate(calls[:-1]):
        train, _ = search.cv[index % config.cv_folds]
        np.testing.assert_array_equal(values, features[train])
    np.testing.assert_array_equal(calls[-1], features)
    scaler = search.best_estimator_.named_steps["standardize"]
    np.testing.assert_allclose(scaler.mean_, features.mean(axis=0))
    search.predict(np.full((5, 2), 1e9))
    assert len(calls) == 7
    np.testing.assert_allclose(scaler.mean_, features.mean(axis=0))


def test_inner_speaker_folds_are_disjoint_and_deterministic():
    labels = np.array(["a", "b"] * 30)
    groups = np.repeat(np.arange(6), 10)
    first = inner_splits(labels, ProbeConfig(), groups)
    second = inner_splits(labels, ProbeConfig(), groups)
    held_out = []
    for (train, test), (train_again, test_again) in zip(first, second):
        assert not set(groups[train]) & set(groups[test])
        np.testing.assert_array_equal(train, train_again)
        np.testing.assert_array_equal(test, test_again)
        held_out.extend(test)
    assert sorted(held_out) == list(range(len(labels)))


def test_metrics_include_never_predicted_classes_and_reject_unknown_labels():
    gold = np.array(["a", "a", "b", "b"])
    predicted = np.array(["a"] * 4)
    scores = classification_metrics(gold, predicted, np.array(["a", "b"]))
    assert scores == pytest.approx({"macro_f1": 1 / 3, "accuracy": 0.5})
    with pytest.raises(ValueError, match="outside"):
        classification_metrics(gold, predicted, np.array(["a"]))
    with pytest.raises(ValueError, match="nonempty"):
        classification_metrics(np.array([]), np.array([]), np.array(["a", "b"]))


def test_probe_is_reproducible_and_unconverged_fits_fail():
    features = np.random.default_rng(4).normal(size=(60, 8))
    labels = np.where(features[:, 0] > 0, "a", "b")
    config = ProbeConfig(c_grid=(0.01, 0.1))
    first = fit_probe(features, labels, config)
    second = fit_probe(features, labels, config)
    assert first.best_params_ == second.best_params_
    np.testing.assert_allclose(first.predict_proba(features), second.predict_proba(features), rtol=0, atol=0)
    with pytest.raises(ConvergenceWarning):
        fit_probe(features, labels, replace(config, max_iter=1))


@pytest.mark.parametrize("grid", [(), (0.0,), (np.nan,), (1.0, 0.1), (0.1, 0.1)])
def test_invalid_regularization_grid_fails(grid):
    with pytest.raises(ValueError):
        ProbeConfig(c_grid=grid)


@pytest.fixture
def small_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = _rows()
    assignments = build_ravdess_splits(rows, speaker_folds=2)
    cache = Path("cache")
    cache.mkdir()
    encoded = []
    for attribute in m3.ATTRIBUTES.values():
        labels = np.array([str(getattr(row, attribute)) for row in rows])
        _, indices = np.unique(labels, return_inverse=True)
        encoded.append(np.eye(len(np.unique(labels)))[indices])
    features = np.column_stack(encoded)
    generator = np.random.default_rng(4)
    for layer in range(2):
        matrix = features + generator.normal(scale=0.2, size=features.shape)
        np.save(cache / f"layer_{layer:02d}.npy", matrix.astype(np.float16), allow_pickle=False)
    inputs = m3.M3Inputs(
        rows, m3.persisted_partitions(rows, assignments, speaker_folds=2), cache,
        {"model_id": "synthetic", "resolved_revision": "test", "fingerprint": "synthetic", "layer_count": 2},
        {str(path): m3.sha256_file(path) for path in cache.iterdir()},
    )
    config = Path("config.json")
    write_json({"probe": {"c_grid": [0.1], "cv_folds": 2, "seed": 7}}, config)
    monkeypatch.setattr(m3, "load_m3_inputs", lambda settings: inputs)
    monkeypatch.setattr(m3, "git_provenance", lambda: {"git_commit": "a" * 40, "git_branch": "synthetic-test"})
    return config


def test_small_end_to_end_run_persists_complete_recomputable_results(small_run):
    output = Path("results") / "synthetic"
    assert main(["probe", "--config", str(small_run), "--output-dir", str(output)]) == 0
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "complete"
    assert status["exit_code"] == 0
    assert status["completed_layers"] == 2
    probes = m3.read_csv(output / "probes.csv")
    generalization = m3.read_csv(output / "generalization.csv")
    candidates = m3.read_csv(output / "cv_results.csv")
    predictions = pq.read_table(output / "predictions.parquet").to_pylist()
    assert len(probes) == 10
    assert len(generalization) == 10
    assert len(candidates) == 18
    assert len(predictions) == sum(int(record["n_test"]) for record in probes + generalization)
    verification = json.loads((output / "verification.json").read_text())
    assert verification["maximum_metric_recomputation_error"] == 0
    assert verification["distinct_model_fits"] == 18
    for record in probes + generalization:
        group = [
            item for item in predictions
            if item["model_fit_id"] == record["model_fit_id"] and item["task"] == record["task"]
        ]
        assert len(group) == int(record["n_test"])
        classes = np.array(json.loads(record["classes"]))
        gold = np.array([item["gold_label"] for item in group])
        predicted = np.array([item["predicted_label"] for item in group])
        scores = classification_metrics(gold, predicted, classes)
        assert float(record["macro_f1"]) == pytest.approx(scores["macro_f1"])
        assert float(record["accuracy"]) == pytest.approx(scores["accuracy"])
        baseline = classification_metrics(gold, np.array([item["majority_predicted_label"] for item in group]), classes)
        assert float(record["majority_macro_f1"]) == pytest.approx(baseline["macro_f1"])
        probabilities = np.array([item["probabilities"] for item in group])
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_array_equal(classes[probabilities.argmax(axis=1)], predicted)
    for row in generalization:
        assert float(row["macro_f1_gap"]) == pytest.approx(float(row["reference_macro_f1"]) - float(row["macro_f1"]))
        assert float(row["accuracy_gap"]) == pytest.approx(float(row["reference_accuracy"]) - float(row["accuracy"]))
        if row["split_name"] == m3.REFERENCE:
            audit = next(item for item in probes if item["layer"] == row["layer"] and item["attribute"] == "emotion")
            assert row["model_fit_id"] == audit["model_fit_id"]
            assert row["macro_f1"] == audit["macro_f1"]
    hashes = json.loads((output / "output_sha256.json").read_text())
    for filename, digest in hashes.items():
        assert m3.sha256_file(output / filename) == digest
    for figure in ("encoding_heatmap.png", "emotion_generalization.png"):
        assert (output / figure).read_bytes().startswith(b"\x89PNG")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["gate"]["decision"] in {"PARTIAL / REVIEW NEEDED", "FAIL / DIAGNOSE"}
    assert len(summary["shifts"]["speaker_shift"]) == 2
    assert all(len(item["fold_macro_f1"]) == 2 for item in summary["shifts"]["speaker_shift"])
    with pytest.raises(FileExistsError):
        m3.run_m3(small_run, output)
    probes[0]["macro_f1"] = "-1"
    m3.write_csv(probes, output / "probes.csv")
    with pytest.raises(ValueError, match="recomputation"):
        m3.verify_result_tables(output, m3.load_m3_inputs({}))


def test_validate_only_never_fits_and_failed_runs_are_not_complete(small_run, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("synthetic fit failure")

    monkeypatch.setattr(m3, "evaluate_task", fail)
    assert main(["probe", "--config", str(small_run), "--validate-only"]) == 0
    assert not Path("results").exists()
    output = Path("results") / "failed"
    with pytest.raises(ValueError, match="synthetic fit failure"):
        m3.run_m3(small_run, output)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["exit_code"] == 1
    assert status["completed_layers"] == 0
    assert not (output / "summary.json").exists()
