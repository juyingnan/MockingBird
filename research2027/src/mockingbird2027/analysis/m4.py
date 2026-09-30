"""Gated M4 interventions with immutable inputs and independent erasure audits."""

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
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from ..data.manifest import write_json, write_parquet
from .controls import ControlMatchError, control_seed, matched_random_removal
from .erasure import AffineRemoval, LinearEraser, Numerics
from .m3 import (
    ATTRIBUTES, LEXICAL, REFERENCE, M3Inputs, git_provenance,
    load_m3_inputs, read_csv, sha256_file, verify_result_tables,
)
from .probes import (
    ProbeConfig, classification_metrics, fit_fixed_probe, fit_probe, inner_splits,
)


PROTOCOL_COMMIT = "0017c9cbb4d04763130958df386cf0345e3d8302"


def selected_layers(generalization: list[dict[str, Any]]) -> dict[str, Any]:
    references = [row for row in generalization if row["split_name"] == REFERENCE]
    anchor = min(references, key=lambda row: (-float(row["macro_f1"]), int(row["layer"])))
    task_layer = int(anchor["layer"])
    values = []
    selection = {}
    for target, names in (("statement", LEXICAL), ("speaker", ("speaker_shift",))):
        candidates = []
        for reference in references:
            layer = int(reference["layer"])
            shifts = [
                float(row["macro_f1_gap"]) for row in generalization
                if int(row["layer"]) == layer and row["split_name"] in names
            ]
            if not shifts:
                raise ValueError("M3 lacks a required shift for M4 layer selection")
            candidates.append({
                "target": target, "layer": layer,
                "reference_macro_f1": float(reference["macro_f1"]),
                "mean_shift_gap": sum(shifts) / len(shifts),
            })
        shift = min(candidates, key=lambda row: (-row["mean_shift_gap"], row["layer"]))
        selection[target] = sorted({task_layer, shift["layer"]})
        values.extend(row for row in sorted(candidates, key=lambda row: row["layer"]) if row["layer"] in selection[target])
    return {"task_anchor": task_layer, "selected_layers": selection, "selection_values": values}


def condition_plan(inputs: M3Inputs, config: dict[str, Any]) -> list[dict[str, Any]]:
    conditions = []
    for target in config["target_order"]:
        partitions = [(REFERENCE, 0)] + (
            [(name, 0) for name in LEXICAL] if target == "statement" else
            sorted(key for key in inputs.partitions if key[0] == "speaker_shift")
        )
        for layer in config["selected_layers"][target]:
            for method in config["methods"]:
                for split_name, fold in partitions:
                    train, test = inputs.partitions[(split_name, fold)]
                    labels = {str(getattr(inputs.rows[index], ATTRIBUTES[target])) for index in train}
                    eligible = len(labels) >= 2
                    conditions.append({
                        "condition_id": f"{target}_L{layer:02d}_{method}_{split_name}_f{fold}",
                        "target": target, "layer": layer, "method": method,
                        "split_name": split_name, "fold": fold,
                        "n_train": len(train), "n_test": len(test),
                        "training_target_classes": len(labels),
                        "status": "pending" if eligible else "blocked_unidentifiable",
                        "reason": "" if eligible else "one outer-training target label; no train-only contrast",
                    })
    return conditions


def load_m4_inputs(config_path: Path) -> tuple[dict[str, Any], M3Inputs, dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for path in (config_path, Path(config["protocol_file"])):
        repository_path = "research2027/" + path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        committed = subprocess.run(
            ["git", "show", f"{PROTOCOL_COMMIT}:{repository_path}"],
            check=True, capture_output=True, text=True,
        ).stdout
        if path.suffix == ".json":
            equal = json.loads(committed) == config
        else:
            equal = committed == path.read_text(encoding="utf-8")
        if not equal:
            raise ValueError("M4 protocol differs from its pre-intervention frozen commit")
    run = Path(config["m3_run"])
    subprocess.run(
        ["git", "diff", "--exit-code", config["m3_results_commit"], "--", "research2027/" + run.as_posix()],
        check=True, capture_output=True, text=True,
        cwd=Path.cwd().parent,
    )
    if sha256_file(run / "output_sha256.json") != config["m3_output_manifest_sha256"]:
        raise ValueError("M3 output manifest differs from frozen M4 inputs")
    for name, digest in json.loads((run / "output_sha256.json").read_text()).items():
        if sha256_file(run / name) != digest:
            raise ValueError(f"M3 result changed: {name}")
    provenance = json.loads((run / "provenance.json").read_text())
    inputs = load_m3_inputs(provenance["config"])
    for filename, digest in provenance["input_sha256"].items():
        if sha256_file(Path(filename)) != digest:
            raise ValueError(f"M3 scientific input changed: {Path(filename).name}")
    verify_result_tables(run, inputs)
    selection = selected_layers(read_csv(run / "generalization.csv"))
    if selection["selected_layers"] != config["selected_layers"] or selection["selection_values"] != config["selection_values"]:
        raise ValueError("M4 layer selection does not exactly follow frozen M3-only rule")
    for package in ("numpy", "scipy", "scikit-learn"):
        if version(package) != provenance["packages"][package]:
            raise ValueError(f"M3 original-head reproduction requires the recorded {package} version")
    hashes = {**inputs.hashes, **{str(path): sha256_file(path) for path in sorted(run.iterdir())}}
    return config, inputs, {"selection": selection, "m3_provenance": provenance, "input_sha256": hashes}


def model_arrays(model: Pipeline) -> dict[str, np.ndarray]:
    scaler = model.named_steps["standardize"]
    classifier = model.named_steps["classifier"]
    return {
        "mean": scaler.mean_, "scale": scaler.scale_,
        "coef": classifier.coef_, "intercept": classifier.intercept_,
        "classes": classifier.classes_, "n_iter": classifier.n_iter_,
        "c": np.array([classifier.C], dtype=np.float64),
    }


def parameters_hash(parameters: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, array in parameters.items():
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def model_hash(model: Pipeline) -> str:
    return parameters_hash(model_arrays(model))


def table_with_schema(records: list[dict[str, Any]], path: Path, empty_fields: list[str]) -> None:
    if records:
        fields = list(dict.fromkeys(key for row in records for key in row))
    else:
        fields = empty_fields
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


class ConditionResults:
    def __init__(self, condition: dict[str, Any], inputs: M3Inputs):
        self.condition = condition
        self.inputs = inputs
        self.metrics: list[dict[str, Any]] = []
        self.predictions: list[dict[str, Any]] = []
        self.probe_fits: list[dict[str, Any]] = []
        self.diagnostic_folds: list[dict[str, Any]] = []

    def evaluate(
        self, purpose: str, variant: str, attribute: str, indices: np.ndarray,
        gold: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray,
        classes: np.ndarray, *, diagnostic_folds: np.ndarray | None = None,
        head_digest: str = "",
    ) -> dict[str, Any]:
        if len(indices) != len(gold) or probabilities.shape != (len(indices), len(classes)):
            raise ValueError("held-out prediction arrays are misaligned")
        scores = classification_metrics(gold, predicted, classes)
        identity = f"{self.condition['condition_id']}:{purpose}:{variant}:{attribute}"
        metadata = {
            name: self.condition[name]
            for name in ("condition_id", "target", "layer", "method", "split_name", "fold")
        }
        record = {
            **metadata, "evaluation_id": identity, "purpose": purpose,
            "variant": variant, "attribute": attribute, "n_test": len(indices),
            "classes": json.dumps(classes.tolist()), **scores, "head_hash": head_digest,
        }
        self.metrics.append(record)
        for position, index in enumerate(indices):
            self.predictions.append({
                "evaluation_id": identity,
                "utterance_id": self.inputs.rows[index].utterance_id,
                "gold_label": str(gold[position]), "predicted_label": str(predicted[position]),
                "classes": classes.tolist(), "probabilities": probabilities[position].tolist(),
                "diagnostic_fold": int(diagnostic_folds[position]) if diagnostic_folds is not None else -1,
            })
        return record

    def save(self, output_dir: Path) -> None:
        identity = self.condition["condition_id"]
        (output_dir / "predictions").mkdir(exist_ok=True)
        write_parquet(self.predictions, output_dir / "predictions" / f"{identity}.parquet")
        if self.diagnostic_folds:
            (output_dir / "diagnostic_folds").mkdir(exist_ok=True)
            write_parquet(self.diagnostic_folds, output_dir / "diagnostic_folds" / f"{identity}.parquet")


class FreshAudit:
    """Paired probes cross-fitted strictly inside an eraser-unseen evaluation pool."""

    def __init__(
        self, features: np.ndarray, test: np.ndarray, results: ConditionResults,
        config: ProbeConfig, attributes: list[str],
    ):
        self.features = features
        self.test = test
        self.results = results
        self.config = config
        self.folds: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        self.c_values: dict[str, list[float]] = {}
        self.original: dict[str, dict[str, Any]] = {}
        for attribute in attributes:
            labels = self.labels(attribute)
            folds = inner_splits(labels, config)
            self.folds[attribute] = folds
            self.c_values[attribute] = []
            classes = np.unique(labels)
            probabilities = np.empty((len(test), len(classes)))
            fold_ids = np.full(len(test), -1)
            predicted = np.empty_like(labels)
            for fold, (train, held_out) in enumerate(folds):
                search = fit_probe(features[train], labels[train], config)
                c = float(search.best_params_["classifier__C"])
                self.c_values[attribute].append(c)
                predicted[held_out] = search.predict(features[held_out])
                probabilities[held_out] = search.predict_proba(features[held_out])
                fold_ids[held_out] = fold
                for index in held_out:
                    results.diagnostic_folds.append({
                        "condition_id": results.condition["condition_id"],
                        "attribute": attribute,
                        "utterance_id": results.inputs.rows[test[index]].utterance_id,
                        "held_out_fold": fold,
                    })
                self.record_fit(attribute, "original", fold, train, c, search.best_estimator_, search)
            self.original[attribute] = results.evaluate(
                "audit", "original", attribute, test, labels, predicted, probabilities,
                classes, diagnostic_folds=fold_ids,
            )

    def labels(self, attribute: str) -> np.ndarray:
        return np.array([
            str(getattr(self.results.inputs.rows[index], ATTRIBUTES[attribute])) for index in self.test
        ])

    def record_fit(
        self, attribute: str, variant: str, fold: int, train: np.ndarray,
        c: float, model: Pipeline, search: Any = None,
    ) -> None:
        ids = [self.results.inputs.rows[self.test[index]].utterance_id for index in train]
        record = {
            "condition_id": self.results.condition["condition_id"],
            "attribute": attribute, "variant": variant, "diagnostic_fold": fold,
            "n_train": len(train), "selected_c": c, "head_hash": model_hash(model),
            "train_ids_sha256": hashlib.sha256(json.dumps(ids).encode("ascii")).hexdigest(),
            "n_iter": int(model.named_steps["classifier"].n_iter_.max()),
            "original_cv_scores": "",
        }
        if search is not None:
            record["original_cv_scores"] = json.dumps([
                {
                    "c": c_value, "mean_macro_f1": float(search.cv_results_["mean_test_score"][index]),
                    "fold_macro_f1": [
                        float(search.cv_results_[f"split{inner}_test_score"][index])
                        for inner in range(self.config.cv_folds)
                    ],
                }
                for index, c_value in enumerate(self.config.c_grid)
            ])
        self.results.probe_fits.append(record)

    def transformed(self, features: np.ndarray, variant: str, attributes: list[str]) -> dict[str, dict[str, Any]]:
        scores = {}
        for attribute in attributes:
            labels = self.labels(attribute)
            classes = np.unique(labels)
            predicted = np.empty_like(labels)
            probabilities = np.empty((len(labels), len(classes)))
            fold_ids = np.full(len(labels), -1)
            for fold, (train, held_out) in enumerate(self.folds[attribute]):
                c = self.c_values[attribute][fold]
                model = fit_fixed_probe(features[train], labels[train], self.config, c=c)
                predicted[held_out] = model.predict(features[held_out])
                probabilities[held_out] = model.predict_proba(features[held_out])
                fold_ids[held_out] = fold
                self.record_fit(attribute, variant, fold, train, c, model)
            scores[attribute] = self.results.evaluate(
                "audit", variant, attribute, self.test, labels, predicted,
                probabilities, classes, diagnostic_folds=fold_ids,
            )
        return scores


def validity_decision(
    original: dict[str, dict[str, Any]], erased: dict[str, dict[str, Any]],
    target: str, class_count: int, config: dict[str, Any], controls_match: bool,
) -> dict[str, Any]:
    settings = config["validity"]
    chance = 1 / class_count
    before, after = original[target], erased[target]
    target_valid = (
        before["macro_f1"] >= chance + settings["minimum_before_f1_excess_over_chance"]
        and before["macro_f1"] - after["macro_f1"] >= settings["minimum_target_f1_drop"]
        and after["macro_f1"] <= chance + settings["maximum_after_f1_excess_over_chance"]
        and after["accuracy"] <= chance + settings["maximum_after_accuracy_excess_over_chance"]
    )
    emotion_loss = original["emotion"]["macro_f1"] - erased["emotion"]["macro_f1"]
    nested = settings["speaker_nested_attributes"] if target == "speaker" else []
    unrelated = [
        name for name in ATTRIBUTES if name not in [target, "emotion", *nested]
        and original[name]["macro_f1"] - erased[name]["macro_f1"] > settings["large_unrelated_f1_drop"]
    ]
    collateral_valid = (
        emotion_loss <= settings["maximum_emotion_f1_drop"]
        and len(unrelated) <= settings["maximum_large_unrelated_losses"]
    )
    failures = []
    if not target_valid:
        failures.append("held-out target removal failed")
    if not collateral_valid:
        failures.append("collateral-damage gate failed")
    if not controls_match:
        failures.append("not all random controls match")
    return {
        "valid": target_valid and collateral_valid and controls_match,
        "target_removal_valid": target_valid, "collateral_valid": collateral_valid,
        "controls_match": controls_match, "chance_reference": chance,
        "target_before_macro_f1": before["macro_f1"],
        "target_after_macro_f1": after["macro_f1"],
        "target_before_accuracy": before["accuracy"], "target_after_accuracy": after["accuracy"],
        "target_delta_macro_f1": after["macro_f1"] - before["macro_f1"],
        "emotion_probe_f1_loss": emotion_loss,
        "large_unrelated_losses": json.dumps(unrelated),
        "reason": "; ".join(failures) if failures else "all predeclared condition-validity checks passed",
    }


@dataclass
class ConditionState:
    condition: dict[str, Any]
    results: ConditionResults
    train: np.ndarray
    test: np.ndarray
    standardized_train: np.ndarray
    standardized_test: np.ndarray
    original_head: Pipeline
    original_c: float
    eraser: LinearEraser
    original_emotion: dict[str, Any]
    valid: bool


class M4Runner:
    def __init__(
        self, config: dict[str, Any], inputs: M3Inputs,
        source: dict[str, Any], output_dir: Path,
    ):
        self.config = config
        self.inputs = inputs
        self.source = source
        self.output_dir = output_dir
        probe = dict(source["m3_provenance"]["config"]["probe"])
        probe["c_grid"] = tuple(probe["c_grid"])
        self.probe_config = ProbeConfig(**probe)
        m3_run = Path(config["m3_run"])
        self.m3_rows = {
            (int(row["layer"]), row["split_name"], int(row["fold"])): row
            for row in read_csv(m3_run / "generalization.csv")
        }
        self.m3_predictions: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in pq.read_table(m3_run / "predictions.parquet").to_pylist():
            if row["task"] == "emotion_generalization":
                self.m3_predictions[(row["layer"], row["split_name"], row["fold"])].append(row)
        self.conditions = condition_plan(inputs, config)
        self.states: dict[str, ConditionState] = {}
        self.metrics: list[dict[str, Any]] = []
        self.probe_fits: list[dict[str, Any]] = []
        self.collateral: list[dict[str, Any]] = []
        self.validation: list[dict[str, Any]] = []
        self.controls: list[dict[str, Any]] = []
        self.reliance: list[dict[str, Any]] = []
        self.retraining: list[dict[str, Any]] = []
        self.head_checks: list[dict[str, Any]] = []
        self.fitting_ids: list[dict[str, Any]] = []

    def original_head(
        self, condition: dict[str, Any], features: np.ndarray, train: np.ndarray, test: np.ndarray
    ) -> tuple[Pipeline, float]:
        key = (condition["layer"], condition["split_name"], condition["fold"])
        c = float(self.m3_rows[key]["selected_c"])
        labels = np.array([row.emotion for row in self.inputs.rows])
        head = fit_fixed_probe(features[train], labels[train], self.probe_config, c=c)
        expected = {row["utterance_id"]: row for row in self.m3_predictions[key]}
        predictions = head.predict(features[test])
        probabilities = head.predict_proba(features[test])
        for position, index in enumerate(test):
            record = expected[self.inputs.rows[index].utterance_id]
            if predictions[position] != record["predicted_label"]:
                raise ValueError("refitted original emotion head does not reproduce M3 predictions")
            if not np.allclose(probabilities[position], record["probabilities"], atol=1e-10, rtol=1e-10):
                raise ValueError("refitted original emotion probabilities differ from M3")
        return head, c

    def save_transform(self, removal: AffineRemoval, condition_id: str, variant: str) -> None:
        directory = self.output_dir / "transforms"
        directory.mkdir(exist_ok=True)
        np.savez_compressed(
            directory / f"{condition_id}_{variant}.npz",
            mean=removal.mean, left=removal.left, right=removal.right,
        )

    def fixed_evaluation(
        self, results: ConditionResults, head: Pipeline, transformed: np.ndarray,
        test: np.ndarray, variant: str, digest: str,
    ) -> dict[str, Any]:
        classifier = head.named_steps["classifier"]
        return results.evaluate(
            "fixed_head", variant, "emotion", test,
            np.array([self.inputs.rows[index].emotion for index in test]),
            classifier.predict(transformed), classifier.predict_proba(transformed),
            classifier.classes_, head_digest=digest,
        )

    def run_condition(self, condition: dict[str, Any]) -> None:
        identity = condition["condition_id"]
        train, test = self.inputs.partitions[(condition["split_name"], condition["fold"])]
        features = np.load(self.inputs.cache / f"layer_{condition['layer']:02d}.npy", allow_pickle=False).astype(np.float64)
        head, c = self.original_head(condition, features, train, test)
        digest = model_hash(head)
        models = self.output_dir / "heads"
        models.mkdir(exist_ok=True)
        np.savez_compressed(models / f"{identity}_original.npz", **model_arrays(head))
        scaler = head.named_steps["standardize"]
        x_train, x_test = scaler.transform(features[train]), scaler.transform(features[test])
        concept = np.array([str(getattr(self.inputs.rows[index], ATTRIBUTES[condition["target"]])) for index in train])
        eraser = LinearEraser(condition["method"], Numerics(**self.config["numerics"])).fit(x_train, concept)
        self.save_transform(eraser.transformation, identity, "targeted")
        for index in train:
            self.fitting_ids.append({
                "condition_id": identity, "utterance_id": self.inputs.rows[index].utterance_id,
                "fitted_components": "original_scaler,original_emotion_head,targeted_eraser,random_control_geometry",
            })
        results = ConditionResults(condition, self.inputs)
        original = self.fixed_evaluation(results, head, x_test, test, "original", digest)
        targeted = self.fixed_evaluation(results, head, eraser.transform(x_test), test, "targeted", digest)
        audit = FreshAudit(x_test, test, results, self.probe_config, self.config["audit"]["attributes"])
        erased_audit = audit.transformed(eraser.transform(x_test), "targeted", self.config["audit"]["attributes"])
        for attribute in self.config["audit"]["attributes"]:
            before, after = audit.original[attribute], erased_audit[attribute]
            self.collateral.append({
                "condition_id": identity, "target": condition["target"], "attribute": attribute,
                "before_macro_f1": before["macro_f1"], "after_macro_f1": after["macro_f1"],
                "delta_macro_f1": after["macro_f1"] - before["macro_f1"],
                "before_accuracy": before["accuracy"], "after_accuracy": after["accuracy"],
                "delta_accuracy": after["accuracy"] - before["accuracy"],
                "before_evaluation_id": before["evaluation_id"], "after_evaluation_id": after["evaluation_id"],
            })
        controls = []
        for draw in range(self.config["controls"]["draws"]):
            variant = f"random_{draw:02d}"
            seed = control_seed(self.config["seed"], identity, draw)
            control = {"condition_id": identity, "variant": variant, "draw": draw, "seed": seed}
            try:
                removal, metadata = matched_random_removal(
                    eraser, x_train, seed=seed, settings=self.config["controls"]
                )
            except ControlMatchError as error:
                control.update({"status": "match_failed", "reason": str(error)})
                print(f"M4 control rejected: {identity} draw {draw}: {error}", flush=True)
                self.controls.append(control)
                continue
            self.save_transform(removal, identity, variant)
            transformed = removal.transform(x_test)
            score = self.fixed_evaluation(results, head, transformed, test, variant, digest)
            target_audit = audit.transformed(transformed, variant, [condition["target"]])[condition["target"]]
            control.update({
                **metadata, "status": "matched", "reason": "",
                "test_rms": removal.perturbation(x_test)["rms"],
                "macro_f1": score["macro_f1"], "accuracy": score["accuracy"],
                "macro_f1_effect": score["macro_f1"] - original["macro_f1"],
                "accuracy_effect": score["accuracy"] - original["accuracy"],
                "target_audit_macro_f1": target_audit["macro_f1"],
                "target_audit_accuracy": target_audit["accuracy"],
                "fixed_evaluation_id": score["evaluation_id"],
                "audit_evaluation_id": target_audit["evaluation_id"],
            })
            controls.append(control)
            self.controls.append(control)
        target_labels = audit.labels(condition["target"])
        counts = Counter(target_labels)
        if len(set(counts.values())) != 1:
            raise ValueError("predeclared target chance screen requires balanced diagnostic labels")
        valid = validity_decision(
            audit.original, erased_audit, condition["target"], len(counts),
            self.config, len(controls) == self.config["controls"]["draws"],
        )
        self.validation.append({
            "condition_id": identity, **valid, "removed_rank": eraser.removed_rank,
            "training_cross_covariance_residual": eraser.training_cross_covariance_residual_,
            "train_rms": eraser.training_rms_,
            "train_mean_l2": eraser.transformation.perturbation(x_train)["mean_l2"],
            "test_rms": eraser.transformation.perturbation(x_test)["rms"],
            "test_mean_l2": eraser.transformation.perturbation(x_test)["mean_l2"],
        })
        for metric in ("macro_f1", "accuracy"):
            effect = targeted[metric] - original[metric]
            row = {
                "condition_id": identity, "metric": metric, "original": original[metric],
                "targeted": targeted[metric], "targeted_effect": effect,
                "matched_draws": len(controls), "condition_valid": valid["valid"],
                "original_evaluation_id": original["evaluation_id"],
                "targeted_evaluation_id": targeted["evaluation_id"],
                "interpretation": "not interpretable: failed validity/matching gate",
            }
            if len(controls) == self.config["controls"]["draws"]:
                effects = np.array([control[f"{metric}_effect"] for control in controls])
                low, high = np.quantile(effects, self.config["reliance"]["control_quantiles"])
                excess = effect - float(effects.mean())
                specificity = float(np.median([control["target_audit_macro_f1"] for control in controls])) - erased_audit[condition["target"]]["macro_f1"]
                row.update({
                    "random_mean_effect": float(effects.mean()), "random_sd_effect": float(effects.std(ddof=1)),
                    "random_q025": float(low), "random_q975": float(high),
                    "targeted_minus_random": excess, "control_target_f1_advantage": specificity,
                })
                if valid["valid"]:
                    if specificity < self.config["reliance"]["minimum_control_target_f1_advantage"]:
                        row["interpretation"] = "not target-specific: controls also erase the target"
                    elif (effect < low or effect > high) and abs(excess) >= self.config["reliance"]["minimum_excess_magnitude"]:
                        row["interpretation"] = "exploratory excess fixed-head sensitivity"
                    else:
                        row["interpretation"] = "no detected excess effect; not an equivalence claim"
            self.reliance.append(row)
        after_digest = model_hash(head)
        if digest != after_digest:
            raise RuntimeError("M4A original head changed during fixed-head interventions")
        self.head_checks.append({
            "condition_id": identity, "before_sha256": digest,
            "after_sha256": after_digest, "unchanged": True,
            "m3_predictions_reproduced": True,
        })
        condition.update({
            "status": "valid" if valid["valid"] else "invalid",
            "reason": valid["reason"],
        })
        results.save(self.output_dir)
        self.metrics.extend(results.metrics)
        self.probe_fits.extend(results.probe_fits)
        self.states[identity] = ConditionState(
            condition, results, train, test, x_train, x_test, head, c, eraser, original, valid["valid"]
        )
        self.save_tables()
        with (self.output_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "condition_id": identity, "status": condition["status"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }) + "\n")
        print(f"M4A {identity}: {condition['status']} - {condition['reason']}", flush=True)

    def retrain(self) -> None:
        reference_valid = {
            (state.condition["target"], state.condition["layer"], state.condition["method"]): state.valid
            for state in self.states.values() if state.condition["split_name"] == REFERENCE
        }
        for condition in self.conditions:
            identity = condition["condition_id"]
            state = self.states.get(identity)
            reference_key = (condition["target"], condition["layer"], condition["method"])
            if state is None or not state.valid or not reference_valid.get(reference_key, False):
                self.retraining.append({
                    "condition_id": identity, "status": "blocked",
                    "reason": condition["reason"] if condition["status"] != "valid" else "matching reference eraser is invalid",
                })
                continue
            features_train = state.eraser.transform(state.standardized_train)
            features_test = state.eraser.transform(state.standardized_test)
            labels_train = np.array([self.inputs.rows[index].emotion for index in state.train])
            new_head = fit_fixed_probe(features_train, labels_train, self.probe_config, c=state.original_c)
            if new_head is state.original_head or model_hash(state.original_head) != state.original_emotion["head_hash"]:
                raise RuntimeError("M4B must not reuse or mutate the M4A head")
            np.savez_compressed(
                self.output_dir / "heads" / f"{identity}_retrained.npz", **model_arrays(new_head)
            )
            for fitting in self.fitting_ids:
                if fitting["condition_id"] == identity:
                    fitting["fitted_components"] += ",retrained_emotion_head"
            record = state.results.evaluate(
                "retrained_head", "targeted", "emotion", state.test,
                np.array([self.inputs.rows[index].emotion for index in state.test]),
                new_head.predict(features_test), new_head.predict_proba(features_test),
                new_head.named_steps["classifier"].classes_, head_digest=model_hash(new_head),
            )
            self.metrics.append(record)
            state.results.save(self.output_dir)
            self.retraining.append({
                "condition_id": identity, "status": "complete", "reason": "validated eraser and matching reference",
                "selected_c": state.original_c, "original_macro_f1": state.original_emotion["macro_f1"],
                "erased_macro_f1": record["macro_f1"],
                "delta_macro_f1": record["macro_f1"] - state.original_emotion["macro_f1"],
                "original_accuracy": state.original_emotion["accuracy"],
                "erased_accuracy": record["accuracy"],
                "delta_accuracy": record["accuracy"] - state.original_emotion["accuracy"],
                "evaluation_id": record["evaluation_id"],
            })
        condition_lookup = {row["condition_id"]: row for row in self.conditions}
        references = {
            (condition_lookup[row["condition_id"]]["target"], condition_lookup[row["condition_id"]]["layer"], condition_lookup[row["condition_id"]]["method"]): row
            for row in self.retraining
            if row["status"] == "complete" and condition_lookup[row["condition_id"]]["split_name"] == REFERENCE
        }
        for row in self.retraining:
            if row["status"] != "complete":
                continue
            condition = condition_lookup[row["condition_id"]]
            reference = references[(condition["target"], condition["layer"], condition["method"])]
            for metric in ("macro_f1", "accuracy"):
                row[f"original_reference_{metric}"] = reference[f"original_{metric}"]
                row[f"erased_reference_{metric}"] = reference[f"erased_{metric}"]
                row[f"delta_reference_{metric}"] = reference[f"delta_{metric}"]
                row[f"original_{metric}_gap"] = reference[f"original_{metric}"] - row[f"original_{metric}"]
                row[f"erased_{metric}_gap"] = reference[f"erased_{metric}"] - row[f"erased_{metric}"]
                row[f"delta_{metric}_gap"] = row[f"erased_{metric}_gap"] - row[f"original_{metric}_gap"]
        self.save_tables()

    def save_tables(self) -> None:
        tables = {
            "conditions": self.conditions, "metrics": self.metrics,
            "diagnostic_probe_fits": self.probe_fits, "collateral_damage": self.collateral,
            "targeted_validation": self.validation, "random_controls": self.controls,
            "fixed_head_reliance": self.reliance, "retrained_heads": self.retraining,
            "head_integrity": self.head_checks,
        }
        for name, rows in tables.items():
            table_with_schema(rows, self.output_dir / f"{name}.csv", ["condition_id", "status", "reason"])
        if self.fitting_ids:
            write_parquet(self.fitting_ids, self.output_dir / "training_ids.parquet")

    def run(self) -> None:
        for target in self.config["target_order"]:
            if target == "speaker" and not any(
                state.valid for state in self.states.values() if state.condition["target"] == "statement"
            ):
                for condition in self.conditions:
                    if condition["target"] == "speaker":
                        condition.update({
                            "status": "blocked_statement_gate",
                            "reason": "no statement reference condition passed the frozen validity gate",
                        })
                print("M4 gate stopped speaker expansion: no valid statement intervention.", flush=True)
                break
            for condition in self.conditions:
                if condition["target"] == target and condition["status"] == "pending":
                    self.run_condition(condition)
        self.retrain()


def run_m4(config_path: Path, output_dir: Path | None, *, validate_only: bool = False) -> None:
    config, inputs, source = load_m4_inputs(config_path)
    if validate_only:
        print(json.dumps({"selection": source["selection"], "conditions": Counter(row["status"] for row in condition_plan(inputs, config))}, indent=2))
        return
    if output_dir is None:
        raise ValueError("M4 requires a new output directory")
    if not output_dir.resolve().is_relative_to((Path.cwd() / "results").resolve()):
        raise ValueError("M4 result directory must be a new child of research2027/results")
    metadata = {
        "experiment_id": output_dir.name, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **git_provenance(), "protocol_commit": PROTOCOL_COMMIT, "config": config,
        "input_sha256": source["input_sha256"],
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": sha256_file(Path(config["protocol_file"])),
        "cache_fingerprint": inputs.metadata["fingerprint"],
        "model_revision": inputs.metadata["resolved_revision"],
        "python_version": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "matplotlib", "pyarrow", "threadpoolctl")},
        "compute": {"system": platform.system(), "device": "cpu", "threads": 1},
        "command": f'.\\.venv\\Scripts\\python.exe -m mockingbird2027 intervene --config "{config_path}" --output-dir "{output_dir}"',
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(metadata, output_dir / "provenance.json")
    write_json(source["selection"], output_dir / "selected_layers.json")
    (output_dir / "protocol.md").write_bytes(Path(config["protocol_file"]).read_bytes())
    (output_dir / "config.json").write_bytes(config_path.read_bytes())
    runner = M4Runner(config, inputs, source, output_dir)
    complete = False
    status: dict[str, Any] = {"status": "running", "milestone": "M4", "planned_conditions": len(runner.conditions)}
    write_json(status, output_dir / "status.json")
    try:
        with threadpool_limits(limits=1):
            runner.run()
        for filename, digest in source["input_sha256"].items():
            if sha256_file(Path(filename)) != digest:
                raise RuntimeError(f"immutable input changed during M4: {Path(filename).name}")
        if sha256_file(config_path) != metadata["config_sha256"] or sha256_file(Path(config["protocol_file"])) != metadata["protocol_sha256"]:
            raise RuntimeError("M4 protocol/configuration changed during the run")
        from .m4_reporting import finalize_m4

        finalize_m4(output_dir, inputs)
        complete = True
    finally:
        runner.save_tables()
        statuses = Counter(row["status"] for row in runner.conditions)
        status.update({
            "status": ("completed_gated" if any(key != "valid" for key in statuses) else "complete") if complete else "failed",
            "exit_code": 0 if complete else 1,
            "condition_status_counts": dict(statuses),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        write_json(status, output_dir / "status.json")
    write_json(
        {str(path.relative_to(output_dir)): sha256_file(path) for path in sorted(output_dir.rglob("*")) if path.is_file()},
        output_dir / "output_sha256.json",
    )
    print(f"M4 finished: {status['status']}; see {output_dir / 'summary.md'}. No replication run.", flush=True)
