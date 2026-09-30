"""One frozen, four-arm erasure-validity experiment with separate A/B/C roles."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from ..data.manifest import write_json, write_parquet
from .erasure import LEACE, Numerics
from .m3 import ATTRIBUTES, M3Inputs, git_provenance, load_m3_inputs, sha256_file, write_csv
from .m4 import parameters_hash
from .probes import ProbeConfig, classification_metrics, fit_probe
from .validity_splits import build_probe_cv, build_roles, cell_id, frozen_probe_cv, role_indices
from .validity_statistics import bootstrap_scores, cluster_bootstrap_counts, validity_decision


PROTOCOL_COMMIT = "d9fdf6575da150d0c8ce0a0ebe7070a5207137d4"


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()


def load_validity_inputs(config_path: Path) -> tuple[dict[str, Any], M3Inputs, dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for path in (config_path, Path(config["protocol_file"])):
        relative = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        frozen = subprocess.run(
            ["git", "show", f"{PROTOCOL_COMMIT}:research2027/{relative}"],
            check=True, capture_output=True, text=True,
        ).stdout
        equal = json.loads(frozen) == config if path.suffix == ".json" else frozen == path.read_text(encoding="utf-8")
        if not equal:
            raise ValueError("validity protocol/config differs from the pre-fitting commit")
    input_hashes = {}
    for name in ("m3", "m4"):
        run = Path(config[f"{name}_run"])
        manifest = run / "output_sha256.json"
        if sha256_file(manifest) != config[f"{name}_output_manifest_sha256"]:
            raise ValueError(f"original {name.upper()} result manifest changed")
        for filename, digest in json.loads(manifest.read_text()).items():
            if sha256_file(run / filename) != digest:
                raise ValueError(f"original {name.upper()} artifact changed: {filename}")
        input_hashes.update({str(path): sha256_file(path) for path in sorted(run.rglob("*")) if path.is_file()})
    m3_provenance = json.loads((Path(config["m3_run"]) / "provenance.json").read_text())
    inputs = load_m3_inputs(m3_provenance["config"])
    if inputs.metadata["fingerprint"] != config["cache_fingerprint"]:
        raise ValueError("validity study cache identity differs from its protocol")
    for name, digest in m3_provenance["input_sha256"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"original M3 scientific input changed: {Path(name).name}")
    input_hashes.update(inputs.hashes)
    for filename in ("M3_RESULTS.md", "M4_RESULTS.md", "M4_PROTOCOL.md"):
        input_hashes[filename] = sha256_file(Path(filename))
    for kind in ("roles", "cv"):
        path = Path(config[f"{kind}_file"])
        if sha256_file(path) != config[f"{kind}_sha256"]:
            raise ValueError(f"frozen {kind} IDs changed")
        input_hashes[str(path)] = sha256_file(path)
    roles = pq.read_table(config["roles_file"]).to_pylist()
    cv_records = pq.read_table(config["cv_file"]).to_pylist()
    probe_settings = {**config["linear_probe"], "c_grid": tuple(config["linear_probe"]["c_grid"])}
    probe_config = ProbeConfig(**probe_settings)
    if roles != build_roles(inputs.rows, config["seed"]):
        raise ValueError("role IDs do not match the predeclared metadata-only allocation")
    if cv_records != build_probe_cv(inputs.rows, roles, probe_config):
        raise ValueError("B-only CV IDs do not match the frozen selection rule")
    cv = frozen_probe_cv(inputs.rows, roles, cv_records, probe_config.cv_folds)
    for package in ("numpy", "scipy", "scikit-learn"):
        if version(package) != m3_provenance["packages"][package]:
            raise ValueError(f"restore the recorded {package} version before this frozen study")
    return config, inputs, {
        "roles": roles, "cv_records": cv_records, "cv": cv,
        "indices": role_indices(inputs.rows, roles), "input_sha256": input_hashes,
    }


@dataclass
class RepresentationArms:
    scaler: StandardScaler
    pca: PCA
    full_eraser: LEACE
    reduced_eraser: LEACE

    def transform(self, features: np.ndarray) -> dict[str, np.ndarray]:
        original = self.scaler.transform(features)
        reduced = self.pca.transform(original)
        return {
            "original": original,
            "leace": self.full_eraser.transform(original),
            "pca64": reduced,
            "pca64_leace": self.reduced_eraser.transform(reduced),
        }

    def parameters(self) -> dict[str, np.ndarray]:
        arrays = {
            "scaler_mean": self.scaler.mean_, "scaler_scale": self.scaler.scale_,
            "pca_mean": self.pca.mean_, "pca_components": self.pca.components_,
            "pca_explained_variance": self.pca.explained_variance_,
            "pca_explained_variance_ratio": self.pca.explained_variance_ratio_,
        }
        for name, eraser in (("full", self.full_eraser), ("reduced", self.reduced_eraser)):
            transform = eraser.transformation
            arrays.update({
                f"{name}_mean": transform.mean, f"{name}_left": transform.left,
                f"{name}_right": transform.right, f"{name}_covariance_support": eraser.eigenvalues_,
            })
        return arrays


def fit_representation_arms(features_A: np.ndarray, labels_A: np.ndarray, config: dict[str, Any]) -> RepresentationArms:
    scaler = StandardScaler().fit(features_A)
    original = scaler.transform(features_A)
    pca = PCA(
        n_components=config["pca"]["components"], svd_solver=config["pca"]["svd_solver"],
        whiten=config["pca"]["whiten"], random_state=config["seed"],
    ).fit(original)
    reduced = pca.transform(original)
    numerics = Numerics(**config["numerics"])
    return RepresentationArms(
        scaler, pca, LEACE(numerics).fit(original, labels_A),
        LEACE(numerics).fit(reduced, labels_A),
    )


def probe_parameters(model: Pipeline, probe: str) -> dict[str, np.ndarray]:
    scaler = model.named_steps["standardize"]
    classifier = model.named_steps["classifier"]
    arrays = {
        "mean": scaler.mean_, "scale": scaler.scale_, "classes": classifier.classes_,
        "c": np.array([classifier.C]), "intercept": classifier.intercept_,
        "n_iter": classifier.n_iter_,
    }
    if probe == "linear":
        arrays["coef"] = classifier.coef_
    elif probe == "rbf_svm":
        # Reference immutable B rows rather than publishing support-vector embeddings.
        arrays.update({
            "support_indices": classifier.support_, "dual_coef": classifier.dual_coef_,
            "n_support": classifier.n_support_, "gamma": np.array([classifier.gamma]),
            "support_vectors_sha256": np.array([
                hashlib.sha256(np.ascontiguousarray(classifier.support_vectors_).tobytes()).hexdigest()
            ]),
        })
    else:
        raise ValueError(f"unexpected validity probe: {probe}")
    return arrays


def fit_validity_probe(
    features_B: np.ndarray, labels_B: np.ndarray, cell_groups_B: np.ndarray,
    probe: str, config: dict[str, Any], frozen_cv: list[tuple[np.ndarray, np.ndarray]],
) -> GridSearchCV:
    settings = {**config["linear_probe"], "c_grid": tuple(config["linear_probe"]["c_grid"])}
    probe_config = ProbeConfig(**settings)
    if probe == "linear":
        search = fit_probe(features_B, labels_B, probe_config, groups=cell_groups_B)
        for actual, expected in zip(search.cv, frozen_cv):
            if not all(np.array_equal(first, second) for first, second in zip(actual, expected)):
                raise ValueError("linear probe CV differs from the committed B-only folds")
        return search
    if probe != "rbf_svm" or len(np.unique(labels_B)) != 2:
        raise ValueError("the additional probe is a binary statement-only RBF-SVM")
    policy = config["rbf_probe"]
    if policy["probability"] is not False:
        raise ValueError("the frozen RBF-SVM protocol forbids probability calibration")
    # The uncalibrated default avoids sklearn 1.9's deprecated probability argument.
    pipeline = Pipeline([
        ("standardize", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf", gamma=1.0 / features_B.shape[1], tol=policy["tol"],
            max_iter=policy["max_iter"],
            random_state=config["seed"],
        )),
    ])
    search = GridSearchCV(
        pipeline, {"classifier__C": policy["c_grid"]}, cv=frozen_cv, n_jobs=1,
        scoring=make_scorer(
            f1_score, labels=np.unique(labels_B), average="macro", pos_label=None, zero_division=0
        ),
        refit=True, error_score="raise",
    )
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(features_B, labels_B)
    if hasattr(search.best_estimator_.named_steps["classifier"], "predict_proba"):
        raise RuntimeError("the fitted SVM unexpectedly enables probability calibration")
    return search


class ValidityRunner:
    def __init__(
        self, config: dict[str, Any], inputs: M3Inputs,
        source: dict[str, Any], output_dir: Path,
    ):
        self.config, self.inputs, self.source, self.output_dir = config, inputs, source, output_dir
        self.indices = source["indices"]
        self.features = np.load(
            inputs.cache / f"layer_{config['layer']:02d}.npy", allow_pickle=False
        ).astype(np.float64)
        self.models: dict[tuple[str, str, str], Pipeline] = {}
        self.majority_labels: dict[str, str] = {}
        self.model_rows: list[dict[str, Any]] = []
        self.cv_rows: list[dict[str, Any]] = []
        self.metrics: list[dict[str, Any]] = []
        self.predictions: list[dict[str, Any]] = []
        self.bootstrap_rows: list[dict[str, Any]] = []
        self.phase_times: dict[str, Any] = {}

    def labels(self, role: str, attribute: str) -> np.ndarray:
        return np.array([str(getattr(self.inputs.rows[index], ATTRIBUTES[attribute])) for index in self.indices[role]])

    def ids(self, role: str) -> list[str]:
        return [self.inputs.rows[index].utterance_id for index in self.indices[role]]

    def fit_all(self) -> None:
        if self.models:
            raise RuntimeError("validity models must be fitted only once")
        self.phase_times["A_fitting_started_utc"] = datetime.now(timezone.utc).isoformat()
        self.arms = fit_representation_arms(self.features[self.indices["A"]], self.labels("A", "statement"), self.config)
        self.preprocessing_hash = parameters_hash(self.arms.parameters())
        np.savez_compressed(self.output_dir / "representation_parameters.npz", **self.arms.parameters())
        transform_rows = []
        for arm, eraser in (("leace", self.arms.full_eraser), ("pca64_leace", self.arms.reduced_eraser)):
            transform_rows.append({
                "arm": arm, "fit_role": "A", "n_fit": len(self.indices["A"]),
                "fit_ids_sha256": ids_hash(self.ids("A")), "removed_rank": eraser.removed_rank,
                "covariance_support_rank": len(eraser.eigenvalues_),
                "covariance_min_positive_eigenvalue": float(eraser.eigenvalues_.min()),
                "covariance_max_eigenvalue": float(eraser.eigenvalues_.max()),
                "effective_covariance_condition": float(eraser.eigenvalues_.max() / eraser.eigenvalues_.min()),
                "training_cross_covariance_residual": eraser.training_cross_covariance_residual_,
            })
        write_csv(transform_rows, self.output_dir / "transform_diagnostics.csv")
        self.phase_times["A_fitting_completed_utc"] = datetime.now(timezone.utc).isoformat()
        features_B = self.arms.transform(self.features[self.indices["B"]])
        groups_B = np.array([cell_id(self.inputs.rows[index]) for index in self.indices["B"]])
        for attribute in self.config["attributes"]:
            labels, counts = np.unique(self.labels("B", attribute), return_counts=True)
            self.majority_labels[attribute] = str(labels[counts.argmax()])
        model_directory = self.output_dir / "models"
        model_directory.mkdir()
        for arm in self.config["arms"]:
            print(f"V1 fitting B-only probes: {arm}", flush=True)
            tasks = [("linear", attribute) for attribute in self.config["attributes"]] + [("rbf_svm", "statement")]
            for probe, attribute in tasks:
                identity = f"{arm}__{probe}__{attribute}"
                search = fit_validity_probe(
                    features_B[arm], self.labels("B", attribute), groups_B, probe,
                    self.config, self.source["cv"][attribute],
                )
                model = search.best_estimator_
                self.models[(arm, probe, attribute)] = model
                parameters = probe_parameters(model, probe)
                digest = parameters_hash(parameters)
                np.savez_compressed(model_directory / f"{identity}.npz", **parameters)
                row = {
                    "model_id": identity, "arm": arm, "probe": probe, "attribute": attribute,
                    "fit_role": "B", "n_fit": len(self.indices["B"]),
                    "fit_ids_sha256": ids_hash(self.ids("B")), "dimension": features_B[arm].shape[1],
                    "selected_c": float(search.best_params_["classifier__C"]),
                    "gamma": 1.0 / features_B[arm].shape[1] if probe == "rbf_svm" else None,
                    "cv_macro_f1": float(search.best_score_),
                    "frozen_parameter_hash": digest,
                    "fit_completed_utc": datetime.now(timezone.utc).isoformat(),
                }
                self.model_rows.append(row)
                candidates = self.config["linear_probe"]["c_grid"] if probe == "linear" else self.config["rbf_probe"]["c_grid"]
                for index, c in enumerate(candidates):
                    self.cv_rows.append({
                        "model_id": identity, "attribute": attribute, "c": c,
                        "selected": bool(index == search.best_index_),
                        "mean_macro_f1": float(search.cv_results_["mean_test_score"][index]),
                        "std_macro_f1": float(search.cv_results_["std_test_score"][index]),
                        **{
                            f"cv_fold_{fold}_macro_f1": float(search.cv_results_[f"split{fold}_test_score"][index])
                            for fold in range(self.config["linear_probe"]["cv_folds"])
                        },
                    })
        expected = len(self.config["arms"]) * (len(self.config["attributes"]) + 1)
        if len(self.models) != expected:
            raise RuntimeError("not all predeclared models have been fitted")
        self.phase_times["all_models_frozen_utc"] = datetime.now(timezone.utc).isoformat()
        write_csv(self.model_rows, self.output_dir / "model_fits.csv")
        write_csv(self.cv_rows, self.output_dir / "cv_results.csv")
        write_json({
            "model_count": len(self.models), "preprocessing_hash": self.preprocessing_hash,
            "model_hashes": {row["model_id"]: row["frozen_parameter_hash"] for row in self.model_rows},
            "timestamp_utc": self.phase_times["all_models_frozen_utc"],
        }, self.output_dir / "fit_complete.json")

    def evaluate_C(self) -> dict[str, Any]:
        expected = len(self.config["arms"]) * (len(self.config["attributes"]) + 1)
        if len(self.models) != expected or self.metrics:
            raise RuntimeError("C may be evaluated once, only after all models are frozen")
        self.phase_times["C_evaluation_started_utc"] = datetime.now(timezone.utc).isoformat()
        print(f"V1 all {len(self.models)} models frozen; evaluating C.", flush=True)
        features_C = self.arms.transform(self.features[self.indices["C"]])
        speakers = np.array([self.inputs.rows[index].speaker_id for index in self.indices["C"]])
        clusters, counts = cluster_bootstrap_counts(speakers, self.config["bootstrap"])
        np.savez_compressed(self.output_dir / "bootstrap_clusters.npz", speakers=clusters, counts=counts)
        fit_lookup = {row["model_id"]: row for row in self.model_rows}
        for (arm, probe, attribute), model in self.models.items():
            identity = f"{arm}__{probe}__{attribute}"
            predicted = model.predict(features_C[arm])
            gold = self.labels("C", attribute)
            classes = model.named_steps["classifier"].classes_
            if set(classes) != set(gold):
                raise ValueError("B/C label spaces differ")
            probabilities = model.predict_proba(features_C[arm]) if probe == "linear" else None
            decisions = model.decision_function(features_C[arm]) if probe == "rbf_svm" else None
            row = {
                "model_id": identity, "arm": arm, "probe": probe, "attribute": attribute,
                "evaluation_role": "C", "n_test": len(gold), "dimension": features_C[arm].shape[1],
                "classes": json.dumps(classes.tolist()),
                "selected_c": fit_lookup[identity]["selected_c"], "gamma": fit_lookup[identity]["gamma"],
                **classification_metrics(gold, predicted, classes),
                "majority_macro_f1": None, "majority_accuracy": None,
                "uniform_chance_accuracy": 1 / len(classes),
                "macro_f1_upper_95": None, "accuracy_upper_95": None,
                "frozen_parameter_hash": fit_lookup[identity]["frozen_parameter_hash"],
            }
            majority = np.full(len(gold), self.majority_labels[attribute])
            baseline_scores = classification_metrics(gold, majority, classes)
            row["majority_macro_f1"], row["majority_accuracy"] = baseline_scores["macro_f1"], baseline_scores["accuracy"]
            if attribute == "statement":
                f1, accuracy, sizes = bootstrap_scores(gold, predicted, speakers, classes, clusters, counts)
                settings = self.config["bootstrap"]
                row["macro_f1_upper_95"] = float(np.quantile(f1, settings["quantile"], method=settings["quantile_method"]))
                row["accuracy_upper_95"] = float(np.quantile(accuracy, settings["quantile"], method=settings["quantile_method"]))
                self.bootstrap_rows.extend({
                    "model_id": identity, "draw": index, "macro_f1": float(f1[index]),
                    "accuracy": float(accuracy[index]), "n_examples": int(sizes[index]),
                } for index in range(len(f1)))
            self.metrics.append(row)
            for position, index in enumerate(self.indices["C"]):
                self.predictions.append({
                    "model_id": identity, "utterance_id": self.inputs.rows[index].utterance_id,
                    "speaker_id": self.inputs.rows[index].speaker_id, "gold_label": str(gold[position]),
                    "predicted_label": str(predicted[position]), "majority_predicted_label": str(majority[position]),
                    "classes": classes.tolist(),
                    "probabilities": probabilities[position].tolist() if probabilities is not None else None,
                    "decision_score": float(decisions[position]) if decisions is not None else None,
                    "score_kind": "probabilities" if probe == "linear" else "decision_function",
                })
        integrity = []
        for (arm, probe, attribute), model in self.models.items():
            identity = f"{arm}__{probe}__{attribute}"
            after = parameters_hash(probe_parameters(model, probe))
            before = fit_lookup[identity]["frozen_parameter_hash"]
            if before != after:
                raise RuntimeError("a frozen diagnostic model changed during C evaluation")
            integrity.append({"model_id": identity, "before_hash": before, "after_hash": after, "unchanged": True})
        if parameters_hash(self.arms.parameters()) != self.preprocessing_hash:
            raise RuntimeError("A-fitted representations changed during B/C use")
        self.phase_times["C_evaluation_completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_csv(self.metrics, self.output_dir / "metrics.csv")
        write_csv(integrity, self.output_dir / "model_integrity.csv")
        write_parquet(self.predictions, self.output_dir / "predictions.parquet")
        write_parquet(self.bootstrap_rows, self.output_dir / "bootstrap_scores.parquet")
        write_json(self.phase_times, self.output_dir / "phase_times.json")
        decision = validity_decision(self.metrics, self.config)
        write_json(decision, self.output_dir / "decision.json")
        return decision


def run_validity(config_path: Path, output_dir: Path | None, *, validate_only: bool = False) -> None:
    config, inputs, source = load_validity_inputs(config_path)
    if validate_only:
        print(json.dumps({
            "layer": config["layer"], "arms": config["arms"],
            "roles": {role: len(indices) for role, indices in source["indices"].items()},
            "planned_models": len(config["arms"]) * (len(config["attributes"]) + 1),
            "models_fitted": 0,
        }, indent=2))
        return
    if output_dir is None or not output_dir.resolve().is_relative_to((Path.cwd() / "results").resolve()):
        raise ValueError("provide a new output directory under research2027/results")
    config_name = str(config_path.resolve().relative_to(Path.cwd().resolve()))
    output_name = str(output_dir.resolve().relative_to(Path.cwd().resolve()))
    provenance = {
        "experiment_id": output_dir.name, "study": config["study"],
        **git_provenance(), "protocol_commit": PROTOCOL_COMMIT,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "config_sha256": sha256_file(config_path),
        "protocol_sha256": sha256_file(Path(config["protocol_file"])),
        "input_sha256": source["input_sha256"],
        "python_version": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "matplotlib", "pyarrow", "threadpoolctl")},
        "compute": {"system": platform.system(), "device": "cpu", "threads": 1},
        "cache_metadata": inputs.metadata,
        "command": f'.\\.venv\\Scripts\\python.exe -m mockingbird2027 validity --config "{config_name}" --output-dir "{output_name}"',
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(provenance, output_dir / "provenance.json")
    for name, path in (
        ("config.json", config_path), ("protocol.md", Path(config["protocol_file"])),
        ("roles.parquet", Path(config["roles_file"])), ("probe_cv.parquet", Path(config["cv_file"])),
    ):
        (output_dir / name).write_bytes(path.read_bytes())
    status: dict[str, Any] = {"status": "running", "scope": "validity_only", "planned_models": 24}
    write_json(status, output_dir / "status.json")
    complete = False
    try:
        runner = ValidityRunner(config, inputs, source, output_dir)
        with threadpool_limits(limits=1):
            runner.fit_all()
            decision = runner.evaluate_C()
        for name, digest in source["input_sha256"].items():
            if sha256_file(Path(name)) != digest:
                raise RuntimeError(f"immutable input changed: {Path(name).name}")
        if sha256_file(config_path) != provenance["config_sha256"] or sha256_file(Path(config["protocol_file"])) != provenance["protocol_sha256"]:
            raise RuntimeError("frozen validity protocol changed during the run")
        from .validity_reporting import finalize_validity

        with threadpool_limits(limits=1):
            finalize_validity(output_dir, inputs)
        status["validity_decision"] = decision["decision"]
        complete = True
    finally:
        status.update({
            "status": "complete" if complete else "failed", "exit_code": 0 if complete else 1,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "downstream_experiments_started": False,
        })
        write_json(status, output_dir / "status.json")
    write_json({
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(output_dir.rglob("*")) if path.is_file()
    }, output_dir / "output_sha256.json")
    print(f"V1 complete: {status['validity_decision']}. Stop for review; no downstream expansion.", flush=True)
