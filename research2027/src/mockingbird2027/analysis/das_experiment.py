"""Staged A-only discovery, frozen B diagnostics, and one locked C evaluation."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import distributions, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..data.manifest import ManifestRow, read_parquet_manifest, write_json, write_parquet
from ..data.ravdess import validate_ravdess_manifest
from ..models.extract import manifest_hash
from .das import (
    DISCOVERY_ROLES, IMPLEMENTATION_VERSION, arbitrary_targets, binary_labels,
    build_discovery_split, collateral_check, erasure_decision, fit_alignment,
    fit_readout, interchange_scores, matched_partners, mean_ablation,
    random_subspace_controls, readout_parameters, select_candidate,
    validate_discovery_split,
)
from .erasure import AffineRemoval
from .m3 import ATTRIBUTES, git_provenance, read_csv, sha256_file, write_csv
from .m4 import parameters_hash
from .probes import ProbeConfig, classification_metrics, inner_splits
from .validity import fit_validity_probe, ids_hash, probe_parameters
from .validity_splits import cell_id, frozen_probe_cv, validate_roles
from .validity_statistics import bootstrap_scores, cluster_bootstrap_counts


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def scientific_hashes(config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    paths = [
        config_path, Path(config["protocol_file"]), Path(config["split_file"]),
        Path("pyproject.toml"), Path("requirements-das.txt"),
        *sorted(Path("src").rglob("*.py")),
    ]
    return {str(path): sha256_file(path) for path in paths}


def check_hashes(hashes: dict[str, str], root: Path = Path(".")) -> None:
    for name, digest in hashes.items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"immutable artifact changed: {name}")


def require_committed_freeze(output: Path) -> None:
    root = Path(subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True,
    ).stdout.strip()).resolve()
    for name in ("candidate_frozen.json", "fit_complete.json"):
        path = (output / name).resolve()
        repository_path = path.relative_to(root).as_posix()
        committed = subprocess.run(
            ["git", "show", f"HEAD:{repository_path}"], check=True, capture_output=True,
        ).stdout
        if committed != path.read_bytes():
            raise RuntimeError("commit the unchanged candidate and B-model freeze before opening C")


@dataclass
class DASInputs:
    rows: list[ManifestRow]
    config: dict[str, Any]
    metadata: dict[str, Any]
    assignments: list[dict[str, str]]
    cv: dict[str, list[tuple[np.ndarray, np.ndarray]]]
    hashes: dict[str, str]

    def indices(self, role: str) -> np.ndarray:
        if role not in DISCOVERY_ROLES:
            raise ValueError(f"unknown discovery role: {role}")
        ids = {item["utterance_id"] for item in self.assignments if item["role"] == role}
        return np.array([index for index, row in enumerate(self.rows) if row.utterance_id in ids], dtype=np.int64)

    def role_rows(self, role: str) -> list[ManifestRow]:
        return [self.rows[index] for index in self.indices(role)]

    def labels(self, role: str, attribute: str) -> np.ndarray:
        return np.array([str(getattr(row, ATTRIBUTES[attribute])) for row in self.role_rows(role)])


def load_inputs(config_path: Path, *, prepare: bool = False) -> DASInputs:
    config = read_json(config_path)
    if Path.cwd().resolve() != Path(__file__).resolve().parents[3]:
        raise ValueError("run DAS commands from the research2027 directory")
    if config["layer"] != 5 or config["cache_fingerprint"] != "d99979b610d68bef":
        raise ValueError("this authorized DAS follow-up uses only the frozen layer-5 cache")
    if (
        config["ablation"] != "mean_A_train_in_standardized_feature_space"
        or config["compute"] != {"device": "cpu", "threads": 1, "dtype": "float64"}
        or config["das"]["orthogonal_map"] != "householder"
        or config["das"]["use_trivialization"] is not False
        or config["readout"]["activation"] != "tanh"
        or config["selection"]["refit_after_selection"] is not False
        or config["no_downstream_expansion"] is not True
    ):
        raise ValueError("unsupported DAS ablation, compute, optimizer or expansion policy")
    for name in ("after_max_macro_f1", "after_max_accuracy", "upper_confidence_ceiling"):
        if config["validity"][name] != 0.55:
            raise ValueError("the exploratory follow-up must preserve the 0.55 ceiling")
    if config["attributes"] != list(ATTRIBUTES):
        raise ValueError("all five independent diagnostics must be retained")
    dimensions = config["das"]["dimensions"]
    if dimensions != sorted(set(dimensions)) or not dimensions or not 0 < min(dimensions) <= max(dimensions) < 768:
        raise ValueError("DAS dimensions must be unique, increasing and inside (0, 768)")
    seeds = config["das"]["seeds"]
    if not seeds or seeds != sorted(set(seeds)):
        raise ValueError("DAS seeds must be nonempty, unique and increasing")
    for optimizer in (config["readout"], config["das"]):
        if (
            optimizer["steps"] < 1 or optimizer["evaluation_interval"] < 1
            or not np.isfinite(optimizer["learning_rate"]) or optimizer["learning_rate"] <= 0
        ):
            raise ValueError("invalid fixed training schedule")

    hashes = {}
    for experiment in (
        "m3-wavlm-ravdess-20260930", "m4-wavlm-ravdess-20260930",
        "m4-validity-v1-wavlm-ravdess-20260930",
    ):
        root = Path("results") / experiment
        manifest = root / "output_sha256.json"
        check_hashes(read_json(manifest), root)
        for path in sorted(root.rglob("*")):
            if path.is_file():
                hashes[str(path)] = sha256_file(path)
    previous = read_json(Path("results") / "m3-wavlm-ravdess-20260930" / "provenance.json")
    check_hashes(previous["input_sha256"])
    hashes.update(previous["input_sha256"])
    rows = sorted(read_parquet_manifest(Path(config["manifest"])), key=lambda row: row.utterance_id)
    validate_ravdess_manifest(rows)
    cache = Path(config["cache"])
    metadata = read_json(cache / "metadata.json")
    if (
        metadata["fingerprint"] != config["cache_fingerprint"] or metadata["manifest_hash"] != manifest_hash(rows)
        or metadata["status"] != "complete" or metadata["storage_dtype"] != "float16"
        or (metadata["utterance_count"], metadata["layer_count"], metadata["hidden_size"]) != (1440, 13, 768)
    ):
        raise ValueError("DAS cache identity/shape differs from the frozen extraction")
    cache_ids = np.load(cache / "ids.npy", allow_pickle=False)
    if cache_ids.tolist() != [row.utterance_id for row in rows]:
        raise ValueError("DAS cache IDs are not aligned with the manifest")
    for kind in ("roles", "cv"):
        path = Path(config[f"{kind}_file"])
        if sha256_file(path) != config[f"{kind}_sha256"]:
            raise ValueError(f"frozen {kind} IDs changed")
        hashes[str(path)] = sha256_file(path)
    roles = pq.read_table(config["roles_file"]).to_pylist()
    validate_roles(rows, roles)
    assignments = build_discovery_split(rows, roles, config["seed"])
    if not prepare:
        frozen = read_json(Path(config["split_file"]))
        if frozen["assignments"] != assignments or frozen["seed"] != config["seed"]:
            raise ValueError("A subdivision differs from its persisted metadata-only specification")
        hashes[config["split_file"]] = sha256_file(Path(config["split_file"]))
    validate_discovery_split(rows, roles, assignments)
    cv_records = pq.read_table(config["cv_file"]).to_pylist()
    cv = frozen_probe_cv(rows, roles, cv_records, config["linear_probe"]["cv_folds"])
    return DASInputs(rows, config, metadata, assignments, cv, hashes)


def prepare_split(config_path: Path) -> None:
    inputs = load_inputs(config_path, prepare=True)
    payload = {
        "seed": inputs.config["seed"], "parent_roles_sha256": inputs.config["roles_sha256"],
        "assignments": inputs.assignments,
        "ids_sha256": {
            role: ids_hash([row.utterance_id for row in inputs.role_rows(role)]) for role in DISCOVERY_ROLES
        },
    }
    path = Path(inputs.config["split_file"])
    if path.exists():
        if read_json(path) != payload:
            raise ValueError("refusing to replace an existing discovery split")
        print("Existing metadata-only DAS split matches; no features fitted or C inspected.")
        return
    write_json(payload, path)
    print("Persisted A_train/A_val/B/C = 480/240/360/360; no feature values or C outcomes inspected.")


class FeatureAccess:
    def __init__(self, inputs: DASInputs, output: Path):
        self.inputs, self.output = inputs, output
        self._c_open = False
        self.reads: list[dict[str, Any]] = []

    def read(self, role: str) -> np.ndarray:
        if role == "C" and not self._c_open:
            raise RuntimeError("C is locked until the candidate and all B models are frozen")
        if role == "B" and not (self.output / "candidate_frozen.json").is_file():
            raise RuntimeError("B is unavailable until the candidate is frozen")
        if self._c_open and (role != "C" or any(item["role"] == "C" for item in self.reads)):
            raise RuntimeError("after opening C, no fitting-role access or second C read is allowed")
        raw = np.load(
            Path(self.inputs.config["cache"]) / f"layer_{self.inputs.config['layer']:02d}.npy",
            allow_pickle=False, mmap_mode="r",
        )
        if raw.shape != (len(self.inputs.rows), self.inputs.metadata["hidden_size"]):
            raise ValueError("cached layer has an unexpected shape")
        values = np.asarray(raw[self.inputs.indices(role)], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite features in permitted role {role}")
        self.reads.append({
            "role": role, "count": len(values), "timestamp_utc": utc_now(),
            "ids_sha256": ids_hash([row.utterance_id for row in self.inputs.role_rows(role)]),
        })
        return values

    def open_C(self, model_hashes: dict[str, str], transform_hash: str, evaluation_commit: str) -> None:
        frozen = read_json(self.output / "fit_complete.json")
        if (
            frozen["model_hashes"] != model_hashes or frozen["model_count"] != len(model_hashes)
            or frozen["model_count"] != frozen["expected_model_count"]
            or frozen["transform_hash"] != transform_hash or frozen["C_access_count"] != 0
        ):
            raise RuntimeError("C requires the complete, unchanged frozen model/transform inventory")
        check_hashes(frozen["training_artifact_sha256"], self.output)
        marker = {
            "timestamp_utc": utc_now(), "evaluation_commit": evaluation_commit,
            "fit_complete_sha256": sha256_file(self.output / "fit_complete.json"),
            "candidate_sha256": sha256_file(self.output / "candidate_frozen.json"),
            "model_count": len(model_hashes),
        }
        with (self.output / "C_opened.json").open("x", encoding="utf-8") as stream:
            json.dump(marker, stream, indent=2)
            stream.write("\n")
        self._c_open = True


@dataclass
class AblationArms:
    scaler: StandardScaler
    removals: dict[str, AffineRemoval]

    def transform(self, values: np.ndarray) -> dict[str, np.ndarray]:
        standardized = self.scaler.transform(values)
        return {"original": standardized, **{
            name: removal.transform(standardized) for name, removal in self.removals.items()
        }}

    def parameters(self) -> dict[str, np.ndarray]:
        arrays = {"scaler_mean": self.scaler.mean_, "scaler_scale": self.scaler.scale_}
        for name, removal in self.removals.items():
            arrays.update({
                f"{name}__mean": removal.mean, f"{name}__basis": removal.right,
            })
        return arrays


def diagnostic_tasks(config: dict[str, Any]) -> list[tuple[str, str]]:
    return [("linear", name) for name in config["attributes"]] + [("rbf_svm", "statement")]


def cv_rows(search: Any, identity: str, role: str, probe: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config["linear_probe" if probe == "linear" else "rbf_probe"]["c_grid"]
    return [{
        "model_id": identity, "training_role": role, "c": float(c),
        "selected": index == search.best_index_,
        "mean_macro_f1": float(search.cv_results_["mean_test_score"][index]),
        **{
            f"fold_{fold}_macro_f1": float(search.cv_results_[f"split{fold}_test_score"][index])
            for fold in range(config["linear_probe"]["cv_folds"])
        },
    } for index, c in enumerate(grid)]


def fit_a_diagnostics(
    inputs: DASInputs, training: np.ndarray, validation: np.ndarray, identity: str,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    config = inputs.config
    groups = np.array([cell_id(row) for row in inputs.role_rows("A_train")])
    probe_config = ProbeConfig(**{**config["linear_probe"], "c_grid": tuple(config["linear_probe"]["c_grid"])})
    f1, metrics, selection_rows = {}, [], []
    for probe, attribute in diagnostic_tasks(config):
        labels = inputs.labels("A_train", attribute)
        splits = inner_splits(labels, probe_config, groups)
        model_id = f"{identity}__{probe}__{attribute}"
        search = fit_validity_probe(training, labels, groups, probe, config, splits)
        prediction = search.best_estimator_.predict(validation)
        classes = search.best_estimator_.named_steps["classifier"].classes_
        scores = classification_metrics(inputs.labels("A_val", attribute), prediction, classes)
        key = attribute if probe == "linear" else "statement_rbf"
        f1[key] = scores["macro_f1"]
        metrics.append({
            "candidate_id": identity, "probe": probe, "attribute": attribute,
            "fitting_role": "A_train", "evaluation_role": "A_val", **scores,
            "selected_c": float(search.best_params_["classifier__C"]),
        })
        selection_rows.extend(cv_rows(search, model_id, "A_train", probe, config))
    return f1, metrics, selection_rows


def subspace_rows(arms: AblationArms, values: np.ndarray, role: str) -> list[dict[str, Any]]:
    features = arms.scaler.transform(values)
    records = []
    for name, removal in arms.removals.items():
        q = removal.right
        records.append({
            "arm": name, "role": role, "count": len(values),
            "removed_rank": removal.removed_rank,
            "orthogonality_error": float(np.linalg.norm(q.T @ q - np.eye(q.shape[1]))),
            **removal.perturbation(features),
        })
    return records


def fit_study(config_path: Path, output: Path, inputs: DASInputs | None = None) -> None:
    inputs = load_inputs(config_path) if inputs is None else inputs
    config = inputs.config
    provenance = {
        "experiment_id": output.name, "study": config["study"], **git_provenance(),
        "parent_experiment_commit": config["parent_commit"], "start_utc": utc_now(),
        "config_sha256": sha256_file(config_path), "scientific_source_sha256": scientific_hashes(config_path, config),
        "input_sha256": inputs.hashes, "cache_metadata": inputs.metadata,
        "adapter_version": IMPLEMENTATION_VERSION,
        "packages": {name: version(name) for name in (
            "torch", "numpy", "scipy", "scikit-learn", "pyarrow", "joblib", "matplotlib", "threadpoolctl",
        )},
        "environment_packages": dict(sorted(
            (dist.metadata["Name"], dist.version) for dist in distributions() if dist.metadata["Name"]
        )),
        "python_version": platform.python_version(),
        "compute": {**config["compute"], "system": platform.system(), "gpu_used": False},
        "commands": {
            phase: f'.\\.venv\\Scripts\\python.exe -m mockingbird2027 das {phase} --output-dir "{output}"'
            for phase in ("fit", "evaluate", "verify")
        },
    }
    if not output.resolve().is_relative_to((Path.cwd() / "results").resolve()):
        raise ValueError("DAS results must use a new directory under research2027/results")
    output.mkdir(parents=True, exist_ok=False)
    write_json(provenance, output / "provenance.json")
    write_json(config, output / "config.json")
    (output / "protocol.md").write_bytes(Path(config["protocol_file"]).read_bytes())
    (output / "split_manifest.json").write_bytes(Path(config["split_file"]).read_bytes())
    (output / "probe_cv.parquet").write_bytes(Path(config["cv_file"]).read_bytes())
    status = {"status": "fitting", "C_opened": False, "downstream_experiments_started": False}
    write_json(status, output / "status.json")
    access = FeatureAccess(inputs, output)
    start = time.perf_counter()
    complete = False
    try:
        torch.set_num_threads(config["compute"]["threads"])
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=config["compute"]["threads"]):
            raw_train, raw_val = access.read("A_train"), access.read("A_val")
            scaler = StandardScaler().fit(raw_train)
            train, val = scaler.transform(raw_train), scaler.transform(raw_val)
            train_rows, val_rows = inputs.role_rows("A_train"), inputs.role_rows("A_val")
            partners, val_partners = matched_partners(train_rows), matched_partners(val_rows)
            train_y, val_y = binary_labels(train_rows), binary_labels(val_rows)
            pairs = [{
                "role": role, "base_id": row.utterance_id,
                "source_id": rows[partner[index]].utterance_id, "cell_id": cell_id(row),
                "actor": row.speaker_id, "emotion": row.emotion,
                "intensity": row.intensity, "repetition": row.repetition,
                "base_statement": row.statement_id, "source_statement": rows[partner[index]].statement_id,
            } for role, rows, partner in (
                ("A_train", train_rows, partners), ("A_val", val_rows, val_partners)
            ) for index, row in enumerate(rows)]
            write_csv(pairs, output / "matched_pairs.csv")
            readout, history, checkpoint = fit_readout(train, train_y, val, val_y, config["readout"])
            write_csv(history, output / "readout_training.csv")
            readout_hash = parameters_hash(readout_parameters(readout))
            np.savez_compressed(output / "statement_readout.npz", **readout_parameters(readout))
            capability = interchange_scores(readout, val, val_partners, val_y, None)
            write_json({
                "checkpoint": checkpoint, "parameter_hash": readout_hash,
                "training_role": "A_train", "selection_role": "A_val", **capability,
            }, output / "readout_frozen.json")
            if capability["accuracy"] < config["readout"]["minimum_validation_accuracy"]:
                status["reason"] = "INCONCLUSIVE_BEHAVIORAL_READOUT_COMPETENCE"
                raise RuntimeError("behavioral readout fails the predeclared A_val competence check; C remains locked")

            baseline, diagnostics, tuning = fit_a_diagnostics(inputs, train, val, "original")
            candidates, training_records, saved = [], [], {}
            candidate_dir = output / "candidates"
            candidate_dir.mkdir()
            mean = train.mean(axis=0)
            for rank in config["das"]["dimensions"]:
                for seed in config["das"]["seeds"]:
                    identity = f"k{rank:03d}_seed{seed}"
                    print(f"DAS A-only candidate: {identity}", flush=True)
                    candidate_start = time.perf_counter()
                    fit = fit_alignment(
                        readout, train, partners, train_y[partners],
                        val, val_partners, val_y[val_partners],
                        rank, seed, config["das"],
                    )
                    removal = mean_ablation(mean, fit.basis)
                    scores, metrics, cv = fit_a_diagnostics(
                        inputs, removal.transform(train), removal.transform(val), identity,
                    )
                    collateral = collateral_check(baseline, scores, config["validity"])
                    behavioral = interchange_scores(readout, val, val_partners, val_y[val_partners], fit.basis)
                    row = {
                        "candidate_id": identity, "dimension": rank, "seed": seed,
                        "checkpoint": fit.checkpoint, "evaluation_role": "A_val",
                        "interchange_accuracy": behavioral["accuracy"],
                        "interchange_cross_entropy": behavioral["cross_entropy"],
                        "linear_statement_f1": scores["statement"],
                        "rbf_statement_f1": scores["statement_rbf"],
                        "emotion_f1_delta": collateral["delta_macro_f1"]["emotion"],
                        "collateral_pass": collateral["pass"],
                        "eligible": behavioral["accuracy"] >= config["das"]["minimum_validation_iia"] and collateral["pass"],
                        "train_rms": removal.perturbation(train)["rms"],
                        "orthogonality_error": float(np.linalg.norm(fit.basis.T @ fit.basis - np.eye(rank))),
                        "runtime_seconds": time.perf_counter() - candidate_start,
                        "selected": False,
                    }
                    candidates.append(row)
                    diagnostics.extend(metrics)
                    tuning.extend(cv)
                    training_records.extend({"candidate_id": identity, **item} for item in fit.records)
                    saved[identity] = removal
                    np.savez_compressed(candidate_dir / f"{identity}.npz", basis=fit.basis, mean=mean)
                    write_csv(candidates, output / "candidate_selection.csv")
                    write_csv(training_records, output / "das_training.csv")
            selected, selection_status = select_candidate(candidates, config["das"]["minimum_validation_iia"])
            selected["selected"] = True
            for row in candidates:
                row["selected"] = row["candidate_id"] == selected["candidate_id"]
            write_csv(candidates, output / "candidate_selection.csv")
            write_csv(diagnostics, output / "A_validation_diagnostics.csv")
            write_csv(tuning, output / "A_probe_selection.csv")
            removal = saved[selected["candidate_id"]]
            frozen_candidate = {
                "timestamp_utc": utc_now(), "selection_status": selection_status,
                "selected": selected, "selection_rule": config["selection"],
                "ablation": config["ablation"], "layer": config["layer"],
                "readout_parameter_hash": readout_hash,
                "selected_parameters_sha256": sha256_file(candidate_dir / f"{selected['candidate_id']}.npz"),
                "fitting_role": "A_train", "selection_role": "A_val", "B_or_C_used_for_selection": False,
            }
            write_json(frozen_candidate, output / "candidate_frozen.json")
            print(f"DAS selected before B/C: {selected['candidate_id']} ({selection_status})", flush=True)

            control_seed = config["arbitrary_control"]["seed"]
            noise_y, train_flips = arbitrary_targets(train_rows, partners, control_seed)
            noise_vy, val_flips = arbitrary_targets(val_rows, val_partners, control_seed + 1)
            arbitrary = fit_alignment(
                readout, train, partners, noise_y, val, val_partners, noise_vy,
                selected["dimension"], control_seed, config["das"],
            )
            np.savez_compressed(output / "arbitrary_control_subspace.npz", basis=arbitrary.basis, mean=mean)
            write_csv(arbitrary.records, output / "arbitrary_control_training.csv")
            write_json({
                "train_cell_flips": train_flips, "validation_cell_flips": val_flips,
                "seed": control_seed, "checkpoint": arbitrary.checkpoint,
                "scope": "A-only binary arbitrary-mapping stress test, not CausalGym dog/give selectivity",
            }, output / "arbitrary_control_mapping.json")
            behavior = []
            for role, values, partner, labels, fake in (
                ("A_train", train, partners, train_y, noise_y),
                ("A_val", val, val_partners, val_y, noise_vy),
            ):
                for condition, q, target, full in (
                    ("no_op_source_target", None, labels[partner], False),
                    ("full_source_swap", None, labels[partner], True),
                    ("selected_DAS", removal.right, labels[partner], False),
                    ("arbitrary_mapping_DAS", arbitrary.basis, fake, False),
                    ("arbitrary_mapping_no_op", None, fake, False),
                ):
                    behavior.append({
                        "role": role, "condition": condition,
                        "readout_parameter_hash": readout_hash,
                        **interchange_scores(readout, values, partner, target, q, full_swap=full),
                    })
            write_csv(behavior, output / "interchange_controls.csv")
            if parameters_hash(readout_parameters(readout)) != readout_hash:
                raise RuntimeError("DAS changed the frozen behavioral readout")

            controls, pool = random_subspace_controls(train, removal, config["random_controls"])
            write_csv(pool, output / "random_control_pool.csv")
            arms = AblationArms(scaler, {"das_mean": removal, **controls})
            arm_parameters = arms.parameters()
            np.savez_compressed(output / "transform_parameters.npz", **arm_parameters)
            raw_b = access.read("B")
            b = arms.transform(raw_b)
            fit_subspaces = [
                row for role, features in (("A_train", raw_train), ("A_val", raw_val), ("B", raw_b))
                for row in subspace_rows(arms, features, role)
            ]
            write_csv(fit_subspaces, output / "subspace_fit_diagnostics.csv")
            models, model_rows, selections = {}, [], []
            model_directory = output / "models"
            model_directory.mkdir()
            groups = np.array([cell_id(row) for row in inputs.role_rows("B")])
            for arm, features in b.items():
                print(f"Fitting independent B diagnostics: {arm}", flush=True)
                for probe, attribute in diagnostic_tasks(config):
                    identity = f"{arm}__{probe}__{attribute}"
                    search = fit_validity_probe(
                        features, inputs.labels("B", attribute), groups, probe, config, inputs.cv[attribute],
                    )
                    model = search.best_estimator_
                    models[identity] = model
                    parameters = probe_parameters(model, probe)
                    np.savez_compressed(model_directory / f"{identity}.npz", **parameters)
                    model_rows.append({
                        "model_id": identity, "arm": arm, "probe": probe, "attribute": attribute,
                        "fitting_role": "B", "n_fit": len(features),
                        "fit_ids_sha256": ids_hash([row.utterance_id for row in inputs.role_rows("B")]),
                        "dimension": features.shape[1],
                        "selected_c": float(search.best_params_["classifier__C"]),
                        "gamma": 1.0 / features.shape[1] if probe == "rbf_svm" else None,
                        "parameter_hash": parameters_hash(parameters), "fit_completed_utc": utc_now(),
                    })
                    selections.extend(cv_rows(search, identity, "B", probe, config))
            write_csv(model_rows, output / "model_fits.csv")
            write_csv(selections, output / "B_probe_selection.csv")
            write_json(access.reads, output / "feature_access_fit.json")
            private = Path("artifacts") / "das_fits" / output.name
            private.mkdir(parents=True, exist_ok=False)
            private_file = private / "frozen_models.joblib"
            joblib.dump({"models": models, "arms": arms}, private_file)
            complete_at = utc_now()
            freeze = {
                "timestamp_utc": complete_at, "model_count": len(models),
                "expected_model_count": len(b) * len(diagnostic_tasks(config)),
                "model_hashes": {row["model_id"]: row["parameter_hash"] for row in model_rows},
                "transform_hash": parameters_hash(arm_parameters),
                "private_state_file": str(private_file),
                "private_state_sha256": sha256_file(private_file),
                "fit_runtime_seconds": time.perf_counter() - start,
                "C_access_count": sum(item["role"] == "C" for item in access.reads),
                "training_artifact_sha256": {
                    str(path.relative_to(output)): sha256_file(path)
                    for path in sorted(output.rglob("*")) if path.is_file() and path.name != "status.json"
                },
            }
            write_json(freeze, output / "fit_complete.json")
            check_hashes(inputs.hashes)
            complete = True
    finally:
        status.update({
            "status": "frozen_awaiting_C" if complete else "fit_failed",
            "fit_finished_utc": utc_now(), "fit_runtime_seconds": time.perf_counter() - start,
        })
        write_json(status, output / "status.json")
    print("Candidate and every B model frozen. Commit this bundle before the separate evaluate command.", flush=True)


def evaluate_study(config_path: Path, output: Path, inputs: DASInputs | None = None) -> None:
    inputs = load_inputs(config_path) if inputs is None else inputs
    config = inputs.config
    provenance = read_json(output / "provenance.json")
    if read_json(output / "config.json") != config:
        raise ValueError("evaluation config differs from the frozen fit")
    if (output / "C_opened.json").exists():
        raise RuntimeError("C was already opened for this run; verify saved predictions instead of rescoring")
    require_committed_freeze(output)
    check_hashes(provenance["scientific_source_sha256"])
    frozen = read_json(output / "fit_complete.json")
    check_hashes(frozen["training_artifact_sha256"], output)
    private = Path(frozen["private_state_file"])
    if sha256_file(private) != frozen["private_state_sha256"]:
        raise ValueError("local frozen model state changed; do not load it or refit after C")
    state = joblib.load(private)
    models, arms = state["models"], state["arms"]
    fits = read_csv(output / "model_fits.csv")
    expected = {
        f"{arm}__{probe}__{attribute}"
        for arm in ("original", *arms.removals) for probe, attribute in diagnostic_tasks(config)
    }
    if set(models) != expected or {row["model_id"] for row in fits} != expected:
        raise ValueError("not every predeclared independent diagnostic was frozen")
    hashes = {
        row["model_id"]: parameters_hash(probe_parameters(models[row["model_id"]], row["probe"]))
        for row in fits
    }
    transform_hash = parameters_hash(arms.parameters())
    evaluation_git = git_provenance()
    access = FeatureAccess(inputs, output)
    access.open_C(hashes, transform_hash, evaluation_git["git_commit"])
    start = time.perf_counter()
    complete = False
    status = read_json(output / "status.json")
    status.update({"status": "evaluating_C", "C_opened": True})
    write_json(status, output / "status.json")
    try:
        with threadpool_limits(limits=config["compute"]["threads"]):
            raw_c = access.read("C")
            features = arms.transform(raw_c)
            c_rows = inputs.role_rows("C")
            speakers = np.array([row.speaker_id for row in c_rows])
            clusters, counts = cluster_bootstrap_counts(speakers, config["bootstrap"])
            np.savez_compressed(output / "bootstrap_clusters.npz", speakers=clusters, counts=counts)
            metrics, predictions, bootstrap = [], [], []
            for fit in fits:
                identity = fit["model_id"]
                model = models[identity]
                arm, probe, attribute = fit["arm"], fit["probe"], fit["attribute"]
                values = features[arm]
                gold = inputs.labels("C", attribute)
                classes = model.named_steps["classifier"].classes_
                prediction = model.predict(values)
                scores = classification_metrics(gold, prediction, classes)
                b_classes, b_counts = np.unique(inputs.labels("B", attribute), return_counts=True)
                majority = np.full(len(gold), b_classes[b_counts.argmax()])
                baseline = classification_metrics(gold, majority, classes)
                metric = {
                    "model_id": identity, "arm": arm, "probe": probe, "attribute": attribute,
                    "evaluation_role": "C", "n_test": len(gold), "classes": classes.tolist(),
                    **scores, "macro_f1_upper_95": None, "accuracy_upper_95": None,
                    "majority_macro_f1": baseline["macro_f1"], "majority_accuracy": baseline["accuracy"],
                    "uniform_chance_accuracy": 1 / len(classes),
                    "parameter_hash": fit["parameter_hash"],
                }
                if attribute == "statement":
                    f1, acc, sizes = bootstrap_scores(gold, prediction, speakers, classes, clusters, counts)
                    settings = config["bootstrap"]
                    metric["macro_f1_upper_95"] = float(np.quantile(f1, settings["quantile"], method=settings["quantile_method"]))
                    metric["accuracy_upper_95"] = float(np.quantile(acc, settings["quantile"], method=settings["quantile_method"]))
                    bootstrap.extend({
                        "model_id": identity, "draw": index, "macro_f1": float(f1[index]),
                        "accuracy": float(acc[index]), "n_examples": int(sizes[index]),
                    } for index in range(len(f1)))
                metrics.append(metric)
                probabilities = model.predict_proba(values) if probe == "linear" else None
                decisions = model.decision_function(values) if probe == "rbf_svm" else None
                predictions.extend({
                    "model_id": identity, "utterance_id": row.utterance_id,
                    "speaker_id": row.speaker_id, "gold_label": str(gold[index]),
                    "predicted_label": str(prediction[index]), "classes": classes.tolist(),
                    "majority_label": str(majority[index]),
                    "probabilities": probabilities[index].tolist() if probabilities is not None else None,
                    "decision_score": float(decisions[index]) if decisions is not None else None,
                } for index, row in enumerate(c_rows))
            for fit in fits:
                if parameters_hash(probe_parameters(models[fit["model_id"]], fit["probe"])) != fit["parameter_hash"]:
                    raise RuntimeError("a B-fitted model changed during C evaluation")
            if parameters_hash(arms.parameters()) != transform_hash:
                raise RuntimeError("an A-fitted transform changed during C evaluation")
            write_json(metrics, output / "metrics.json")
            csv_metrics = [{**row, "classes": json.dumps(row["classes"])} for row in metrics]
            write_csv(csv_metrics, output / "metrics.csv")
            write_csv([row for row in csv_metrics if row["attribute"] == "statement"], output / "target_validity.csv")
            write_parquet(predictions, output / "predictions.parquet")
            write_parquet(bootstrap, output / "bootstrap_scores.parquet")
            collateral = collateral_rows(metrics)
            write_csv(collateral, output / "collateral.csv")
            fit_subspaces = read_csv(output / "subspace_fit_diagnostics.csv")
            write_csv([*fit_subspaces, *subspace_rows(arms, raw_c, "C")], output / "subspace_diagnostics.csv")
            write_json(erasure_decision(metrics, config["validity"]), output / "decision.json")
            write_json({
                **evaluation_git, "finished_utc": utc_now(),
                "runtime_seconds": time.perf_counter() - start, "accesses": access.reads,
                "all_model_hashes_unchanged": True, "transform_hash_unchanged": True,
            }, output / "evaluation.json")
            check_hashes(inputs.hashes)
            verify_study(output, inputs, check_outputs=False)
            plot_results(output)
            complete = True
    finally:
        status.update({
            "status": "complete" if complete else "evaluation_failed",
            "evaluation_finished_utc": utc_now(),
            "evaluation_runtime_seconds": time.perf_counter() - start,
        })
        write_json(status, output / "status.json")
    write_json({
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "output_sha256.json"
    }, output / "output_sha256.json")
    print(f"Locked C result: {read_json(output / 'decision.json')['decision']}. Stop; no reliance or rescue experiment.", flush=True)


def collateral_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    original = {row["attribute"]: row for row in metrics if row["arm"] == "original" and row["probe"] == "linear"}
    return [{
        "arm": row["arm"], "attribute": row["attribute"], "evaluation_role": "C",
        "before_macro_f1": original[row["attribute"]]["macro_f1"],
        "after_macro_f1": row["macro_f1"],
        "delta_macro_f1": row["macro_f1"] - original[row["attribute"]]["macro_f1"],
        "before_accuracy": original[row["attribute"]]["accuracy"],
        "after_accuracy": row["accuracy"],
        "delta_accuracy": row["accuracy"] - original[row["attribute"]]["accuracy"],
    } for row in metrics if row["arm"] != "original" and row["probe"] == "linear" and row["attribute"] != "statement"]


def verify_study(output: Path, inputs: DASInputs, *, check_outputs: bool = True) -> dict[str, Any]:
    config = inputs.config
    frozen = read_json(output / "fit_complete.json")
    check_hashes(frozen["training_artifact_sha256"], output)
    check_hashes(inputs.hashes)
    if check_outputs:
        check_hashes(read_json(output / "output_sha256.json"), output)
    opened = read_json(output / "C_opened.json")
    if (
        datetime.fromisoformat(frozen["timestamp_utc"]) > datetime.fromisoformat(opened["timestamp_utc"])
        or frozen["C_access_count"] != 0
        or opened["fit_complete_sha256"] != sha256_file(output / "fit_complete.json")
    ):
        raise ValueError("C was not protected by the pre-evaluation freeze")
    candidate = read_json(output / "candidate_frozen.json")
    if candidate["B_or_C_used_for_selection"] is not False:
        raise ValueError("candidate selection used a forbidden role")
    if datetime.fromisoformat(candidate["timestamp_utc"]) > datetime.fromisoformat(frozen["timestamp_utc"]):
        raise ValueError("candidate was frozen after the B model inventory")
    candidates = []
    for row in read_csv(output / "candidate_selection.csv"):
        candidates.append({
            **row, "dimension": int(row["dimension"]), "seed": int(row["seed"]),
            "interchange_accuracy": float(row["interchange_accuracy"]),
            "linear_statement_f1": float(row["linear_statement_f1"]),
            "rbf_statement_f1": float(row["rbf_statement_f1"]),
            "collateral_pass": row["collateral_pass"] == "True",
        })
    selected, selection_status = select_candidate(candidates, config["das"]["minimum_validation_iia"])
    if (
        selected["candidate_id"] != candidate["selected"]["candidate_id"]
        or selection_status != candidate["selection_status"]
        or [row["candidate_id"] for row in candidates if row["selected"] == "True"] != [selected["candidate_id"]]
        or {(row["dimension"], row["seed"]) for row in candidates} != {
            (rank, seed) for rank in config["das"]["dimensions"] for seed in config["das"]["seeds"]
        }
    ):
        raise ValueError("candidate selection does not reproduce the declared A-only rule/grid")
    metrics = read_json(output / "metrics.json")
    csv_metrics = read_csv(output / "metrics.csv")
    fits = {row["model_id"]: row for row in read_csv(output / "model_fits.csv")}
    if len(metrics) != frozen["expected_model_count"] or set(fits) != set(frozen["model_hashes"]):
        raise ValueError("diagnostic model inventory is incomplete")
    expected = {
        f"{arm}__{probe}__{attribute}"
        for arm in ("original", "das_mean", *(f"random_{i:02d}" for i in range(config["random_controls"]["count"])))
        for probe, attribute in diagnostic_tasks(config)
    }
    if {row["model_id"] for row in metrics} != expected or set(fits) != expected:
        raise ValueError("metric arms/tasks differ from the prespecified inventory")
    if len(csv_metrics) != len(metrics):
        raise ValueError("CSV and JSON metric tables differ")
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(output / "predictions.parquet").to_pylist():
        predictions[row["model_id"]].append(row)
    if set(predictions) != expected:
        raise ValueError("saved predictions do not cover the frozen model inventory")
    c_rows = {row.utterance_id: row for row in inputs.role_rows("C")}
    b_ids = [row.utterance_id for row in inputs.role_rows("B")]
    c_speakers = np.array([row.speaker_id for row in inputs.role_rows("C")])
    clusters, counts = cluster_bootstrap_counts(c_speakers, config["bootstrap"])
    with np.load(output / "bootstrap_clusters.npz", allow_pickle=False) as cluster_file:
        if not np.array_equal(cluster_file["speakers"], clusters) or not np.array_equal(cluster_file["counts"], counts):
            raise ValueError("persisted actor bootstrap draws differ from the declared seed/policy")
    bootstrap_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(output / "bootstrap_scores.parquet").to_pylist():
        bootstrap_records[row["model_id"]].append(row)
    if set(bootstrap_records) != {row["model_id"] for row in metrics if row["attribute"] == "statement"}:
        raise ValueError("bootstrap inventory differs from the statement diagnostic inventory")
    errors = []
    for metric, csv_metric in zip(metrics, csv_metrics):
        identity = metric["model_id"]
        fit = fits[identity]
        if (
            metric["evaluation_role"] != "C" or metric["n_test"] != len(c_rows)
            or fit["fitting_role"] != "B" or int(fit["n_fit"]) != len(b_ids)
            or fit["fit_ids_sha256"] != ids_hash(b_ids)
            or datetime.fromisoformat(fit["fit_completed_utc"]) > datetime.fromisoformat(opened["timestamp_utc"])
        ):
            raise ValueError("model fitting/evaluation roles or timestamps violate the freeze")
        with np.load(output / "models" / f"{identity}.npz", allow_pickle=False) as arrays:
            stored_hash = parameters_hash({name: arrays[name] for name in arrays.files})
            if "support_vectors" in arrays.files:
                raise ValueError("raw support-vector embeddings must not be published")
        if not stored_hash == fit["parameter_hash"] == metric["parameter_hash"] == frozen["model_hashes"][identity]:
            raise ValueError("saved diagnostic parameters differ from their pre-C hash")
        group = predictions[identity]
        if Counter(row["utterance_id"] for row in group) != Counter({key: 1 for key in c_rows}):
            raise ValueError("predictions must cover C exactly once per model")
        classes = np.array(metric["classes"])
        b_labels = inputs.labels("B", metric["attribute"])
        if not np.array_equal(classes, np.unique(b_labels)):
            raise ValueError("evaluation label space differs from B")
        gold, predicted, speakers = [], [], []
        for row in group:
            metadata = c_rows[row["utterance_id"]]
            if row["gold_label"] != str(getattr(metadata, ATTRIBUTES[metric["attribute"]])) or row["speaker_id"] != metadata.speaker_id:
                raise ValueError("prediction gold labels/speakers differ from metadata")
            if row["classes"] != classes.tolist() or row["predicted_label"] not in set(classes):
                raise ValueError("prediction class metadata is inconsistent")
            if metric["probe"] == "linear":
                probability = np.array(row["probabilities"])
                if (
                    probability.shape != classes.shape or not np.isfinite(probability).all()
                    or np.any(probability < 0) or np.any(probability > 1)
                    or abs(probability.sum() - 1) > 1e-10
                ):
                    raise ValueError("invalid stored linear-probe probabilities")
                if classes[probability.argmax()] != row["predicted_label"]:
                    raise ValueError("probability and predicted label disagree")
            elif row["probabilities"] is not None or not np.isfinite(row["decision_score"]):
                raise ValueError("RBF probes must retain finite uncalibrated decisions")
            gold.append(row["gold_label"])
            predicted.append(row["predicted_label"])
            speakers.append(row["speaker_id"])
        gold_a, predicted_a, speakers_a = np.array(gold), np.array(predicted), np.array(speakers)
        class_index = {label: index for index, label in enumerate(classes)}
        confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
        for truth, prediction in zip(gold, predicted):
            confusion[class_index[truth], class_index[prediction]] += 1
        tp = np.diag(confusion)
        denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
        f1 = float(np.divide(2 * tp, denominator, out=np.zeros(len(classes)), where=denominator != 0).mean())
        accuracy = float(tp.sum() / confusion.sum())
        labels, label_counts = np.unique(b_labels, return_counts=True)
        majority = np.full(len(group), labels[label_counts.argmax()])
        if not np.array_equal(majority, [row["majority_label"] for row in group]):
            raise ValueError("saved majority baseline is not B-only")
        baseline = classification_metrics(gold_a, majority, classes)
        if csv_metric["model_id"] != identity or json.loads(csv_metric["classes"]) != classes.tolist():
            raise ValueError("CSV and JSON metric identities/classes differ")
        for name, value in (("macro_f1", f1), ("accuracy", accuracy)):
            errors.extend((abs(value - metric[name]), abs(value - float(csv_metric[name]))))
            errors.extend((
                abs(baseline[name] - metric[f"majority_{name}"]),
                abs(baseline[name] - float(csv_metric[f"majority_{name}"])),
            ))
        errors.append(abs(metric["uniform_chance_accuracy"] - 1 / len(classes)))
        if metric["attribute"] == "statement":
            bf1, bacc, sizes = bootstrap_scores(gold_a, predicted_a, speakers_a, classes, clusters, counts)
            saved = bootstrap_records[identity]
            if len(saved) != config["bootstrap"]["draws"] or [row["draw"] for row in saved] != list(range(len(saved))):
                raise ValueError("missing or misordered statement bootstrap scores")
            for name, values in (("macro_f1", bf1), ("accuracy", bacc)):
                errors.append(float(np.max(np.abs(values - np.array([row[name] for row in saved])))))
                upper = float(np.quantile(values, config["bootstrap"]["quantile"], method=config["bootstrap"]["quantile_method"]))
                errors.append(abs(upper - metric[f"{name}_upper_95"]))
                errors.append(abs(upper - float(csv_metric[f"{name}_upper_95"])))
            if not np.array_equal(sizes, [row["n_examples"] for row in saved]):
                raise ValueError("actor bootstrap sample sizes differ")
    if max(errors, default=0) > 1e-12:
        raise ValueError("saved metric/bound recomputation failed")
    if erasure_decision(metrics, config["validity"]) != read_json(output / "decision.json"):
        raise ValueError("saved validity decision differs from the frozen criteria")
    expected_collateral = collateral_rows(metrics)
    actual_collateral = read_csv(output / "collateral.csv")
    if len(expected_collateral) != len(actual_collateral):
        raise ValueError("collateral table is incomplete")
    for before, after in zip(expected_collateral, actual_collateral):
        if any(
            abs(value - float(after[key])) > 1e-12 if isinstance(value, float) else value != after[key]
            for key, value in before.items()
        ):
            raise ValueError("collateral table recomputation failed")
    result = {
        "status": "pass", "metric_rows": len(metrics),
        "prediction_rows": sum(map(len, predictions.values())),
        "bootstrap_score_rows": sum(map(len, bootstrap_records.values())),
        "maximum_metric_or_bound_recomputation_error": max(errors, default=0),
        "C_opened_after_all_models_frozen": True,
        "frozen_input_hashes_checked": len(inputs.hashes),
        "reliance_or_retraining_experiment": False,
    }
    if not check_outputs:
        write_json(result, output / "verification.json")
    elif result != read_json(output / "verification.json"):
        raise ValueError("verification record differs from recomputation")
    return result


def plot_results(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = read_csv(output / "candidate_selection.csv")
    metrics = read_json(output / "metrics.json")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for seed in sorted({row["seed"] for row in candidates}):
        subset = [row for row in candidates if row["seed"] == seed]
        ax.plot(
            [int(row["dimension"]) for row in subset],
            [max(float(row["linear_statement_f1"]), float(row["rbf_statement_f1"])) for row in subset],
            marker="o", label=f"seed {seed}",
        )
    ax.set(xscale="log", xlabel="DAS subspace dimension (A-only selection)",
           ylabel="A-val max(linear, RBF) statement macro-F1",
           title="Exploratory development scores, not locked-C results")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "candidate_selection.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for offset, probe, label in ((-0.08, "linear", "Linear"), (0.08, "rbf_svm", "RBF-SVM")):
        group = [row for row in metrics if row["attribute"] == "statement" and row["probe"] == probe]
        positions = np.arange(len(group)) + offset
        axes[0].errorbar(
            positions, [row["macro_f1"] for row in group],
            yerr=[np.zeros(len(group)), [max(0, row["macro_f1_upper_95"] - row["macro_f1"]) for row in group]],
            fmt="o", capsize=3, label=label,
        )
        axes[0].set_xticks(range(len(group)), [row["arm"].replace("_", "\n") for row in group])
    axes[0].axhline(0.55, color="black", ls="--")
    axes[0].set(ylim=(0, 1.04), ylabel="Locked-C statement macro-F1", title="Point scores and one-sided 95% upper bounds")
    axes[0].legend()
    for arm in ("original", "das_mean"):
        group = [row for row in metrics if row["probe"] == "linear" and row["attribute"] != "statement" and row["arm"] == arm]
        axes[1].plot([row["attribute"] for row in group], [row["macro_f1"] for row in group], "o-", label=arm)
    axes[1].set(ylim=(0, 1.04), ylabel="Locked-C macro-F1", title="Independent collateral diagnostics")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "locked_validity.png", dpi=160)
    plt.close(fig)


def run_das(phase: str, config_path: Path, output: Path | None) -> None:
    if phase == "prepare":
        if output is not None:
            raise ValueError("prepare writes the declared metadata split, not a result directory")
        prepare_split(config_path)
        return
    if output is None:
        raise ValueError("DAS fit/evaluate/verify require an explicit results directory")
    if phase == "fit":
        fit_study(config_path, output)
    elif phase == "evaluate":
        evaluate_study(config_path, output)
    elif phase == "verify":
        print(json.dumps(verify_study(output, load_inputs(config_path)), indent=2))
    else:
        raise ValueError(f"unknown DAS phase: {phase}")
