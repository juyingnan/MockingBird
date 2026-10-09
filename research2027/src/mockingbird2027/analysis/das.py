"""Matched-pair distributed interchange search on cached feature vectors."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import orthogonal

from ..data.manifest import ManifestRow, ManifestValidationError
from .erasure import AffineRemoval, feature_matrix
from .m3 import ATTRIBUTES
from .validity_splits import cell_id, validate_roles


IMPLEMENTATION_VERSION = "feature-level-dii-v1"
DISCOVERY_ROLES = ("A_train", "A_val", "B", "C")


def build_discovery_split(
    rows: Sequence[ManifestRow], roles: list[dict[str, Any]], seed: int,
) -> list[dict[str, str]]:
    validate_roles(rows, roles)
    original = {item["utterance_id"]: item["role"] for item in roles}
    cells = {cell_id(row): row for row in rows if original[row.utterance_id] == "A"}
    types = sorted({(row.emotion, row.intensity) for row in cells.values()})
    rng = np.random.default_rng(seed)
    assigned: Counter[str] = Counter()
    validation = set()
    for position in rng.permutation(len(types)):
        emotion, intensity = types[position]
        for gender in ("female", "male"):
            available = sorted(
                key for key, row in cells.items()
                if (row.emotion, row.intensity, row.gender) == (emotion, intensity, gender)
            )
            if len(available) != 6:
                raise ManifestValidationError("A must contain six actors per gender/cell type")
            shuffled = rng.permutation(available).tolist()
            selected = sorted(shuffled, key=lambda key: assigned[cells[key].speaker_id])[:2]
            validation.update(selected)
            assigned.update(cells[key].speaker_id for key in selected)
    records = []
    for row in sorted(rows, key=lambda item: item.utterance_id):
        role = original[row.utterance_id]
        if role == "A":
            role = "A_val" if cell_id(row) in validation else "A_train"
        records.append({"utterance_id": row.utterance_id, "cell_id": cell_id(row), "role": role})
    validate_discovery_split(rows, roles, records)
    return records


def validate_discovery_split(
    rows: Sequence[ManifestRow], original: list[dict[str, Any]], records: list[dict[str, str]],
) -> None:
    validate_roles(rows, original)
    by_id = {row.utterance_id: row for row in rows}
    if Counter(item["utterance_id"] for item in records) != Counter({key: 1 for key in by_id}):
        raise ManifestValidationError("discovery split must cover every ID exactly once")
    if Counter(item["role"] for item in records) != {"A_train": 480, "A_val": 240, "B": 360, "C": 360}:
        raise ManifestValidationError("discovery roles must have sizes 480/240/360/360")
    old = {item["utterance_id"]: item["role"] for item in original}
    cell_roles: dict[str, set[str]] = defaultdict(set)
    for item in records:
        row = by_id[item["utterance_id"]]
        if item["cell_id"] != cell_id(row):
            raise ManifestValidationError("discovery cell does not match metadata")
        parent_role = "A" if item["role"].startswith("A_") else item["role"]
        if parent_role != old[row.utterance_id]:
            raise ManifestValidationError("discovery split changes the frozen A/B/C roles")
        cell_roles[cell_id(row)].add(item["role"])
    if any(len(values) != 1 for values in cell_roles.values()):
        raise ManifestValidationError("a complete matched cell crosses discovery roles")
    for role in DISCOVERY_ROLES:
        group = [by_id[item["utterance_id"]] for item in records if item["role"] == role]
        for field in ATTRIBUTES.values():
            if {getattr(row, field) for row in group} != {getattr(row, field) for row in rows}:
                raise ManifestValidationError(f"{role} is missing labels for {field}")
        matched_partners(group)


def matched_partners(rows: Sequence[ManifestRow]) -> np.ndarray:
    groups: dict[tuple[str, str, str, int], dict[int, int]] = defaultdict(dict)
    if len({row.utterance_id for row in rows}) != len(rows) or not rows:
        raise ManifestValidationError("matched pairs require unique nonempty utterance IDs")
    for index, row in enumerate(rows):
        key = (row.speaker_id, row.emotion, row.intensity, row.repetition)
        if row.statement_id in groups[key]:
            raise ManifestValidationError("duplicate statement within a matched pair")
        groups[key][row.statement_id] = index
    partners = np.empty(len(rows), dtype=np.int64)
    for group in groups.values():
        if set(group) != {1, 2}:
            raise ManifestValidationError("each pair must contain statements 1 and 2")
        partners[group[1]], partners[group[2]] = group[2], group[1]
    return partners


def binary_labels(rows: Sequence[ManifestRow]) -> np.ndarray:
    labels = np.array([row.statement_id - 1 for row in rows], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("DAS requires both binary statement labels")
    return labels


def arbitrary_targets(
    rows: Sequence[ManifestRow], partners: np.ndarray, seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    cells = sorted({cell_id(row) for row in rows})
    if len(cells) % 2:
        raise ValueError("the arbitrary-mapping control needs an even number of complete cells")
    shuffled = np.random.default_rng(seed).permutation(cells).tolist()
    flips = {key: int(index < len(cells) // 2) for index, key in enumerate(shuffled)}
    target = binary_labels(rows)[partners]
    return target ^ np.array([flips[cell_id(row)] for row in rows]), flips


def tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.array(values, copy=True), dtype=torch.float64)


def readout_parameters(model: nn.Module) -> dict[str, np.ndarray]:
    return {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}


def _classification(model: nn.Module, x: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite readout/interchange loss")
        return {
            "cross_entropy": float(loss),
            "accuracy": float((logits.argmax(dim=1) == targets).double().mean()),
        }


def fit_readout(
    training: np.ndarray, labels: np.ndarray, validation: np.ndarray,
    validation_labels: np.ndarray, settings: dict[str, Any],
) -> tuple[nn.Module, list[dict[str, Any]], int]:
    training, validation = feature_matrix(training), feature_matrix(validation)
    if training.shape[1] != validation.shape[1]:
        raise ValueError("readout train/validation dimensions differ")
    if set(labels.tolist()) != {0, 1} or set(validation_labels.tolist()) != {0, 1}:
        raise ValueError("readout labels must be binary in both A partitions")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(settings["seed"])
        model = nn.Sequential(
            nn.Linear(training.shape[1], settings["hidden_size"], dtype=torch.float64),
            nn.Tanh(), nn.Linear(settings["hidden_size"], 2, dtype=torch.float64),
        )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    x, v = tensor(training), tensor(validation)
    y, vy = torch.as_tensor(labels), torch.as_tensor(validation_labels)
    best_loss, best_step, best_state = float("inf"), -1, {}
    records = []
    for step in range(settings["steps"] + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite readout training loss")
            loss.backward()
            optimizer.step()
        if step % settings["evaluation_interval"] == 0 or step == settings["steps"]:
            train_scores, val_scores = _classification(model, x, y), _classification(model, v, vy)
            records.append({
                "step": step, "training_role": "A_train", "validation_role": "A_val",
                **{f"train_{key}": value for key, value in train_scores.items()},
                **{f"validation_{key}": value for key, value in val_scores.items()},
            })
            if val_scores["cross_entropy"] < best_loss:
                best_loss, best_step = val_scores["cross_entropy"], step
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return model, records, best_step


class OrthogonalSubspace(nn.Module):
    def __init__(self, dimension: int, rank: int, seed: int):
        super().__init__()
        if not 0 < rank < dimension:
            raise ValueError("DAS rank must be strictly between zero and ambient dimension")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            layer = nn.Linear(rank, dimension, bias=False, dtype=torch.float64)
            nn.init.orthogonal_(layer.weight)
        self.layer = orthogonal(
            layer, orthogonal_map="householder", use_trivialization=False,
        )

    @property
    def basis(self) -> torch.Tensor:
        return self.layer.weight


def interchange(base: torch.Tensor, source: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if (
        base.ndim != 2 or source.shape != base.shape or basis.ndim != 2
        or basis.shape[0] != base.shape[1]
    ):
        raise ValueError("DII requires aligned base/source matrices and a compatible subspace")
    return base + ((source - base) @ basis) @ basis.T


@dataclass
class AlignmentFit:
    basis: np.ndarray
    checkpoint: int
    records: list[dict[str, Any]]


def fit_alignment(
    readout: nn.Module, training: np.ndarray, partners: np.ndarray, targets: np.ndarray,
    validation: np.ndarray, val_partners: np.ndarray, val_targets: np.ndarray,
    rank: int, seed: int, settings: dict[str, Any],
) -> AlignmentFit:
    if any(parameter.requires_grad for parameter in readout.parameters()):
        raise ValueError("DAS must not optimize a trainable behavioral readout")
    training, validation = feature_matrix(training), feature_matrix(validation)
    for values, source_indices, labels in (
        (training, partners, targets), (validation, val_partners, val_targets)
    ):
        if (
            source_indices.shape != (len(values),) or source_indices.dtype.kind not in "iu"
            or np.any(source_indices < 0) or np.any(source_indices >= len(values))
            or labels.shape != (len(values),) or set(labels.tolist()) != {0, 1}
        ):
            raise ValueError("DAS pair indices and binary counterfactual targets must align")
    alignment = OrthogonalSubspace(training.shape[1], rank, seed)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=settings["learning_rate"])
    x, v = tensor(training), tensor(validation)
    y, vy = torch.as_tensor(targets), torch.as_tensor(val_targets)
    source, val_source = x[partners], v[val_partners]
    best_loss, best_step, best_basis = float("inf"), -1, None
    records = []
    for step in range(settings["steps"] + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            changed = interchange(x, source, alignment.basis)
            loss = nn.functional.cross_entropy(readout(changed), y)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite DAS training loss")
            loss.backward()
            optimizer.step()
        if step % settings["evaluation_interval"] == 0 or step == settings["steps"]:
            with torch.no_grad():
                q = alignment.basis
                train_scores = _classification(readout, interchange(x, source, q), y)
                val_scores = _classification(readout, interchange(v, val_source, q), vy)
                records.append({
                    "step": step, "training_role": "A_train", "validation_role": "A_val",
                    **{f"train_{key}": value for key, value in train_scores.items()},
                    **{f"validation_{key}": value for key, value in val_scores.items()},
                })
                if val_scores["cross_entropy"] < best_loss:
                    best_loss, best_step = val_scores["cross_entropy"], step
                    best_basis = q.cpu().numpy().copy()
    if best_basis is None:
        raise RuntimeError("DAS did not produce a finite checkpoint")
    mean_ablation(training.mean(axis=0), best_basis)
    return AlignmentFit(best_basis, best_step, records)


def interchange_scores(
    model: nn.Module, features: np.ndarray, partners: np.ndarray,
    targets: np.ndarray, basis: np.ndarray | None, *, full_swap: bool = False,
) -> dict[str, float]:
    x = tensor(features)
    if full_swap and basis is not None:
        raise ValueError("full swap and subspace swap are different interventions")
    changed = x[partners] if full_swap else x
    if basis is not None:
        changed = interchange(x, x[partners], tensor(basis))
    return _classification(model, changed, torch.as_tensor(targets))


def mean_ablation(mean: np.ndarray, basis: np.ndarray) -> AffineRemoval:
    mean = np.asarray(mean, dtype=np.float64)
    q = np.asarray(basis, dtype=np.float64)
    return AffineRemoval(mean.copy(), q.copy(), q.copy(), "das_mean")


def collateral_check(
    before: dict[str, float], after: dict[str, float], policy: dict[str, Any],
) -> dict[str, Any]:
    expected = {"emotion", "speaker", "gender", "intensity"}
    if not expected.issubset(before) or not expected.issubset(after):
        raise ValueError("collateral check requires all four non-target attributes")
    if any(not np.isfinite(values[name]) or not 0 <= values[name] <= 1 for values in (before, after) for name in expected):
        raise ValueError("collateral scores must be finite values in [0, 1]")
    delta = {name: after[name] - before[name] for name in sorted(expected)}
    large = [
        name for name in ("speaker", "gender", "intensity")
        if -delta[name] > policy["large_unrelated_f1_drop"]
    ]
    return {
        "delta_macro_f1": delta,
        "large_unrelated_losses": large,
        "pass": (
            -delta["emotion"] <= policy["maximum_emotion_f1_drop"]
            and len(large) <= policy["maximum_large_unrelated_losses"]
        ),
    }


def select_candidate(
    candidates: list[dict[str, Any]], minimum_iia: float,
) -> tuple[dict[str, Any], str]:
    if not candidates or len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise ValueError("candidate selection requires unique nonempty candidate records")
    if len({(row["dimension"], row["seed"]) for row in candidates}) != len(candidates):
        raise ValueError("candidate dimension/seed specifications must be unique")
    for row in candidates:
        if row["evaluation_role"] != "A_val":
            raise ValueError("candidate selection may only use A_val")
        for name in ("linear_statement_f1", "rbf_statement_f1", "interchange_accuracy"):
            if not np.isfinite(row[name]) or not 0 <= row[name] <= 1:
                raise ValueError("candidate selection scores must be finite values in [0, 1]")
    eligible = [
        row for row in candidates
        if row["interchange_accuracy"] >= minimum_iia and row["collateral_pass"]
    ]
    pool = eligible if eligible else candidates
    selected = min(pool, key=lambda row: (
        max(row["linear_statement_f1"], row["rbf_statement_f1"]),
        (row["linear_statement_f1"] + row["rbf_statement_f1"]) / 2,
        row["dimension"], row["seed"],
    ))
    return dict(selected), "ELIGIBLE_A_CANDIDATE" if eligible else "NO_ELIGIBLE_A_CANDIDATE_DIAGNOSTIC_ONLY"


def random_subspace_controls(
    training: np.ndarray, target: AffineRemoval, settings: dict[str, Any],
) -> tuple[dict[str, AffineRemoval], list[dict[str, Any]]]:
    training = feature_matrix(training)
    if not 0 < settings["count"] <= settings["pool_size"]:
        raise ValueError("random-control pool must contain the requested number of controls")
    target_rms = target.perturbation(training)["rms"]
    if target_rms <= 0:
        raise ValueError("cannot match a zero-magnitude target ablation")
    records, bases = [], {}
    centered = training - target.mean
    for draw in range(settings["pool_size"]):
        seed = settings["seed"] + draw
        q, _ = np.linalg.qr(
            np.random.default_rng(seed).normal(size=(training.shape[1], target.removed_rank)),
            mode="reduced",
        )
        rms = float(np.sqrt(np.mean(np.sum((centered @ q) ** 2, axis=1))))
        relative = abs(rms / target_rms - 1)
        records.append({
            "draw": draw, "seed": seed, "dimension": target.removed_rank,
            "fitting_role": "A_train", "target_train_rms": target_rms,
            "train_rms": rms, "relative_rms_error": relative,
            "within_rms_tolerance": relative <= settings["relative_rms_tolerance"],
            "selected": False, "arm": "",
        })
        bases[draw] = q
    controls = {}
    for index, row in enumerate(sorted(records, key=lambda item: (item["relative_rms_error"], item["draw"]))[:settings["count"]]):
        arm = f"random_{index:02d}"
        row.update({"selected": True, "arm": arm})
        controls[arm] = mean_ablation(target.mean, bases[row["draw"]])
    return controls, records


def erasure_decision(metrics: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    lookup = {(row["arm"], row["probe"], row["attribute"]): row for row in metrics}
    required = {
        (arm, "linear", attribute) for arm in ("original", "das_mean") for attribute in ATTRIBUTES
    } | {(arm, "rbf_svm", "statement") for arm in ("original", "das_mean")}
    if len(lookup) != len(metrics) or not required.issubset(lookup):
        raise ValueError("decision needs complete, unique original and DAS diagnostics")
    checks = {}
    for probe in ("linear", "rbf_svm"):
        before, after = lookup[("original", probe, "statement")], lookup[("das_mean", probe, "statement")]
        for row in (before, after):
            for name in ("macro_f1", "accuracy", "macro_f1_upper_95", "accuracy_upper_95"):
                if not np.isfinite(row[name]) or not 0 <= row[name] <= 1:
                    raise ValueError("decision needs finite statement metrics and bounds")
        checks[probe] = {
            "before_competent": before["macro_f1"] >= policy["before_min_macro_f1"],
            "target_f1_reduction": before["macro_f1"] - after["macro_f1"],
            "reduction_pass": before["macro_f1"] - after["macro_f1"] >= policy["minimum_target_f1_drop"],
            "point_pass": after["macro_f1"] <= policy["after_max_macro_f1"] and after["accuracy"] <= policy["after_max_accuracy"],
            "precision_pass": after["macro_f1_upper_95"] <= policy["upper_confidence_ceiling"] and after["accuracy_upper_95"] <= policy["upper_confidence_ceiling"],
            **{name: after[name] for name in ("macro_f1", "accuracy", "macro_f1_upper_95", "accuracy_upper_95")},
        }
    collateral = collateral_check(
        {name: lookup[("original", "linear", name)]["macro_f1"] for name in ATTRIBUTES},
        {name: lookup[("das_mean", "linear", name)]["macro_f1"] for name in ATTRIBUTES},
        policy,
    )
    if not all(item["before_competent"] for item in checks.values()):
        outcome = "INCONCLUSIVE_PROBE_COMPETENCE"
    elif not all(item["point_pass"] for item in checks.values()):
        outcome = "FAIL_RESIDUAL_DECODABILITY"
    elif not collateral["pass"]:
        outcome = "FAIL_COLLATERAL_DAMAGE"
    elif not all(item["reduction_pass"] for item in checks.values()):
        outcome = "FAIL_TARGET_REDUCTION"
    elif not all(item["precision_pass"] for item in checks.values()):
        outcome = "INCONCLUSIVE_PRECISION"
    else:
        outcome = "PASS_VALIDITY_ONLY"
    return {
        "decision": outcome, "primary_arm": "das_mean",
        "probe_checks": checks, "collateral": collateral,
        "stop": True, "reliance_experiment_run": False,
        "claim_boundary": (
            "Exploratory validity of mean ablation under specified probes on an already explored corpus; "
            "not zero information, native-WavLM causal alignment, emotion-classifier reliance, "
            "harmful generalization, or an explanation of LEACE's failure."
        ),
    }
