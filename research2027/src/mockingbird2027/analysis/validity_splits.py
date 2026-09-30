"""Metadata-only, cell-grouped roles and probe CV for the bounded validity study."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np

from ..data.manifest import ManifestRow, ManifestValidationError
from .m3 import ATTRIBUTES
from .probes import ProbeConfig, inner_splits


ROLES = ("A", "B", "C")


def cell_id(row: ManifestRow) -> str:
    return f"{row.speaker_id}|{row.emotion}|{row.intensity}"


def build_roles(rows: Sequence[ManifestRow], seed: int) -> list[dict[str, Any]]:
    cells: dict[str, list[ManifestRow]] = defaultdict(list)
    speakers: dict[str, str] = {}
    for row in rows:
        cells[cell_id(row)].append(row)
        if speakers.setdefault(row.speaker_id, row.gender) != row.gender:
            raise ManifestValidationError("inconsistent speaker gender")
    combinations = sorted({(row.emotion, row.intensity) for row in rows})
    if len(rows) != 1440 or len(cells) != 360 or len(combinations) != 15:
        raise ManifestValidationError("validity roles require the complete 360-cell RAVDESS design")
    for group in cells.values():
        if len(group) != 4 or {(row.statement_id, row.repetition) for row in group} != {
            (1, 1), (1, 2), (2, 1), (2, 2)
        }:
            raise ManifestValidationError("a validity cell must contain both statements and repetitions")
    generator = np.random.default_rng(seed)
    offsets = {}
    for gender in ("female", "male"):
        group = sorted(speaker for speaker, value in speakers.items() if value == gender)
        if len(group) != 12:
            raise ManifestValidationError("validity allocation requires 12 speakers of each gender")
        for position, speaker in enumerate(generator.permutation(group).tolist()):
            offsets[speaker] = position % 4
    order = generator.permutation(len(combinations)).tolist()
    phases = {combinations[index]: position for position, index in enumerate(order)}
    cycle = ("A", "A", "B", "C")
    records = [
        {
            "utterance_id": row.utterance_id,
            "cell_id": cell_id(row),
            "role": cycle[(phases[(row.emotion, row.intensity)] + offsets[row.speaker_id]) % 4],
        }
        for row in sorted(rows, key=lambda row: row.utterance_id)
    ]
    validate_roles(rows, records)
    return records


def validate_roles(rows: Sequence[ManifestRow], assignments: Sequence[dict[str, Any]]) -> None:
    expected_ids = {row.utterance_id for row in rows}
    if Counter(row["utterance_id"] for row in assignments) != Counter({identity: 1 for identity in expected_ids}):
        raise ManifestValidationError("validity roles must cover each manifest ID exactly once")
    if Counter(row["role"] for row in assignments) != {"A": 720, "B": 360, "C": 360}:
        raise ManifestValidationError("validity roles must contain exactly 720/360/360 utterances")
    by_id = {row.utterance_id: row for row in rows}
    roles_by_cell: dict[str, set[str]] = defaultdict(set)
    for assignment in assignments:
        row = by_id[assignment["utterance_id"]]
        if assignment["cell_id"] != cell_id(row):
            raise ManifestValidationError("cell metadata differs from the manifest")
        roles_by_cell[assignment["cell_id"]].add(assignment["role"])
    if any(len(roles) != 1 for roles in roles_by_cell.values()):
        raise ManifestValidationError("a matched cell crosses validity roles")
    for role, divisor in (("A", 2), ("B", 4), ("C", 4)):
        group = [by_id[item["utterance_id"]] for item in assignments if item["role"] == role]
        for name, field in ATTRIBUTES.items():
            if {getattr(row, field) for row in group} != {getattr(row, field) for row in rows}:
                raise ManifestValidationError(f"{role} is missing {name} labels")
            if name != "speaker":
                whole = Counter(getattr(row, field) for row in rows)
                if Counter(getattr(row, field) for row in group) != {
                    label: count // divisor for label, count in whole.items()
                }:
                    raise ManifestValidationError(f"{role} does not preserve the {name} marginal")


def role_indices(
    rows: Sequence[ManifestRow], assignments: Sequence[dict[str, Any]]
) -> dict[str, np.ndarray]:
    validate_roles(rows, assignments)
    lookup = {item["utterance_id"]: item["role"] for item in assignments}
    return {
        role: np.array([index for index, row in enumerate(rows) if lookup[row.utterance_id] == role], dtype=np.int64)
        for role in ROLES
    }


def build_probe_cv(
    rows: Sequence[ManifestRow], assignments: Sequence[dict[str, Any]], config: ProbeConfig
) -> list[dict[str, Any]]:
    indices = role_indices(rows, assignments)["B"]
    groups = np.array([cell_id(rows[index]) for index in indices])
    records = []
    for attribute, field in ATTRIBUTES.items():
        labels = np.array([str(getattr(rows[index], field)) for index in indices])
        for fold, (_, validation) in enumerate(inner_splits(labels, config, groups)):
            records.extend({
                "attribute": attribute, "utterance_id": rows[indices[index]].utterance_id,
                "validation_fold": fold,
            } for index in validation)
    return records


def frozen_probe_cv(
    rows: Sequence[ManifestRow], assignments: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]], folds: int,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    indices = role_indices(rows, assignments)["B"]
    ids = [rows[index].utterance_id for index in indices]
    expected = {(attribute, identity) for attribute in ATTRIBUTES for identity in ids}
    if len(records) != len(expected) or {
        (row["attribute"], row["utterance_id"]) for row in records
    } != expected:
        raise ManifestValidationError("CV records must cover B only, once per attribute and ID")
    output = {}
    for attribute, field in ATTRIBUTES.items():
        assigned = {
            record["utterance_id"]: record["validation_fold"]
            for record in records if record["attribute"] == attribute
        }
        if set(assigned.values()) != set(range(folds)):
            raise ManifestValidationError("unexpected probe CV fold indices")
        groups: dict[str, set[int]] = defaultdict(set)
        labels = np.array([str(getattr(rows[index], field)) for index in indices])
        for index in indices:
            groups[cell_id(rows[index])].add(assigned[rows[index].utterance_id])
        if any(len(values) != 1 for values in groups.values()):
            raise ManifestValidationError("probe CV splits a matched cell")
        splits = []
        for fold in range(folds):
            train = np.array([index for index, identity in enumerate(ids) if assigned[identity] != fold])
            validation = np.array([index for index, identity in enumerate(ids) if assigned[identity] == fold])
            if any(set(labels[part]) != set(labels) for part in (train, validation)):
                raise ManifestValidationError("probe CV is missing training or validation classes")
            splits.append((train, validation))
        output[attribute] = splits
    return output
