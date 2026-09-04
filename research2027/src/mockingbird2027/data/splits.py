"""Deterministic split metadata with leakage validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .manifest import ManifestRow, ManifestValidationError


@dataclass(frozen=True)
class SplitAssignment:
    """Assignment of one utterance to one named split and fold."""

    split_name: str
    fold: int
    utterance_id: str
    role: str


def _actor_number(speaker_id: str) -> int:
    try:
        return int(speaker_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ManifestValidationError(f"invalid RAVDESS speaker_id: {speaker_id}") from error


def _append_binary_split(
    assignments: list[SplitAssignment],
    rows: Sequence[ManifestRow],
    split_name: str,
    fold: int,
    is_test: Callable[[ManifestRow], bool],
) -> None:
    for row in rows:
        assignments.append(
            SplitAssignment(
                split_name=split_name,
                fold=fold,
                utterance_id=row.utterance_id,
                role="test" if is_test(row) else "train",
            )
        )


def build_ravdess_splits(rows: Sequence[ManifestRow], *, speaker_folds: int = 6) -> list[SplitAssignment]:
    """Build audit, lexical, repetition, and speaker-group split assignments."""
    speakers_by_gender: dict[str, list[str]] = defaultdict(list)
    speaker_gender: dict[str, str] = {}
    for row in rows:
        previous = speaker_gender.setdefault(row.speaker_id, row.gender)
        if previous != row.gender:
            raise ManifestValidationError(f"inconsistent gender for {row.speaker_id}")
    for speaker_id, gender in speaker_gender.items():
        speakers_by_gender[gender].append(speaker_id)

    male = sorted(speakers_by_gender["male"], key=_actor_number)
    female = sorted(speakers_by_gender["female"], key=_actor_number)
    if len(male) != len(female):
        raise ManifestValidationError("speaker folds require equal male and female speaker counts")
    pairs = list(zip(male, female))
    if not 2 <= speaker_folds <= len(pairs):
        raise ManifestValidationError(
            f"speaker_folds must be between 2 and {len(pairs)}, found {speaker_folds}"
        )

    assignments: list[SplitAssignment] = []
    _append_binary_split(
        assignments,
        rows,
        "attribute_audit_repetition_1_to_2",
        0,
        lambda row: row.repetition == 2,
    )
    _append_binary_split(
        assignments,
        rows,
        "repetition_shift_1_to_2",
        0,
        lambda row: row.repetition == 2,
    )
    _append_binary_split(
        assignments,
        rows,
        "lexical_shift_1_to_2",
        0,
        lambda row: row.statement_id == 2,
    )
    _append_binary_split(
        assignments,
        rows,
        "lexical_shift_2_to_1",
        0,
        lambda row: row.statement_id == 1,
    )
    _append_binary_split(
        assignments,
        rows,
        "gender_shift_male_to_female",
        0,
        lambda row: row.gender == "female",
    )
    _append_binary_split(
        assignments,
        rows,
        "gender_shift_female_to_male",
        0,
        lambda row: row.gender == "male",
    )

    for fold in range(speaker_folds):
        held_out = {
            speaker
            for pair_index, pair in enumerate(pairs)
            if pair_index % speaker_folds == fold
            for speaker in pair
        }
        _append_binary_split(
            assignments,
            rows,
            "speaker_shift",
            fold,
            lambda row, held_out=held_out: row.speaker_id in held_out,
        )

    validate_split_assignments(rows, assignments)
    return assignments


def validate_split_assignments(
    rows: Sequence[ManifestRow], assignments: Sequence[SplitAssignment]
) -> None:
    """Ensure every split is complete, non-overlapping, and leakage-safe."""
    rows_by_id = {row.utterance_id: row for row in rows}
    expected_ids = set(rows_by_id)
    grouped: dict[tuple[str, int], list[SplitAssignment]] = defaultdict(list)
    for assignment in assignments:
        grouped[(assignment.split_name, assignment.fold)].append(assignment)

    errors: list[str] = []
    for key, group in sorted(grouped.items()):
        counts = Counter(item.utterance_id for item in group)
        assigned_ids = set(counts)
        if assigned_ids != expected_ids:
            errors.append(f"{key} does not cover every manifest utterance exactly once")
        unknown_ids = assigned_ids - expected_ids
        if unknown_ids:
            errors.append(f"{key} references unknown utterances: {sorted(unknown_ids)[:5]}")
        if any(count != 1 for count in counts.values()):
            errors.append(f"{key} contains duplicate utterance assignments")
        roles = Counter(item.role for item in group)
        if roles["train"] == 0 or roles["test"] == 0:
            errors.append(f"{key} must contain non-empty train and test sets")
        if set(roles) - {"train", "test"}:
            errors.append(f"{key} contains invalid split roles")

        split_predicates: dict[str, Callable[[ManifestRow], bool]] = {
            "attribute_audit_repetition_1_to_2": lambda row: row.repetition == 2,
            "repetition_shift_1_to_2": lambda row: row.repetition == 2,
            "lexical_shift_1_to_2": lambda row: row.statement_id == 2,
            "lexical_shift_2_to_1": lambda row: row.statement_id == 1,
            "gender_shift_male_to_female": lambda row: row.gender == "female",
            "gender_shift_female_to_male": lambda row: row.gender == "male",
        }
        if unknown_ids:
            continue

        predicate = split_predicates.get(key[0])
        if predicate is not None:
            for item in group:
                expected_role = "test" if predicate(rows_by_id[item.utterance_id]) else "train"
                if item.role != expected_role:
                    errors.append(f"{key} has an invalid {item.role} assignment for {item.utterance_id}")

        if key[0] == "speaker_shift":
            train_speakers = {
                rows_by_id[item.utterance_id].speaker_id for item in group if item.role == "train"
            }
            test_speakers = {
                rows_by_id[item.utterance_id].speaker_id for item in group if item.role == "test"
            }
            overlap = train_speakers & test_speakers
            if overlap:
                errors.append(f"{key} leaks speakers: {sorted(overlap)}")
            gender_by_speaker = {row.speaker_id: row.gender for row in rows}
            test_gender_counts = Counter(gender_by_speaker[speaker] for speaker in test_speakers)
            if test_gender_counts["male"] != test_gender_counts["female"]:
                errors.append(f"{key} is not gender-balanced at the speaker level")

    if errors:
        raise ManifestValidationError("; ".join(errors))


def split_report(assignments: Sequence[SplitAssignment]) -> dict[str, Any]:
    """Summarize split sizes without duplicating utterance-level metadata."""
    grouped: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    for assignment in assignments:
        grouped[(assignment.split_name, assignment.fold)][assignment.role] += 1
    return {
        f"{name}:fold_{fold}": dict(sorted(counts.items()))
        for (name, fold), counts in sorted(grouped.items())
    }
