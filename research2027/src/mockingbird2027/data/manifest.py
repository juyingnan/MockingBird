"""Shared manifest validation and serialization helpers."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REQUIRED_COLUMNS = (
    "utterance_id",
    "dataset",
    "path",
    "emotion",
    "speaker_id",
    "gender",
    "statement_id",
    "repetition",
    "intensity",
    "duration_seconds",
    "sample_rate_original",
)


class ManifestValidationError(ValueError):
    """Raised when a manifest violates its data contract."""


@dataclass(frozen=True)
class ManifestRow:
    """Metadata for one complete utterance."""

    utterance_id: str
    dataset: str
    path: str
    emotion: str
    speaker_id: str
    gender: str
    statement_id: int
    repetition: int
    intensity: str
    duration_seconds: float
    sample_rate_original: int


def validate_manifest(rows: Sequence[ManifestRow]) -> None:
    """Validate invariants shared by all dataset manifests."""
    errors: list[str] = []
    if not rows:
        errors.append("manifest is empty")

    utterance_ids = [row.utterance_id for row in rows]
    paths = [row.path for row in rows]
    duplicate_ids = sorted(key for key, count in Counter(utterance_ids).items() if count > 1)
    duplicate_paths = sorted(key for key, count in Counter(paths).items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate utterance_id values: {duplicate_ids[:5]}")
    if duplicate_paths:
        errors.append(f"duplicate paths: {duplicate_paths[:5]}")

    for index, row in enumerate(rows):
        values = asdict(row)
        missing = [name for name in REQUIRED_COLUMNS if values[name] in (None, "")]
        if missing:
            errors.append(f"row {index} is missing required values: {missing}")
        if row.duration_seconds <= 0:
            errors.append(f"{row.utterance_id} has non-positive duration")
        if row.sample_rate_original <= 0:
            errors.append(f"{row.utterance_id} has invalid sample rate")

    if errors:
        raise ManifestValidationError("; ".join(errors))


def manifest_report(rows: Sequence[ManifestRow]) -> dict[str, Any]:
    """Create a compact, JSON-serializable manifest summary."""

    def counts(values: Iterable[Any]) -> dict[str, int]:
        return dict(sorted(Counter(str(value) for value in values).items()))

    return {
        "dataset": rows[0].dataset if rows else None,
        "utterance_count": len(rows),
        "speaker_count": len({row.speaker_id for row in rows}),
        "duration_seconds_total": round(sum(row.duration_seconds for row in rows), 6),
        "sample_rate_counts": counts(row.sample_rate_original for row in rows),
        "emotion_counts": counts(row.emotion for row in rows),
        "speaker_utterance_counts": counts(row.speaker_id for row in rows),
        "speaker_emotion_counts": {
            speaker_id: counts(row.emotion for row in rows if row.speaker_id == speaker_id)
            for speaker_id in sorted({row.speaker_id for row in rows})
        },
        "gender_counts": counts(row.gender for row in rows),
        "statement_counts": counts(row.statement_id for row in rows),
        "repetition_counts": counts(row.repetition for row in rows),
        "intensity_counts": counts(row.intensity for row in rows),
    }


def read_parquet_manifest(path: Path) -> list[ManifestRow]:
    """Load a manifest while enforcing the declared column contract."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Parquet input requires the project dependencies") from error

    table = pq.read_table(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(table.column_names))
    if missing:
        raise ManifestValidationError(f"manifest is missing columns: {missing}")
    rows = [ManifestRow(**{name: record[name] for name in REQUIRED_COLUMNS}) for record in table.to_pylist()]
    validate_manifest(rows)
    return rows


def write_parquet(records: Sequence[Any], path: Path) -> None:
    """Write dataclass or mapping records as Parquet."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Parquet output requires the project dependencies") from error

    path.parent.mkdir(parents=True, exist_ok=True)
    dictionaries = [asdict(record) if hasattr(record, "__dataclass_fields__") else dict(record) for record in records]
    table = pa.Table.from_pylist(dictionaries)
    pq.write_table(table, path)


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write deterministic, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
