"""RAVDESS filename parsing and manifest construction."""

from __future__ import annotations

import re
import wave
from collections import Counter
from pathlib import Path
from typing import Sequence

from .manifest import ManifestRow, ManifestValidationError, validate_manifest


_FILENAME = re.compile(
    r"^(?P<modality>\d{2})-(?P<vocal_channel>\d{2})-"
    r"(?P<emotion>\d{2})-(?P<intensity>\d{2})-"
    r"(?P<statement>\d{2})-(?P<repetition>\d{2})-(?P<actor>\d{2})\.wav$",
    re.IGNORECASE,
)

_EMOTIONS = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}
_INTENSITIES = {1: "normal", 2: "strong"}
_EXPECTED_EMOTION_COUNTS = {"neutral": 96, **{emotion: 192 for emotion in list(_EMOTIONS.values())[1:]}}


def _coded_value(name: str, value: int, allowed: set[int]) -> int:
    if value not in allowed:
        raise ManifestValidationError(f"invalid RAVDESS {name} code {value:02d}")
    return value


def parse_ravdess_file(path: Path) -> ManifestRow:
    """Parse one official audio-only speech RAVDESS WAV filename and header."""
    match = _FILENAME.fullmatch(path.name)
    if match is None:
        raise ManifestValidationError(f"invalid RAVDESS filename: {path.name}")

    codes = {name: int(value) for name, value in match.groupdict().items()}
    _coded_value("modality", codes["modality"], {3})
    _coded_value("vocal channel", codes["vocal_channel"], {1})
    emotion_code = _coded_value("emotion", codes["emotion"], set(_EMOTIONS))
    intensity_code = _coded_value("intensity", codes["intensity"], set(_INTENSITIES))
    statement = _coded_value("statement", codes["statement"], {1, 2})
    repetition = _coded_value("repetition", codes["repetition"], {1, 2})
    actor = _coded_value("actor", codes["actor"], set(range(1, 25)))
    if emotion_code == 1 and intensity_code != 1:
        raise ManifestValidationError("neutral RAVDESS utterances must use normal intensity")

    try:
        with wave.open(str(path), "rb") as audio:
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
    except (OSError, wave.Error) as error:
        raise ManifestValidationError(f"cannot read WAV header for {path}: {error}") from error

    return ManifestRow(
        utterance_id=f"ravdess:{path.stem.lower()}",
        dataset="ravdess",
        path=str(path.resolve()),
        emotion=_EMOTIONS[emotion_code],
        speaker_id=f"actor_{actor:02d}",
        gender="male" if actor % 2 else "female",
        statement_id=statement,
        repetition=repetition,
        intensity=_INTENSITIES[intensity_code],
        duration_seconds=frame_count / sample_rate if sample_rate else 0.0,
        sample_rate_original=sample_rate,
    )


def build_ravdess_manifest(root: Path, *, require_complete: bool = True) -> list[ManifestRow]:
    """Discover WAV files recursively and construct a deterministic manifest."""
    root = root.resolve()
    if not root.is_dir():
        raise ManifestValidationError(f"RAVDESS root is not a directory: {root}")

    paths = sorted(root.rglob("*.wav"), key=lambda item: item.as_posix().lower())
    rows = [parse_ravdess_file(path) for path in paths]
    validate_ravdess_manifest(rows, require_complete=require_complete)
    return sorted(rows, key=lambda row: row.utterance_id)


def validate_ravdess_manifest(rows: Sequence[ManifestRow], *, require_complete: bool = True) -> None:
    """Validate RAVDESS metadata and, optionally, the complete speech corpus design."""
    validate_manifest(rows)
    errors: list[str] = []
    if any(row.dataset != "ravdess" for row in rows):
        errors.append("all rows must use dataset='ravdess'")

    speaker_emotions: dict[str, Counter[str]] = {}
    for row in rows:
        speaker_emotions.setdefault(row.speaker_id, Counter())[row.emotion] += 1
    if any(not counts for counts in speaker_emotions.values()):
        errors.append("every speaker must have emotion observations")

    if require_complete:
        expected_counts = {
            "utterances": 1440,
            "speakers": 24,
            "male": 720,
            "female": 720,
            "statement_1": 720,
            "statement_2": 720,
            "repetition_1": 720,
            "repetition_2": 720,
            "normal_intensity": 768,
            "strong_intensity": 672,
        }
        observed_counts = {
            "utterances": len(rows),
            "speakers": len(speaker_emotions),
            "male": sum(row.gender == "male" for row in rows),
            "female": sum(row.gender == "female" for row in rows),
            "statement_1": sum(row.statement_id == 1 for row in rows),
            "statement_2": sum(row.statement_id == 2 for row in rows),
            "repetition_1": sum(row.repetition == 1 for row in rows),
            "repetition_2": sum(row.repetition == 2 for row in rows),
            "normal_intensity": sum(row.intensity == "normal" for row in rows),
            "strong_intensity": sum(row.intensity == "strong" for row in rows),
        }
        for name, expected in expected_counts.items():
            if observed_counts[name] != expected:
                errors.append(f"expected {expected} {name}, found {observed_counts[name]}")

        emotion_counts = Counter(row.emotion for row in rows)
        if dict(emotion_counts) != _EXPECTED_EMOTION_COUNTS:
            errors.append(f"unexpected emotion counts: {dict(sorted(emotion_counts.items()))}")

        expected_per_speaker = {"neutral": 4, **{emotion: 8 for emotion in list(_EMOTIONS.values())[1:]}}
        for speaker_id, counts in sorted(speaker_emotions.items()):
            if dict(counts) != expected_per_speaker:
                errors.append(f"unexpected emotion balance for {speaker_id}: {dict(sorted(counts.items()))}")

    if errors:
        raise ManifestValidationError("; ".join(errors))
