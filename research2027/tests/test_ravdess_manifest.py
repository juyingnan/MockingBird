import json
import wave
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from mockingbird2027.cli import main
from mockingbird2027.data.manifest import ManifestValidationError, manifest_report
from mockingbird2027.data.ravdess import build_ravdess_manifest, parse_ravdess_file
from mockingbird2027.data.splits import build_ravdess_splits


def _write_wav(path: Path, sample_rate: int = 16_000, frames: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * frames)


def _write_subset(root: Path) -> None:
    for actor in range(1, 5):
        for statement in (1, 2):
            for repetition in (1, 2):
                filename = f"03-01-03-01-{statement:02d}-{repetition:02d}-{actor:02d}.wav"
                _write_wav(root / f"Actor_{actor:02d}" / filename)


def test_parse_ravdess_filename_and_audio_header(tmp_path):
    path = tmp_path / "03-01-05-02-02-01-12.wav"
    _write_wav(path, sample_rate=8_000, frames=80)

    row = parse_ravdess_file(path)

    assert row.utterance_id == "ravdess:03-01-05-02-02-01-12"
    assert row.emotion == "angry"
    assert row.intensity == "strong"
    assert row.statement_id == 2
    assert row.repetition == 1
    assert row.speaker_id == "actor_12"
    assert row.gender == "female"
    assert row.duration_seconds == pytest.approx(0.01)
    assert row.sample_rate_original == 8_000


def test_invalid_filename_fails_loudly(tmp_path):
    path = tmp_path / "not-ravdess.wav"
    _write_wav(path)

    with pytest.raises(ManifestValidationError, match="invalid RAVDESS filename"):
        parse_ravdess_file(path)


def test_incomplete_corpus_is_rejected_by_default(tmp_path):
    _write_subset(tmp_path)

    with pytest.raises(ManifestValidationError, match="expected 1440 utterances"):
        build_ravdess_manifest(tmp_path)


def test_manifest_report_and_speaker_folds_use_explicit_metadata(tmp_path):
    _write_subset(tmp_path)
    rows = build_ravdess_manifest(tmp_path, require_complete=False)
    assignments = build_ravdess_splits(rows, speaker_folds=2)

    report = manifest_report(rows)
    assert report["utterance_count"] == 16
    assert report["speaker_count"] == 4
    assert report["gender_counts"] == {"female": 8, "male": 8}
    assert report["statement_counts"] == {"1": 8, "2": 8}

    row_by_id = {row.utterance_id: row for row in rows}
    speaker_groups = {}
    for assignment in assignments:
        if assignment.split_name == "speaker_shift":
            key = (assignment.fold, assignment.role)
            speaker_groups.setdefault(key, set()).add(row_by_id[assignment.utterance_id].speaker_id)
    for fold in range(2):
        assert speaker_groups[(fold, "train")].isdisjoint(speaker_groups[(fold, "test")])
        test_genders = {
            row_by_id[assignment.utterance_id].gender
            for assignment in assignments
            if assignment.split_name == "speaker_shift"
            and assignment.fold == fold
            and assignment.role == "test"
        }
        assert test_genders == {"male", "female"}


def test_cli_writes_manifest_splits_and_compact_report(tmp_path):
    dataset_root = tmp_path / "audio"
    output_dir = tmp_path / "artifacts"
    _write_subset(dataset_root)

    result = main(
        [
            "manifest",
            "ravdess",
            str(dataset_root),
            "--output-dir",
            str(output_dir),
            "--speaker-folds",
            "2",
            "--allow-incomplete",
        ]
    )

    assert result == 0
    manifest_path = output_dir / "manifests" / "ravdess.parquet"
    splits_path = output_dir / "manifests" / "ravdess_splits.parquet"
    report_path = output_dir / "reports" / "ravdess_manifest_report.json"
    assert pq.read_table(manifest_path).num_rows == 16
    assert pq.read_table(splits_path).num_rows == 16 * 8
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["utterance_count"] == 16
    assert report["splits"]["speaker_shift:fold_0"] == {"test": 8, "train": 8}
