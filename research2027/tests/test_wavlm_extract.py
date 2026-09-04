import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mockingbird2027.data.manifest import ManifestRow
from mockingbird2027.models.extract import (
    CacheValidationError,
    WavLMExtractor,
    WavLMRuntime,
    load_audio_16khz,
    run_smoke_gated_extraction,
    validate_cache,
)
from mockingbird2027.models.pooling import attention_masked_mean


def _write_wav(path: Path, sample_rate: int, frames: int, value: int = 1000) -> None:
    samples = np.full(frames, value, dtype=np.int16)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples.tobytes())


def _row(path: Path, index: int) -> ManifestRow:
    return ManifestRow(
        utterance_id=f"ravdess:test-{index:02d}",
        dataset="ravdess",
        path=str(path),
        emotion="happy",
        speaker_id=f"actor_{index + 1:02d}",
        gender="male" if index % 2 == 0 else "female",
        statement_id=1,
        repetition=1,
        intensity="normal",
        duration_seconds=0.01,
        sample_rate_original=16_000,
    )


class _FakeProcessor:
    def __call__(
        self, waveforms, *, sampling_rate, padding, return_attention_mask, return_tensors
    ):
        assert sampling_rate == 16_000
        assert padding and return_attention_mask and return_tensors == "pt"
        maximum = max(len(waveform) for waveform in waveforms)
        values = torch.zeros((len(waveforms), maximum), dtype=torch.float32)
        mask = torch.zeros((len(waveforms), maximum), dtype=torch.long)
        for index, waveform in enumerate(waveforms):
            length = len(waveform)
            values[index, :length] = torch.from_numpy(waveform.copy())
            mask[index, :length] = 1
        return {"input_values": values, "attention_mask": mask}


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(num_hidden_layers=2)

    def forward(self, *, input_values, attention_mask, output_hidden_states, return_dict):
        assert output_hidden_states and return_dict
        base = input_values.unsqueeze(-1).repeat(1, 1, 3)
        return SimpleNamespace(hidden_states=tuple(base + index for index in range(3)))

    def _get_feature_vector_attention_mask(self, length, attention_mask):
        return attention_mask[:, :length].bool()


def _runtime_factory(counter):
    def factory(revision, device):
        counter.append((revision, device))
        model = _FakeModel().eval()
        model.requires_grad_(False)
        return WavLMRuntime(
            processor=_FakeProcessor(),
            model=model,
            device="cpu",
            resolved_revision="fake-commit",
            torch_version=torch.__version__,
            transformers_version="test",
        )

    return factory


def test_attention_masked_mean_excludes_padding():
    hidden = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[1, 1, 0]])

    pooled = attention_masked_mean(hidden, mask)

    assert pooled.item() == pytest.approx(2.0)
    assert pooled.dtype == torch.float32


def test_audio_is_resampled_to_processor_rate(tmp_path):
    path = tmp_path / "sample.wav"
    _write_wav(path, sample_rate=8_000, frames=80)

    waveform = load_audio_16khz(path)

    assert waveform.dtype == np.float32
    assert len(waveform) == 160
    assert np.isfinite(waveform).all()


def test_all_layers_are_cached_validated_and_reused(tmp_path):
    first_path = tmp_path / "first.wav"
    second_path = tmp_path / "second.wav"
    _write_wav(first_path, sample_rate=16_000, frames=160, value=500)
    _write_wav(second_path, sample_rate=8_000, frames=40, value=1000)
    rows = [_row(second_path, 1), _row(first_path, 0)]
    loads = []

    extractor = WavLMExtractor(
        tmp_path / "cache", batch_size=2, device="cpu", runtime_factory=_runtime_factory(loads)
    )
    result = extractor.extract_or_reuse(rows, mode="smoke")

    assert not result.reused
    assert result.utterance_count == 2
    assert result.layer_count == 3
    assert result.hidden_size == 3
    assert len(loads) == 1
    ids = np.load(result.path / "ids.npy", allow_pickle=False).tolist()
    assert ids == sorted(row.utterance_id for row in rows)
    for layer_index in range(3):
        layer = np.load(result.path / f"layer_{layer_index:02d}.npy", allow_pickle=False)
        assert layer.shape == (2, 3)
        assert layer.dtype == np.float16
        assert np.isfinite(layer).all()

    metadata = json.loads((result.path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["resolved_revision"] == "fake-commit"
    assert metadata["pooling"]["type"] == "attention_masked_mean"
    assert metadata["preprocessing"]["sample_rate_hz"] == 16_000

    def fail_if_loaded(revision, device):
        raise AssertionError("a valid cache must be reused before loading the model")

    reused = WavLMExtractor(
        tmp_path / "cache", runtime_factory=fail_if_loaded
    ).extract_or_reuse(rows, mode="smoke")
    assert reused.reused
    assert reused.path == result.path


def test_cache_validation_rejects_order_mismatch_and_nan(tmp_path):
    paths = [tmp_path / "one.wav", tmp_path / "two.wav"]
    for path in paths:
        _write_wav(path, sample_rate=16_000, frames=80)
    rows = [_row(path, index) for index, path in enumerate(paths)]
    result = WavLMExtractor(
        tmp_path / "cache", runtime_factory=_runtime_factory([])
    ).extract_or_reuse(rows, mode="smoke")
    expected_ids = sorted(row.utterance_id for row in rows)

    with pytest.raises(CacheValidationError, match="ordering"):
        validate_cache(result.path, list(reversed(expected_ids)))

    layer = np.load(result.path / "layer_00.npy", allow_pickle=False)
    layer[0, 0] = np.nan
    np.save(result.path / "layer_00.npy", layer, allow_pickle=False)
    with pytest.raises(CacheValidationError, match="non-finite"):
        validate_cache(result.path, expected_ids)


def test_failed_smoke_prevents_full_extraction(tmp_path):
    rows = [_row(tmp_path / f"{index}.wav", index) for index in range(3)]

    class FailingExtractor:
        def __init__(self):
            self.modes = []

        def find_existing_cache(self, rows, mode):
            return None

        def extract_or_reuse(self, rows, *, mode):
            self.modes.append(mode)
            raise CacheValidationError("synthetic smoke failure")

    extractor = FailingExtractor()
    with pytest.raises(CacheValidationError, match="smoke failure"):
        run_smoke_gated_extraction(rows, extractor, smoke_size=2)
    assert extractor.modes == ["smoke"]
