"""Frozen WavLM extraction with immutable, validated caches."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from ..data.manifest import ManifestRow, read_parquet_manifest
from ..data.ravdess import validate_ravdess_manifest
from .pooling import attention_masked_mean
from .registry import WAVLM_DEFAULT_REVISION, WAVLM_MODEL_ID, WAVLM_SAMPLE_RATE


_CACHE_SCHEMA_VERSION = 1
_PREPROCESSING = {
    "audio": "complete_utterance",
    "channels": "mono_mean",
    "sample_rate_hz": WAVLM_SAMPLE_RATE,
    "resampling": "scipy.signal.resample_poly",
    "processor": "model_default",
    "processor_sampling_rate_hz": WAVLM_SAMPLE_RATE,
}
_POOLING = {
    "type": "attention_masked_mean",
    "mask": "model_feature_vector_attention_mask",
    "accumulation_dtype": "float32",
}


class CacheValidationError(RuntimeError):
    """Raised when an embedding cache is incomplete or inconsistent."""


@dataclass(frozen=True)
class CacheResult:
    """Validated cache details returned to callers."""

    path: Path
    fingerprint: str
    utterance_count: int
    layer_count: int
    hidden_size: int
    reused: bool


@dataclass
class WavLMRuntime:
    """Loaded processor and frozen model state."""

    processor: Any
    model: Any
    device: str
    resolved_revision: str
    torch_version: str
    transformers_version: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def manifest_hash(rows: Sequence[ManifestRow]) -> str:
    """Hash ordered manifest content, not Parquet storage details."""
    ordered = [asdict(row) for row in sorted(rows, key=lambda row: row.utterance_id)]
    return hashlib.sha256(_canonical_json(ordered)).hexdigest()


def load_audio_16khz(path: Path) -> np.ndarray:
    """Load a complete WAV, mix channels, and resample to processor rate."""
    sample_rate, samples = wavfile.read(path)
    if sample_rate <= 0 or samples.size == 0:
        raise ValueError(f"invalid or empty audio file: {path}")

    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        if np.issubdtype(samples.dtype, np.unsignedinteger):
            midpoint = (info.max + 1) / 2
            waveform = (samples.astype(np.float32) - midpoint) / midpoint
        else:
            scale = float(max(abs(info.min), info.max))
            waveform = samples.astype(np.float32) / scale
    elif np.issubdtype(samples.dtype, np.floating):
        waveform = samples.astype(np.float32)
    else:
        raise ValueError(f"unsupported WAV sample dtype {samples.dtype} for {path}")

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    elif waveform.ndim != 1:
        raise ValueError(f"unsupported WAV shape {waveform.shape} for {path}")

    if sample_rate != WAVLM_SAMPLE_RATE:
        divisor = math.gcd(sample_rate, WAVLM_SAMPLE_RATE)
        waveform = resample_poly(
            waveform,
            WAVLM_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        ).astype(np.float32, copy=False)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"audio contains no finite samples after preprocessing: {path}")
    return np.ascontiguousarray(waveform, dtype=np.float32)


def load_wavlm_runtime(
    revision: str = WAVLM_DEFAULT_REVISION, device: str = "auto"
) -> WavLMRuntime:
    """Load the authorized frozen WavLM model and its matching processor."""
    import torch
    import transformers
    from transformers import AutoFeatureExtractor, WavLMModel

    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected_device == "auto":
        selected_device = "cpu"

    processor = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL_ID, revision=revision)
    model = WavLMModel.from_pretrained(WAVLM_MODEL_ID, revision=revision)
    model.eval()
    model.requires_grad_(False)
    model.to(selected_device)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("WavLM must be in eval mode with every parameter frozen")

    resolved_revision = getattr(model.config, "_commit_hash", None) or revision
    return WavLMRuntime(
        processor=processor,
        model=model,
        device=selected_device,
        resolved_revision=str(resolved_revision),
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
    )


def _cache_identity(
    rows: Sequence[ManifestRow], requested_revision: str, resolved_revision: str, mode: str
) -> dict[str, Any]:
    return {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "dataset": "ravdess",
        "model_id": WAVLM_MODEL_ID,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "manifest_hash": manifest_hash(rows),
        "preprocessing": _PREPROCESSING,
        "pooling": _POOLING,
        "storage_dtype": "float16",
        "extraction_mode": mode,
    }


def _fingerprint(identity: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(identity)).hexdigest()[:16]


def validate_cache(
    cache_path: Path,
    expected_ids: Sequence[str],
    *,
    expected_fingerprint: str | None = None,
    require_directory_name: bool = True,
) -> CacheResult:
    """Validate cache metadata, identifier ordering, layer shapes, and finiteness."""
    metadata_path = cache_path / "metadata.json"
    if not metadata_path.is_file():
        raise CacheValidationError(f"missing cache metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CacheValidationError(f"invalid cache metadata: {metadata_path}") from error

    fingerprint = metadata.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise CacheValidationError("cache metadata has no fingerprint")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise CacheValidationError("cache fingerprint does not match the requested extraction")
    identity_keys = (
        "schema_version",
        "dataset",
        "model_id",
        "requested_revision",
        "resolved_revision",
        "manifest_hash",
        "preprocessing",
        "pooling",
        "storage_dtype",
        "extraction_mode",
    )
    recorded_identity = {key: metadata.get(key) for key in identity_keys}
    if _fingerprint(recorded_identity) != fingerprint:
        raise CacheValidationError("cache identity does not match its fingerprint")
    if require_directory_name and cache_path.name != fingerprint:
        raise CacheValidationError("cache directory does not match its fingerprint")
    if metadata.get("status") != "complete":
        raise CacheValidationError("cache metadata is not marked complete")

    ids_path = cache_path / "ids.npy"
    if not ids_path.is_file():
        raise CacheValidationError("cache is missing ids.npy")
    ids = np.load(ids_path, allow_pickle=False)
    actual_ids = ids.tolist()
    if actual_ids != list(expected_ids):
        raise CacheValidationError("cached utterance ordering does not match the manifest")

    utterance_count = len(expected_ids)
    layer_count = metadata.get("layer_count")
    hidden_size = metadata.get("hidden_size")
    if not isinstance(layer_count, int) or layer_count <= 0:
        raise CacheValidationError("invalid layer count in cache metadata")
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise CacheValidationError("invalid hidden size in cache metadata")
    if metadata.get("utterance_count") != utterance_count:
        raise CacheValidationError("cache utterance count does not match ids.npy")

    expected_files = {f"layer_{index:02d}.npy" for index in range(layer_count)}
    actual_files = {path.name for path in cache_path.glob("layer_*.npy")}
    if actual_files != expected_files:
        raise CacheValidationError("cache layer files do not match metadata")
    for filename in sorted(expected_files):
        array = np.load(cache_path / filename, mmap_mode="r", allow_pickle=False)
        if array.shape != (utterance_count, hidden_size):
            raise CacheValidationError(f"unexpected shape for {filename}: {array.shape}")
        if array.dtype != np.float16:
            raise CacheValidationError(f"unexpected dtype for {filename}: {array.dtype}")
        if not np.isfinite(array).all():
            raise CacheValidationError(f"non-finite values found in {filename}")

    return CacheResult(
        path=cache_path,
        fingerprint=fingerprint,
        utterance_count=utterance_count,
        layer_count=layer_count,
        hidden_size=hidden_size,
        reused=True,
    )


class WavLMExtractor:
    """Lazy WavLM runner that reuses validated immutable caches."""

    def __init__(
        self,
        cache_root: Path,
        *,
        revision: str = WAVLM_DEFAULT_REVISION,
        device: str = "auto",
        batch_size: int = 4,
        runtime_factory: Callable[[str, str], WavLMRuntime] = load_wavlm_runtime,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.cache_root = cache_root
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.runtime_factory = runtime_factory
        self._runtime: WavLMRuntime | None = None

    @property
    def model_cache_root(self) -> Path:
        return self.cache_root / "ravdess" / WAVLM_MODEL_ID.replace("/", "_")

    def find_existing_cache(
        self, rows: Sequence[ManifestRow], mode: str
    ) -> CacheResult | None:
        root = self.model_cache_root
        if not root.is_dir():
            return None
        expected_ids = [row.utterance_id for row in rows]
        expected_manifest_hash = manifest_hash(rows)
        for candidate in sorted(path for path in root.iterdir() if path.is_dir()):
            metadata_path = candidate / "metadata.json"
            if not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                metadata.get("model_id") == WAVLM_MODEL_ID
                and metadata.get("requested_revision") == self.revision
                and metadata.get("manifest_hash") == expected_manifest_hash
                and metadata.get("preprocessing") == _PREPROCESSING
                and metadata.get("pooling") == _POOLING
                and metadata.get("storage_dtype") == "float16"
                and metadata.get("extraction_mode") == mode
            ):
                return validate_cache(candidate, expected_ids)
        return None

    def _get_runtime(self) -> WavLMRuntime:
        if self._runtime is None:
            self._runtime = self.runtime_factory(self.revision, self.device)
        return self._runtime

    def extract_or_reuse(self, rows: Sequence[ManifestRow], *, mode: str) -> CacheResult:
        """Return an existing valid cache or create one without overwriting."""
        ordered_rows = sorted(rows, key=lambda row: row.utterance_id)
        expected_ids = [row.utterance_id for row in ordered_rows]
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError("embedding extraction requires unique utterance identifiers")

        existing = self.find_existing_cache(ordered_rows, mode)
        if existing is not None:
            return existing

        runtime = self._get_runtime()
        identity = _cache_identity(ordered_rows, self.revision, runtime.resolved_revision, mode)
        fingerprint = _fingerprint(identity)
        target = self.model_cache_root / fingerprint
        if target.exists():
            return validate_cache(target, expected_ids, expected_fingerprint=fingerprint)

        arrays = self._extract_arrays(ordered_rows, runtime)
        return self._write_cache(target, fingerprint, identity, expected_ids, arrays, runtime)

    def _extract_arrays(
        self, rows: Sequence[ManifestRow], runtime: WavLMRuntime
    ) -> list[np.ndarray]:
        import torch

        model = runtime.model
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("refusing extraction with a training or unfrozen model")

        layer_batches: list[list[np.ndarray]] | None = None
        with torch.inference_mode():
            for start in range(0, len(rows), self.batch_size):
                batch = rows[start : start + self.batch_size]
                waveforms = [load_audio_16khz(Path(row.path)) for row in batch]
                inputs = runtime.processor(
                    waveforms,
                    sampling_rate=WAVLM_SAMPLE_RATE,
                    padding=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                input_values = inputs["input_values"].to(runtime.device)
                attention_mask = inputs["attention_mask"].to(runtime.device)
                outputs = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states
                expected_layers = int(model.config.num_hidden_layers) + 1
                if hidden_states is None or len(hidden_states) != expected_layers:
                    raise RuntimeError(
                        f"expected {expected_layers} hidden states, received "
                        f"{0 if hidden_states is None else len(hidden_states)}"
                    )
                if layer_batches is None:
                    layer_batches = [[] for _ in hidden_states]

                for layer_index, hidden_state in enumerate(hidden_states):
                    frame_mask = model._get_feature_vector_attention_mask(
                        hidden_state.shape[1], attention_mask
                    )
                    pooled = attention_masked_mean(hidden_state, frame_mask)
                    if pooled.requires_grad:
                        raise RuntimeError("pooled embeddings unexpectedly track gradients")
                    array = pooled.cpu().numpy()
                    if array.shape[0] != len(batch) or not np.isfinite(array).all():
                        raise RuntimeError(f"invalid pooled output at layer {layer_index}")
                    layer_batches[layer_index].append(array)

        if layer_batches is None:
            raise RuntimeError("cannot extract an empty manifest")
        arrays = [np.concatenate(batches, axis=0).astype(np.float16) for batches in layer_batches]
        hidden_sizes = {array.shape[1] for array in arrays if array.ndim == 2}
        if len(hidden_sizes) != 1 or any(array.shape[0] != len(rows) for array in arrays):
            raise RuntimeError("hidden-state layers have inconsistent shapes")
        if any(not np.isfinite(array).all() for array in arrays):
            raise RuntimeError("non-finite values found after float16 conversion")
        return arrays

    def _write_cache(
        self,
        target: Path,
        fingerprint: str,
        identity: dict[str, Any],
        expected_ids: Sequence[str],
        arrays: Sequence[np.ndarray],
        runtime: WavLMRuntime,
    ) -> CacheResult:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f".{fingerprint}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            np.save(temporary / "ids.npy", np.asarray(expected_ids, dtype=str), allow_pickle=False)
            for index, array in enumerate(arrays):
                np.save(temporary / f"layer_{index:02d}.npy", array, allow_pickle=False)
            metadata = {
                **identity,
                "fingerprint": fingerprint,
                "status": "complete",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "utterance_count": len(expected_ids),
                "hidden_size": int(arrays[0].shape[1]),
                "layer_count": len(arrays),
                "torch_version": runtime.torch_version,
                "transformers_version": runtime.transformers_version,
                "numpy_version": np.__version__,
            }
            (temporary / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            validate_cache(
                temporary,
                expected_ids,
                expected_fingerprint=fingerprint,
                require_directory_name=False,
            )
            if target.exists():
                existing = validate_cache(target, expected_ids, expected_fingerprint=fingerprint)
                return existing
            temporary.rename(target)
            validated = validate_cache(target, expected_ids, expected_fingerprint=fingerprint)
            return CacheResult(**{**asdict(validated), "reused": False})
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def run_smoke_gated_extraction(
    rows: Sequence[ManifestRow],
    extractor: WavLMExtractor,
    *,
    smoke_size: int = 16,
    smoke_only: bool = False,
) -> tuple[CacheResult | None, CacheResult | None]:
    """Require a validated small cache before starting uncached full extraction."""
    ordered_rows = sorted(rows, key=lambda row: row.utterance_id)
    if not 1 <= smoke_size < len(ordered_rows):
        raise ValueError(f"smoke_size must be between 1 and {len(ordered_rows) - 1}")

    full_existing = extractor.find_existing_cache(ordered_rows, "full")
    if full_existing is not None:
        return None, full_existing

    smoke_rows = ordered_rows[:smoke_size]
    smoke_result = extractor.extract_or_reuse(smoke_rows, mode="smoke")
    validate_cache(smoke_result.path, [row.utterance_id for row in smoke_rows])
    if smoke_only:
        return smoke_result, None

    full_result = extractor.extract_or_reuse(ordered_rows, mode="full")
    validate_cache(full_result.path, [row.utterance_id for row in ordered_rows])
    return smoke_result, full_result


def run_m2_extraction(
    manifest_path: Path,
    cache_root: Path,
    *,
    smoke_size: int = 16,
    smoke_only: bool = False,
    revision: str = WAVLM_DEFAULT_REVISION,
    device: str = "auto",
    batch_size: int = 4,
) -> tuple[CacheResult | None, CacheResult | None]:
    """Run smoke-gated extraction, returning smoke and optional full results."""
    rows = sorted(read_parquet_manifest(manifest_path), key=lambda row: row.utterance_id)
    validate_ravdess_manifest(rows, require_complete=True)
    extractor = WavLMExtractor(
        cache_root,
        revision=revision,
        device=device,
        batch_size=batch_size,
    )
    return run_smoke_gated_extraction(
        rows,
        extractor,
        smoke_size=smoke_size,
        smoke_only=smoke_only,
    )
