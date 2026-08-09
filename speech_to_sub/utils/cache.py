from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from speech_to_sub.constants import (
    PIPELINE_VERSION,
    QWEN_ALIGNMENT_MAX_SEGMENT_SECONDS,
    SIDECAR_SCHEMA_VERSION,
    SRT_BUILDER_VERSION,
)
from speech_to_sub.models import ProcessingSettings
from speech_to_sub.utils.io_utils import atomic_write_json, read_text_utf8, sha256_file


def build_source_fingerprint(path: Path) -> dict[str, Any]:
    """Строит устойчивый fingerprint источника."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


_LEGACY_RECOGNITION_PIPELINE_VERSION = "7"
_LEGACY_RECOGNITION_BUILDER_VERSION = "2"
_LAYOUT_SETTING_KEYS = frozenset(
    {
        "srt_builder_version",
        "max_chars_per_line",
        "line_length_gap",
        "max_cps",
    }
)


def build_recognition_fingerprint(
    settings: ProcessingSettings,
    stream_ordinal: int,
    runtime: Mapping[str, Any] | None = None,
    aligner_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Возвращает параметры тяжёлого распознавания и выравнивания."""
    backend_parameters: dict[str, Any]
    if settings.backend in {"faster-whisper", "parakeet-tdt-v3"}:
        backend_parameters = {
            "long_form_window_seconds": settings.long_form_window_seconds,
            "long_form_overlap_seconds": settings.long_form_overlap_seconds,
            "vad_filter": settings.vad_filter,
            "vad_min_silence_ms": settings.vad_min_silence_ms,
            "beam_size": settings.beam_size,
            "condition_on_previous_text": settings.condition_on_previous_text,
        }
    elif settings.backend == "transformers":
        backend_parameters = {
            "chunk_length_seconds": settings.chunk_length_seconds,
            "stride_length_seconds": settings.stride_length_seconds,
        }
    else:
        backend_parameters = {}
    aligner_path = (
        str(settings.aligner_model_path.expanduser().resolve())
        if settings.aligner_model_path
        else None
    )
    return {
        "pipeline_version": PIPELINE_VERSION,
        "model_path": str(settings.model_path.expanduser().resolve()),
        "language": settings.language,
        "audio_language": settings.audio_language,
        "audio_stream_index": stream_ordinal,
        "requested_device": settings.device,
        "requested_quantization_enabled": settings.quantization_enabled,
        "runtime": dict(
            runtime
            or {
                "device": settings.device,
                "quantized": settings.quantization_enabled,
            }
        ),
        "aligner": {
            "id": settings.aligner,
            "model_path": aligner_path,
            "runtime": dict(aligner_runtime) if aligner_runtime else None,
            "parameters": {
                "max_segment_seconds": QWEN_ALIGNMENT_MAX_SEGMENT_SECONDS,
            }
            if settings.aligner == "qwen3-forced-aligner"
            else {},
        },
        "backend_parameters": backend_parameters,
    }


def build_layout_fingerprint(settings: ProcessingSettings) -> dict[str, Any]:
    """Возвращает параметры лёгкой SRT-разметки."""
    return {
        "srt_builder_version": SRT_BUILDER_VERSION,
        "max_chars_per_line": settings.max_chars_per_line,
        "line_length_gap": settings.line_length_gap,
        "max_cps": settings.max_cps,
    }


def load_sidecar(path: Path) -> dict[str, Any] | None:
    """Читает sidecar или возвращает None при неверном формате."""
    if not path.is_file():
        return None
    try:
        value = json.loads(read_text_utf8(path))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def sidecar_matches(
    sidecar: dict[str, Any] | None,
    source: dict[str, Any],
    recognition_settings: dict[str, Any],
    layout_settings: dict[str, Any],
) -> bool:
    """Проверяет оба отпечатка готового SRT и завершённый статус."""
    fingerprints = sidecar_fingerprints(sidecar)
    return bool(
        sidecar
        and sidecar.get("status") == "done"
        and sidecar.get("source") == source
        and fingerprints == (recognition_settings, layout_settings)
    )


def sidecar_recognition_matches(
    sidecar: dict[str, Any] | None,
    source: dict[str, Any],
    recognition_settings: dict[str, Any],
) -> bool:
    """Проверяет пригодность тяжёлого результата для новой SRT-разметки."""
    fingerprints = sidecar_fingerprints(sidecar)
    return bool(
        sidecar
        and sidecar.get("status") == "done"
        and sidecar.get("source") == source
        and fingerprints is not None
        and fingerprints[0] == recognition_settings
    )


def sidecar_fingerprints(
    sidecar: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Читает отпечатки sidecar v2 либо разрешённой прежней схемы v1."""
    if not sidecar:
        return None
    schema_version = sidecar.get("sidecar_schema_version")
    if (
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version == SIDECAR_SCHEMA_VERSION
    ):
        recognition = sidecar.get("recognition_settings")
        layout = sidecar.get("layout_settings")
        if isinstance(recognition, Mapping) and isinstance(layout, Mapping):
            return dict(recognition), dict(layout)
        return None
    if schema_version is not None:
        return None
    return _legacy_sidecar_fingerprints(sidecar)


def _legacy_sidecar_fingerprints(
    sidecar: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    settings = sidecar.get("settings")
    if not isinstance(settings, Mapping):
        return None
    if (
        settings.get("pipeline_version") != _LEGACY_RECOGNITION_PIPELINE_VERSION
        or settings.get("srt_builder_version")
        != _LEGACY_RECOGNITION_BUILDER_VERSION
    ):
        return None
    recognition = {
        key: value for key, value in settings.items() if key not in _LAYOUT_SETTING_KEYS
    }
    layout = {
        key: value for key, value in settings.items() if key in _LAYOUT_SETTING_KEYS
    }
    return recognition, layout


def write_sidecar(path: Path, payload: dict[str, Any]) -> None:
    """Атомарно сохраняет диагностический sidecar."""
    atomic_write_json(path, payload)
