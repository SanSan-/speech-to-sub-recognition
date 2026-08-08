from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from speech_to_sub.constants import (
    PIPELINE_VERSION,
    QWEN_ALIGNMENT_MAX_SEGMENT_SECONDS,
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


def build_settings_fingerprint(
    settings: ProcessingSettings,
    stream_ordinal: int,
    runtime: Mapping[str, Any] | None = None,
    aligner_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Возвращает значимые для результата параметры."""
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
        "srt_builder_version": SRT_BUILDER_VERSION,
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
        "max_chars_per_line": settings.max_chars_per_line,
        "backend_parameters": backend_parameters,
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
    settings: dict[str, Any],
) -> bool:
    """Проверяет fingerprint sidecar и завершённый статус."""
    return bool(
        sidecar
        and sidecar.get("status") == "done"
        and sidecar.get("source") == source
        and sidecar.get("settings") == settings
    )


def write_sidecar(path: Path, payload: dict[str, Any]) -> None:
    """Атомарно сохраняет диагностический sidecar."""
    atomic_write_json(path, payload)
