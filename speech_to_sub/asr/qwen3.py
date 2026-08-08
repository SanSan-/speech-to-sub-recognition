"""Qwen3-ASR backend через изолированный persistent Python worker."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from speech_to_sub.asr.base import CancelCheck, ProgressCallback
from speech_to_sub.asr.checkpoints import require_local_checkpoint
from speech_to_sub.asr.external_worker import ExternalWorkerError, PersistentNdjsonWorker
from speech_to_sub.exceptions import AsrModelError, ProcessingCancelled, ValidationError
from speech_to_sub.models import ProcessingSettings, RuntimeSignature, Transcript, TranscriptSegment

WORKER_MODULE = "speech_to_sub.workers.qwen_worker"
DEFAULT_WORKER_RELATIVE_PATH = Path("resources/runtimes/qwen/Scripts/python.exe")
_SUPPORTED_LANGUAGES = frozenset({"auto", "en", "ru"})
_ASR_CHUNK_TARGET_SECONDS = 175

WorkerFactory = Callable[[Path, str], PersistentNdjsonWorker]


@dataclass(frozen=True)
class _PreparedRequest:
    model_path: Path
    worker_python: Path
    runtime: RuntimeSignature


class Qwen3AsrBackend:
    """Адаптер Qwen3-ASR 0.6B без встроенного ForcedAligner."""

    backend_id = "qwen3-asr"

    def __init__(self, worker_factory: WorkerFactory = PersistentNdjsonWorker) -> None:
        self._worker_factory = worker_factory
        self._client: PersistentNdjsonWorker | None = None
        self._client_python: Path | None = None
        self._load_key: tuple[str, str, str, bool] | None = None
        self._runtime: RuntimeSignature | None = None
        self._lock = threading.RLock()

    def preflight(self, settings: ProcessingSettings) -> Path:
        """Проверяет worker и локальный checkpoint без загрузки весов."""
        with self._lock:
            return self._prepare(settings).model_path

    def expected_runtime_signature(self, settings: ProcessingSettings) -> RuntimeSignature:
        """Получает доступный device/precision из изолированного runtime."""
        with self._lock:
            return self._prepare(settings).runtime

    def runtime_signature(self, settings: ProcessingSettings) -> RuntimeSignature | None:
        """Возвращает runtime только для совпадающего ASR load key."""
        requested_key = _settings_key(settings)
        with self._lock:
            if self._runtime is None or self._load_key != requested_key:
                return None
            return dict(self._runtime)  # type: ignore[return-value]

    def transcribe(
        self,
        audio_path: Path,
        settings: ProcessingSettings,
        duration: float,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> Transcript:
        """Распознаёт аудио в coarse-сегменты, пригодные для отдельного aligner-а."""
        resolved_audio = audio_path.expanduser().resolve()
        _validate_audio(resolved_audio, duration)
        _raise_if_cancelled(cancel_check)
        if progress_callback:
            progress_callback(0)
        with self._lock:
            prepared = self._prepare(settings, cancel_check=cancel_check)
            if progress_callback:
                progress_callback(5)
            try:
                response = self._request_transcription(
                    resolved_audio,
                    settings,
                    prepared,
                    progress_callback,
                    cancel_check,
                )
            except ExternalWorkerError as exc:
                response, prepared = self._retry_on_cpu(
                    exc,
                    resolved_audio,
                    settings,
                    prepared,
                    progress_callback,
                    cancel_check,
                )
            runtime = _parse_runtime(response.get("runtime"))
            self._runtime = runtime
            self._load_key = _settings_key(settings)
            transcript = _normalize_transcript(response, settings, prepared, runtime, duration)
        _raise_if_cancelled(cancel_check)
        if progress_callback:
            progress_callback(100)
        return transcript

    def unload(self) -> None:
        """Завершает worker и гарантированно освобождает ASR-модель."""
        with self._lock:
            self._shutdown_client()
            self._load_key = None
            self._runtime = None

    def _prepare(
        self,
        settings: ProcessingSettings,
        *,
        device_override: str | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> _PreparedRequest:
        language = str(settings.language).strip().casefold()
        if language not in _SUPPORTED_LANGUAGES:
            raise ValidationError("Qwen3-ASR поддерживает язык en, ru или auto.")
        model_path = require_local_checkpoint(settings.model_path, "Qwen3-ASR")
        worker_python = _worker_python_path(settings)
        if not worker_python.is_file():
            raise ValidationError(
                "Python изолированного Qwen runtime не найден: "
                f"{worker_python}. Укажите QWEN_ASR_PYTHON или worker_python_path."
            )
        result = self._client_for(worker_python).request(
            "preflight",
            {"model_path": str(model_path), "device": device_override or settings.device},
            cancel_check=cancel_check,
        )
        return _PreparedRequest(model_path, worker_python, _parse_runtime(result))

    def _request_transcription(
        self,
        audio_path: Path,
        settings: ProcessingSettings,
        prepared: _PreparedRequest,
        progress_callback: ProgressCallback | None,
        cancel_check: CancelCheck | None,
    ) -> dict[str, Any]:
        def emit_worker_progress(value: int) -> None:
            if progress_callback:
                progress_callback(5 + round(max(0, min(100, value)) * 0.9))

        return self._client_for(prepared.worker_python).request(
            "transcribe",
            {
                "audio_path": str(audio_path),
                "model_path": str(prepared.model_path),
                "language": settings.language,
                "device": prepared.runtime["device"],
            },
            progress_callback=emit_worker_progress,
            cancel_check=cancel_check,
        )

    def _retry_on_cpu(
        self,
        original_error: ExternalWorkerError,
        audio_path: Path,
        settings: ProcessingSettings,
        prepared: _PreparedRequest,
        progress_callback: ProgressCallback | None,
        cancel_check: CancelCheck | None,
    ) -> tuple[dict[str, Any], _PreparedRequest]:
        if not settings.allow_cpu_fallback or prepared.runtime["device"] != "cuda":
            raise AsrModelError(f"Qwen3-ASR worker завершился ошибкой: {original_error}") from original_error
        self._shutdown_client()
        try:
            cpu_prepared = self._prepare(
                settings,
                device_override="cpu",
                cancel_check=cancel_check,
            )
            response = self._request_transcription(
                audio_path,
                settings,
                cpu_prepared,
                progress_callback,
                cancel_check,
            )
            return response, cpu_prepared
        except ExternalWorkerError as cpu_error:
            self._shutdown_client()
            raise AsrModelError(
                "Qwen3-ASR завершился ошибкой на CUDA и при повторе на CPU."
            ) from cpu_error

    def _client_for(self, python_path: Path) -> PersistentNdjsonWorker:
        if self._client is not None and self._client_python == python_path:
            return self._client
        self._shutdown_client()
        self._client = self._worker_factory(python_path, WORKER_MODULE)
        self._client_python = python_path
        return self._client

    def _shutdown_client(self) -> None:
        if self._client is not None:
            self._client.shutdown()
        self._client = None
        self._client_python = None


def _normalize_transcript(
    response: Mapping[str, Any],
    settings: ProcessingSettings,
    prepared: _PreparedRequest,
    runtime: RuntimeSignature,
    duration: float,
) -> Transcript:
    text = " ".join(str(response.get("text") or "").split())
    if not text:
        raise AsrModelError("Qwen3-ASR не вернул распознанного текста.")
    raw_chunks = response.get("chunks")
    chunks = raw_chunks if isinstance(raw_chunks, Sequence) else ()
    segments, languages = _normalize_segments(chunks, settings.language, duration)
    if not segments:
        segments = (TranscriptSegment(0.0, duration, text, segment_id=0),)
        languages = (_normalize_language(response.get("language"), settings.language),)
    return Transcript(
        text=text,
        language=_normalize_language(response.get("language"), settings.language),
        duration=duration,
        segments=segments,
        model=str(prepared.model_path),
        device=runtime["device"],
        quantized=False,
        metadata={
            "runtime": Qwen3AsrBackend.backend_id,
            "engine_version": runtime["engine_version"],
            "compute_type": runtime["compute_type"],
            "word_timestamps": False,
            "aligner": "none",
            "asr_chunk_target_seconds": _ASR_CHUNK_TARGET_SECONDS,
            "segment_languages": languages,
            "isolated_worker": True,
        },
    )


def _normalize_segments(
    chunks: Sequence[Any],
    requested_language: str,
    duration: float,
) -> tuple[tuple[TranscriptSegment, ...], tuple[str, ...]]:
    result: list[TranscriptSegment] = []
    languages: list[str] = []
    previous_end = 0.0
    for raw in chunks:
        if not isinstance(raw, Mapping):
            continue
        chunk_text = " ".join(str(raw.get("text") or "").split())
        try:
            start = max(previous_end, float(raw.get("start", 0.0)))
            end = min(duration, float(raw.get("end", duration)))
        except (TypeError, ValueError):
            continue
        if not chunk_text or not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        language = _normalize_language(raw.get("language"), requested_language)
        result.append(TranscriptSegment(start, end, chunk_text, segment_id=len(result)))
        languages.append(language)
        previous_end = end
    return tuple(result), tuple(languages)


def _parse_runtime(value: Any) -> RuntimeSignature:
    if not isinstance(value, Mapping):
        raise AsrModelError("Qwen worker не вернул описание runtime.")
    device = str(value.get("device") or "").casefold()
    if device not in {"cpu", "cuda"}:
        raise AsrModelError(f"Qwen worker вернул неизвестное устройство: {device or '?'}")
    backend = str(value.get("backend") or "")
    if backend != Qwen3AsrBackend.backend_id:
        raise AsrModelError(f"Qwen worker вернул неизвестный backend: {backend or '?'}")
    return {
        "backend": backend,
        "engine_version": str(value.get("engine_version") or "unavailable"),
        "device": cast(Any, device),
        "compute_type": str(value.get("compute_type") or "unknown"),
        "quantized": bool(value.get("quantized", False)),
    }


def _worker_python_path(settings: ProcessingSettings) -> Path:
    configured = getattr(settings, "worker_python_path", None)
    raw_path = configured or os.environ.get("QWEN_ASR_PYTHON")
    if raw_path:
        return Path(str(raw_path)).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / DEFAULT_WORKER_RELATIVE_PATH).resolve()


def _settings_key(settings: ProcessingSettings) -> tuple[str, str, str, bool]:
    return (
        str(settings.model_path.expanduser().resolve()).casefold(),
        str(_worker_python_path(settings)).casefold(),
        settings.device.strip().casefold(),
        settings.allow_cpu_fallback,
    )


def _normalize_language(value: Any, requested_language: str) -> str:
    if requested_language != "auto":
        return requested_language
    raw = str(value or "").strip().casefold()
    aliases = {"english": "en", "russian": "ru", "en": "en", "ru": "ru"}
    if not raw:
        return "auto"
    parts = [aliases.get(part.strip(), part.strip()) for part in raw.split(",")]
    return ",".join(part for part in parts if part) or "auto"


def _validate_audio(audio_path: Path, duration: float) -> None:
    if not audio_path.is_file():
        raise ValidationError(f"Аудиофайл для Qwen3-ASR не найден: {audio_path}")
    if not math.isfinite(duration) or duration <= 0:
        raise ValidationError("Длительность аудио для Qwen3-ASR должна быть положительной.")


def _raise_if_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and cancel_check():
        raise ProcessingCancelled("Распознавание Qwen3-ASR отменено.")


__all__ = ["Qwen3AsrBackend"]
