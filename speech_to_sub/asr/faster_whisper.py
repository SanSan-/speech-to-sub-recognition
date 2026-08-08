"""Локальный long-form backend faster-whisper/CTranslate2."""

from __future__ import annotations

import gc
import importlib.metadata
import logging
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

from speech_to_sub.asr.base import CancelCheck, ProgressCallback, RuntimeSignature
from speech_to_sub.exceptions import AsrModelError, ProcessingCancelled, ValidationError
from speech_to_sub.media.audio_windows import AudioWindow, iter_audio_windows
from speech_to_sub.models import (
    ProcessingSettings,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)

logger = logging.getLogger(__name__)

DEFAULT_LONG_FORM_WINDOW_SECONDS = 300.0
DEFAULT_LONG_FORM_OVERLAP_SECONDS = 2.0
DEFAULT_VAD_MIN_SILENCE_MS = 600


class FasterWhisperBackend:
    """CTranslate2 Whisper с ограниченным по памяти оконным декодированием."""

    backend_id = "faster-whisper"

    def __init__(self) -> None:
        self._model: Any | None = None
        self._load_key: tuple[str, str, bool, bool] | None = None
        self._device_label = "unloaded"
        self._compute_type = "unloaded"
        self._lock = threading.RLock()

    def preflight(self, settings: ProcessingSettings) -> Path:
        """Проверяет локальный CTranslate2 checkpoint без загрузки весов."""
        path = settings.model_path.expanduser().resolve()
        if not path.is_dir():
            raise ValidationError(f"Каталог CTranslate2-модели не найден: {path}")
        ctranslate2 = _import_ctranslate2()
        if not ctranslate2.contains_model(str(path)):
            raise ValidationError(
                f"Каталог не содержит CTranslate2 model.bin/config.json: {path}"
            )
        for filename in ("tokenizer.json", "preprocessor_config.json"):
            if not (path / filename).is_file():
                raise ValidationError(f"В CTranslate2 checkpoint отсутствует {filename}: {path}")
        return path

    def expected_runtime_signature(
        self,
        settings: ProcessingSettings,
    ) -> RuntimeSignature:
        """Определяет ожидаемый CTranslate2 device/compute type без загрузки модели."""
        ctranslate2 = _import_ctranslate2()
        requested = settings.device.strip().casefold()
        if requested not in {"auto", "cuda", "cpu"}:
            raise ValidationError("Устройство должно быть auto, cuda или cpu.")
        cuda_available = int(ctranslate2.get_cuda_device_count()) > 0
        if requested == "cuda" and not cuda_available:
            raise AsrModelError("Запрошена CUDA, но CTranslate2 не обнаружил CUDA-устройство.")
        device = "cuda" if cuda_available and requested != "cpu" else "cpu"
        compute_type = _compute_type(device, settings.quantization_enabled)
        supported = set(ctranslate2.get_supported_compute_types(device))
        if compute_type not in supported:
            raise AsrModelError(
                f"CTranslate2 не поддерживает compute_type={compute_type} на {device}."
            )
        return _runtime_signature(device, compute_type)

    def runtime_signature(
        self,
        settings: ProcessingSettings,
    ) -> RuntimeSignature | None:
        """Возвращает фактический runtime только для совпадающего запроса загрузки."""
        requested_key = _model_load_key(settings.model_path.expanduser().resolve(), settings)
        with self._lock:
            if self._model is None or self._load_key != requested_key:
                return None
            return _runtime_signature(self._device_label, self._compute_type)

    def transcribe(
        self,
        audio_path: Path,
        settings: ProcessingSettings,
        duration: float,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> Transcript:
        """Распознаёт аудио ограниченными окнами и объединяет абсолютные timestamps."""
        _raise_if_cancelled(cancel_check)
        if progress_callback:
            progress_callback(0)
        with self._lock:
            self._ensure_loaded(settings)
            try:
                transcript = self._transcribe_all_windows(
                    audio_path,
                    settings,
                    duration,
                    progress_callback,
                    cancel_check,
                )
            except ProcessingCancelled:
                raise
            except Exception as exc:
                transcript = self._retry_on_cpu(
                    exc,
                    audio_path,
                    settings,
                    duration,
                    progress_callback,
                    cancel_check,
                )
        _raise_if_cancelled(cancel_check)
        if progress_callback:
            progress_callback(100)
        return transcript

    def unload(self) -> None:
        """Освобождает CTranslate2 model и сборщик Python."""
        with self._lock:
            self._model = None
            self._load_key = None
            self._device_label = "unloaded"
            self._compute_type = "unloaded"
        gc.collect()

    def _ensure_loaded(self, settings: ProcessingSettings) -> None:
        model_path = self.preflight(settings)
        key = _model_load_key(model_path, settings)
        if self._model is not None and self._load_key == key:
            return
        if self._model is not None:
            self.unload()
        expected = self.expected_runtime_signature(settings)
        try:
            self._load_once(
                model_path,
                expected["device"],
                expected["compute_type"],
            )
        except Exception as exc:
            if settings.allow_cpu_fallback and expected["device"] == "cuda":
                logger.warning("Загрузка faster-whisper на CUDA не удалась, пробую CPU: %s", exc)
                self.unload()
                self._load_once(
                    model_path,
                    "cpu",
                    _compute_type("cpu", settings.quantization_enabled),
                )
            else:
                raise AsrModelError(f"Не удалось загрузить faster-whisper: {exc}") from exc
        self._load_key = key

    def _load_once(self, model_path: Path, device: str, compute_type: str) -> None:
        whisper_model = _import_whisper_model()
        self._model = whisper_model(
            str(model_path),
            device=device,
            compute_type=compute_type,
            local_files_only=True,
            num_workers=1,
        )
        self._device_label = device
        self._compute_type = compute_type
        logger.info(
            "Локальная faster-whisper модель загружена: device=%s, compute_type=%s.",
            device,
            compute_type,
        )

    def _retry_on_cpu(
        self,
        original_error: Exception,
        audio_path: Path,
        settings: ProcessingSettings,
        duration: float,
        progress_callback: ProgressCallback | None,
        cancel_check: CancelCheck | None,
    ) -> Transcript:
        if not settings.allow_cpu_fallback or self._device_label != "cuda":
            raise AsrModelError(f"faster-whisper завершился ошибкой: {original_error}") from original_error
        logger.warning("Инференс faster-whisper на CUDA не удался, повторяю на CPU: %s", original_error)
        requested_key = self._load_key
        self.unload()
        try:
            model_path = self.preflight(settings)
            self._load_once(
                model_path,
                "cpu",
                _compute_type("cpu", settings.quantization_enabled),
            )
            self._load_key = requested_key
            if progress_callback:
                progress_callback(0)
            return self._transcribe_all_windows(
                audio_path,
                settings,
                duration,
                progress_callback,
                cancel_check,
            )
        except ProcessingCancelled:
            raise
        except Exception as cpu_error:
            self.unload()
            raise AsrModelError(
                "faster-whisper завершился ошибкой на CUDA и при повторе на CPU."
            ) from cpu_error

    def _transcribe_all_windows(
        self,
        audio_path: Path,
        settings: ProcessingSettings,
        duration: float,
        progress_callback: ProgressCallback | None,
        cancel_check: CancelCheck | None,
    ) -> Transcript:
        assert self._model is not None
        window_seconds, overlap_seconds = _window_settings(settings)
        detected_language = None if settings.language == "auto" else settings.language
        normalized_segments: list[TranscriptSegment] = []
        next_segment_id = 0
        for window in iter_audio_windows(
            audio_path,
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
        ):
            _raise_if_cancelled(cancel_check)
            raw_segments, info = self._model.transcribe(
                window.samples,
                language=detected_language,
                task="transcribe",
                beam_size=int(getattr(settings, "beam_size", 5)),
                word_timestamps=True,
                vad_filter=bool(getattr(settings, "vad_filter", True)),
                vad_parameters={
                    "min_silence_duration_ms": int(
                        getattr(settings, "vad_min_silence_ms", DEFAULT_VAD_MIN_SILENCE_MS)
                    )
                },
                condition_on_previous_text=bool(
                    getattr(settings, "condition_on_previous_text", True)
                ),
            )
            materialized = list(raw_segments)
            if detected_language is None:
                detected_language = str(getattr(info, "language", "") or "auto").casefold()
            accepted = _normalize_window_segments(
                materialized,
                window,
                overlap_seconds=overlap_seconds,
                audio_duration=duration,
                first_segment_id=next_segment_id,
            )
            normalized_segments.extend(accepted)
            next_segment_id += len(accepted)
            if progress_callback and duration > 0:
                progress_callback(
                    min(99, round((window.offset + window.duration) / duration * 100))
                )

        if not normalized_segments:
            raise AsrModelError("faster-whisper не вернул распознанного текста.")
        text = " ".join(segment.text for segment in normalized_segments).strip()
        return Transcript(
            text=text,
            language=detected_language or settings.language,
            duration=duration,
            segments=tuple(normalized_segments),
            model=str(settings.model_path),
            device=self._device_label,
            quantized=self._compute_type.startswith("int8"),
            metadata={
                "runtime": self.backend_id,
                "engine_version": _engine_version(),
                "compute_type": self._compute_type,
                "word_timestamps": any(segment.words for segment in normalized_segments),
                "window_seconds": window_seconds,
                "overlap_seconds": overlap_seconds,
                "vad_filter": bool(getattr(settings, "vad_filter", True)),
            },
        )


def _raise_if_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and cancel_check():
        raise ProcessingCancelled("Распознавание faster-whisper отменено пользователем.")


def _normalize_window_segments(
    raw_segments: Sequence[Any],
    window: AudioWindow,
    *,
    overlap_seconds: float,
    audio_duration: float,
    first_segment_id: int,
) -> list[TranscriptSegment]:
    half_overlap = overlap_seconds / 2.0
    commit_start = window.offset if window.offset == 0 else window.offset + half_overlap
    commit_end = window.offset + window.duration
    if not window.is_final:
        commit_end -= half_overlap
    result: list[TranscriptSegment] = []
    for raw_segment in raw_segments:
        raw_words = getattr(raw_segment, "words", None) or ()
        words = _normalize_words(
            raw_words,
            window.offset,
            commit_start,
            commit_end,
            audio_duration,
        )
        if words:
            text = "".join(word.text for word in words).strip()
            start, end = words[0].start, words[-1].end
        else:
            if raw_words:
                continue
            start = max(0.0, window.offset + float(getattr(raw_segment, "start", 0.0)))
            end = min(
                audio_duration,
                window.offset + float(getattr(raw_segment, "end", 0.0)),
            )
            midpoint = (start + end) / 2.0
            if midpoint < commit_start or midpoint >= commit_end or end <= start:
                continue
            text = str(getattr(raw_segment, "text", "")).strip()
        if not text:
            continue
        result.append(
            TranscriptSegment(
                start=start,
                end=end,
                text=text,
                words=tuple(words),
                segment_id=first_segment_id + len(result),
            )
        )
    return result


def _normalize_words(
    raw_words: Iterable[Any],
    offset: float,
    commit_start: float,
    commit_end: float,
    audio_duration: float,
) -> list[TranscriptWord]:
    result: list[TranscriptWord] = []
    for raw_word in raw_words:
        start = max(0.0, offset + float(getattr(raw_word, "start", 0.0)))
        end = min(audio_duration, offset + float(getattr(raw_word, "end", 0.0)))
        midpoint = (start + end) / 2.0
        text = str(getattr(raw_word, "word", getattr(raw_word, "text", "")))
        if not text.strip() or end <= start:
            continue
        if midpoint < commit_start or midpoint >= commit_end:
            continue
        probability = getattr(raw_word, "probability", None)
        result.append(
            TranscriptWord(
                start=start,
                end=end,
                text=text,
                probability=float(probability) if probability is not None else None,
            )
        )
    return result


def _window_settings(settings: ProcessingSettings) -> tuple[float, float]:
    window_seconds = float(
        getattr(settings, "long_form_window_seconds", DEFAULT_LONG_FORM_WINDOW_SECONDS)
    )
    overlap_seconds = float(
        getattr(settings, "long_form_overlap_seconds", DEFAULT_LONG_FORM_OVERLAP_SECONDS)
    )
    if window_seconds <= 0 or overlap_seconds < 0 or overlap_seconds >= window_seconds:
        raise ValidationError("Некорректные параметры long-form окон.")
    return window_seconds, overlap_seconds


def _compute_type(device: str, quantization_enabled: bool) -> str:
    if device == "cuda":
        return "int8_float16" if quantization_enabled else "float16"
    return "int8" if quantization_enabled else "float32"


def _runtime_signature(device: str, compute_type: str) -> RuntimeSignature:
    return {
        "backend": FasterWhisperBackend.backend_id,
        "engine_version": _engine_version(),
        "device": cast(Any, device),
        "compute_type": compute_type,
        "quantized": compute_type.startswith("int8"),
    }


def _model_load_key(
    model_path: Path,
    settings: ProcessingSettings,
) -> tuple[str, str, bool, bool]:
    return (
        str(model_path).casefold(),
        settings.device.casefold(),
        settings.quantization_enabled,
        settings.allow_cpu_fallback,
    )


def _engine_version() -> str:
    try:
        return importlib.metadata.version("faster-whisper")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _import_ctranslate2() -> Any:
    try:
        import ctranslate2
    except ImportError as exc:
        raise AsrModelError("Для backend faster-whisper требуется CTranslate2.") from exc
    return ctranslate2


def _import_whisper_model() -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise AsrModelError("Для выбранного backend требуется faster-whisper.") from exc
    return WhisperModel


__all__ = ["FasterWhisperBackend"]
