"""Parakeet TDT v3 через изолированный Transformers worker."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from speech_to_sub.asr.base import CancelCheck, ProgressCallback
from speech_to_sub.asr.external_worker import ExternalWorkerError, PersistentNdjsonWorker
from speech_to_sub.exceptions import AsrModelError, ProcessingCancelled, ValidationError
from speech_to_sub.models import (
    ProcessingSettings,
    RuntimeSignature,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)

WORKER_MODULE = "speech_to_sub.workers.parakeet_worker"
DEFAULT_WORKER_RELATIVE_PATH = Path("resources/runtimes/parakeet/Scripts/python.exe")
_SUPPORTED_LANGUAGES = frozenset({"auto", "en", "ru"})

WorkerFactory = Callable[[Path, str], PersistentNdjsonWorker]


@dataclass(frozen=True)
class _PreparedRequest:
    model_path: Path
    worker_python: Path
    runtime: RuntimeSignature


@dataclass(frozen=True)
class _Window:
    offset: float
    duration: float
    is_final: bool


class ParakeetTdtBackend:
    """Оконный Parakeet TDT v3 с нативными временными метками."""

    backend_id = "parakeet-tdt-v3"

    def __init__(self, worker_factory: WorkerFactory = PersistentNdjsonWorker) -> None:
        self._worker_factory = worker_factory
        self._client: PersistentNdjsonWorker | None = None
        self._client_python: Path | None = None
        self._load_key: tuple[str, str, str, bool] | None = None
        self._runtime: RuntimeSignature | None = None
        self._lock = threading.RLock()

    def preflight(self, settings: ProcessingSettings) -> Path:
        """Проверяет локальные веса и изолированный runtime без загрузки модели."""
        with self._lock:
            return self._prepare(settings).model_path

    def expected_runtime_signature(self, settings: ProcessingSettings) -> RuntimeSignature:
        """Возвращает runtime, обнаруженный изолированным worker-ом."""
        with self._lock:
            return self._prepare(settings).runtime

    def runtime_signature(self, settings: ProcessingSettings) -> RuntimeSignature | None:
        """Возвращает runtime только для совпадающего load key."""
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
        """Распознаёт подготовленный FLAC окнами и объединяет абсолютные timestamps."""
        _raise_if_cancelled(cancel_check)
        resolved_audio = audio_path.expanduser().resolve()
        _validate_audio(resolved_audio, duration, settings)
        if progress_callback:
            progress_callback(0)
        with self._lock:
            prepared = self._prepare(settings, cancel_check=cancel_check)
            try:
                transcript = self._transcribe_windows(
                    resolved_audio,
                    settings,
                    duration,
                    prepared,
                    progress_callback,
                    cancel_check,
                )
            except ProcessingCancelled:
                raise
            except ExternalWorkerError as exc:
                transcript = self._retry_on_cpu(
                    exc,
                    resolved_audio,
                    settings,
                    duration,
                    prepared,
                    progress_callback,
                    cancel_check,
                )
        _raise_if_cancelled(cancel_check)
        if progress_callback:
            progress_callback(100)
        return transcript

    def unload(self) -> None:
        """Завершает worker и освобождает модель."""
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
        model_path = _local_checkpoint(settings.model_path)
        worker_python = _worker_python_path(settings)
        if not worker_python.is_file():
            raise ValidationError(
                "Python изолированного Parakeet runtime не найден: "
                f"{worker_python}. Укажите PARAKEET_ASR_PYTHON или worker_python_path."
            )
        result = self._client_for(worker_python).request(
            "preflight",
            {
                "model_path": str(model_path),
                "device": device_override or settings.device,
            },
            cancel_check=cancel_check,
        )
        return _PreparedRequest(model_path, worker_python, _parse_runtime(result))

    def _transcribe_windows(
        self,
        audio_path: Path,
        settings: ProcessingSettings,
        duration: float,
        prepared: _PreparedRequest,
        progress_callback: ProgressCallback | None,
        cancel_check: CancelCheck | None,
    ) -> Transcript:
        segments: list[TranscriptSegment] = []
        detected_languages: list[str] = []
        runtime = prepared.runtime
        for window in _iter_windows(duration, settings):
            _raise_if_cancelled(cancel_check)
            response = self._request_window(audio_path, window, prepared, cancel_check)
            runtime = _parse_runtime(response.get("runtime"))
            accepted = _normalize_window(response, window, settings, duration, len(segments))
            if accepted is not None:
                segments.append(accepted)
            language = str(response.get("language") or "").strip().casefold()
            if language:
                detected_languages.append(language)
            if progress_callback:
                progress_callback(min(99, round((window.offset + window.duration) / duration * 100)))
        segments = _merge_window_segments(segments)
        if not segments:
            raise AsrModelError("Parakeet TDT v3 не вернул распознанного текста.")
        self._runtime = runtime
        self._load_key = _settings_key(settings)
        return _build_transcript(segments, detected_languages, settings, runtime, duration)

    def _request_window(
        self,
        audio_path: Path,
        window: _Window,
        prepared: _PreparedRequest,
        cancel_check: CancelCheck | None,
    ) -> dict[str, Any]:
        return self._client_for(prepared.worker_python).request(
            "transcribe-window",
            {
                "audio_path": str(audio_path),
                "model_path": str(prepared.model_path),
                "offset": window.offset,
                "duration": window.duration,
                "device": prepared.runtime["device"],
            },
            cancel_check=cancel_check,
        )

    def _retry_on_cpu(
        self,
        original_error: ExternalWorkerError,
        audio_path: Path,
        settings: ProcessingSettings,
        duration: float,
        prepared: _PreparedRequest,
        progress_callback: ProgressCallback | None,
        cancel_check: CancelCheck | None,
    ) -> Transcript:
        if not settings.allow_cpu_fallback or prepared.runtime["device"] != "cuda":
            raise AsrModelError(f"Parakeet worker завершился ошибкой: {original_error}") from original_error
        self._shutdown_client()
        try:
            prepared = self._prepare(
                settings,
                device_override="cpu",
                cancel_check=cancel_check,
            )
            if progress_callback:
                progress_callback(0)
            return self._transcribe_windows(
                audio_path,
                settings,
                duration,
                prepared,
                progress_callback,
                cancel_check,
            )
        except ProcessingCancelled:
            raise
        except ExternalWorkerError as cpu_error:
            self._shutdown_client()
            raise AsrModelError(
                "Parakeet завершился ошибкой на CUDA и при повторе на CPU."
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


def _validate_audio(path: Path, duration: float, settings: ProcessingSettings) -> None:
    if not path.is_file():
        raise ValidationError(f"Аудиофайл для Parakeet не найден: {path}")
    if not math.isfinite(duration) or duration <= 0:
        raise ValidationError("Длительность аудио для Parakeet должна быть положительной.")
    language = settings.language.strip().casefold()
    if language not in _SUPPORTED_LANGUAGES:
        raise ValidationError("Parakeet поддерживает язык en, ru или auto.")


def _iter_windows(duration: float, settings: ProcessingSettings) -> Sequence[_Window]:
    window_seconds = float(settings.long_form_window_seconds)
    overlap_seconds = float(settings.long_form_overlap_seconds)
    step = window_seconds - overlap_seconds
    result: list[_Window] = []
    offset = 0.0
    while offset < duration:
        current_duration = min(window_seconds, duration - offset)
        is_final = offset + current_duration >= duration - 1e-6
        result.append(_Window(offset, current_duration, is_final))
        if is_final:
            break
        offset += step
    return tuple(result)


def _normalize_window(
    response: Mapping[str, Any],
    window: _Window,
    settings: ProcessingSettings,
    audio_duration: float,
    segment_id: int,
) -> TranscriptSegment | None:
    raw_tokens = response.get("tokens")
    tokens = raw_tokens if isinstance(raw_tokens, Sequence) else ()
    words = _tokens_to_words(tokens, window, settings.long_form_overlap_seconds, audio_duration)
    if not words:
        return None
    text = _join_words(words)
    return TranscriptSegment(
        start=words[0].start,
        end=words[-1].end,
        text=text,
        words=tuple(words),
        segment_id=segment_id,
    )


def _tokens_to_words(
    raw_tokens: Sequence[Any],
    window: _Window,
    overlap_seconds: float,
    audio_duration: float,
) -> list[TranscriptWord]:
    grouped = _group_subword_tokens(raw_tokens)
    half_overlap = overlap_seconds / 2.0
    commit_start = window.offset if window.offset == 0 else window.offset + half_overlap
    commit_end = window.offset + window.duration - (0.0 if window.is_final else half_overlap)
    result: list[TranscriptWord] = []
    for text, relative_start, relative_end in grouped:
        start = max(0.0, min(audio_duration, window.offset + relative_start))
        end = max(start + 0.001, min(audio_duration, window.offset + relative_end))
        midpoint = (start + end) / 2.0
        if commit_start <= midpoint <= commit_end and end > start:
            result.append(TranscriptWord(start=start, end=end, text=text))
    return result


def _merge_window_segments(
    segments: Sequence[TranscriptSegment],
) -> list[TranscriptSegment]:
    """Убирает дубли и пересечения слов на границах соседних окон."""
    result: list[TranscriptSegment] = []
    previous_word: TranscriptWord | None = None
    for segment in segments:
        words, previous_word = _merge_segment_words(segment.words, previous_word)
        if not words:
            continue
        result.append(
            TranscriptSegment(
                start=words[0].start,
                end=words[-1].end,
                text=_join_words(words),
                words=tuple(words),
                segment_id=len(result),
            )
        )
    return result


def _merge_segment_words(
    source: Sequence[TranscriptWord],
    previous_word: TranscriptWord | None,
) -> tuple[list[TranscriptWord], TranscriptWord | None]:
    words: list[TranscriptWord] = []
    for word in source:
        start = _merged_word_start(word, previous_word)
        if start is None or word.end <= start:
            continue
        previous_word = TranscriptWord(
            start=start,
            end=word.end,
            text=word.text,
            probability=word.probability,
        )
        words.append(previous_word)
    return words, previous_word


def _merged_word_start(
    word: TranscriptWord,
    previous_word: TranscriptWord | None,
) -> float | None:
    if previous_word is None or word.start >= previous_word.end:
        return word.start
    if _normalized_word(word.text) == _normalized_word(previous_word.text):
        return None
    return previous_word.end


def _normalized_word(value: str) -> str:
    return value.strip().casefold()


def _group_subword_tokens(raw_tokens: Sequence[Any]) -> list[tuple[str, float, float]]:
    result: list[tuple[str, float, float]] = []
    current_text = ""
    current_start = 0.0
    current_end = 0.0
    for raw in raw_tokens:
        parsed = _parse_subword_token(raw)
        if parsed is None:
            continue
        token, start, end = parsed
        starts_word = bool(token[:1].isspace())
        normalized = token.strip() if starts_word else token
        if not normalized:
            continue
        if starts_word and current_text:
            result.append((current_text, current_start, max(current_start + 0.001, current_end)))
            current_text = ""
        if not current_text:
            current_start = start
            current_end = start
        current_text += normalized
        current_end = max(current_end, end)
    if current_text:
        result.append((current_text, current_start, max(current_start + 0.001, current_end)))
    return result


def _parse_subword_token(raw: Any) -> tuple[str, float, float] | None:
    if not isinstance(raw, Mapping):
        return None
    token = str(raw.get("token") or "")
    try:
        start = float(raw.get("start", 0.0))
        end = float(raw.get("end", start))
    except (TypeError, ValueError):
        return None
    if not token or not math.isfinite(start) or not math.isfinite(end):
        return None
    return token, start, end


def _build_transcript(
    segments: Sequence[TranscriptSegment],
    detected_languages: Sequence[str],
    settings: ProcessingSettings,
    runtime: RuntimeSignature,
    duration: float,
) -> Transcript:
    language = settings.language if settings.language != "auto" else _detected_language(detected_languages)
    return Transcript(
        text=" ".join(segment.text for segment in segments).strip(),
        language=language,
        duration=duration,
        segments=tuple(segments),
        model=str(settings.model_path.expanduser().resolve()),
        device=runtime["device"],
        quantized=False,
        metadata={
            "runtime": ParakeetTdtBackend.backend_id,
            "engine_version": runtime["engine_version"],
            "compute_type": runtime["compute_type"],
            "word_timestamps": True,
            "window_seconds": settings.long_form_window_seconds,
            "overlap_seconds": settings.long_form_overlap_seconds,
            "isolated_worker": True,
        },
    )


def _join_words(words: Sequence[TranscriptWord]) -> str:
    text = ""
    closing = frozenset(".,!?;:%…)]}—")
    for word in words:
        token = word.text.strip()
        if not token:
            continue
        text += token if not text or token[0] in closing else f" {token}"
    return text


def _detected_language(values: Sequence[str]) -> str:
    aliases = {"english": "en", "russian": "ru", "en": "en", "ru": "ru"}
    normalized = [aliases.get(value, value) for value in values if value]
    return max(set(normalized), key=normalized.count) if normalized else "auto"


def _parse_runtime(value: Any) -> RuntimeSignature:
    if not isinstance(value, Mapping):
        raise AsrModelError("Parakeet worker не вернул описание runtime.")
    device = str(value.get("device") or "").casefold()
    backend = str(value.get("backend") or "")
    if device not in {"cpu", "cuda"} or backend != ParakeetTdtBackend.backend_id:
        raise AsrModelError("Parakeet worker вернул несовместимый runtime.")
    return {
        "backend": backend,
        "engine_version": str(value.get("engine_version") or "unavailable"),
        "device": cast(Any, device),
        "compute_type": str(value.get("compute_type") or "unknown"),
        "quantized": False,
    }


def _local_checkpoint(value: Path | str) -> Path:
    path = Path(value).expanduser().resolve()
    required = ("config.json", "processor_config.json", "tokenizer.json")
    if not path.is_dir() or any(not (path / name).is_file() for name in required):
        raise ValidationError(f"Некорректный локальный checkpoint Parakeet: {path}")
    if not any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        raise ValidationError(f"В checkpoint Parakeet отсутствуют веса: {path}")
    return path


def _worker_python_path(settings: ProcessingSettings) -> Path:
    configured = getattr(settings, "worker_python_path", None)
    raw_path = configured or os.environ.get("PARAKEET_ASR_PYTHON")
    if raw_path:
        return Path(str(raw_path)).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / DEFAULT_WORKER_RELATIVE_PATH).resolve()


def _settings_key(settings: ProcessingSettings) -> tuple[str, str, str, bool]:
    return (
        str(settings.model_path.expanduser().resolve()).casefold(),
        str(_worker_python_path(settings)).casefold(),
        settings.device.strip().casefold(),
        settings.allow_cpu_fallback,
    )


def _raise_if_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and cancel_check():
        raise ProcessingCancelled("Распознавание Parakeet отменено пользователем.")


__all__ = ["ParakeetTdtBackend"]
