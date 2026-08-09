"""Облачное распознавание через файловый OpenAI Speech-to-Text API."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import math
import os
import subprocess
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speech_to_sub.asr.base import CancelCheck, ProgressCallback
from speech_to_sub.exceptions import AsrModelError, ProcessingCancelled, ValidationError
from speech_to_sub.models import (
    ProcessingSettings,
    RuntimeSignature,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)

DEFAULT_OPENAI_TRANSCRIPTION_MODEL = "whisper-1"
DEFAULT_CHUNK_SECONDS = 900.0
WHISPER_CHUNK_OVERLAP_SECONDS = 2.0
MAX_UPLOAD_BYTES = 24_000_000
MP3_BITRATE = "128k"
OPENAI_MAX_RETRIES = 2
OPENAI_TIMEOUT_SECONDS = 600.0
MAX_CONTEXT_CHARS = 200

SUPPORTED_MODELS = frozenset({"whisper-1"})

ClientFactory = Callable[..., Any]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class OpenAiApiError(AsrModelError):
    """Базовая ошибка облачного распознавания OpenAI."""


class OpenAiAuthenticationError(OpenAiApiError):
    """Ключ API отсутствует либо отклонён сервером."""


class OpenAiRateLimitError(OpenAiApiError):
    """Превышен лимит запросов или доступная квота OpenAI."""


class OpenAiConnectionError(OpenAiApiError):
    """Сервис OpenAI недоступен по сети или не ответил вовремя."""


class OpenAiRequestError(OpenAiApiError):
    """OpenAI отклонил параметры запроса."""


class OpenAiResponseError(OpenAiApiError):
    """OpenAI вернул неполный результат распознавания."""


class OpenAiAudioPreparationError(OpenAiApiError):
    """Не удалось подготовить безопасную часть аудио для загрузки."""


@dataclass(frozen=True, slots=True)
class _Chunk:
    index: int
    offset: float
    duration: float
    is_final: bool


@dataclass(frozen=True, slots=True)
class _PreparedRequest:
    model: str
    api_key: str
    runtime: RuntimeSignature


@dataclass(frozen=True, slots=True)
class _ChunkResult:
    segments: tuple[TranscriptSegment, ...]
    language: str
    used_timestamp_fallback: bool


class OpenAiApiBackend:
    """Адаптер OpenAI API с явным выбором облака и безопасной загрузкой частями."""

    backend_id = "openai-api"

    def __init__(
        self,
        client_factory: ClientFactory | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self._client_factory = client_factory or _default_client_factory
        self._command_runner = command_runner or subprocess.run
        self._client: Any | None = None
        self._client_key_digest: bytes | None = None
        self._active_model: str | None = None
        self._runtime: RuntimeSignature | None = None
        self._lock = threading.RLock()

    def preflight(self, settings: ProcessingSettings) -> Path:
        """Проверяет явный выбор облака, ключ и установленный клиент без сетевого запроса."""
        prepared = _prepare_request(settings)
        return Path(prepared.model)

    def expected_runtime_signature(
        self, settings: ProcessingSettings
    ) -> RuntimeSignature:
        """Возвращает подпись клиента и выбранной удалённой модели."""
        return _runtime_signature(_selected_model(settings))

    def runtime_signature(
        self, settings: ProcessingSettings
    ) -> RuntimeSignature | None:
        """Возвращает подпись только после успешного запроса той же модели."""
        model = _selected_model(settings)
        with self._lock:
            if self._active_model != model or self._runtime is None:
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
        """Передаёт нормализованное аудио в OpenAI частями и объединяет временные метки."""
        resolved_audio = audio_path.expanduser().resolve()
        _validate_audio(resolved_audio, duration)
        _raise_if_cancelled(cancel_check)
        prepared = _prepare_request(settings)
        if progress_callback:
            progress_callback(0)
        with self._lock:
            client = self._client_for(prepared.api_key)
            transcript = self._transcribe_chunks(
                client,
                resolved_audio,
                settings,
                duration,
                prepared,
                progress_callback,
                cancel_check,
            )
            self._active_model = prepared.model
            self._runtime = prepared.runtime
        _raise_if_cancelled(cancel_check)
        if progress_callback:
            progress_callback(100)
        return transcript

    def unload(self) -> None:
        """Закрывает HTTP-клиент и удаляет ключ из памяти адаптера."""
        with self._lock:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
            self._client = None
            self._client_key_digest = None
            self._active_model = None
            self._runtime = None

    def _client_for(self, api_key: str) -> Any:
        digest = hashlib.sha256(api_key.encode("utf-8")).digest()
        if self._client is not None and self._client_key_digest == digest:
            return self._client
        self.unload()
        try:
            self._client = self._client_factory(
                api_key=api_key,
                max_retries=OPENAI_MAX_RETRIES,
                timeout=OPENAI_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise _map_openai_error(exc) from None
        self._client_key_digest = digest
        return self._client

    def _transcribe_chunks(
        self,
        client: Any,
        audio_path: Path,
        settings: ProcessingSettings,
        duration: float,
        prepared: _PreparedRequest,
        progress_callback: ProgressCallback | None,
        cancel_check: CancelCheck | None,
    ) -> Transcript:
        overlap = WHISPER_CHUNK_OVERLAP_SECONDS
        chunks = _iter_chunks(duration, DEFAULT_CHUNK_SECONDS, overlap)
        segments: list[TranscriptSegment] = []
        languages: list[str] = []
        fallback_used = False
        prompt = ""
        with tempfile.TemporaryDirectory(prefix="speech-to-sub-openai-") as temp_dir:
            for chunk in chunks:
                _raise_if_cancelled(cancel_check)
                chunk_path = Path(temp_dir) / f"chunk-{chunk.index:05d}.mp3"
                _encode_chunk(audio_path, chunk_path, chunk, self._command_runner)
                _raise_if_cancelled(cancel_check)
                response = _request_transcription(
                    client,
                    chunk_path,
                    prepared.model,
                    settings.language,
                    prompt,
                )
                _raise_if_cancelled(cancel_check)
                normalized = _normalize_response(response, chunk, duration, overlap)
                segments.extend(normalized.segments)
                languages.append(normalized.language)
                fallback_used = fallback_used or normalized.used_timestamp_fallback
                prompt = _context_tail(_response_text(response))
                if progress_callback:
                    processed_until = min(duration, chunk.offset + chunk.duration)
                    progress_callback(min(99, round(processed_until / duration * 100)))
        return _build_transcript(
            segments,
            languages,
            settings,
            prepared,
            duration,
            overlap,
            fallback_used,
        )


def _prepare_request(settings: ProcessingSettings) -> _PreparedRequest:
    backend = str(settings.backend).strip().casefold()
    if backend != OpenAiApiBackend.backend_id:
        raise ValidationError(
            "Отправка аудио в OpenAI разрешена только после явного выбора backend openai-api."
        )
    if not bool(getattr(settings, "allow_cloud_processing", False)):
        raise ValidationError(
            "Передача аудио в OpenAI требует явного согласия пользователя."
        )
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise OpenAiAuthenticationError(
            "Для OpenAI API требуется переменная окружения OPENAI_API_KEY."
        )
    model = _selected_model(settings)
    _import_openai()
    return _PreparedRequest(model, api_key, _runtime_signature(model))


def _selected_model(settings: ProcessingSettings) -> str:
    configured = str(getattr(settings, "openai_model", "") or "").strip()
    model = (
        configured
        or os.getenv(
            "OPENAI_TRANSCRIPTION_MODEL",
            DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
        ).strip()
    )
    model = model or DEFAULT_OPENAI_TRANSCRIPTION_MODEL
    if model not in SUPPORTED_MODELS:
        variants = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValidationError(
            f"Модель OpenAI '{model}' не поддерживается. Доступны: {variants}."
        )
    return model


def _runtime_signature(model: str) -> RuntimeSignature:
    return {
        "backend": OpenAiApiBackend.backend_id,
        "engine_version": _sdk_version(),
        "device": "cloud",
        "compute_type": f"remote:{model}",
        "quantized": False,
    }


def _iter_chunks(
    duration: float, chunk_seconds: float, overlap: float
) -> tuple[_Chunk, ...]:
    if overlap < 0 or overlap >= chunk_seconds:
        raise ValidationError(
            "Перекрытие частей OpenAI должно быть меньше длины части."
        )
    chunks: list[_Chunk] = []
    offset = 0.0
    step = chunk_seconds - overlap
    while offset < duration:
        current_duration = min(chunk_seconds, duration - offset)
        is_final = offset + current_duration >= duration - 1e-6
        chunks.append(_Chunk(len(chunks), offset, current_duration, is_final))
        if is_final:
            break
        offset += step
    return tuple(chunks)


def _encode_chunk(
    source: Path,
    destination: Path,
    chunk: _Chunk,
    runner: CommandRunner,
) -> None:
    args = _build_ffmpeg_chunk_args(source, destination, chunk)
    try:
        result = runner(
            args,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise OpenAiAudioPreparationError(
            "Не удалось запустить FFmpeg для OpenAI API."
        ) from exc
    if result.returncode != 0:
        raise OpenAiAudioPreparationError(
            f"FFmpeg не подготовил часть для OpenAI API, код {result.returncode}."
        )
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise OpenAiAudioPreparationError(
            "FFmpeg не создал непустую часть для OpenAI API."
        )
    if destination.stat().st_size > MAX_UPLOAD_BYTES:
        raise OpenAiAudioPreparationError(
            "Подготовленная часть превышает безопасный предел загрузки OpenAI API."
        )


def _build_ffmpeg_chunk_args(
    source: Path, destination: Path, chunk: _Chunk
) -> list[str]:
    ffmpeg = os.getenv("FFMPEG_PATH", "ffmpeg").strip() or "ffmpeg"
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{chunk.offset:.3f}",
        "-t",
        f"{chunk.duration:.3f}",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        MP3_BITRATE,
        str(destination),
    ]


def _request_transcription(
    client: Any,
    chunk_path: Path,
    model: str,
    language: str,
    prompt: str,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment", "word"],
    }
    normalized_language = language.strip().casefold()
    if normalized_language != "auto":
        kwargs["language"] = normalized_language
    if prompt:
        kwargs["prompt"] = prompt
    try:
        with chunk_path.open("rb") as audio_file:
            return client.audio.transcriptions.create(file=audio_file, **kwargs)
    except Exception as exc:
        raise _map_openai_error(exc) from None


def _normalize_response(
    response: Any,
    chunk: _Chunk,
    audio_duration: float,
    overlap: float,
) -> _ChunkResult:
    text = _response_text(response)
    if not text:
        raise OpenAiResponseError("OpenAI API вернул пустое распознавание.")
    words = _normalize_words(
        _field(response, "words", ()), chunk, audio_duration, overlap
    )
    raw_segments = _as_sequence(_field(response, "segments", ()))
    if words:
        segments = _segments_from_words(words, raw_segments, chunk, audio_duration)
        fallback = False
    else:
        segments = _segments_without_words(
            raw_segments, text, chunk, audio_duration, overlap
        )
        fallback = not bool(raw_segments)
    if not segments:
        raise OpenAiResponseError("OpenAI API не вернул пригодных временных границ.")
    return _ChunkResult(tuple(segments), _response_language(response), fallback)


def _normalize_words(
    value: Any,
    chunk: _Chunk,
    audio_duration: float,
    overlap: float,
) -> list[TranscriptWord]:
    commit_start, commit_end = _commit_bounds(chunk, overlap, audio_duration)
    words: list[TranscriptWord] = []
    for raw in _as_sequence(value):
        relative_start = _finite_number(_field(raw, "start", None))
        relative_end = _finite_number(_field(raw, "end", None))
        text = str(_field(raw, "word", _field(raw, "text", "")) or "").strip()
        if (
            relative_start is None
            or relative_end is None
            or relative_end <= relative_start
            or not text
        ):
            continue
        start = max(0.0, min(audio_duration, chunk.offset + relative_start))
        end = max(start, min(audio_duration, chunk.offset + relative_end))
        midpoint = (start + end) / 2.0
        inside = commit_start <= midpoint and (chunk.is_final or midpoint < commit_end)
        if inside and end > start:
            words.append(TranscriptWord(start, end, text))
    return words


def _segments_from_words(
    words: Sequence[TranscriptWord],
    raw_segments: Sequence[Any],
    chunk: _Chunk,
    audio_duration: float,
) -> list[TranscriptSegment]:
    result: list[TranscriptSegment] = []
    used: set[int] = set()
    for raw in raw_segments:
        relative_start = _finite_number(_field(raw, "start", None))
        relative_end = _finite_number(_field(raw, "end", None))
        if relative_start is None or relative_end is None:
            continue
        start = max(0.0, min(audio_duration, chunk.offset + relative_start))
        end = max(start, min(audio_duration, chunk.offset + relative_end))
        indexes = [
            index
            for index, word in enumerate(words)
            if index not in used and start <= (word.start + word.end) / 2.0 <= end
        ]
        if not indexes:
            continue
        segment_words = tuple(words[index] for index in indexes)
        used.update(indexes)
        result.append(_segment_from_words(segment_words))
    remaining = tuple(word for index, word in enumerate(words) if index not in used)
    if remaining:
        result.append(_segment_from_words(remaining))
    return sorted(result, key=lambda segment: (segment.start, segment.end))


def _segments_without_words(
    raw_segments: Sequence[Any],
    text: str,
    chunk: _Chunk,
    audio_duration: float,
    overlap: float,
) -> list[TranscriptSegment]:
    commit_start, commit_end = _commit_bounds(chunk, overlap, audio_duration)
    result: list[TranscriptSegment] = []
    for raw in raw_segments:
        relative_start = _finite_number(_field(raw, "start", None))
        relative_end = _finite_number(_field(raw, "end", None))
        segment_text = " ".join(str(_field(raw, "text", "") or "").split())
        if relative_start is None or relative_end is None or not segment_text:
            continue
        start = max(commit_start, chunk.offset + relative_start)
        end = min(commit_end, chunk.offset + relative_end)
        if end > start:
            result.append(TranscriptSegment(start, end, segment_text))
    if result:
        return result
    if commit_end <= commit_start:
        return []
    return [TranscriptSegment(commit_start, commit_end, text)]


def _segment_from_words(words: tuple[TranscriptWord, ...]) -> TranscriptSegment:
    return TranscriptSegment(
        words[0].start,
        words[-1].end,
        _join_words(words),
        words=words,
    )


def _build_transcript(
    source_segments: Sequence[TranscriptSegment],
    languages: Sequence[str],
    settings: ProcessingSettings,
    prepared: _PreparedRequest,
    duration: float,
    overlap: float,
    fallback_used: bool,
) -> Transcript:
    segments = _finalize_segments(source_segments, duration)
    if not segments:
        raise OpenAiResponseError("OpenAI API не вернул пригодного текста.")
    language = _choose_language(languages, settings.language)
    return Transcript(
        text=" ".join(segment.text for segment in segments).strip(),
        language=language,
        duration=duration,
        segments=segments,
        model=prepared.model,
        device="cloud",
        quantized=False,
        metadata={
            "runtime": OpenAiApiBackend.backend_id,
            "engine_version": prepared.runtime["engine_version"],
            "compute_type": prepared.runtime["compute_type"],
            "provider": "openai",
            "endpoint": "audio/transcriptions",
            "word_timestamps": any(segment.words for segment in segments),
            "timestamp_fallback": "chunk" if fallback_used else "none",
            "chunk_seconds": DEFAULT_CHUNK_SECONDS,
            "chunk_overlap_seconds": overlap,
        },
    )


def _finalize_segments(
    source: Sequence[TranscriptSegment],
    duration: float,
) -> tuple[TranscriptSegment, ...]:
    result: list[TranscriptSegment] = []
    previous_end = 0.0
    for segment in sorted(source, key=lambda item: (item.start, item.end)):
        start = max(previous_end, min(duration, segment.start))
        end = min(duration, segment.end)
        if end <= start or not segment.text.strip():
            continue
        words = tuple(
            word for word in segment.words if start <= word.start < word.end <= end
        )
        result.append(
            TranscriptSegment(
                start, end, segment.text.strip(), words, segment_id=len(result)
            )
        )
        previous_end = end
    return tuple(result)


def _commit_bounds(
    chunk: _Chunk, overlap: float, audio_duration: float
) -> tuple[float, float]:
    half_overlap = overlap / 2.0
    start = chunk.offset if chunk.index == 0 else chunk.offset + half_overlap
    end = (
        chunk.offset + chunk.duration
        if chunk.is_final
        else chunk.offset + chunk.duration - half_overlap
    )
    return max(0.0, start), min(audio_duration, end)


def _response_text(response: Any) -> str:
    return " ".join(str(_field(response, "text", "") or "").split())


def _response_language(response: Any) -> str:
    direct = str(_field(response, "language", "") or "").strip().casefold()
    if direct:
        return _normalize_language(direct)
    languages = _as_sequence(_field(response, "languages", ()))
    if languages:
        code = str(_field(languages[0], "code", "") or "").strip().casefold()
        if code:
            return _normalize_language(code)
    return "auto"


def _choose_language(values: Sequence[str], requested: str) -> str:
    if requested != "auto":
        return requested
    known = [value for value in values if value and value != "auto"]
    return max(set(known), key=known.count) if known else "auto"


def _normalize_language(value: str) -> str:
    aliases = {"english": "en", "russian": "ru"}
    return aliases.get(value, value)


def _context_tail(text: str) -> str:
    return text[-MAX_CONTEXT_CHARS:] if text else ""


def _join_words(words: Sequence[TranscriptWord]) -> str:
    text = ""
    closing = frozenset(".,!?;:%…)]}—")
    for word in words:
        token = word.text.strip()
        if token:
            text += token if not text or token[0] in closing else f" {token}"
    return text


def _validate_audio(path: Path, duration: float) -> None:
    if not path.is_file():
        raise ValidationError(f"Аудиофайл для OpenAI API не найден: {path}")
    if not math.isfinite(duration) or duration <= 0:
        raise ValidationError(
            "Длительность аудио для OpenAI API должна быть положительной."
        )


def _raise_if_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and cancel_check():
        raise ProcessingCancelled("Распознавание OpenAI API отменено.")


def _field(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_sequence(value: Any) -> Sequence[Any]:
    return (
        value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else ()
    )


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _default_client_factory(**kwargs: Any) -> Any:
    module = _import_openai()
    return module.OpenAI(**kwargs)


def _import_openai() -> Any:
    try:
        return importlib.import_module("openai")
    except ImportError as exc:
        raise OpenAiApiError("Для backend openai-api требуется пакет openai.") from exc


def _sdk_version() -> str:
    try:
        return importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _map_openai_error(exc: Exception) -> OpenAiApiError:
    module = _import_openai()
    if isinstance(exc, getattr(module, "AuthenticationError", ())):
        return OpenAiAuthenticationError("OpenAI отклонил ключ API.")
    if isinstance(exc, getattr(module, "PermissionDeniedError", ())):
        return OpenAiAuthenticationError(
            "У ключа OpenAI нет доступа к выбранной модели."
        )
    if isinstance(exc, getattr(module, "RateLimitError", ())):
        return OpenAiRateLimitError("OpenAI отклонил запрос из-за лимита или квоты.")
    connection_errors = tuple(
        value
        for value in (
            getattr(module, "APIConnectionError", None),
            getattr(module, "APITimeoutError", None),
        )
        if isinstance(value, type)
    )
    if connection_errors and isinstance(exc, connection_errors):
        return OpenAiConnectionError("Не удалось получить ответ от OpenAI API.")
    if isinstance(exc, getattr(module, "BadRequestError", ())):
        return OpenAiRequestError("OpenAI отклонил аудио или параметры распознавания.")
    if isinstance(exc, getattr(module, "APIStatusError", ())):
        status = getattr(exc, "status_code", "неизвестен")
        return OpenAiRequestError(
            f"OpenAI API завершил запрос с HTTP-статусом {status}."
        )
    if isinstance(exc, OpenAiApiError):
        return exc
    return OpenAiConnectionError("Не удалось выполнить запрос к OpenAI API.")


__all__ = [
    "DEFAULT_OPENAI_TRANSCRIPTION_MODEL",
    "OpenAiApiBackend",
    "OpenAiApiError",
    "OpenAiAuthenticationError",
    "OpenAiAudioPreparationError",
    "OpenAiConnectionError",
    "OpenAiRateLimitError",
    "OpenAiRequestError",
    "OpenAiResponseError",
    "SUPPORTED_MODELS",
]
