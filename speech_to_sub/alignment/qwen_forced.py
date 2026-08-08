"""Самостоятельный Qwen3 ForcedAligner через изолированный persistent worker."""

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
from speech_to_sub.constants import QWEN_ALIGNMENT_MAX_SEGMENT_SECONDS
from speech_to_sub.exceptions import AsrModelError, ProcessingCancelled, ValidationError
from speech_to_sub.models import (
    ProcessingSettings,
    RuntimeSignature,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)

WORKER_MODULE = "speech_to_sub.workers.qwen_aligner_worker"
DEFAULT_WORKER_RELATIVE_PATH = Path("resources/runtimes/qwen/Scripts/python.exe")
MAX_ALIGNMENT_SEGMENT_SECONDS = float(QWEN_ALIGNMENT_MAX_SEGMENT_SECONDS)

WorkerFactory = Callable[[Path, str], PersistentNdjsonWorker]


@dataclass(frozen=True)
class _PreparedRequest:
    model_path: Path
    worker_python: Path
    runtime: RuntimeSignature


class QwenForcedAlignerAdapter:
    """Выравнивает готовый transcript независимо от ASR backend-а."""

    aligner_id = "qwen3-forced-aligner"
    requires_exclusive_runtime = True

    def __init__(self, worker_factory: WorkerFactory = PersistentNdjsonWorker) -> None:
        self._worker_factory = worker_factory
        self._client: PersistentNdjsonWorker | None = None
        self._client_python: Path | None = None
        self._load_key: tuple[str, str, str, bool] | None = None
        self._runtime: RuntimeSignature | None = None
        self._lock = threading.RLock()

    def preflight(self, settings: ProcessingSettings) -> Path:
        """Проверяет локальные веса и Python worker-а без загрузки модели."""
        model_path = _model_path(settings)
        worker_python = _worker_python_path(settings)
        if not worker_python.is_file():
            raise ValidationError(
                "Python изолированного Qwen runtime не найден: "
                f"{worker_python}. Укажите ASR_ALIGNER_WORKER_PYTHON или "
                "aligner_worker_python_path."
            )
        return model_path

    def expected_runtime_signature(self, settings: ProcessingSettings) -> RuntimeSignature:
        """Получает доступный device/precision из отдельного aligner worker-а."""
        with self._lock:
            return self._prepare(settings).runtime

    def runtime_signature(
        self,
        settings: ProcessingSettings,
        transcript: Transcript | None = None,
    ) -> RuntimeSignature | None:
        """Читает фактический alignment runtime из результата либо load key."""
        if transcript is not None:
            raw = transcript.metadata.get("alignment_runtime")
            return _parse_runtime(raw, required=False)
        with self._lock:
            if self._runtime is None or self._load_key != _settings_key(settings):
                return None
            return dict(self._runtime)  # type: ignore[return-value]

    def align(
        self,
        audio_path: Path,
        transcript: Transcript,
        settings: ProcessingSettings,
        duration: float,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> Transcript:
        """Запускает ForcedAligner над аудио и текстом ASR-сегментов."""
        resolved_audio = audio_path.expanduser().resolve()
        _validate_request(resolved_audio, transcript, duration)
        _raise_if_cancelled(cancel_check)
        if progress_callback:
            progress_callback(0)
        segments = _serialize_segments(transcript, duration, cancel_check)
        _raise_if_cancelled(cancel_check)
        with self._lock:
            prepared = self._prepare(settings, cancel_check=cancel_check)
            if progress_callback:
                progress_callback(5)
            try:
                response = self._request_alignment(
                    resolved_audio,
                    prepared,
                    segments,
                    progress_callback,
                    cancel_check,
                )
            except ExternalWorkerError as exc:
                response, prepared = self._retry_on_cpu(
                    exc,
                    resolved_audio,
                    settings,
                    prepared,
                    segments,
                    progress_callback,
                    cancel_check,
                )
            runtime = _parse_runtime(response.get("runtime"), required=True)
            assert runtime is not None
            self._runtime = runtime
            self._load_key = _settings_key(settings)
            aligned = _normalize_alignment(
                response,
                transcript,
                segments,
                prepared,
                runtime,
                duration,
            )
        _raise_if_cancelled(cancel_check)
        if progress_callback:
            progress_callback(100)
        return aligned

    def unload(self) -> None:
        """Завершает отдельный worker и освобождает ForcedAligner."""
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
        model_path = self.preflight(settings)
        worker_python = _worker_python_path(settings)
        result = self._client_for(worker_python).request(
            "preflight",
            {"model_path": str(model_path), "device": device_override or settings.device},
            cancel_check=cancel_check,
        )
        runtime = _parse_runtime(result, required=True)
        assert runtime is not None
        return _PreparedRequest(model_path, worker_python, runtime)

    def _request_alignment(
        self,
        audio_path: Path,
        prepared: _PreparedRequest,
        segments: list[dict[str, Any]],
        progress_callback: ProgressCallback | None,
        cancel_check: CancelCheck | None,
    ) -> dict[str, Any]:
        def emit_worker_progress(value: int) -> None:
            if progress_callback:
                progress_callback(5 + round(max(0, min(100, value)) * 0.9))

        return self._client_for(prepared.worker_python).request(
            "align",
            {
                "audio_path": str(audio_path),
                "model_path": str(prepared.model_path),
                "segments": segments,
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
        segments: list[dict[str, Any]],
        progress_callback: ProgressCallback | None,
        cancel_check: CancelCheck | None,
    ) -> tuple[dict[str, Any], _PreparedRequest]:
        if not settings.allow_cpu_fallback or prepared.runtime["device"] != "cuda":
            raise AsrModelError(
                f"Qwen3 ForcedAligner worker завершился ошибкой: {original_error}"
            ) from original_error
        self._shutdown_client()
        try:
            cpu_prepared = self._prepare(
                settings,
                device_override="cpu",
                cancel_check=cancel_check,
            )
            response = self._request_alignment(
                audio_path,
                cpu_prepared,
                segments,
                progress_callback,
                cancel_check,
            )
            return response, cpu_prepared
        except ExternalWorkerError as cpu_error:
            self._shutdown_client()
            raise AsrModelError(
                "Qwen3 ForcedAligner завершился ошибкой на CUDA и при повторе на CPU."
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


def _serialize_segments(
    transcript: Transcript,
    duration: float,
    cancel_check: CancelCheck | None = None,
) -> list[dict[str, Any]]:
    raw_languages = transcript.metadata.get("segment_languages")
    languages = (
        raw_languages
        if isinstance(raw_languages, Sequence) and not isinstance(raw_languages, (str, bytes))
        else ()
    )
    result: list[dict[str, Any]] = []
    previous_source_end = 0.0
    for index, segment in enumerate(transcript.segments):
        _raise_if_cancelled(cancel_check)
        text = " ".join(segment.text.split())
        if not text:
            raise ValidationError(f"ASR-сегмент {index} не содержит текста для выравнивания.")
        source_start = float(segment.start)
        end = min(duration, float(segment.end))
        _validate_segment_bounds(index, source_start, end, previous_source_end, duration)
        start = max(previous_source_end, source_start)
        if end <= start:
            raise ValidationError(f"ASR-сегмент {index} имеет пустой временной интервал.")
        language = _segment_language(transcript.language, languages, index)
        pieces = _split_segment(segment, text, start, end, language, cancel_check)
        result.extend(pieces)
        previous_source_end = end
    if not result:
        raise ValidationError("Qwen3 ForcedAligner не получил ASR-сегментов с текстом.")
    return result


def _split_segment(
    segment: TranscriptSegment,
    text: str,
    start: float,
    end: float,
    language: str,
    cancel_check: CancelCheck | None,
) -> list[dict[str, Any]]:
    if end - start <= MAX_ALIGNMENT_SEGMENT_SECONDS:
        return [_serialized_segment(text, start, end, language)]
    words = _normalized_source_words(segment, start, end, cancel_check)
    if words:
        split_words = _split_at_word_boundaries(words, language, cancel_check)
        if split_words:
            return split_words
    return _split_proportionally(text, start, end, language, cancel_check)


def _normalized_source_words(
    segment: TranscriptSegment,
    start: float,
    end: float,
    cancel_check: CancelCheck | None,
) -> list[TranscriptWord]:
    if not segment.words:
        return []
    result: list[TranscriptWord] = []
    previous_end = start
    for word in segment.words:
        _raise_if_cancelled(cancel_check)
        text = " ".join(word.text.split())
        word_start = float(word.start)
        word_end = float(word.end)
        outside = word_start < start - 0.05 or word_end > end + 0.05
        invalid = not math.isfinite(word_start) or not math.isfinite(word_end) or outside
        if not text or invalid or word_start < previous_end - 0.05:
            return []
        word_start = max(start, previous_end, word_start)
        word_end = min(end, word_end)
        if word_end <= word_start:
            return []
        result.append(TranscriptWord(word_start, word_end, text, word.probability))
        previous_end = word_end
    return result


def _split_at_word_boundaries(
    words: Sequence[TranscriptWord],
    language: str,
    cancel_check: CancelCheck | None,
) -> list[dict[str, Any]]:
    groups: list[list[TranscriptWord]] = []
    current: list[TranscriptWord] = []
    for word in words:
        _raise_if_cancelled(cancel_check)
        if word.end - word.start > MAX_ALIGNMENT_SEGMENT_SECONDS:
            return []
        if (
            current
            and word.end - current[0].start > MAX_ALIGNMENT_SEGMENT_SECONDS
        ):
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)
    return [
        _serialized_segment(
            " ".join(word.text for word in group),
            group[0].start,
            group[-1].end,
            language,
        )
        for group in groups
    ]


def _split_proportionally(
    text: str,
    start: float,
    end: float,
    language: str,
    cancel_check: CancelCheck | None,
) -> list[dict[str, Any]]:
    part_count = math.ceil((end - start) / MAX_ALIGNMENT_SEGMENT_SECONDS)
    text_parts = _proportional_text_parts(text, part_count)
    span = end - start
    result: list[dict[str, Any]] = []
    for index, text_part in enumerate(text_parts):
        _raise_if_cancelled(cancel_check)
        part_start = start + span * index / part_count
        part_end = start + span * (index + 1) / part_count
        result.append(_serialized_segment(text_part, part_start, part_end, language))
    return result


def _proportional_text_parts(text: str, part_count: int) -> list[str]:
    words = text.split()
    if len(words) >= part_count:
        return [
            " ".join(
                words[
                    index * len(words) // part_count : (index + 1)
                    * len(words)
                    // part_count
                ]
            )
            for index in range(part_count)
        ]
    compact = "".join(words)
    if len(compact) < part_count:
        raise ValidationError(
            "ASR-сегмент слишком длинный для ForcedAligner и содержит недостаточно текста "
            "для детерминированного разбиения."
        )
    return [
        compact[
            index * len(compact) // part_count : (index + 1)
            * len(compact)
            // part_count
        ]
        for index in range(part_count)
    ]


def _serialized_segment(
    text: str,
    start: float,
    end: float,
    language: str,
) -> dict[str, Any]:
    return {"text": text, "start": start, "end": end, "language": language}


def _validate_segment_bounds(
    index: int,
    start: float,
    end: float,
    previous_end: float,
    duration: float,
) -> None:
    invalid_order = start < 0 or start < previous_end - 0.05 or end > duration + 0.05
    if not math.isfinite(start) or not math.isfinite(end) or invalid_order:
        raise ValidationError(f"ASR-сегмент {index} имеет некорректные границы.")
    if end <= start:
        raise ValidationError(f"ASR-сегмент {index} имеет пустой временной интервал.")


def _segment_language(transcript_language: str, values: Sequence[Any], index: int) -> str:
    raw = values[index] if index < len(values) else transcript_language
    normalized = str(raw or "").strip().casefold()
    aliases = {"english": "en", "russian": "ru"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"en", "ru"}:
        raise ValidationError(
            f"Для ASR-сегмента {index} ForcedAligner требует определённый язык en или ru."
        )
    return normalized


def _normalize_alignment(
    response: Mapping[str, Any],
    transcript: Transcript,
    prepared_segments: Sequence[Mapping[str, Any]],
    prepared: _PreparedRequest,
    runtime: RuntimeSignature,
    duration: float,
) -> Transcript:
    raw_segments = response.get("segments")
    if (
        not isinstance(raw_segments, Sequence)
        or isinstance(raw_segments, (str, bytes))
        or len(raw_segments) != len(prepared_segments)
    ):
        raise AsrModelError("Qwen3 ForcedAligner вернул неполный набор ASR-сегментов.")
    segments: list[TranscriptSegment] = []
    for index, (source, raw) in enumerate(zip(prepared_segments, raw_segments)):
        if not isinstance(raw, Mapping):
            raise AsrModelError(f"Qwen3 ForcedAligner вернул некорректный сегмент {index}.")
        source_segment = TranscriptSegment(
            float(source["start"]),
            float(source["end"]),
            str(source["text"]),
            segment_id=index,
        )
        words = _normalize_words(raw.get("words"), source_segment, duration)
        if not words:
            raise AsrModelError(f"Qwen3 ForcedAligner не вернул слова для сегмента {index}.")
        segments.append(
            TranscriptSegment(
                words[0].start,
                words[-1].end,
                source_segment.text,
                tuple(words),
                source_segment.segment_id,
            )
        )
    metadata = dict(transcript.metadata)
    metadata.update(
        {
            "aligner": QwenForcedAlignerAdapter.aligner_id,
            "aligner_model": str(prepared.model_path),
            "alignment_runtime": dict(runtime),
            "alignment_segment_count": len(segments),
            "alignment_max_segment_seconds": MAX_ALIGNMENT_SEGMENT_SECONDS,
            "word_timestamps": True,
            "isolated_alignment_worker": True,
            "segment_languages": tuple(
                str(segment["language"]) for segment in prepared_segments
            ),
        }
    )
    return Transcript(
        transcript.text,
        transcript.language,
        transcript.duration,
        tuple(segments),
        transcript.model,
        transcript.device,
        transcript.quantized,
        metadata,
    )


def _normalize_words(value: Any, segment: TranscriptSegment, duration: float) -> list[TranscriptWord]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    words: list[TranscriptWord] = []
    previous_end = max(0.0, segment.start)
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        text = " ".join(str(raw.get("text") or "").split())
        try:
            start = max(previous_end, segment.start, float(raw.get("start", 0.0)))
            end = min(duration, segment.end, float(raw.get("end", 0.0)))
        except (TypeError, ValueError):
            continue
        if not text or not math.isfinite(start) or not math.isfinite(end):
            continue
        if end <= start:
            end = min(segment.end, duration, start + 0.001)
        if end <= start:
            continue
        words.append(TranscriptWord(start, end, text))
        previous_end = end
    return words


def _parse_runtime(value: Any, *, required: bool) -> RuntimeSignature | None:
    if not isinstance(value, Mapping):
        if required:
            raise AsrModelError("ForcedAligner worker не вернул описание runtime.")
        return None
    device = str(value.get("device") or "").casefold()
    backend = str(value.get("backend") or "")
    if device not in {"cpu", "cuda"} or backend != QwenForcedAlignerAdapter.aligner_id:
        if required:
            raise AsrModelError("ForcedAligner worker вернул неизвестный runtime.")
        return None
    return {
        "backend": backend,
        "engine_version": str(value.get("engine_version") or "unavailable"),
        "device": cast(Any, device),
        "compute_type": str(value.get("compute_type") or "unknown"),
        "quantized": bool(value.get("quantized", False)),
    }


def _model_path(settings: ProcessingSettings) -> Path:
    if settings.aligner_model_path is None:
        raise ValidationError("Для Qwen3 ForcedAligner требуется aligner_model_path.")
    return require_local_checkpoint(settings.aligner_model_path, "Qwen3 ForcedAligner")


def _worker_python_path(settings: ProcessingSettings) -> Path:
    configured = settings.aligner_worker_python_path
    raw_path = (
        configured
        or os.environ.get("ASR_ALIGNER_WORKER_PYTHON")
        or os.environ.get("QWEN_ASR_PYTHON")
    )
    if raw_path:
        return Path(str(raw_path)).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / DEFAULT_WORKER_RELATIVE_PATH).resolve()


def _settings_key(settings: ProcessingSettings) -> tuple[str, str, str, bool]:
    model_path = settings.aligner_model_path or Path()
    return (
        str(model_path.expanduser().resolve()).casefold(),
        str(_worker_python_path(settings)).casefold(),
        settings.device.strip().casefold(),
        settings.allow_cpu_fallback,
    )


def _validate_request(audio_path: Path, transcript: Transcript, duration: float) -> None:
    if not audio_path.is_file():
        raise ValidationError(f"Аудиофайл для ForcedAligner не найден: {audio_path}")
    if not math.isfinite(duration) or duration <= 0:
        raise ValidationError("Длительность аудио для ForcedAligner должна быть положительной.")
    if not transcript.text.strip():
        raise ValidationError("ForcedAligner не получил текста распознавания.")


def _raise_if_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and cancel_check():
        raise ProcessingCancelled("Выравнивание Qwen3 ForcedAligner отменено.")


__all__ = ["MAX_ALIGNMENT_SEGMENT_SECONDS", "QwenForcedAlignerAdapter"]
