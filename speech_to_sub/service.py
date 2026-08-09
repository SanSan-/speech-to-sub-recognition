from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout as FileLockTimeout

from speech_to_sub.alignment.base import AlignmentAdapter
from speech_to_sub.alignment.registry import aligner_names, get_aligner
from speech_to_sub.asr.base import AsrBackend
from speech_to_sub.asr.registry import activate_backend, backend_names, get_backend
from speech_to_sub.constants import (
    MAX_LINE_LENGTH_GAP,
    SIDECAR_SCHEMA_VERSION,
    WORK_DIR,
)
from speech_to_sub.exceptions import ProcessingCancelled, SpeechToSubError, ValidationError
from speech_to_sub.media.ffmpeg import (
    get_media_duration,
    normalize_audio,
    probe_media,
    select_audio_stream,
)
from speech_to_sub.models import (
    AudioStreamInfo,
    FileResult,
    MediaProbe,
    OutputPaths,
    ProcessingSettings,
    RuntimeSignature,
    Transcript,
)
from speech_to_sub.subtitles.builder import build_cues_with_diagnostics, render_srt
from speech_to_sub.subtitles.validator import validate_srt_text
from speech_to_sub.utils.cache import (
    build_layout_fingerprint,
    build_recognition_fingerprint,
    build_source_fingerprint,
    load_sidecar,
    sidecar_recognition_matches,
    sidecar_matches,
    write_sidecar,
)
from speech_to_sub.utils.env_utils import get_command_path
from speech_to_sub.utils.io_utils import (
    atomic_copy_file,
    atomic_write_text_utf8,
    discover_media,
    read_text_utf8,
)

logger = logging.getLogger(__name__)

WORKSPACE_LOCK_NAME = ".active.lock"
DEFAULT_WORKSPACE_TTL_SECONDS = 60 * 60
SUBTITLE_LAYOUT_SIDECAR_KEY = "subtitle_layout"
_LAYOUT_DIAGNOSTIC_LABELS = {
    "reconciled_segments": "восстановлены текст и пунктуация сегментов",
    "alignment_text_segments": "обнаружены расхождения текста и словных меток",
    "synthetic_timing_segments": "синтезированы временные метки сегментов",
    "retimed_leading_islands": "перенесены фрагменты с аномальными начальными метками",
    "adjusted_boundaries": "скорректированы временные границы реплик",
    "max_boundary_drift_ms": "максимальный сдвиг границы, мс",
}

EventCallback = Callable[[dict[str, Any]], None]
LogCallback = Callable[[str], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class _FileProcessingContext:
    path: Path
    index: int
    total: int
    settings: ProcessingSettings
    outputs: OutputPaths
    temporary_audio: Path
    ffmpeg_path: str
    ffprobe_path: str
    emit_event: EventCallback
    log: LogCallback
    cancel_check: CancelCheck | None


@dataclass(frozen=True, slots=True)
class _CachedRecognition:
    """Проверенный тяжёлый результат, пригодный для новой SRT-разметки."""

    transcript: Transcript
    normalized_duration: float
    recognition_settings: dict[str, Any]


def get_preflight_status(settings_values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Проверяет команды и checkpoint без загрузки весов."""
    settings = ProcessingSettings.from_mapping(settings_values)
    ffmpeg = get_command_path("FFMPEG_PATH", "ffmpeg")
    ffprobe = get_command_path("FFPROBE_PATH", "ffprobe")
    ffmpeg_found = _command_exists(ffmpeg)
    ffprobe_found = _command_exists(ffprobe)
    model_error = None
    aligner_error = None
    aligner_path: Path | None = None
    try:
        selected_backend = get_backend(settings.backend)
        model_path = selected_backend.preflight(settings)
        model_found = True
    except (SpeechToSubError, OSError, ValueError) as exc:
        model_path = settings.model_path
        model_found = False
        model_error = str(exc)
    try:
        aligner_path = get_aligner(settings.aligner).preflight(settings)
        aligner_found = True
    except (SpeechToSubError, OSError, ValueError) as exc:
        aligner_found = False
        aligner_error = str(exc)
    return {
        "status": (
            "ok"
            if ffmpeg_found and ffprobe_found and model_found and aligner_found
            else "degraded"
        ),
        "ffmpeg": {"path": ffmpeg, "available": ffmpeg_found},
        "ffprobe": {"path": ffprobe, "available": ffprobe_found},
        "backend": {
            "id": settings.backend,
            "available": model_found,
            "error": model_error,
        },
        "model": {
            "path": str(model_path),
            "available": model_found,
            "error": model_error,
        },
        "aligner": {
            "id": settings.aligner,
            "path": str(aligner_path) if aligner_path else None,
            "available": aligner_found,
            "error": aligner_error,
        },
    }


def build_items(
    paths: Iterable[str | Path],
    settings_values: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Строит карточки web UI без загрузки ASR-модели."""
    settings = ProcessingSettings.from_mapping(settings_values)
    media_paths = discover_media((Path(value) for value in paths), settings.recursive)
    common_root = _common_input_root(media_paths)
    output_map = _build_output_map(media_paths, settings, common_root)
    ffprobe_path = get_command_path("FFPROBE_PATH", "ffprobe")
    items: list[dict[str, Any]] = []
    for path in media_paths:
        outputs = output_map[path]
        item: dict[str, Any] = {
            "path": str(path),
            "name": path.name,
            "format": path.suffix.casefold().lstrip("."),
            "state": "idle",
            "stage": "Ожидание",
            "progress": 0,
            "srt_output": str(outputs.srt_path),
            "sidecar_output": str(outputs.sidecar_path),
            "audio_output": str(outputs.normalized_audio_path) if settings.keep_audio else None,
            "cached": False,
            "skipped": False,
            "error": None,
        }
        try:
            source = build_source_fingerprint(path)
            item["source_fingerprint"] = source
            probe = probe_media(path, ffprobe_path=ffprobe_path)
            stream, warning = select_audio_stream(
                probe,
                requested_ordinal=settings.audio_stream_index,
                preferred_language=settings.audio_language,
            )
            item["probe"] = probe.to_dict()
            item["selected_stream"] = stream.to_dict()
            item["warning"] = warning
            fingerprints = _cache_fingerprints_for_lookup(
                outputs,
                settings,
                stream.ordinal,
                backend=get_backend(settings.backend),
            )
            cache_valid = _is_valid_cache_candidate(outputs, source, fingerprints)
            if cache_valid and settings.keep_audio and not _has_saved_audio(outputs):
                item.update(stage="Требуется сохранить аудио")
            elif cache_valid:
                item.update(state="cached", stage="Кеш", progress=100, cached=True)
            elif outputs.srt_path.exists() and not settings.force:
                item.update(state="skipped", stage="Существующий SRT", progress=100, skipped=True)
            elif fingerprints and _load_cached_recognition(
                outputs,
                source,
                fingerprints[0],
            ):
                item.update(stage="Пересборка SRT из кеша распознавания")
        except Exception as exc:
            item.update(state="error", stage="Ошибка проверки", progress=100, error=str(exc))
        items.append(item)
    return items


def process_paths(
    paths: Iterable[str | Path],
    settings_values: Mapping[str, Any] | None,
    emit_event: EventCallback,
    log: LogCallback,
    *,
    backend: AsrBackend | None = None,
    cancel_check: CancelCheck | None = None,
) -> list[dict[str, Any]]:
    """Последовательно обрабатывает пачку, продолжая работу после ошибки файла."""
    settings = ProcessingSettings.from_mapping(settings_values)
    _validate_settings(settings)
    media_paths = discover_media((Path(value) for value in paths), settings.recursive)
    if not media_paths:
        raise ValidationError("Не найдено поддерживаемых медиафайлов.")
    common_root = _common_input_root(media_paths)
    output_map = _build_output_map(media_paths, settings, common_root)
    ffmpeg_path = get_command_path("FFMPEG_PATH", "ffmpeg")
    ffprobe_path = get_command_path("FFPROBE_PATH", "ffprobe")
    if not _command_exists(ffmpeg_path) or not _command_exists(ffprobe_path):
        raise ValidationError("FFmpeg и ffprobe должны быть доступны до запуска пачки.")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    emit_event({"type": "job", "total": len(media_paths)})
    results: list[FileResult] = []
    active_backend = backend
    with tempfile.TemporaryDirectory(prefix="job-", dir=WORK_DIR) as temporary_value:
        temporary_root = Path(temporary_value)
        workspace_lock = FileLock(str(temporary_root / WORKSPACE_LOCK_NAME), timeout=0)
        with workspace_lock:
            for index, path in enumerate(media_paths, start=1):
                if _is_cancelled(cancel_check):
                    results.extend(
                        _cancel_remaining(
                            media_paths[index - 1 :],
                            first_index=index,
                            total=len(media_paths),
                            emit_event=emit_event,
                            log=log,
                        )
                    )
                    break
                result, active_backend = _process_one(
                    path,
                    index=index,
                    total=len(media_paths),
                    settings=settings,
                    outputs=output_map[path],
                    temporary_root=temporary_root,
                    ffmpeg_path=ffmpeg_path,
                    ffprobe_path=ffprobe_path,
                    backend=active_backend,
                    emit_event=emit_event,
                    log=log,
                    cancel_check=cancel_check,
                )
                results.append(result)
    return [result.to_dict() for result in results]


def cleanup_stale_workspaces(
    work_dir: str | Path = WORK_DIR,
    *,
    ttl_seconds: float = DEFAULT_WORKSPACE_TTL_SECONDS,
    now: float | None = None,
) -> list[Path]:
    """Удаляет только устаревшие управляемые job-каталоги без активной блокировки."""
    if ttl_seconds < 0:
        raise ValueError("TTL рабочих каталогов не может быть отрицательным.")
    root = Path(work_dir).expanduser()
    if not root.exists():
        return []
    if not root.is_dir():
        raise ValueError(f"WORK_DIR не является каталогом: {root}")
    resolved_root = root.resolve(strict=True)
    current_time = time.time() if now is None else now
    removed: list[Path] = []
    for candidate in tuple(resolved_root.iterdir()):
        target = _stale_workspace_target(
            candidate,
            resolved_root,
            ttl_seconds=ttl_seconds,
            now=current_time,
        )
        if target is None:
            continue
        lock = FileLock(str(target / WORKSPACE_LOCK_NAME), timeout=0)
        try:
            lock.acquire()
        except FileLockTimeout:
            continue
        else:
            lock.release()
        try:
            shutil.rmtree(target)
        except OSError as exc:
            logger.warning(
                "Не удалось удалить устаревший рабочий каталог %s: %s",
                target,
                exc,
            )
        else:
            removed.append(target)
    return removed


def _stale_workspace_target(
    candidate: Path,
    resolved_root: Path,
    *,
    ttl_seconds: float,
    now: float,
) -> Path | None:
    """Проверяет имя, границы, маркер и возраст рабочего каталога."""
    if not candidate.name.startswith("job-") or candidate.is_symlink():
        return None
    try:
        target = candidate.resolve(strict=True)
    except OSError:
        return None
    if target.parent != resolved_root or not target.is_dir():
        return None
    lock_path = target / WORKSPACE_LOCK_NAME
    if not lock_path.is_file() or lock_path.is_symlink():
        return None
    try:
        newest_mtime = max(target.stat().st_mtime, lock_path.stat().st_mtime)
    except OSError:
        return None
    return target if now - newest_mtime > ttl_seconds else None


def build_output_paths(
    input_path: Path,
    settings: ProcessingSettings,
    common_root: Path | None = None,
    *,
    include_source_extension: bool = False,
    collision_suffix: str | None = None,
) -> OutputPaths:
    """Формирует пути SRT, sidecar и сохраняемого FLAC."""
    language = settings.language.casefold() or "auto"
    if settings.output_dir:
        output_root = settings.output_dir.expanduser().resolve()
        relative_parent = Path()
        if common_root:
            try:
                relative_parent = input_path.parent.relative_to(common_root)
            except ValueError:
                relative_parent = Path()
        parent = output_root / relative_parent
    else:
        parent = input_path.parent
    stem = input_path.name if include_source_extension else input_path.stem
    if collision_suffix:
        stem = f"{stem}.{collision_suffix}"
    return OutputPaths(
        srt_path=parent / f"{stem}.{language}.srt",
        sidecar_path=parent / f"{stem}.{language}.asr.json",
        normalized_audio_path=parent / f"{stem}.{language}.asr.flac",
    )


def _process_one(
    path: Path,
    *,
    index: int,
    total: int,
    settings: ProcessingSettings,
    outputs: OutputPaths,
    temporary_root: Path,
    ffmpeg_path: str,
    ffprobe_path: str,
    backend: AsrBackend | None,
    emit_event: EventCallback,
    log: LogCallback,
    cancel_check: CancelCheck | None,
) -> tuple[FileResult, AsrBackend | None]:
    """Не допускает одновременную публикацию одного набора артефактов."""
    lock = FileLock(str(_output_lock_path(outputs)), timeout=0)
    try:
        with lock:
            return _process_one_locked(
                path,
                index=index,
                total=total,
                settings=settings,
                outputs=outputs,
                temporary_root=temporary_root,
                ffmpeg_path=ffmpeg_path,
                ffprobe_path=ffprobe_path,
                backend=backend,
                emit_event=emit_event,
                log=log,
                cancel_check=cancel_check,
            )
    except FileLockTimeout:
        message = "Этот выходной файл уже обрабатывается другим процессом."
        _emit_file(
            emit_event,
            path,
            index,
            total,
            "error",
            "Выход занят",
            100,
            error=message,
        )
        _write_log(log, f"Файл {index}/{total}: {path.name} — {message}")
        return FileResult(input_path=path, state="error", error=message), backend


def _process_one_locked(
    path: Path,
    *,
    index: int,
    total: int,
    settings: ProcessingSettings,
    outputs: OutputPaths,
    temporary_root: Path,
    ffmpeg_path: str,
    ffprobe_path: str,
    backend: AsrBackend | None,
    emit_event: EventCallback,
    log: LogCallback,
    cancel_check: CancelCheck | None,
) -> tuple[FileResult, AsrBackend | None]:
    started_at = datetime.now(timezone.utc).isoformat()
    _emit_file(emit_event, path, index, total, "probing", "Проверка потоков", 2)
    _write_log(log, f"Файл {index}/{total}: {path.name} — проверка потоков.")
    try:
        _raise_if_cancelled(cancel_check)
        probe = probe_media(path, ffprobe_path=ffprobe_path)
        _raise_if_cancelled(cancel_check)
        stream, warning = select_audio_stream(
            probe,
            requested_ordinal=settings.audio_stream_index,
            preferred_language=settings.audio_language,
        )
        if warning:
            _write_log(log, f"{path.name}: {warning}")
        source = build_source_fingerprint(path)
        temporary_audio = temporary_root / f"{index:04d}-{path.stem}.asr.flac"
        context = _FileProcessingContext(
            path=path,
            index=index,
            total=total,
            settings=settings,
            outputs=outputs,
            temporary_audio=temporary_audio,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            emit_event=emit_event,
            log=log,
            cancel_check=cancel_check,
        )
        existing_result, cached_recognition = _reuse_existing_result(
            context,
            probe=probe,
            stream=stream,
            source=source,
            backend=backend,
        )
        if existing_result is not None:
            return existing_result, backend
        if cached_recognition is not None:
            return (
                _rebuild_srt_from_cached_recognition(
                    context,
                    cached_recognition,
                    probe=probe,
                    stream=stream,
                    source=source,
                    warning=warning,
                    started_at=started_at,
                ),
                backend,
            )

        _emit_file(emit_event, path, index, total, "extracting", "Нормализация аудио", 10)
        _raise_if_cancelled(cancel_check)
        normalize_audio(
            path,
            temporary_audio,
            stream,
            ffmpeg_path=ffmpeg_path,
            overwrite=True,
        )
        normalized_duration = get_media_duration(
            temporary_audio,
            ffprobe_path=ffprobe_path,
        )
        _raise_if_cancelled(cancel_check)
        _emit_file(emit_event, path, index, total, "transcribing", "Локальная ASR", 20)
        _write_log(log, f"Файл {index}/{total}: {path.name} — локальное распознавание.")
        transcript, runtime, aligner, backend = _transcribe_with_alignment(
            context,
            normalized_duration,
            backend=backend,
        )
        _emit_file(emit_event, path, index, total, "writing", "Формирование SRT", 88)
        srt_text, subtitle_layout = _build_srt_artifacts(
            transcript,
            settings,
            normalized_duration,
            path=path,
            log=log,
        )
        finished_at = datetime.now(timezone.utc).isoformat()
        aligner_runtime = _loaded_aligner_runtime(aligner, settings, transcript)
        if settings.aligner != "none" and aligner_runtime is None:
            aligner_runtime = _expected_aligner_runtime(aligner, settings)
        recognition_settings = build_recognition_fingerprint(
            settings,
            stream.ordinal,
            runtime=runtime,
            aligner_runtime=aligner_runtime,
        )
        sidecar = _build_sidecar_payload(
            started_at=started_at,
            finished_at=finished_at,
            source=source,
            recognition_settings=recognition_settings,
            layout_settings=build_layout_fingerprint(settings),
            probe=probe,
            stream=stream,
            warning=warning,
            normalized_duration=normalized_duration,
            transcript=transcript,
            subtitle_layout=subtitle_layout,
            outputs=outputs,
            keep_audio=settings.keep_audio,
        )
        _publish_artifacts(
            outputs,
            srt_text=srt_text,
            sidecar=sidecar,
            audio_source=temporary_audio if settings.keep_audio else None,
        )
        _emit_file(
            emit_event,
            path,
            index,
            total,
            "done",
            "Готово",
            100,
            outputs=outputs,
            audio_output=(outputs.normalized_audio_path if settings.keep_audio else None),
        )
        _write_log(log, f"Файл {index}/{total}: {path.name} — SRT готов.")
        return (
            FileResult(
                input_path=path,
                state="done",
                srt_path=outputs.srt_path,
                sidecar_path=outputs.sidecar_path,
                audio_path=outputs.normalized_audio_path if settings.keep_audio else None,
                probe=probe,
            ),
            backend,
        )
    except ProcessingCancelled as exc:
        message = str(exc)
        _emit_file(
            emit_event,
            path,
            index,
            total,
            "cancelled",
            "Отменено",
            100,
            error=message,
        )
        _write_log(log, f"Файл {index}/{total}: {path.name} — отменено.")
        return FileResult(input_path=path, state="cancelled", error=message), backend
    except Exception as exc:
        message = str(exc) or exc.__class__.__name__
        _emit_file(
            emit_event,
            path,
            index,
            total,
            "error",
            "Ошибка",
            100,
            error=message,
        )
        _write_log(log, f"Файл {index}/{total}: {path.name} — ошибка: {message}")
        return FileResult(input_path=path, state="error", error=message), backend


def _reuse_existing_result(
    context: _FileProcessingContext,
    *,
    probe: MediaProbe,
    stream: AudioStreamInfo,
    source: dict[str, Any],
    backend: AsrBackend | None,
) -> tuple[FileResult | None, _CachedRecognition | None]:
    path = context.path
    settings = context.settings
    outputs = context.outputs
    fingerprints = _cache_fingerprints_for_lookup(
        outputs,
        settings,
        stream.ordinal,
        backend=backend if backend is not None else get_backend(settings.backend),
    )
    cache_valid = _is_valid_cache_candidate(outputs, source, fingerprints)
    if not settings.force and cache_valid:
        if settings.keep_audio and not _has_saved_audio(outputs):
            _save_cached_audio(
                context,
                stream=stream,
            )
        _emit_file(
            context.emit_event,
            path,
            context.index,
            context.total,
            "cached",
            "Готовый кеш",
            100,
            outputs=outputs,
            audio_output=outputs.normalized_audio_path if settings.keep_audio else None,
        )
        _write_log(
            context.log,
            f"Файл {context.index}/{context.total}: {path.name} — готовый кеш.",
        )
        return (
            FileResult(
                input_path=path,
                state="cached",
                srt_path=outputs.srt_path,
                sidecar_path=outputs.sidecar_path,
                audio_path=outputs.normalized_audio_path if settings.keep_audio else None,
                cached=True,
                probe=probe,
            ),
            None,
        )
    if outputs.srt_path.exists() and not settings.force:
        _emit_file(
            context.emit_event,
            path,
            context.index,
            context.total,
            "skipped",
            "SRT уже существует",
            100,
            outputs=outputs,
        )
        _write_log(
            context.log,
            f"Файл {context.index}/{context.total}: {path.name} — SRT безопасно пропущен.",
        )
        return (
            FileResult(
                input_path=path,
                state="skipped",
                srt_path=outputs.srt_path,
                sidecar_path=outputs.sidecar_path if outputs.sidecar_path.exists() else None,
                skipped=True,
                probe=probe,
            ),
            None,
        )
    cached_recognition = (
        _load_cached_recognition(outputs, source, fingerprints[0])
        if fingerprints is not None
        else None
    )
    return None, cached_recognition


def _rebuild_srt_from_cached_recognition(
    context: _FileProcessingContext,
    cached: _CachedRecognition,
    *,
    probe: MediaProbe,
    stream: AudioStreamInfo,
    source: dict[str, Any],
    warning: str | None,
    started_at: str,
) -> FileResult:
    """Пересобирает только SRT и sidecar из проверенного тяжёлого кеша."""
    normalized_duration, audio_source = _prepare_reused_audio(
        context,
        stream,
        cached.normalized_duration,
    )
    _raise_if_cancelled(context.cancel_check)
    _emit_file(
        context.emit_event,
        context.path,
        context.index,
        context.total,
        "writing",
        "Пересборка SRT из кеша распознавания",
        88,
    )
    srt_text, subtitle_layout = _build_srt_artifacts(
        cached.transcript,
        context.settings,
        normalized_duration,
        path=context.path,
        log=context.log,
    )
    sidecar = _build_reused_sidecar(
        context,
        cached,
        started_at=started_at,
        source=source,
        probe=probe,
        stream=stream,
        warning=warning,
        normalized_duration=normalized_duration,
        subtitle_layout=subtitle_layout,
    )
    _publish_artifacts(
        context.outputs,
        srt_text=srt_text,
        sidecar=sidecar,
        audio_source=audio_source,
    )
    return _complete_reused_srt(context, probe)


def _build_reused_sidecar(
    context: _FileProcessingContext,
    cached: _CachedRecognition,
    *,
    started_at: str,
    source: Mapping[str, Any],
    probe: MediaProbe,
    stream: AudioStreamInfo,
    warning: str | None,
    normalized_duration: float,
    subtitle_layout: Mapping[str, int],
) -> dict[str, Any]:
    finished_at = datetime.now(timezone.utc).isoformat()
    return _build_sidecar_payload(
        started_at=started_at,
        finished_at=finished_at,
        source=source,
        recognition_settings=cached.recognition_settings,
        layout_settings=build_layout_fingerprint(context.settings),
        probe=probe,
        stream=stream,
        warning=warning,
        normalized_duration=normalized_duration,
        transcript=cached.transcript,
        subtitle_layout=subtitle_layout,
        outputs=context.outputs,
        keep_audio=context.settings.keep_audio,
    )


def _complete_reused_srt(
    context: _FileProcessingContext,
    probe: MediaProbe,
) -> FileResult:
    audio_output = (
        context.outputs.normalized_audio_path if context.settings.keep_audio else None
    )
    _emit_file(
        context.emit_event,
        context.path,
        context.index,
        context.total,
        "done",
        "Готово из кеша распознавания",
        100,
        outputs=context.outputs,
        audio_output=audio_output,
    )
    _write_log(
        context.log,
        f"Файл {context.index}/{context.total}: {context.path.name} — "
        "SRT пересобран из кеша распознавания.",
    )
    return FileResult(
        input_path=context.path,
        state="done",
        srt_path=context.outputs.srt_path,
        sidecar_path=context.outputs.sidecar_path,
        audio_path=audio_output,
        probe=probe,
    )


def _prepare_reused_audio(
    context: _FileProcessingContext,
    stream: AudioStreamInfo,
    cached_duration: float,
) -> tuple[float, Path | None]:
    if not context.settings.keep_audio or _has_saved_audio(context.outputs):
        return cached_duration, None
    _emit_file(
        context.emit_event,
        context.path,
        context.index,
        context.total,
        "extracting",
        "Сохранение нормализованного аудио",
        25,
    )
    normalize_audio(
        context.path,
        context.temporary_audio,
        stream,
        ffmpeg_path=context.ffmpeg_path,
        overwrite=True,
    )
    duration = get_media_duration(
        context.temporary_audio,
        ffprobe_path=context.ffprobe_path,
    )
    _raise_if_cancelled(context.cancel_check)
    return duration, context.temporary_audio


def _build_srt_artifacts(
    transcript: Transcript,
    settings: ProcessingSettings,
    normalized_duration: float,
    *,
    path: Path,
    log: LogCallback,
) -> tuple[str, dict[str, int]]:
    build_result = build_cues_with_diagnostics(
        transcript.segments,
        max_chars_per_line=settings.max_chars_per_line,
        line_length_gap=settings.line_length_gap,
        max_cps=settings.max_cps,
        audio_duration=normalized_duration,
    )
    srt_text = render_srt(build_result.cues)
    validate_srt_text(
        srt_text,
        duration=normalized_duration,
        max_chars_per_line=settings.max_chars_per_line,
        line_length_gap=settings.line_length_gap,
        max_cps=settings.max_cps,
    )
    diagnostics = build_result.diagnostics.to_dict()
    _log_subtitle_layout_diagnostics(path, log, diagnostics)
    return srt_text, diagnostics


def _build_sidecar_payload(
    *,
    started_at: str,
    finished_at: str,
    source: Mapping[str, Any],
    recognition_settings: Mapping[str, Any],
    layout_settings: Mapping[str, Any],
    probe: MediaProbe,
    stream: AudioStreamInfo,
    warning: str | None,
    normalized_duration: float,
    transcript: Transcript,
    subtitle_layout: Mapping[str, int],
    outputs: OutputPaths,
    keep_audio: bool,
) -> dict[str, Any]:
    return {
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
        "status": "done",
        "created_at": finished_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "source": dict(source),
        "recognition_settings": dict(recognition_settings),
        "layout_settings": dict(layout_settings),
        "probe": probe.to_dict(),
        "selected_stream": stream.to_dict(),
        "warning": warning,
        "normalized_audio_duration": normalized_duration,
        "transcript": transcript.to_dict(),
        SUBTITLE_LAYOUT_SIDECAR_KEY: dict(subtitle_layout),
        "outputs": {
            "srt": str(outputs.srt_path),
            "sidecar": str(outputs.sidecar_path),
            "audio": str(outputs.normalized_audio_path) if keep_audio else None,
        },
    }


def _save_cached_audio(
    context: _FileProcessingContext,
    *,
    stream: AudioStreamInfo,
) -> None:
    _raise_if_cancelled(context.cancel_check)
    _emit_file(
        context.emit_event,
        context.path,
        context.index,
        context.total,
        "extracting",
        "Сохранение нормализованного аудио",
        25,
    )
    normalize_audio(
        context.path,
        context.temporary_audio,
        stream,
        ffmpeg_path=context.ffmpeg_path,
        overwrite=True,
    )
    normalized_duration = get_media_duration(
        context.temporary_audio,
        ffprobe_path=context.ffprobe_path,
    )
    _raise_if_cancelled(context.cancel_check)
    _record_saved_audio(context.outputs, context.temporary_audio, normalized_duration)


def _transcribe_with_alignment(
    context: _FileProcessingContext,
    normalized_duration: float,
    *,
    backend: AsrBackend | None,
) -> tuple[Transcript, RuntimeSignature, AlignmentAdapter, AsrBackend]:
    settings = context.settings
    if backend is None:
        backend = activate_backend(settings.backend)
    aligner = get_aligner(settings.aligner)
    if aligner.requires_exclusive_runtime:
        aligner.unload()

    def asr_progress(value: int) -> None:
        _raise_if_cancelled(context.cancel_check)
        mapped = 20 + int(max(0, min(100, value)) * 0.65)
        _emit_file(
            context.emit_event,
            context.path,
            context.index,
            context.total,
            "transcribing",
            "Локальная ASR",
            mapped,
        )

    transcript = backend.transcribe(
        context.temporary_audio,
        settings,
        normalized_duration,
        progress_callback=asr_progress,
        cancel_check=context.cancel_check,
    )
    _raise_if_cancelled(context.cancel_check)
    runtime = _loaded_runtime_signature(backend, settings)
    if runtime is None:
        runtime = _runtime_from_transcript(backend, transcript, settings)
    if settings.aligner != "none":
        _emit_file(
            context.emit_event,
            context.path,
            context.index,
            context.total,
            "aligning",
            "Выравнивание слов",
            86,
        )

    def align_progress(value: int) -> None:
        _raise_if_cancelled(context.cancel_check)
        mapped = 86 + int(max(0, min(100, value)) * 0.01)
        _emit_file(
            context.emit_event,
            context.path,
            context.index,
            context.total,
            "aligning",
            "Выравнивание слов",
            mapped,
        )

    if aligner.requires_exclusive_runtime:
        backend.unload()
    transcript = aligner.align(
        context.temporary_audio,
        transcript,
        settings,
        normalized_duration,
        progress_callback=align_progress,
        cancel_check=context.cancel_check,
    )
    _raise_if_cancelled(context.cancel_check)
    _log_alignment_fallback(context, transcript)
    return transcript, runtime, aligner, backend


def _log_alignment_fallback(
    context: _FileProcessingContext,
    transcript: Transcript,
) -> None:
    """Отмечает безопасный возврат к исходным словным меткам faster-whisper."""
    if (
        context.settings.aligner != "qwen3-forced-aligner"
        or transcript.metadata.get("alignment_status") != "fallback"
    ):
        return
    details = transcript.metadata.get("alignment_fallback")
    segment_index = details.get("segment_index") if isinstance(details, Mapping) else None
    segment_label = (
        f"сегмента {segment_index}"
        if isinstance(segment_index, int) and not isinstance(segment_index, bool)
        else "одного из сегментов"
    )
    _write_log(
        context.log,
        f"{context.path.name}: Qwen3 ForcedAligner не вернул слова для {segment_label}; "
        "используются исходные временные метки faster-whisper.",
    )


def _log_subtitle_layout_diagnostics(
    path: Path,
    log: LogCallback,
    diagnostics: Mapping[str, int],
) -> None:
    """Журналирует только выполненные восстановления и обнаруженный сдвиг."""
    events = [
        f"{label}: {diagnostics[key]}"
        for key, label in _LAYOUT_DIAGNOSTIC_LABELS.items()
        if diagnostics.get(key, 0) > 0
    ]
    if events:
        _write_log(log, f"{path.name}: разметка SRT — {'; '.join(events)}.")


def _is_cancelled(cancel_check: CancelCheck | None) -> bool:
    return bool(cancel_check and cancel_check())


def _raise_if_cancelled(cancel_check: CancelCheck | None) -> None:
    if _is_cancelled(cancel_check):
        raise ProcessingCancelled("Обработка отменена пользователем.")


def _cancel_remaining(
    paths: list[Path],
    *,
    first_index: int,
    total: int,
    emit_event: EventCallback,
    log: LogCallback,
) -> list[FileResult]:
    results: list[FileResult] = []
    message = "Обработка отменена пользователем."
    for offset, path in enumerate(paths):
        index = first_index + offset
        _emit_file(
            emit_event,
            path,
            index,
            total,
            "cancelled",
            "Отменено",
            100,
            error=message,
        )
        _write_log(log, f"Файл {index}/{total}: {path.name} — отменено до запуска.")
        results.append(FileResult(input_path=path, state="cancelled", error=message))
    return results


def _is_valid_cache(
    outputs: OutputPaths,
    source: dict[str, Any],
    *,
    recognition_settings: dict[str, Any],
    layout_settings: dict[str, Any],
) -> bool:
    if not outputs.srt_path.is_file():
        return False
    sidecar = load_sidecar(outputs.sidecar_path)
    if not sidecar_matches(
        sidecar,
        source,
        recognition_settings,
        layout_settings,
    ):
        return False
    duration = _sidecar_audio_duration(sidecar)
    if duration is None:
        return False
    try:
        validate_srt_text(
            read_text_utf8(outputs.srt_path),
            duration=duration,
            max_chars_per_line=int(layout_settings["max_chars_per_line"]),
            line_length_gap=int(layout_settings["line_length_gap"]),
            max_cps=float(layout_settings["max_cps"]),
        )
    except Exception:
        return False
    return True


def _is_valid_cache_candidate(
    outputs: OutputPaths,
    source: dict[str, Any],
    fingerprints: tuple[dict[str, Any], dict[str, Any]] | None,
) -> bool:
    """Проверяет найденную пару fingerprints как готовый SRT-кеш."""
    if fingerprints is None:
        return False
    return _is_valid_cache(
        outputs,
        source,
        recognition_settings=fingerprints[0],
        layout_settings=fingerprints[1],
    )


def _cache_fingerprints_for_lookup(
    outputs: OutputPaths,
    settings: ProcessingSettings,
    stream_ordinal: int,
    *,
    backend: AsrBackend | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Строит runtime-aware fingerprint только для существующего кандидата кеша."""
    if not outputs.sidecar_path.is_file():
        return None
    selected_backend = backend if backend is not None else get_backend(settings.backend)
    runtime = _loaded_runtime_signature(selected_backend, settings)
    if runtime is None:
        runtime = _expected_runtime_signature(selected_backend, settings)
    if runtime is None:
        return None
    aligner = get_aligner(settings.aligner)
    aligner_runtime = _expected_aligner_runtime(aligner, settings)
    if settings.aligner != "none" and aligner_runtime is None:
        return None
    return (
        build_recognition_fingerprint(
            settings,
            stream_ordinal,
            runtime=runtime,
            aligner_runtime=aligner_runtime,
        ),
        build_layout_fingerprint(settings),
    )


def _load_cached_recognition(
    outputs: OutputPaths,
    source: dict[str, Any],
    recognition_settings: dict[str, Any],
) -> _CachedRecognition | None:
    sidecar = load_sidecar(outputs.sidecar_path)
    if not sidecar_recognition_matches(sidecar, source, recognition_settings):
        return None
    duration = _sidecar_audio_duration(sidecar)
    transcript_raw = sidecar.get("transcript") if sidecar else None
    if duration is None or not isinstance(transcript_raw, Mapping):
        return None
    try:
        transcript = Transcript.from_mapping(transcript_raw)
    except ValidationError:
        return None
    if abs(transcript.duration - duration) > 0.25:
        return None
    return _CachedRecognition(
        transcript=transcript,
        normalized_duration=duration,
        recognition_settings=recognition_settings,
    )


def _loaded_aligner_runtime(
    aligner: AlignmentAdapter,
    settings: ProcessingSettings,
    transcript: Transcript,
) -> RuntimeSignature | None:
    """Читает фактический runtime optional aligner-а из результата."""
    accessor = getattr(aligner, "runtime_signature", None)
    if not callable(accessor):
        return None
    return _normalize_runtime_signature(accessor(settings, transcript))


def _expected_aligner_runtime(
    aligner: AlignmentAdapter,
    settings: ProcessingSettings,
) -> RuntimeSignature | None:
    """Получает ожидаемый runtime aligner-а без загрузки весов."""
    accessor = getattr(aligner, "expected_runtime_signature", None)
    if not callable(accessor):
        return None
    return _normalize_runtime_signature(accessor(settings))


def _loaded_runtime_signature(
    backend: AsrBackend | None,
    settings: ProcessingSettings,
) -> RuntimeSignature | None:
    """Читает фактический runtime backend-а, не заставляя его загружать модель."""
    accessor = getattr(backend, "runtime_signature", None)
    if not callable(accessor):
        return None
    return _normalize_runtime_signature(accessor(settings))


def _expected_runtime_signature(
    backend: AsrBackend,
    settings: ProcessingSettings,
) -> RuntimeSignature | None:
    """Получает ожидаемый runtime выбранного backend-а без загрузки весов."""
    accessor = getattr(backend, "expected_runtime_signature", None)
    if not callable(accessor):
        return None
    return _normalize_runtime_signature(accessor(settings))


def _runtime_from_transcript(
    backend: AsrBackend,
    transcript: Transcript,
    settings: ProcessingSettings,
) -> RuntimeSignature:
    """Строит совместимый runtime для тестовых и внешних backend-ов старого контракта."""
    metadata = transcript.metadata
    raw = {
        "backend": metadata.get("runtime")
        or getattr(backend, "backend_id", settings.backend),
        "engine_version": metadata.get("engine_version", "unknown"),
        "device": transcript.device,
        "compute_type": metadata.get("compute_type")
        or _legacy_compute_type(transcript.device, transcript.quantized),
        "quantized": transcript.quantized,
    }
    runtime = _normalize_runtime_signature(raw)
    if runtime is None:
        raise ValidationError("ASR backend вернул неполное описание фактического runtime.")
    return runtime


def _normalize_runtime_signature(value: Any) -> RuntimeSignature | None:
    if not isinstance(value, Mapping):
        return None
    backend = str(value.get("backend", "")).strip().casefold()
    engine_version = str(value.get("engine_version", "")).strip()
    device = str(value.get("device", "")).strip().casefold()
    compute_type = str(value.get("compute_type", "")).strip().casefold()
    quantized = value.get("quantized")
    if (
        not backend
        or not engine_version
        or device not in {"cpu", "cuda"}
        or not compute_type
        or not isinstance(quantized, bool)
    ):
        return None
    return {
        "backend": backend,
        "engine_version": engine_version,
        "device": "cuda" if device == "cuda" else "cpu",
        "compute_type": compute_type,
        "quantized": quantized,
    }


def _legacy_compute_type(device: str, quantized: bool) -> str:
    if quantized:
        return "int8"
    return "float16" if device.casefold() == "cuda" else "float32"


def _sidecar_audio_duration(sidecar: Mapping[str, Any] | None) -> float | None:
    if not sidecar:
        return None
    value = sidecar.get("normalized_audio_duration")
    if value is None and isinstance(sidecar.get("transcript"), Mapping):
        value = sidecar["transcript"].get("duration")
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


def _has_saved_audio(outputs: OutputPaths) -> bool:
    return (
        outputs.normalized_audio_path.is_file()
        and outputs.normalized_audio_path.stat().st_size > 0
    )


def _record_saved_audio(
    outputs: OutputPaths,
    audio_source: Path,
    duration: float,
) -> None:
    sidecar = load_sidecar(outputs.sidecar_path)
    if sidecar is None:
        raise ValidationError("Не найден sidecar подтверждённого кеша.")
    raw_outputs = sidecar.get("outputs")
    saved_outputs = dict(raw_outputs) if isinstance(raw_outputs, Mapping) else {}
    saved_outputs["audio"] = str(outputs.normalized_audio_path)
    sidecar["outputs"] = saved_outputs
    sidecar["normalized_audio_duration"] = duration
    _publish_artifacts(outputs, sidecar=sidecar, audio_source=audio_source)


def _publish_artifacts(
    outputs: OutputPaths,
    *,
    srt_text: str | None = None,
    sidecar: Mapping[str, Any] | None = None,
    audio_source: Path | None = None,
) -> None:
    """Готовит весь набор рядом с целями и откатывает частичный commit."""
    token = uuid.uuid4().hex
    staged: dict[Path, Path] = {}
    try:
        if audio_source is not None:
            staged_audio = _staging_path(outputs.normalized_audio_path, token)
            atomic_copy_file(audio_source, staged_audio)
            staged[outputs.normalized_audio_path] = staged_audio
        if srt_text is not None:
            staged_srt = _staging_path(outputs.srt_path, token)
            atomic_write_text_utf8(staged_srt, srt_text)
            staged[outputs.srt_path] = staged_srt
        if sidecar is not None:
            staged_sidecar = _staging_path(outputs.sidecar_path, token)
            write_sidecar(staged_sidecar, dict(sidecar))
            staged[outputs.sidecar_path] = staged_sidecar
        _commit_staged_artifacts(staged, token)
    finally:
        for staged_path in staged.values():
            staged_path.unlink(missing_ok=True)


def _commit_staged_artifacts(staged: Mapping[Path, Path], token: str) -> None:
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for target in staged:
            if target.exists():
                backup = _backup_path(target, token)
                os.replace(target, backup)
                backups[target] = backup
        for target, staged_path in staged.items():
            os.replace(staged_path, target)
            published.append(target)
    except Exception:
        for target in reversed(published):
            target.unlink(missing_ok=True)
        for target, backup in backups.items():
            if backup.exists():
                os.replace(backup, target)
        raise
    finally:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def _staging_path(target: Path, token: str) -> Path:
    return target.with_name(f".{target.name}.{token}.stage")


def _backup_path(target: Path, token: str) -> Path:
    return target.with_name(f".{target.name}.{token}.backup")


def _validate_settings(settings: ProcessingSettings) -> None:
    _validate_backend_settings(settings)
    _validate_output_settings(settings)
    _validate_chunk_settings(settings)
    _validate_long_form_settings(settings)
    _validate_decoding_settings(settings)


def _validate_backend_settings(settings: ProcessingSettings) -> None:
    if settings.backend.casefold() not in backend_names():
        variants = ", ".join(backend_names())
        raise ValidationError(f"ASR backend должен иметь одно из значений: {variants}.")
    if settings.aligner.casefold() not in aligner_names():
        variants = ", ".join(aligner_names())
        raise ValidationError(f"Aligner должен иметь одно из значений: {variants}.")
    if settings.language.casefold() not in {"en", "ru", "auto"}:
        raise ValidationError("Язык должен быть en, ru или auto.")


def _validate_output_settings(settings: ProcessingSettings) -> None:
    if settings.audio_stream_index is not None and settings.audio_stream_index < 0:
        raise ValidationError("Индекс аудиопотока не может быть отрицательным.")
    if (
        isinstance(settings.max_chars_per_line, bool)
        or not isinstance(settings.max_chars_per_line, int)
        or settings.max_chars_per_line < 20
        or settings.max_chars_per_line > 80
    ):
        raise ValidationError("Лимит строки SRT должен быть от 20 до 80 символов.")
    if (
        isinstance(settings.line_length_gap, bool)
        or not isinstance(settings.line_length_gap, int)
        or settings.line_length_gap < 0
        or settings.line_length_gap > MAX_LINE_LENGTH_GAP
    ):
        raise ValidationError(
            f"Допуск длины строки SRT должен быть от 0 до {MAX_LINE_LENGTH_GAP} символов."
        )
    if (
        isinstance(settings.max_cps, bool)
        or not isinstance(settings.max_cps, (int, float))
        or not math.isfinite(settings.max_cps)
        or settings.max_cps < 5
        or settings.max_cps > 60
    ):
        raise ValidationError("Скорость чтения SRT должна быть от 5 до 60 CPS.")


def _validate_chunk_settings(settings: ProcessingSettings) -> None:
    if settings.chunk_length_seconds < 10:
        raise ValidationError("Длина ASR-фрагмента должна быть не меньше 10 секунд.")
    if settings.stride_length_seconds < 0:
        raise ValidationError("ASR stride не может быть отрицательным.")
    if settings.stride_length_seconds * 2 >= settings.chunk_length_seconds:
        raise ValidationError("ASR stride должен быть меньше половины длины фрагмента.")


def _validate_long_form_settings(settings: ProcessingSettings) -> None:
    if settings.long_form_window_seconds < 30 or settings.long_form_window_seconds > 3600:
        raise ValidationError("Long-form окно должно быть от 30 до 3600 секунд.")
    if (
        settings.long_form_overlap_seconds < 0
        or settings.long_form_overlap_seconds >= settings.long_form_window_seconds
    ):
        raise ValidationError("Long-form overlap должен быть короче основного окна.")


def _validate_decoding_settings(settings: ProcessingSettings) -> None:
    if settings.vad_min_silence_ms < 100 or settings.vad_min_silence_ms > 10_000:
        raise ValidationError("VAD min silence должен быть от 100 до 10000 мс.")
    if settings.beam_size < 1 or settings.beam_size > 20:
        raise ValidationError("Beam size должен быть от 1 до 20.")


def _emit_file(
    emit_event: EventCallback,
    path: Path,
    index: int,
    total: int,
    state: str,
    stage: str,
    progress: int,
    *,
    outputs: OutputPaths | None = None,
    audio_output: Path | None = None,
    error: str | None = None,
) -> None:
    event: dict[str, Any] = {
        "type": "file",
        "path": str(path),
        "name": path.name,
        "index": index,
        "total": total,
        "state": state,
        "stage": stage,
        "progress": max(0, min(100, progress)),
    }
    if outputs:
        event.update(
            srt_output=str(outputs.srt_path),
            sidecar_output=str(outputs.sidecar_path),
            audio_output=str(audio_output) if audio_output else None,
        )
    if error:
        event["error"] = error
    emit_event(event)


def _write_log(log: LogCallback, message: str) -> None:
    logger.info(message)
    if callable(log):
        log(message)


def _common_input_root(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    try:
        common = os.path.commonpath([str(path.parent) for path in paths])
    except ValueError:
        return None
    return Path(common)


def _build_output_map(
    paths: list[Path],
    settings: ProcessingSettings,
    common_root: Path | None,
) -> dict[Path, OutputPaths]:
    """Устраняет коллизии одинаковых basename, сохраняя обычные имена в остальных случаях."""
    candidates = {
        path: build_output_paths(path, settings, common_root)
        for path in paths
    }
    groups: dict[str, list[Path]] = {}
    for path, outputs in candidates.items():
        groups.setdefault(str(outputs.srt_path).casefold(), []).append(path)
    for collided_paths in groups.values():
        if len(collided_paths) < 2:
            continue
        for path in collided_paths:
            candidates[path] = build_output_paths(
                path,
                settings,
                common_root,
                include_source_extension=True,
            )
    repeated_groups = _output_groups(candidates)
    for collided_paths in repeated_groups.values():
        if len(collided_paths) < 2:
            continue
        for path in collided_paths:
            suffix = hashlib.sha256(
                str(path.resolve()).casefold().encode("utf-8")
            ).hexdigest()[:12]
            candidates[path] = build_output_paths(
                path,
                settings,
                common_root,
                include_source_extension=True,
                collision_suffix=suffix,
            )
    return candidates


def _output_groups(candidates: Mapping[Path, OutputPaths]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path, outputs in candidates.items():
        groups.setdefault(str(outputs.srt_path).casefold(), []).append(path)
    return groups


def _output_lock_path(outputs: OutputPaths) -> Path:
    lock_root = WORK_DIR / "locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        str(outputs.srt_path.resolve()).casefold().encode("utf-8")
    ).hexdigest()
    return lock_root / f"{digest}.lock"


def _command_exists(command: str) -> bool:
    candidate = Path(command).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        return candidate.is_file()
    return shutil.which(command) is not None


__all__ = [
    "build_items",
    "build_output_paths",
    "get_preflight_status",
    "process_paths",
]
