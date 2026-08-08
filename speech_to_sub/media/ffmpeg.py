"""Безопасный запуск ffprobe и FFmpeg для подготовки аудио."""

from __future__ import annotations

import json
import logging
import math
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np

from speech_to_sub.exceptions import MediaError
from speech_to_sub.models import AudioStreamInfo, MediaProbe

LOGGER = logging.getLogger(__name__)

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
WarningCallback = Callable[[str], None]


class FFprobeTagsPayload(TypedDict, total=False):
    """Используемые теги потока ffprobe."""

    language: str
    title: str
    DURATION: str


class FFprobeStreamPayload(TypedDict, total=False):
    """Используемая часть JSON одного потока ffprobe."""

    index: int
    codec_type: str
    codec_name: str
    sample_rate: str
    channels: int
    duration: str
    tags: FFprobeTagsPayload


class FFprobeFormatPayload(TypedDict, total=False):
    """Используемая часть раздела format ffprobe."""

    duration: str
    format_name: str


class FFprobePayload(TypedDict, total=False):
    """JSON-контракт ffprobe, необходимый приложению."""

    streams: list[FFprobeStreamPayload]
    format: FFprobeFormatPayload


_LANGUAGE_ALIASES: dict[str, frozenset[str]] = {
    "en": frozenset({"en", "eng", "english"}),
    "eng": frozenset({"en", "eng", "english"}),
    "english": frozenset({"en", "eng", "english"}),
    "ru": frozenset({"ru", "rus", "russian", "русский"}),
    "rus": frozenset({"ru", "rus", "russian", "русский"}),
    "russian": frozenset({"ru", "rus", "russian", "русский"}),
}


def build_ffprobe_args(path: str | Path, ffprobe_path: str | Path = "ffprobe") -> list[str]:
    """Возвращает аргументы ffprobe без shell-интерпретации пути."""
    return [
        str(ffprobe_path),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]


def probe_media(
    path: str | Path,
    ffprobe_path: str | Path = "ffprobe",
    *,
    runner: CommandRunner | None = None,
) -> MediaProbe:
    """Запускает ffprobe и преобразует JSON в типизированные метаданные."""
    media_path = Path(path)
    process = (runner or subprocess.run)(
        build_ffprobe_args(media_path, ffprobe_path),
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if process.returncode != 0:
        detail = (process.stderr or "").strip() or "ffprobe не вернул описание ошибки"
        raise MediaError(f"ffprobe завершился с кодом {process.returncode}: {detail}")

    try:
        raw_payload = json.loads(process.stdout or "")
    except (json.JSONDecodeError, TypeError) as error:
        raise MediaError("ffprobe вернул некорректный JSON") from error
    if not isinstance(raw_payload, Mapping):
        raise MediaError("Корень JSON ffprobe должен быть объектом")
    return parse_ffprobe_payload(media_path, cast(Mapping[str, Any], raw_payload))


def parse_ffprobe_payload(path: str | Path, payload: Mapping[str, Any]) -> MediaProbe:
    """Разбирает проверенную часть JSON ffprobe без неявных преобразований."""
    raw_streams = payload.get("streams", [])
    if not isinstance(raw_streams, list):
        raise MediaError("Поле streams в JSON ffprobe должно быть списком")

    streams: list[AudioStreamInfo] = []
    for raw_stream in raw_streams:
        if not isinstance(raw_stream, Mapping) or raw_stream.get("codec_type") != "audio":
            continue
        stream_index = _optional_int(raw_stream.get("index"))
        if stream_index is None or stream_index < 0:
            raise MediaError("Аудиопоток ffprobe не содержит корректный index")
        tags_value = raw_stream.get("tags")
        tags = tags_value if isinstance(tags_value, Mapping) else {}
        duration = _parse_duration(raw_stream.get("duration"))
        if duration is None:
            duration = _parse_duration(tags.get("DURATION"))
        streams.append(
            AudioStreamInfo(
                ordinal=len(streams),
                index=stream_index,
                codec_name=str(raw_stream.get("codec_name") or "unknown"),
                sample_rate=_optional_int(raw_stream.get("sample_rate")),
                channels=_optional_int(raw_stream.get("channels")),
                language=_optional_text(tags.get("language")),
                title=_optional_text(tags.get("title")),
                duration=duration,
            )
        )

    format_value = payload.get("format")
    format_payload = format_value if isinstance(format_value, Mapping) else {}
    duration = _parse_duration(format_payload.get("duration"))
    if duration is None:
        known_durations = [stream.duration for stream in streams if stream.duration is not None]
        duration = max(known_durations, default=0.0)
    return MediaProbe(
        path=Path(path),
        duration=duration,
        streams=tuple(streams),
        format_name=_optional_text(format_payload.get("format_name")),
    )


def select_audio_stream(
    media: MediaProbe | Sequence[AudioStreamInfo],
    requested_ordinal: int | None = None,
    preferred_language: str | None = None,
    preferred_title: str | None = None,
    *,
    warning_callback: WarningCallback | None = None,
) -> tuple[AudioStreamInfo, str | None]:
    """Возвращает выбранный поток и видимое предупреждение безопасного fallback."""
    available = media.streams if isinstance(media, MediaProbe) else tuple(media)
    warnings: list[str] = []

    def report_warning(message: str) -> None:
        warnings.append(message)
        _warn(message, warning_callback)

    selected = _choose_audio_stream(
        available,
        requested_ordinal,
        preferred_language,
        preferred_title,
        report_warning,
    )
    return selected, warnings[0] if warnings else None


def _choose_audio_stream(
    available: Sequence[AudioStreamInfo],
    requested_ordinal: int | None,
    preferred_language: str | None,
    preferred_title: str | None,
    report_warning: WarningCallback,
) -> AudioStreamInfo:
    """Реализует приоритеты выбора без формирования внешнего контракта."""
    if not available:
        raise MediaError("В медиафайле не найдено ни одного аудиопотока")

    if requested_ordinal is not None:
        if isinstance(requested_ordinal, bool) or requested_ordinal < 0:
            raise MediaError("Порядковый номер аудиопотока должен быть целым числом от нуля")
        for stream in available:
            if stream.ordinal == requested_ordinal:
                return stream
        ordinals = ", ".join(str(stream.ordinal) for stream in available)
        raise MediaError(
            f"Аудиопоток с порядковым номером {requested_ordinal} не найден; доступны: {ordinals}"
        )

    matches = tuple(
        stream
        for stream in available
        if _matches_preference(stream, preferred_language, preferred_title)
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        report_warning(
            "Несколько аудиопотоков совпали с языком или title; выбран первый "
            f"(ordinal={matches[0].ordinal}, index={matches[0].index})"
        )
        return matches[0]

    if len(available) == 1:
        return available[0]

    selected = available[0]
    report_warning(
        "Не удалось однозначно выбрать аудиопоток; выбран первый "
        f"(ordinal={selected.ordinal}, index={selected.index})"
    )
    return selected


def build_ffmpeg_normalize_args(
    source: str | Path,
    destination: str | Path,
    stream: AudioStreamInfo,
    ffmpeg_path: str | Path = "ffmpeg",
    *,
    overwrite: bool = False,
) -> list[str]:
    """Собирает команду нормализации в mono 16 kHz FLAC."""
    return [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-map",
        f"0:{stream.index}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "flac",
        str(destination),
    ]


def normalize_audio(
    source: str | Path,
    destination: str | Path,
    stream: AudioStreamInfo,
    ffmpeg_path: str | Path = "ffmpeg",
    *,
    overwrite: bool = False,
    runner: CommandRunner | None = None,
) -> Path:
    """Создаёт отдельный FLAC и никогда не перезаписывает его неявно."""
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise MediaError(f"Исходный медиафайл не найден: {source_path}")
    if source_path.resolve() == destination_path.resolve():
        raise MediaError("Путь нормализованного аудио совпадает с исходным файлом")
    if destination_path.exists() and not overwrite:
        raise MediaError(f"Файл уже существует, перезапись не разрешена: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    process = (runner or subprocess.run)(
        build_ffmpeg_normalize_args(
            source_path,
            destination_path,
            stream,
            ffmpeg_path,
            overwrite=overwrite,
        ),
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode != 0:
        detail = (process.stderr or "").strip() or "FFmpeg не вернул описание ошибки"
        raise MediaError(f"FFmpeg завершился с кодом {process.returncode}: {detail}")
    if not destination_path.is_file() or destination_path.stat().st_size == 0:
        raise MediaError("FFmpeg завершился без непустого выходного FLAC")
    return destination_path


def get_media_duration(
    media: MediaProbe | str | Path,
    ffprobe_path: str | Path = "ffprobe",
    *,
    runner: CommandRunner | None = None,
) -> float:
    """Возвращает положительную длительность из probe или нового запуска ffprobe."""
    probe = media if isinstance(media, MediaProbe) else probe_media(media, ffprobe_path, runner=runner)
    if not math.isfinite(probe.duration) or probe.duration <= 0:
        raise MediaError(f"ffprobe не определил положительную длительность: {probe.path}")
    return probe.duration


def decode_audio_float32(
    source: str | Path,
    ffmpeg_path: str | Path = "ffmpeg",
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] | None = None,
) -> np.ndarray:
    """Декодирует нормализованное аудио через настроенный FFmpeg в mono float32."""
    source_path = Path(source)
    if not source_path.is_file():
        raise MediaError(f"Нормализованное аудио не найдено: {source_path}")
    args = [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(source_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "-c:a",
        "pcm_f32le",
        "pipe:1",
    ]
    process = (runner or subprocess.run)(
        args,
        shell=False,
        check=False,
        capture_output=True,
    )
    if process.returncode != 0:
        raw_detail = process.stderr or b""
        detail = (
            raw_detail.decode("utf-8", errors="replace")
            if isinstance(raw_detail, bytes)
            else str(raw_detail)
        ).strip()
        raise MediaError(
            f"FFmpeg не смог декодировать аудио: {detail or 'описание ошибки отсутствует'}"
        )
    raw_audio = process.stdout or b""
    if not isinstance(raw_audio, bytes) or len(raw_audio) < 4:
        raise MediaError("FFmpeg не вернул PCM-данные для локального распознавания.")
    samples = np.frombuffer(raw_audio, dtype="<f4")
    if samples.size == 0 or not np.isfinite(samples).all():
        raise MediaError("FFmpeg вернул некорректные PCM-данные.")
    return samples


def _matches_preference(
    stream: AudioStreamInfo,
    preferred_language: str | None,
    preferred_title: str | None,
) -> bool:
    language = (preferred_language or "").strip().casefold()
    title = (stream.title or "").strip().casefold()
    stream_language = (stream.language or "").strip().casefold()
    aliases = _LANGUAGE_ALIASES.get(language, frozenset({language}) if language else frozenset())
    language_match = bool(aliases and stream_language in aliases)

    if preferred_title and title:
        title_match = preferred_title.strip().casefold() in title
    else:
        title_names = {alias for alias in aliases if len(alias) > 2}
        title_match = bool(title and any(name in title for name in title_names))
    return language_match or title_match


def _warn(message: str, callback: WarningCallback | None) -> None:
    (callback or LOGGER.warning)(message)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_duration(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text or text.casefold() == "n/a":
        return None
    try:
        if ":" in text:
            hours_text, minutes_text, seconds_text = text.split(":", maxsplit=2)
            result = int(hours_text) * 3600 + int(minutes_text) * 60 + float(seconds_text)
        else:
            result = float(text)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None
