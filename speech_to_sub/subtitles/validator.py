"""Синтаксическая и временная проверка субтитров SRT."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from speech_to_sub.exceptions import ValidationError
from speech_to_sub.subtitles.builder import Cue

_TIMESTAMP_RE = re.compile(
    r"^(?P<hours>\d{2,}):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d),(?P<milliseconds>\d{3})$"
)
_TIMING_LINE_RE = re.compile(r"^(?P<start>\S+) --> (?P<end>\S+)$")


def validate_cues(
    cues: Sequence[Cue],
    *,
    audio_duration: float | None = None,
    duration_tolerance: float = 0.25,
) -> None:
    """Проверяет нумерацию, текст, интервалы и границу длительности."""
    if not cues:
        raise ValidationError("SRT не содержит ни одной реплики")
    _validate_duration_settings(audio_duration, duration_tolerance)

    previous_end = 0.0
    for expected_index, cue in enumerate(cues, start=1):
        _validate_cue(cue, expected_index, previous_end)
        previous_end = cue.end

    if audio_duration is not None and cues[-1].end > audio_duration + duration_tolerance:
        raise ValidationError("Последняя реплика выходит за длительность аудио")


def _validate_duration_settings(
    audio_duration: float | None,
    duration_tolerance: float,
) -> None:
    if duration_tolerance < 0:
        raise ValidationError("Допуск длительности не может быть отрицательным")
    if audio_duration is not None and (
        not math.isfinite(audio_duration) or audio_duration <= 0
    ):
        raise ValidationError("Длительность аудио должна быть положительной")


def _validate_cue(cue: Cue, expected_index: int, previous_end: float) -> None:
    if cue.index != expected_index:
        raise ValidationError("Нумерация SRT должна быть последовательной и начинаться с единицы")
    if not cue.text.strip():
        raise ValidationError(f"Реплика {cue.index} не содержит текста")
    lines = cue.text.splitlines()
    if len(lines) not in (1, 2) or any(not line.strip() for line in lines):
        raise ValidationError(f"Реплика {cue.index} должна содержать одну или две непустые строки")
    if not math.isfinite(cue.start) or not math.isfinite(cue.end):
        raise ValidationError(f"Реплика {cue.index} содержит неконечную временную метку")
    if cue.start < 0 or cue.end <= cue.start:
        raise ValidationError(f"Реплика {cue.index} содержит некорректный интервал")
    if cue.start < previous_end:
        raise ValidationError(f"Реплика {cue.index} пересекается с предыдущей")


def parse_srt(content: str) -> tuple[Cue, ...]:
    """Разбирает SRT-текст в cues и отклоняет неоднозначный синтаксис."""
    if content.startswith("\ufeff"):
        raise ValidationError("SRT не должен содержать UTF-8 BOM")
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValidationError("SRT пуст")

    blocks = re.split(r"\n[ \t]*\n", normalized)
    cues: list[Cue] = []
    for block_number, block in enumerate(blocks, start=1):
        lines = block.split("\n")
        if len(lines) < 3 or len(lines) > 4:
            raise ValidationError(
                f"Блок {block_number} должен содержать номер, тайминг и одну или две строки текста"
            )
        try:
            index = int(lines[0])
        except ValueError as error:
            raise ValidationError(f"Некорректный номер блока SRT: {lines[0]!r}") from error
        timing_match = _TIMING_LINE_RE.fullmatch(lines[1])
        if timing_match is None:
            raise ValidationError(f"Некорректная строка тайминга в блоке {block_number}")
        cues.append(
            Cue(
                index=index,
                start=_parse_timestamp(timing_match.group("start")),
                end=_parse_timestamp(timing_match.group("end")),
                text="\n".join(lines[2:]),
            )
        )
    return tuple(cues)


def validate_srt(
    content: str,
    *,
    audio_duration: float | None = None,
    duration_tolerance: float = 0.25,
) -> tuple[Cue, ...]:
    """Разбирает и полностью проверяет SRT, возвращая проверенные cues."""
    cues = parse_srt(content)
    validate_cues(
        cues,
        audio_duration=audio_duration,
        duration_tolerance=duration_tolerance,
    )
    return cues


def validate_srt_text(content: str, duration: float | None = None) -> None:
    """Проверяет SRT-текст через стабильный контракт service layer."""
    validate_srt(content, audio_duration=duration)


def _parse_timestamp(value: str) -> float:
    match = _TIMESTAMP_RE.fullmatch(value)
    if match is None:
        raise ValidationError(f"Некорректная временная метка SRT: {value!r}")
    return (
        int(match.group("hours")) * 3600
        + int(match.group("minutes")) * 60
        + int(match.group("seconds"))
        + int(match.group("milliseconds")) / 1000
    )
