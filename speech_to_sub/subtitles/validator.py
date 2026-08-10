"""Синтаксическая и временная проверка субтитров SRT."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from speech_to_sub.constants import DEFAULT_LINE_LENGTH_GAP, MAX_LINE_LENGTH_GAP
from speech_to_sub.exceptions import ValidationError
from speech_to_sub.subtitles.builder import Cue
from speech_to_sub.subtitles.layout import visible_character_count

_TIMESTAMP_RE = re.compile(
    r"^(?P<hours>\d{2,}):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d),(?P<milliseconds>\d{3})$"
)
_TIMING_LINE_RE = re.compile(r"^(?P<start>\S+) --> (?P<end>\S+)$")


def validate_cues(
    cues: Sequence[Cue],
    *,
    audio_duration: float | None = None,
    duration_tolerance: float = 0.25,
    max_chars_per_line: int | None = None,
    line_length_gap: int = DEFAULT_LINE_LENGTH_GAP,
    max_lines: int | None = None,
    max_cps: float | None = None,
) -> None:
    """Проверяет структуру и переданные явно ограничения раскладки."""
    if not cues:
        raise ValidationError("SRT не содержит ни одной реплики")
    _validate_duration_settings(audio_duration, duration_tolerance)
    _validate_layout_settings(max_chars_per_line, line_length_gap, max_lines, max_cps)

    previous_end = 0.0
    for expected_index, cue in enumerate(cues, start=1):
        _validate_cue(
            cue,
            expected_index,
            previous_end,
            max_chars_per_line=max_chars_per_line,
            line_length_gap=line_length_gap,
            max_lines=max_lines,
            max_cps=max_cps,
        )
        previous_end = cue.end

    if (
        audio_duration is not None
        and cues[-1].end > audio_duration + duration_tolerance + 1e-9
    ):
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


def _validate_cue(
    cue: Cue,
    expected_index: int,
    previous_end: float,
    *,
    max_chars_per_line: int | None,
    line_length_gap: int,
    max_lines: int | None,
    max_cps: float | None,
) -> None:
    if cue.index != expected_index:
        raise ValidationError(
            "Нумерация SRT должна быть последовательной и начинаться с единицы"
        )
    _validate_cue_text(
        cue,
        max_chars_per_line=max_chars_per_line,
        line_length_gap=line_length_gap,
        max_lines=max_lines,
    )
    _validate_cue_timing(cue, previous_end, max_cps=max_cps)


def _validate_cue_text(
    cue: Cue,
    *,
    max_chars_per_line: int | None,
    line_length_gap: int,
    max_lines: int | None,
) -> None:
    if not cue.text.strip():
        raise ValidationError(f"Реплика {cue.index} не содержит текста")
    lines = cue.text.splitlines()
    if len(lines) not in (1, 2) or any(not line.strip() for line in lines):
        raise ValidationError(
            f"Реплика {cue.index} должна содержать одну или две непустые строки"
        )
    if max_lines is not None and len(lines) > max_lines:
        raise ValidationError(f"Реплика {cue.index} превышает лимит количества строк")
    if max_chars_per_line is not None and any(
        len(line) > max_chars_per_line + line_length_gap for line in lines
    ):
        raise ValidationError(f"Реплика {cue.index} превышает лимит символов в строке")


def _validate_cue_timing(
    cue: Cue,
    previous_end: float,
    *,
    max_cps: float | None,
) -> None:
    if not math.isfinite(cue.start) or not math.isfinite(cue.end):
        raise ValidationError(
            f"Реплика {cue.index} содержит неконечную временную метку"
        )
    if cue.start < 0 or cue.end <= cue.start:
        raise ValidationError(f"Реплика {cue.index} содержит некорректный интервал")
    if cue.start < previous_end:
        raise ValidationError(f"Реплика {cue.index} пересекается с предыдущей")
    if max_cps is not None:
        actual_cps = visible_character_count(cue.text) / (cue.end - cue.start)
        if actual_cps > max_cps and not math.isclose(
            actual_cps,
            max_cps,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValidationError(
                f"Реплика {cue.index} превышает лимит CPS: {actual_cps:.2f} > {max_cps:g}"
            )


def _validate_layout_settings(
    max_chars_per_line: int | None,
    line_length_gap: int,
    max_lines: int | None,
    max_cps: float | None,
) -> None:
    if max_chars_per_line is not None and max_chars_per_line < 1:
        raise ValidationError("Лимит символов в строке должен быть положительным")
    if (
        isinstance(line_length_gap, bool)
        or not isinstance(line_length_gap, int)
        or line_length_gap < 0
        or line_length_gap > MAX_LINE_LENGTH_GAP
    ):
        raise ValidationError(
            f"Допуск длины строки должен быть целым числом от 0 до {MAX_LINE_LENGTH_GAP}"
        )
    if max_lines is not None and max_lines not in (1, 2):
        raise ValidationError("SRT допускает лимит в одну или две строки")
    if max_cps is not None and (not math.isfinite(max_cps) or max_cps <= 0):
        raise ValidationError("Лимит CPS должен быть положительным конечным числом")


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
            raise ValidationError(
                f"Некорректный номер блока SRT: {lines[0]!r}"
            ) from error
        timing_match = _TIMING_LINE_RE.fullmatch(lines[1])
        if timing_match is None:
            raise ValidationError(
                f"Некорректная строка тайминга в блоке {block_number}"
            )
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
    max_chars_per_line: int | None = None,
    line_length_gap: int = DEFAULT_LINE_LENGTH_GAP,
    max_lines: int | None = None,
    max_cps: float | None = None,
) -> tuple[Cue, ...]:
    """Разбирает и полностью проверяет SRT, возвращая проверенные cues."""
    cues = parse_srt(content)
    validate_cues(
        cues,
        audio_duration=audio_duration,
        duration_tolerance=duration_tolerance,
        max_chars_per_line=max_chars_per_line,
        line_length_gap=line_length_gap,
        max_lines=max_lines,
        max_cps=max_cps,
    )
    return cues


def validate_srt_text(
    content: str,
    duration: float | None = None,
    *,
    max_chars_per_line: int | None = None,
    line_length_gap: int = DEFAULT_LINE_LENGTH_GAP,
    max_lines: int | None = None,
    max_cps: float | None = None,
) -> None:
    """Проверяет SRT-текст через стабильный контракт service layer."""
    validate_srt(
        content,
        audio_duration=duration,
        max_chars_per_line=max_chars_per_line,
        line_length_gap=line_length_gap,
        max_lines=max_lines,
        max_cps=max_cps,
    )


def _parse_timestamp(value: str) -> float:
    match = _TIMESTAMP_RE.fullmatch(value)
    if match is None:
        raise ValidationError(f"Некорректная временная метка SRT: {value!r}")
    total_milliseconds = (
        int(match.group("hours")) * 3_600_000
        + int(match.group("minutes")) * 60_000
        + int(match.group("seconds")) * 1000
        + int(match.group("milliseconds"))
    )
    return total_milliseconds / 1000
