"""Синтаксическая и временная проверка поддерживаемых субтитров."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from speech_to_sub.constants import DEFAULT_LINE_LENGTH_GAP, MAX_LINE_LENGTH_GAP
from speech_to_sub.exceptions import ValidationError
from speech_to_sub.subtitles.builder import Cue
from speech_to_sub.subtitles.formats import (
    ASS_DEFAULT_STYLE,
    ASS_EVENTS_FORMAT,
    ASS_SCRIPT_INFO_LINES,
    ASS_STYLE_FORMAT,
    SubtitleFormat,
    ass_separator,
    unescape_ass_text,
    unescape_vtt_text,
)
from speech_to_sub.subtitles.layout import visible_character_count

_TIMESTAMP_RE = re.compile(
    r"^(?P<hours>\d{2,}):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d),(?P<milliseconds>\d{3})$"
)
_TIMING_LINE_RE = re.compile(r"^(?P<start>\S+) --> (?P<end>\S+)$")
_VTT_TIMESTAMP_RE = re.compile(
    r"^(?P<hours>\d{2,}):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)\.(?P<milliseconds>\d{3})$"
)
_ASS_TIMESTAMP_RE = re.compile(
    r"^(?P<hours>\d+):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)\.(?P<centiseconds>\d{2})$"
)


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
        raise ValidationError("Субтитры не содержат ни одной реплики")
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
            "Нумерация реплик должна быть последовательной и начинаться с единицы"
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
        raise ValidationError("Субтитры допускают лимит в одну или две строки")
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


def parse_vtt(content: str) -> tuple[Cue, ...]:
    """Разбирает детерминированный профиль WebVTT генератора."""
    normalized = _normalized_subtitle_text(content, "VTT")
    lines = normalized.split("\n")
    if lines[0] != "WEBVTT":
        raise ValidationError("VTT должен начинаться с заголовка WEBVTT")
    if len(lines) < 3 or lines[1] != "":
        raise ValidationError("После заголовка WEBVTT должна быть пустая строка")
    blocks = re.split(r"\n[ \t]*\n", "\n".join(lines[2:]))
    cues: list[Cue] = []
    for block_number, block in enumerate(blocks, start=1):
        block_lines = block.split("\n")
        timing_index = _vtt_timing_index(block_lines, block_number)
        text_lines = block_lines[timing_index + 1 :]
        if len(text_lines) not in (1, 2) or any(not line.strip() for line in text_lines):
            raise ValidationError(
                f"Реплика VTT {block_number} должна содержать одну или две непустые строки"
            )
        timing_match = _TIMING_LINE_RE.fullmatch(block_lines[timing_index])
        if timing_match is None:
            raise ValidationError(f"Некорректный тайминг VTT в блоке {block_number}")
        cues.append(
            Cue(
                index=block_number,
                start=_parse_vtt_timestamp(timing_match.group("start")),
                end=_parse_vtt_timestamp(timing_match.group("end")),
                text=unescape_vtt_text("\n".join(text_lines)),
            )
        )
    return tuple(cues)


def _vtt_timing_index(lines: list[str], block_number: int) -> int:
    if not lines or len(lines) > 4:
        raise ValidationError(f"Некорректная структура блока VTT {block_number}")
    if _TIMING_LINE_RE.fullmatch(lines[0]):
        return 0
    if len(lines) < 3:
        raise ValidationError(f"Некорректная структура блока VTT {block_number}")
    try:
        cue_id = int(lines[0])
    except ValueError as exc:
        raise ValidationError(f"Некорректный идентификатор VTT: {lines[0]!r}") from exc
    if cue_id != block_number:
        raise ValidationError("Нумерация VTT должна быть последовательной")
    return 1


def validate_vtt(
    content: str,
    *,
    audio_duration: float | None = None,
    duration_tolerance: float = 0.001,
    max_chars_per_line: int | None = None,
    line_length_gap: int = DEFAULT_LINE_LENGTH_GAP,
    max_lines: int | None = None,
    max_cps: float | None = None,
) -> tuple[Cue, ...]:
    """Разбирает и полностью проверяет WebVTT."""
    cues = parse_vtt(content)
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


def parse_ass(content: str) -> tuple[Cue, ...]:
    """Разбирает безопасный профиль ASS, создаваемый генератором."""
    normalized = _normalized_subtitle_text(content, "ASS")
    lines = normalized.split("\n")
    expected_header = [
        *ASS_SCRIPT_INFO_LINES,
        "",
        "[V4+ Styles]",
        ASS_STYLE_FORMAT,
        ASS_DEFAULT_STYLE,
        "",
        "[Events]",
        ASS_EVENTS_FORMAT,
    ]
    if lines[: len(expected_header)] != expected_header:
        raise ValidationError("ASS содержит неподдерживаемый заголовок или стиль")
    dialogue_lines = lines[len(expected_header) :]
    if not dialogue_lines:
        raise ValidationError("ASS не содержит ни одной реплики")
    if any(not line for line in dialogue_lines):
        raise ValidationError("ASS содержит пустую строку внутри раздела Events")
    return tuple(
        _parse_ass_dialogue(line, index)
        for index, line in enumerate(dialogue_lines, start=1)
    )


def _parse_ass_dialogue(line: str, index: int) -> Cue:
    event = ass_separator(index, line).get(index)
    if event is None or not event.translatable:
        raise ValidationError(f"Некорректная строка Events ASS: {line!r}")
    if (
        event.layer != 0
        or event.style != "Default"
        or event.actor
        or event.effect
    ):
        raise ValidationError(f"Реплика ASS {index} не соответствует профилю генератора")
    if any(value != 0 for value in (event.margin_l, event.margin_r, event.margin_v)):
        raise ValidationError(f"Реплика ASS {index} содержит неподдерживаемые отступы")
    return Cue(
        index=index,
        start=_parse_ass_timestamp(event.start_time or ""),
        end=_parse_ass_timestamp(event.end_time or ""),
        text=unescape_ass_text(event.text or ""),
    )


def validate_ass(
    content: str,
    *,
    audio_duration: float | None = None,
    duration_tolerance: float = 0.01,
    max_chars_per_line: int | None = None,
    line_length_gap: int = DEFAULT_LINE_LENGTH_GAP,
    max_lines: int | None = None,
    max_cps: float | None = None,
) -> tuple[Cue, ...]:
    """Разбирает и полностью проверяет ASS."""
    cues = parse_ass(content)
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


def validate_subtitle_text(
    content: str,
    output_format: SubtitleFormat,
    duration: float | None = None,
    *,
    max_chars_per_line: int | None = None,
    line_length_gap: int = DEFAULT_LINE_LENGTH_GAP,
    max_lines: int | None = None,
    max_cps: float | None = None,
) -> tuple[Cue, ...]:
    """Проверяет готовый текст через единый контракт service layer."""
    validators = {
        SubtitleFormat.SRT: validate_srt,
        SubtitleFormat.ASS: validate_ass,
        SubtitleFormat.VTT: validate_vtt,
    }
    duration_tolerances = {
        SubtitleFormat.SRT: 0.001,
        SubtitleFormat.ASS: 0.01,
        SubtitleFormat.VTT: 0.001,
    }
    validator = validators.get(output_format)
    if validator is None:
        raise ValidationError(f"Неподдерживаемый формат субтитров: {output_format!r}.")
    return validator(
        content,
        audio_duration=duration,
        duration_tolerance=duration_tolerances[output_format],
        max_chars_per_line=max_chars_per_line,
        line_length_gap=line_length_gap,
        max_lines=max_lines,
        max_cps=max_cps,
    )


def _normalized_subtitle_text(content: str, label: str) -> str:
    if content.startswith("\ufeff"):
        raise ValidationError(f"{label} не должен содержать UTF-8 BOM")
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not normalized.strip():
        raise ValidationError(f"{label} пуст")
    return normalized


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


def _parse_vtt_timestamp(value: str) -> float:
    match = _VTT_TIMESTAMP_RE.fullmatch(value)
    if match is None:
        raise ValidationError(f"Некорректная временная метка VTT: {value!r}")
    total_milliseconds = (
        int(match.group("hours")) * 3_600_000
        + int(match.group("minutes")) * 60_000
        + int(match.group("seconds")) * 1000
        + int(match.group("milliseconds"))
    )
    return total_milliseconds / 1000


def _parse_ass_timestamp(value: str) -> float:
    match = _ASS_TIMESTAMP_RE.fullmatch(value)
    if match is None:
        raise ValidationError(f"Некорректная временная метка ASS: {value!r}")
    hours = int(match.group("hours"))
    if hours > 595:
        raise ValidationError("Временная метка ASS превышает предел 595 часов")
    total_centiseconds = (
        hours * 360_000
        + int(match.group("minutes")) * 6_000
        + int(match.group("seconds")) * 100
        + int(match.group("centiseconds"))
    )
    return total_centiseconds / 100
