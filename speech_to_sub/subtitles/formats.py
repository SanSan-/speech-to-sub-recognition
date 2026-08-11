"""Форматы вывода и их детерминированный рендеринг."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from speech_to_sub.constants import SubtitleFormat
from speech_to_sub.exceptions import ValidationError

if TYPE_CHECKING:
    from speech_to_sub.subtitles.builder import Cue

_ASS_BACKSLASH_GUARD = "\u2060"
_ASS_ESCAPE_REPLACEMENTS = {"{": "{", "}": "}", "N": "\n"}
ASS_SCRIPT_INFO_LINES = (
    "[Script Info]",
    "ScriptType: v4.00+",
    "WrapStyle: 0",
    "ScaledBorderAndShadow: yes",
)
ASS_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
    "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
    "MarginR, MarginV, Encoding"
)
ASS_DEFAULT_STYLE = (
    "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
    "0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1"
)
ASS_EVENTS_FORMAT = (
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
)
_ASS_PREFIX_PATTERN = r"^(Dialogue|Comment): (\d++),"
_ASS_TIMING_PATTERN = r"(\d++:\d++:\d++\.\d++),(\d++:\d++:\d++\.\d++),"
_ASS_STYLE_PATTERN = r"([^,]*+),([^,]*+),"
_ASS_MARGIN_PATTERN = r"(\d++),(\d++),(\d++),"
_ASS_SUFFIX_PATTERN = r"([^,]*+),(.*+)$"
ASS_EVENT_MASK = re.compile(
    _ASS_PREFIX_PATTERN
    + _ASS_TIMING_PATTERN
    + _ASS_STYLE_PATTERN
    + _ASS_MARGIN_PATTERN
    + _ASS_SUFFIX_PATTERN,
    re.IGNORECASE,
)


@dataclass(slots=True)
class AssEvent:
    """Поля события ASS, разобранные по контракту экспортёра."""

    event_type: str = "Dialogue"
    translatable: bool = True
    layer: int | None = None
    start_time: str | None = None
    end_time: str | None = None
    style: str | None = None
    actor: str | None = None
    margin_l: int | None = None
    margin_r: int | None = None
    margin_v: int | None = None
    effect: str | None = None
    text: str | None = None


def parse_subtitle_format(value: object) -> SubtitleFormat:
    """Преобразует внешнее значение формата в доменный тип."""
    if isinstance(value, SubtitleFormat):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("Формат субтитров должен быть строкой: srt, ass или vtt.")
    normalized = value.strip().casefold()
    try:
        return SubtitleFormat(normalized)
    except ValueError as exc:
        raise ValidationError(
            f"Неизвестный формат субтитров '{value}'. Поддерживаются: srt, ass, vtt."
        ) from exc


def render_subtitles(cues: Sequence[Cue], output_format: SubtitleFormat) -> str:
    """Рендерит реплики в выбранный формат без записи на диск."""
    if output_format is SubtitleFormat.SRT:
        from speech_to_sub.subtitles.builder import render_srt

        return render_srt(cues)
    if output_format is SubtitleFormat.ASS:
        return render_ass(cues)
    if output_format is SubtitleFormat.VTT:
        return render_vtt(cues)
    raise ValidationError(f"Неподдерживаемый формат субтитров: {output_format!r}.")


def render_vtt(cues: Sequence[Cue]) -> str:
    """Форматирует реплики как WebVTT с миллисекундной шкалой."""
    from speech_to_sub.subtitles.builder import quantize_cue_timestamps

    timestamps = quantize_cue_timestamps(cues, units_per_second=1000)
    blocks = [
        f"{cue.index}\n{_format_vtt_timestamp(start)} --> "
        f"{_format_vtt_timestamp(end)}\n{escape_vtt_text(cue.text)}"
        for cue, (start, end) in zip(cues, timestamps, strict=True)
    ]
    return "WEBVTT\n\n" + "\n\n".join(blocks) + ("\n" if blocks else "")


def render_ass(cues: Sequence[Cue]) -> str:
    """Форматирует реплики как ASS через проверенный событийный экспортёр."""
    from speech_to_sub.subtitles.builder import quantize_cue_timestamps

    timestamps = quantize_cue_timestamps(cues, units_per_second=100)
    dialogue_lines = [
        "Dialogue: 0,"
        f"{_format_ass_timestamp(start)},{_format_ass_timestamp(end)},"
        "Default,,0,0,0,,"
        for start, end in timestamps
    ]
    origin = [
        *ASS_SCRIPT_INFO_LINES,
        "",
        "[V4+ Styles]",
        ASS_STYLE_FORMAT,
        ASS_DEFAULT_STYLE,
        "",
        "[Events]",
        ASS_EVENTS_FORMAT,
        *dialogue_lines,
    ]
    dialogs = parse_ass_dialogs(origin)
    translated_dialogs = {
        line_index: escape_ass_text(cue.text)
        for line_index, cue in zip(dialogs, cues, strict=True)
    }
    return "\n".join(
        _build_ass_export_lines(origin, dialogs, translated_dialogs)
    ) + "\n"


def ass_validator(text: str) -> bool:
    """Проверяет, соответствует ли строка событию ASS из исходного экспортёра."""
    return bool(ASS_EVENT_MASK.match(text))


def ass_separator(line: int, text: str) -> dict[int, AssEvent]:
    """Разбирает одну строку события ASS, сохраняя все её структурные поля."""
    match = ASS_EVENT_MASK.match(text)
    if not match or (match.lastindex or 0) < 11:
        return {}
    event_type = match.group(1)
    return {
        line: AssEvent(
            event_type=event_type,
            translatable=event_type.casefold() == "dialogue",
            layer=int(match.group(2) or 0),
            start_time=match.group(3),
            end_time=match.group(4),
            style=match.group(5),
            actor=match.group(6),
            margin_l=int(match.group(7) or 0),
            margin_r=int(match.group(8) or 0),
            margin_v=int(match.group(9) or 0),
            effect=match.group(10),
            text=match.group(11),
        )
    }


def parse_ass_events(origins: Sequence[str]) -> dict[int, AssEvent]:
    """Разбирает все события ASS и сохраняет их исходные номера строк."""
    events: dict[int, AssEvent] = {}
    for index, text in enumerate(origins):
        if ass_validator(text):
            events.update(ass_separator(index, text))
    return events


def parse_ass_dialogs(origins: Sequence[str]) -> dict[int, AssEvent]:
    """Возвращает только переводимые события ``Dialogue``."""
    return {
        key: event
        for key, event in parse_ass_events(origins).items()
        if event.translatable
    }


def _replace_ass_dialog_text(line: str, text: str) -> str:
    fields = line.split(",", 9)
    if len(fields) != 10:
        return line
    return ",".join([*fields[:9], text or ""])


def _build_ass_export_lines(
    origin: list[str],
    dialogs: dict[int, AssEvent],
    translated_dialogs: dict[int, str],
) -> list[str]:
    return [
        _replace_ass_dialog_text(line, translated_dialogs[index])
        if index in dialogs and index in translated_dialogs
        else line
        for index, line in enumerate(origin)
    ]


def escape_ass_text(text: str) -> str:
    """Экранирует управляющие символы ASS, сохраняя запятые и Unicode."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    result: list[str] = []
    for character in normalized:
        if character == "\\":
            result.extend(("\\", _ASS_BACKSLASH_GUARD))
        elif character == "{":
            result.append("\\{")
        elif character == "}":
            result.append("\\}")
        elif character == "\n":
            result.append("\\N")
        elif character == _ASS_BACKSLASH_GUARD:
            result.append(_ASS_BACKSLASH_GUARD * 2)
        else:
            result.append(character)
    return "".join(result)


def unescape_ass_text(text: str) -> str:
    """Восстанавливает текст из ограниченного экранирования генератора ASS."""
    result: list[str] = []
    index = 0
    while index < len(text):
        character, index = _decode_ass_character(text, index)
        result.append(character)
    return "".join(result)


def _decode_ass_character(text: str, index: int) -> tuple[str, int]:
    current = text[index]
    if current == _ASS_BACKSLASH_GUARD:
        return _decode_ass_guard(text, index)
    if current == "\\":
        return _decode_ass_escape(text, index)
    if current in "{}":
        raise ValidationError("Текст ASS содержит неэкранированную фигурную скобку.")
    return current, index + 1


def _decode_ass_guard(text: str, index: int) -> tuple[str, int]:
    next_index = index + 1
    if next_index >= len(text) or text[next_index] != _ASS_BACKSLASH_GUARD:
        raise ValidationError("Текст ASS содержит неожиданный защитный символ.")
    return _ASS_BACKSLASH_GUARD, index + 2


def _decode_ass_escape(text: str, index: int) -> tuple[str, int]:
    next_index = index + 1
    if next_index >= len(text):
        raise ValidationError("Текст ASS заканчивается незавершённым экранированием.")
    escaped = text[next_index]
    if escaped == _ASS_BACKSLASH_GUARD:
        return "\\", index + 2
    if escaped not in _ASS_ESCAPE_REPLACEMENTS:
        raise ValidationError(
            f"Текст ASS содержит неподдерживаемое экранирование \\{escaped}."
        )
    return _ASS_ESCAPE_REPLACEMENTS[escaped], index + 2


def escape_vtt_text(text: str) -> str:
    """Экранирует управляющую разметку WebVTT, сохраняя видимый текст."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def unescape_vtt_text(text: str) -> str:
    """Восстанавливает только сущности, создаваемые безопасным рендерером WebVTT."""
    result: list[str] = []
    index = 0
    entities = {"&amp;": "&", "&lt;": "<", "&gt;": ">"}
    while index < len(text):
        current = text[index]
        if current in "<>":
            raise ValidationError("Текст VTT содержит неэкранированную разметку.")
        if current != "&":
            result.append(current)
            index += 1
            continue
        entity = next(
            (candidate for candidate in entities if text.startswith(candidate, index)),
            None,
        )
        if entity is None:
            raise ValidationError("Текст VTT содержит неподдерживаемую сущность.")
        result.append(entities[entity])
        index += len(entity)
    return "".join(result)


def _format_vtt_timestamp(total_milliseconds: int) -> str:
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _format_ass_timestamp(total_centiseconds: int) -> str:
    hours, remainder = divmod(total_centiseconds, 360_000)
    if hours > 595:
        raise ValidationError("ASS не поддерживает временные метки длиннее 595 часов.")
    minutes, remainder = divmod(remainder, 6_000)
    seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


__all__ = [
    "ASS_DEFAULT_STYLE",
    "ASS_EVENT_MASK",
    "ASS_EVENTS_FORMAT",
    "ASS_SCRIPT_INFO_LINES",
    "ASS_STYLE_FORMAT",
    "AssEvent",
    "SubtitleFormat",
    "ass_separator",
    "ass_validator",
    "escape_ass_text",
    "escape_vtt_text",
    "parse_ass_dialogs",
    "parse_ass_events",
    "parse_subtitle_format",
    "render_ass",
    "render_subtitles",
    "render_vtt",
    "unescape_ass_text",
    "unescape_vtt_text",
]
