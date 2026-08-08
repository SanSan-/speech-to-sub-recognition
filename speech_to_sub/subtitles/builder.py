"""Преобразование временных меток ASR в читаемые SRT-реплики."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from collections.abc import Sequence

from speech_to_sub.constants import DEFAULT_MAX_CHARS_PER_LINE
from speech_to_sub.exceptions import ValidationError
from speech_to_sub.models import TranscriptSegment, TranscriptWord

_SENTENCE_END_RE = re.compile(r"[.!?…][\"')\]}]*$")
_CLOSING_PUNCTUATION = frozenset(".,!?;:%…)]}-—")


@dataclass(frozen=True, slots=True)
class Cue:
    """Одна пронумерованная реплика SRT."""

    index: int
    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class _RawCue:
    start: float
    end: float
    text: str


def build_cues(
    segments: Sequence[TranscriptSegment],
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE,
    *,
    max_lines: int = 2,
    min_duration: float = 0.8,
    max_duration: float = 7.0,
    pause_threshold: float = 0.8,
    audio_duration: float | None = None,
) -> list[Cue]:
    """Строит непересекающиеся cues, предпочитая словные временные метки."""
    _validate_settings(
        max_chars_per_line,
        max_lines,
        min_duration,
        max_duration,
        pause_threshold,
        audio_duration,
    )
    raw_cues: list[_RawCue] = []
    for segment in segments:
        _validate_interval(segment.start, segment.end, "сегмента")
        text = _clean_text(segment.text)
        valid_words = tuple(word for word in segment.words if _clean_text(word.text))
        if valid_words:
            raw_cues.extend(
                _cues_from_words(
                    valid_words,
                    max_chars=max_chars_per_line * max_lines,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    pause_threshold=pause_threshold,
                )
            )
        elif text:
            raw_cues.extend(
                _cues_from_segment(
                    segment.start,
                    segment.end,
                    text,
                    max_chars=max_chars_per_line * max_lines,
                    max_duration=max_duration,
                )
            )

    if not raw_cues:
        raise ValidationError("Невозможно построить SRT: распознанный текст пуст")

    ordered = sorted(raw_cues, key=lambda cue: (cue.start, cue.end))
    normalized = _normalize_timing(ordered, min_duration, audio_duration)
    return [
        Cue(
            index=index,
            start=cue.start,
            end=cue.end,
            text=_wrap_text(cue.text, max_chars_per_line, max_lines),
        )
        for index, cue in enumerate(normalized, start=1)
    ]


def build_srt(
    segments: Sequence[TranscriptSegment],
    *,
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE,
    max_lines: int = 2,
    min_duration: float = 0.8,
    max_duration: float = 7.0,
    pause_threshold: float = 0.8,
    audio_duration: float | None = None,
) -> str:
    """Строит и проверяет готовый текст SRT в памяти."""
    cues = build_cues(
        segments,
        max_chars_per_line=max_chars_per_line,
        max_lines=max_lines,
        min_duration=min_duration,
        max_duration=max_duration,
        pause_threshold=pause_threshold,
        audio_duration=audio_duration,
    )
    from speech_to_sub.subtitles.validator import validate_cues

    validate_cues(cues, audio_duration=audio_duration)
    return render_srt(cues)


def render_srt(cues: Sequence[Cue]) -> str:
    """Форматирует cues как SRT с LF и завершающей пустой строкой."""
    blocks = [
        f"{cue.index}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n{cue.text}"
        for cue in cues
    ]
    return "\n\n".join(blocks) + "\n" if blocks else ""


def format_timestamp(seconds: float) -> str:
    """Форматирует секунды как HH:MM:SS,mmm с округлением до миллисекунд."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValidationError("Временная метка SRT должна быть конечной и неотрицательной")
    total_milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def _cues_from_words(
    words: Sequence[TranscriptWord],
    *,
    max_chars: int,
    min_duration: float,
    max_duration: float,
    pause_threshold: float,
) -> list[_RawCue]:
    ordered = sorted(words, key=lambda word: (word.start, word.end))
    for word in ordered:
        _validate_interval(word.start, word.end, "слова")

    result: list[_RawCue] = []
    group: list[TranscriptWord] = []
    for position, word in enumerate(ordered):
        candidate = [*group, word]
        candidate_text = _join_word_texts(candidate)
        exceeds_length = bool(group and len(candidate_text) > max_chars)
        exceeds_duration = bool(group and word.end - group[0].start > max_duration)
        if exceeds_length or exceeds_duration:
            result.append(_raw_cue_from_words(group))
            group = []
        group.append(word)

        next_word = ordered[position + 1] if position + 1 < len(ordered) else None
        current_duration = group[-1].end - group[0].start
        punctuation_boundary = bool(
            _SENTENCE_END_RE.search(_clean_text(word.text)) and current_duration >= min_duration
        )
        pause_boundary = bool(
            next_word is not None
            and next_word.start - word.end >= pause_threshold
            and current_duration >= min_duration
        )
        duration_boundary = current_duration >= max_duration
        if next_word is None or punctuation_boundary or pause_boundary or duration_boundary:
            result.append(_raw_cue_from_words(group))
            group = []
    return result


def _raw_cue_from_words(words: Sequence[TranscriptWord]) -> _RawCue:
    return _RawCue(
        start=words[0].start,
        end=words[-1].end,
        text=_join_word_texts(words),
    )


def _cues_from_segment(
    start: float,
    end: float,
    text: str,
    *,
    max_chars: int,
    max_duration: float,
) -> list[_RawCue]:
    desired_chunks = max(1, math.ceil((end - start) / max_duration))
    chunks = _split_text(text, max_chars, desired_chunks)
    if len(chunks) == 1:
        return [_RawCue(start=start, end=end, text=chunks[0])]

    weights = [max(1, len(chunk.replace(" ", ""))) for chunk in chunks]
    total_weight = sum(weights)
    duration = end - start
    result: list[_RawCue] = []
    elapsed_weight = 0
    for index, (chunk, weight) in enumerate(zip(chunks, weights, strict=True)):
        chunk_start = start + duration * elapsed_weight / total_weight
        elapsed_weight += weight
        chunk_end = end if index == len(chunks) - 1 else start + duration * elapsed_weight / total_weight
        result.append(_RawCue(start=chunk_start, end=chunk_end, text=chunk))
    return result


def _split_text(text: str, max_chars: int, desired_chunks: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    target = min(max_chars, max(1, math.ceil(len(text) / desired_chunks)))
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and len(candidate) > target:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _normalize_timing(
    cues: Sequence[_RawCue],
    min_duration: float,
    audio_duration: float | None,
) -> list[_RawCue]:
    normalized: list[_RawCue] = []
    previous_end = 0.0
    for index, cue in enumerate(cues):
        start = max(0.0, cue.start, previous_end)
        if audio_duration is not None and start >= audio_duration:
            raise ValidationError("Начало реплики находится за пределами длительности аудио")

        next_start = cues[index + 1].start if index + 1 < len(cues) else None
        end = max(cue.end, start + 0.001)
        if next_start is not None and next_start > start:
            end = min(end, next_start)
        if end - start < min_duration:
            limit = next_start if next_start is not None and next_start > start else start + min_duration
            if audio_duration is not None:
                limit = min(limit, audio_duration)
            end = max(end, min(start + min_duration, limit))
        if audio_duration is not None:
            end = min(end, audio_duration)
        if end <= start:
            raise ValidationError("После нормализации получена реплика нулевой длительности")

        normalized.append(replace(cue, start=start, end=end))
        previous_end = end
    return normalized


def _wrap_text(text: str, max_chars: int, max_lines: int) -> str:
    words = text.split()
    if len(text) <= max_chars or len(words) < 2 or max_lines == 1:
        return text
    if max_lines != 2:
        return _wrap_greedy(words, max_chars, max_lines)

    candidates: list[tuple[int, int, str, str]] = []
    for split_at in range(1, len(words)):
        first = " ".join(words[:split_at])
        second = " ".join(words[split_at:])
        overflow = max(0, len(first) - max_chars) + max(0, len(second) - max_chars)
        candidates.append((overflow, abs(len(first) - len(second)), first, second))
    _, _, first, second = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    return f"{first}\n{second}"


def _wrap_greedy(words: Sequence[str], max_chars: int, max_lines: int) -> str:
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars and len(lines) < max_lines - 1:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _join_word_texts(words: Sequence[TranscriptWord]) -> str:
    result = ""
    for word in words:
        token = _clean_text(word.text)
        if not token:
            continue
        if not result or token[0] in _CLOSING_PUNCTUATION or result[-1] in "([{—-":
            result += token
        else:
            result += f" {token}"
    return result


def _clean_text(text: str) -> str:
    return " ".join(str(text).split())


def _validate_interval(start: float, end: float, label: str) -> None:
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValidationError(f"Некорректные временные границы {label}: {start}–{end}")


def _validate_settings(
    max_chars_per_line: int,
    max_lines: int,
    min_duration: float,
    max_duration: float,
    pause_threshold: float,
    audio_duration: float | None,
) -> None:
    if max_chars_per_line < 1 or max_lines < 1:
        raise ValidationError("Лимиты строк SRT должны быть положительными")
    if min_duration <= 0 or max_duration <= 0 or min_duration > max_duration:
        raise ValidationError("Некорректный диапазон длительности SRT-реплики")
    if pause_threshold < 0:
        raise ValidationError("Порог паузы не может быть отрицательным")
    if audio_duration is not None and (
        not math.isfinite(audio_duration) or audio_duration <= 0
    ):
        raise ValidationError("Длительность аудио должна быть положительной")
