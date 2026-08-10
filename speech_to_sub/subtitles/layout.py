"""Подготовка единого потока слов для раскладки субтитров."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from collections.abc import Sequence
from typing import Literal

from speech_to_sub.exceptions import ValidationError
from speech_to_sub.models import TranscriptSegment, TranscriptWord

_DASHES = frozenset({"-", "—"})
_CLOSING_PUNCTUATION = frozenset(".,!?;:%…)]}»”")
_OPENING_PUNCTUATION = frozenset("([{«“")
_AMBIGUOUS_QUOTES = frozenset("\"'’")
_TERMINAL_PUNCTUATION = frozenset(".!?…")
_SENTENCE_END_RE = re.compile(r"[.!?…][\"'»”’)]*\Z")
_EMBEDDED_SENTENCE_RE = re.compile(
    r"([.!?…][\"'»”’)]*)(?=[A-ZА-ЯЁ«“\"(\[])",
)
_MIN_RECONCILIATION_RATIO = 0.72
_LEADING_ISLAND_MAX_WORDS = 2
_LEADING_ISLAND_MAX_CHARS = 12
_LEADING_ISLAND_MIN_GAP = 3.0
_LEADING_ISLAND_MIN_RIGHT_WORDS = 4
_LEADING_ISLAND_PROBABILITY_GAP = 0.05
_HYPHEN_SUFFIX_RE = re.compile(r"-(то|либо|нибудь|ка|таки|де|с)\W*\Z", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class LayoutDiagnostics:
    """Счётчики неразрушающей подготовки словных меток."""

    reconciled_segments: int = 0
    alignment_text_segments: int = 0
    synthetic_timing_segments: int = 0
    retimed_leading_islands: int = 0
    adjusted_boundaries: int = 0
    max_boundary_drift_ms: int = 0
    timing_anomaly_adjustments: int = 0
    reading_speed_target_exceeded_cues: int = 0
    max_actual_cps: float = 0.0
    duration_target_exceeded_cues: int = 0
    max_actual_duration_ms: int = 0
    line_length_target_exceeded_lines: int = 0
    max_actual_line_length: int = 0

    def to_dict(self) -> dict[str, int | float]:
        """Возвращает JSON-совместимое представление для sidecar."""
        return {
            "reconciled_segments": self.reconciled_segments,
            "alignment_text_segments": self.alignment_text_segments,
            "synthetic_timing_segments": self.synthetic_timing_segments,
            "retimed_leading_islands": self.retimed_leading_islands,
            "adjusted_boundaries": self.adjusted_boundaries,
            "max_boundary_drift_ms": self.max_boundary_drift_ms,
            "timing_anomaly_adjustments": self.timing_anomaly_adjustments,
            "reading_speed_target_exceeded_cues": (
                self.reading_speed_target_exceeded_cues
            ),
            "max_actual_cps": self.max_actual_cps,
            "duration_target_exceeded_cues": self.duration_target_exceeded_cues,
            "max_actual_duration_ms": self.max_actual_duration_ms,
            "line_length_target_exceeded_lines": (
                self.line_length_target_exceeded_lines
            ),
            "max_actual_line_length": self.max_actual_line_length,
        }


@dataclass(frozen=True, slots=True)
class PreparedWords:
    """Единый поток слов и диагностика его подготовки."""

    words: tuple[TranscriptWord, ...]
    diagnostics: LayoutDiagnostics


@dataclass(frozen=True, slots=True)
class TimedSentence:
    """Предложение: текст из ASR и словные метки только как временные якоря."""

    text: str
    words: tuple[TranscriptWord, ...]
    start: float
    end: float
    boundary: Literal["sentence", "end"]


@dataclass(frozen=True, slots=True)
class TimedLine:
    """Смысловая строка с неизменяемой текстовой границей и словными якорями."""

    text: str
    words: tuple[TranscriptWord, ...]
    start: float
    end: float
    sentence_index: int
    sentence_end: bool = False


@dataclass(slots=True)
class _MutableWord:
    text: str
    start: float
    end: float
    probabilities: list[float]


@dataclass(slots=True)
class _AlignmentWordState:
    result: list[_MutableWord]
    pending_decorations: list[TranscriptWord] = field(default_factory=list)


@dataclass(slots=True)
class _DisplayUnitState:
    result: list[str]
    pending_decorations: list[str] = field(default_factory=list)


def visible_character_count(text: str) -> int:
    """Считает CPS-символы, заменяя перенос строки логическим пробелом."""
    return len(" ".join(text.splitlines()))


def prepare_timed_words(segments: Sequence[TranscriptSegment]) -> PreparedWords:
    """Сводит сегменты в поток слов, сохраняя исходный текст ASR."""
    prepared: list[TranscriptWord] = []
    reconciled = 0
    alignment_text = 0
    synthetic = 0
    retimed = 0

    for segment in segments:
        text = _clean_text(segment.text)
        words = tuple(word for word in segment.words if _clean_text(word.text))
        if words and not _word_intervals_are_valid(
            words,
            segment_start=segment.start,
            segment_end=segment.end,
        ):
            words = ()
        if words:
            words, island_retimed = _retime_leading_island(segment, words)
            retimed += int(island_retimed)
            if text and _texts_correspond(text, words):
                prepared.extend(_reconcile_text_with_words(text, words))
                reconciled += 1
            elif text:
                prepared.extend(_words_from_segment(segment.start, segment.end, text))
                synthetic += 1
            else:
                prepared.extend(_normalize_alignment_words(words))
                alignment_text += 1
        elif text:
            prepared.extend(_words_from_segment(segment.start, segment.end, text))
            synthetic += 1

    merged = _merge_global_hyphen_suffixes(prepared)
    return PreparedWords(
        words=merged,
        diagnostics=LayoutDiagnostics(
            reconciled_segments=reconciled,
            alignment_text_segments=alignment_text,
            synthetic_timing_segments=synthetic,
            retimed_leading_islands=retimed,
        ),
    )


def build_timed_sentences(
    words: Sequence[TranscriptWord],
    pause_threshold: float,
) -> tuple[TimedSentence, ...]:
    """Собирает предложения до упаковки строк и независимо от ограничений времени показа."""
    if pause_threshold < 0:
        raise ValidationError("Порог паузы не может быть отрицательным")
    if not words:
        return ()

    result: list[TimedSentence] = []
    start = 0
    for index, word in enumerate(words):
        following = words[index + 1] if index + 1 < len(words) else None
        sentence_end = bool(_SENTENCE_END_RE.search(_clean_text(word.text)))
        if following is not None and not sentence_end:
            continue
        chunk = tuple(words[start : index + 1])
        boundary: Literal["sentence", "end"] = "sentence" if sentence_end else "end"
        result.append(
            TimedSentence(
                text=_join_prepared_words(chunk),
                words=chunk,
                start=chunk[0].start,
                end=max(item.end for item in chunk),
                boundary=boundary,
            )
        )
        start = index + 1
    return tuple(result)


def _join_prepared_words(words: Sequence[TranscriptWord]) -> str:
    text = ""
    for word in words:
        token = _clean_text(word.text)
        if not token:
            continue
        text += token if not text or text.endswith("-") else f" {token}"
    return text


def _texts_correspond(text: str, words: Sequence[TranscriptWord]) -> bool:
    source = _compact_text(text)
    aligned = "".join(_compact_text(word.text) for word in words)
    if not source or not aligned:
        return False
    return SequenceMatcher(None, source, aligned, autojunk=False).ratio() >= (
        _MIN_RECONCILIATION_RATIO
    )


def _reconcile_text_with_words(
    text: str,
    words: Sequence[TranscriptWord],
) -> tuple[TranscriptWord, ...]:
    units = _display_units(text)
    source = "".join(_compact_text(unit) for unit in units)
    target, spans = _target_character_spans(words)
    anchors = _alignment_anchors(source, target)
    result: list[TranscriptWord] = []
    source_cursor = 0
    previous_start = words[0].start

    for unit in units:
        unit_length = len(_compact_text(unit))
        mapped_start = _map_coordinate(source_cursor, anchors)
        source_cursor += unit_length
        mapped_end = _map_coordinate(source_cursor, anchors)
        start = max(previous_start, _time_at_coordinate(words, spans, mapped_start, False))
        end = _time_at_coordinate(words, spans, mapped_end, True)
        end = max(start + 0.001, end)
        probability = _probability_at_coordinate(words, spans, (mapped_start + mapped_end) / 2)
        result.append(TranscriptWord(start, end, unit, probability))
        previous_start = start
    return tuple(result)


def _target_character_spans(
    words: Sequence[TranscriptWord],
) -> tuple[str, tuple[tuple[int, int], ...]]:
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for word in words:
        compact = _compact_text(word.text)
        start = cursor
        cursor += len(compact)
        spans.append((start, cursor))
        parts.append(compact)
    return "".join(parts), tuple(spans)


def _alignment_anchors(source: str, target: str) -> tuple[tuple[float, float], ...]:
    matcher = SequenceMatcher(None, source, target, autojunk=False)
    points = {(0.0, 0.0), (float(len(source)), float(len(target)))}
    for block in matcher.get_matching_blocks():
        if block.size:
            points.add((float(block.a), float(block.b)))
            points.add((float(block.a + block.size), float(block.b + block.size)))
    return tuple(sorted(points))


def _map_coordinate(
    coordinate: float,
    anchors: Sequence[tuple[float, float]],
) -> float:
    previous = anchors[0]
    for current in anchors[1:]:
        if coordinate <= current[0]:
            width = current[0] - previous[0]
            if width <= 0:
                return current[1]
            ratio = (coordinate - previous[0]) / width
            return previous[1] + ratio * (current[1] - previous[1])
        previous = current
    return anchors[-1][1]


def _time_at_coordinate(
    words: Sequence[TranscriptWord],
    spans: Sequence[tuple[int, int]],
    coordinate: float,
    is_end: bool,
) -> float:
    lexical = [(word, span) for word, span in zip(words, spans, strict=True) if span[1] > span[0]]
    if not lexical:
        return words[-1].end if is_end else words[0].start
    if coordinate <= 0:
        return lexical[0][0].start
    if coordinate >= lexical[-1][1][1]:
        return lexical[-1][0].end

    for index, (word, (start, end)) in enumerate(lexical):
        if coordinate < end or (is_end and math.isclose(coordinate, end)):
            if not is_end and math.isclose(coordinate, end) and index + 1 < len(lexical):
                return lexical[index + 1][0].start
            ratio = (coordinate - start) / (end - start)
            return word.start + min(1.0, max(0.0, ratio)) * (word.end - word.start)
    return lexical[-1][0].end


def _probability_at_coordinate(
    words: Sequence[TranscriptWord],
    spans: Sequence[tuple[int, int]],
    coordinate: float,
) -> float | None:
    for word, (start, end) in zip(words, spans, strict=True):
        if start <= coordinate <= end and end > start:
            return word.probability
    return None


def _normalize_alignment_words(
    words: Sequence[TranscriptWord],
) -> tuple[TranscriptWord, ...]:
    state = _AlignmentWordState(result=[])
    for word in _split_embedded_sentence_words(words):
        _consume_alignment_word(state, word)
    _finish_alignment_words(state)
    return tuple(_freeze_word(word) for word in state.result)


def _consume_alignment_word(
    state: _AlignmentWordState,
    word: TranscriptWord,
) -> None:
    token = _clean_text(word.text)
    if _capture_alignment_decoration(state, token, word):
        return
    token = _normalize_leading_hyphen(token)
    if (
        state.result
        and state.result[-1].text.endswith("-")
        and not state.pending_decorations
    ):
        _append_to_previous(state.result[-1], token, word)
        return
    _append_alignment_lexeme(state, token, word)


def _capture_alignment_decoration(
    state: _AlignmentWordState,
    token: str,
    word: TranscriptWord,
) -> bool:
    if token in _DASHES:
        state.pending_decorations.append(
            TranscriptWord(word.start, word.end, "—", word.probability)
        )
        return True
    if _is_closing_punctuation(token):
        if state.pending_decorations:
            state.pending_decorations.append(word)
            return True
        if state.result:
            _append_to_previous(state.result[-1], token, word)
            return True
    if _is_punctuation_only(token, _OPENING_PUNCTUATION):
        state.pending_decorations.append(word)
        return True
    if _is_hyphen_suffix(token) and not any(
        decoration.text == "—" for decoration in state.pending_decorations
    ):
        _append_alignment_suffix(state.result, token, word)
        return True
    return False


def _append_alignment_suffix(
    result: list[_MutableWord],
    token: str,
    word: TranscriptWord,
) -> None:
    if result:
        _append_to_previous(result[-1], token, word)
        return
    result.append(_MutableWord(token, word.start, word.end, _probabilities(word)))


def _append_alignment_lexeme(
    state: _AlignmentWordState,
    token: str,
    word: TranscriptWord,
) -> None:
    decorations = tuple(state.pending_decorations)
    start = min(
        (item.start for item in decorations),
        default=word.start,
    )
    probabilities = _probabilities(word)
    for decoration in decorations:
        probabilities.extend(_probabilities(decoration))
    text = _format_pending_decorations(
        tuple(decoration.text for decoration in decorations),
        token,
    )
    end = max((item.end for item in decorations), default=word.end)
    state.result.append(_MutableWord(text, start, max(end, word.end), probabilities))
    state.pending_decorations.clear()


def _finish_alignment_words(state: _AlignmentWordState) -> None:
    decoration_words = tuple(state.pending_decorations)
    if not decoration_words:
        return
    decoration = _format_pending_decorations(
        tuple(word.text for word in decoration_words)
    )
    start = min(word.start for word in decoration_words)
    end = max(word.end for word in decoration_words)
    probabilities = [
        probability
        for word in decoration_words
        for probability in _probabilities(word)
    ]
    if state.result:
        state.result[-1].text += f" {decoration}"
        state.result[-1].end = max(state.result[-1].end, end)
        state.result[-1].probabilities.extend(probabilities)
        return
    state.result.append(
        _MutableWord(
            decoration,
            start,
            end,
            probabilities,
        )
    )


def _append_to_previous(target: _MutableWord, text: str, word: TranscriptWord) -> None:
    target.text += text
    target.end = max(target.end, word.end)
    target.probabilities.extend(_probabilities(word))


def _freeze_word(word: _MutableWord) -> TranscriptWord:
    probability = (
        sum(word.probabilities) / len(word.probabilities) if word.probabilities else None
    )
    return TranscriptWord(word.start, word.end, word.text, probability)


def _probabilities(word: TranscriptWord) -> list[float]:
    return [word.probability] if word.probability is not None else []


def _words_from_segment(start: float, end: float, text: str) -> tuple[TranscriptWord, ...]:
    units = _display_units(text)
    if not units:
        return ()
    weights = [max(1, len(_compact_text(unit))) for unit in units]
    total_weight = sum(weights)
    elapsed = 0
    result: list[TranscriptWord] = []
    for index, (unit, weight) in enumerate(zip(units, weights, strict=True)):
        word_start = start + (end - start) * elapsed / total_weight
        elapsed += weight
        word_end = end if index == len(units) - 1 else start + (end - start) * elapsed / total_weight
        result.append(TranscriptWord(word_start, word_end, unit))
    return tuple(result)


def _display_units(text: str) -> tuple[str, ...]:
    state = _DisplayUnitState(result=[])
    separated = _EMBEDDED_SENTENCE_RE.sub(r"\1 ", _clean_text(text))
    for raw_token in separated.split():
        _consume_display_token(state, raw_token)
    _finish_display_units(state)
    return tuple(state.result)


def _consume_display_token(state: _DisplayUnitState, raw_token: str) -> None:
    token = "—" if raw_token in _DASHES else raw_token
    if _capture_display_decoration(state, token):
        return
    token = _normalize_leading_hyphen(token)
    state.result.append(_format_pending_decorations(tuple(state.pending_decorations), token))
    state.pending_decorations.clear()


def _capture_display_decoration(state: _DisplayUnitState, token: str) -> bool:
    if token == "—":
        state.pending_decorations.append(token)
        return True
    if _is_closing_punctuation(token):
        if state.pending_decorations:
            state.pending_decorations.append(token)
            return True
        if state.result:
            state.result[-1] += token
            return True
    if _is_punctuation_only(token, _OPENING_PUNCTUATION):
        state.pending_decorations.append(token)
        return True
    if _is_hyphen_suffix(token) and "—" not in state.pending_decorations:
        _append_display_suffix(state.result, token)
        return True
    return False


def _append_display_suffix(result: list[str], token: str) -> None:
    if result:
        result[-1] += token
        return
    result.append(token)


def _finish_display_units(state: _DisplayUnitState) -> None:
    if not state.pending_decorations:
        return
    decoration = _format_pending_decorations(tuple(state.pending_decorations))
    if state.result:
        state.result[-1] += f" {decoration}"
    else:
        state.result.append(decoration)


def _format_pending_decorations(
    decorations: Sequence[str],
    lexeme: str = "",
) -> str:
    result = ""
    for decoration in decorations:
        if decoration == "—":
            if result and not result.endswith(" "):
                result += " "
            result += "— "
        else:
            result += decoration
    return f"{result}{lexeme}".rstrip()


def _normalize_leading_hyphen(token: str) -> str:
    if token.startswith("-") and len(token) > 1:
        return f"— {token[1:]}"
    return token


def _split_embedded_sentence_words(
    words: Sequence[TranscriptWord],
) -> tuple[TranscriptWord, ...]:
    result: list[TranscriptWord] = []
    for word in words:
        parts = tuple(part for part in _EMBEDDED_SENTENCE_RE.sub(r"\1 ", word.text).split() if part)
        if len(parts) <= 1:
            result.append(word)
            continue
        weights = [max(1, len(_compact_text(part))) for part in parts]
        total_weight = sum(weights)
        elapsed = 0
        for index, (part, weight) in enumerate(zip(parts, weights, strict=True)):
            start = word.start + (word.end - word.start) * elapsed / total_weight
            elapsed += weight
            end = (
                word.end
                if index == len(parts) - 1
                else word.start + (word.end - word.start) * elapsed / total_weight
            )
            result.append(TranscriptWord(start, end, part, word.probability))
    return tuple(result)


def _merge_global_hyphen_suffixes(
    words: Sequence[TranscriptWord],
) -> tuple[TranscriptWord, ...]:
    result: list[TranscriptWord] = []
    for word in words:
        if result and _is_hyphen_suffix(word.text):
            result[-1] = _merge_words(result[-1], word, separator="")
        elif result and result[-1].text.endswith("-"):
            result[-1] = _merge_words(result[-1], word, separator="")
        else:
            result.append(word)
    return tuple(result)


def _merge_words(
    left: TranscriptWord,
    right: TranscriptWord,
    *,
    separator: str,
) -> TranscriptWord:
    probabilities = [
        probability
        for probability in (left.probability, right.probability)
        if probability is not None
    ]
    probability = sum(probabilities) / len(probabilities) if probabilities else None
    return TranscriptWord(
        start=min(left.start, right.start),
        end=max(left.end, right.end),
        text=f"{left.text}{separator}{right.text}",
        probability=probability,
    )


def _retime_leading_island(
    segment: TranscriptSegment,
    words: Sequence[TranscriptWord],
) -> tuple[tuple[TranscriptWord, ...], bool]:
    island_size = _leading_island_size(segment, words)
    if island_size is None:
        return tuple(words), False
    right_start = words[island_size].start
    durations = [word.end - word.start for word in words[:island_size]]
    if sum(durations) >= right_start - segment.start:
        return tuple(words), False
    cursor = right_start
    retimed = list(words)
    for index in range(island_size - 1, -1, -1):
        start = cursor - durations[index]
        word = words[index]
        retimed[index] = TranscriptWord(start, cursor, word.text, word.probability)
        cursor = start
    return tuple(retimed), True


def _leading_island_size(
    segment: TranscriptSegment,
    words: Sequence[TranscriptWord],
) -> int | None:
    if words[0].start - segment.start > 0.25:
        return None
    max_size = min(_LEADING_ISLAND_MAX_WORDS, len(words) - _LEADING_ISLAND_MIN_RIGHT_WORDS)
    for size in range(1, max_size + 1):
        left = words[:size]
        right = words[size:]
        if right[0].start - left[-1].end < _LEADING_ISLAND_MIN_GAP:
            continue
        if visible_character_count(_plain_word_text(left)) > _LEADING_ISLAND_MAX_CHARS:
            continue
        if _SENTENCE_END_RE.search(_clean_text(left[-1].text)):
            continue
        if not _has_lower_probability(left, right[:_LEADING_ISLAND_MIN_RIGHT_WORDS]):
            continue
        if _segment_confirms_order(segment.text, words, size):
            return size
    return None


def _has_lower_probability(
    left: Sequence[TranscriptWord],
    right: Sequence[TranscriptWord],
) -> bool:
    left_values = [word.probability for word in left if word.probability is not None]
    right_values = [word.probability for word in right if word.probability is not None]
    if len(left_values) != len(left) or len(right_values) != len(right):
        return False
    return sum(left_values) / len(left_values) + _LEADING_ISLAND_PROBABILITY_GAP < (
        sum(right_values) / len(right_values)
    )


def _segment_confirms_order(
    text: str,
    words: Sequence[TranscriptWord],
    island_size: int,
) -> bool:
    compared = words[: island_size + _LEADING_ISLAND_MIN_RIGHT_WORDS]
    return _compact_text(text).startswith("".join(_compact_text(word.text) for word in compared))


def _plain_word_text(words: Sequence[TranscriptWord]) -> str:
    return " ".join(_clean_text(word.text) for word in words)


def _word_intervals_are_valid(
    words: Sequence[TranscriptWord],
    *,
    segment_start: float,
    segment_end: float,
) -> bool:
    """Проверяет словные якоря без превращения ошибки выравнивания в отказ SRT."""
    if not _interval_is_valid(segment_start, segment_end):
        return False
    previous_end = segment_start
    for word in words:
        if (
            not _interval_is_valid(word.start, word.end)
            or word.start + 1e-9 < previous_end
            or word.start + 1e-9 < segment_start
            or word.end > segment_end + 1e-9
        ):
            return False
        previous_end = word.end
    return True


def _interval_is_valid(start: float, end: float) -> bool:
    return math.isfinite(start) and math.isfinite(end) and start >= 0 and end > start


def _is_punctuation_only(token: str, allowed: frozenset[str]) -> bool:
    return bool(token) and all(character in allowed for character in token)


def _is_closing_punctuation(token: str) -> bool:
    if _is_punctuation_only(token, _CLOSING_PUNCTUATION):
        return True
    allowed = _CLOSING_PUNCTUATION | _AMBIGUOUS_QUOTES
    return _is_punctuation_only(token, allowed) and any(
        character in _TERMINAL_PUNCTUATION for character in token
    )


def _is_hyphen_suffix(token: str) -> bool:
    return _HYPHEN_SUFFIX_RE.fullmatch(token) is not None


def _compact_text(text: str) -> str:
    normalized = _clean_text(text).casefold().replace("ё", "е")
    return "".join(character for character in normalized if character.isalnum())


def _clean_text(text: str) -> str:
    return " ".join(str(text).split())
