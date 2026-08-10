"""Преобразование временных меток ASR в читаемые SRT-реплики."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from collections.abc import Sequence

from speech_to_sub.constants import (
    DEFAULT_LINE_LENGTH_GAP,
    DEFAULT_MAX_CHARS_PER_LINE,
    DEFAULT_MAX_CPS,
    MAX_LINE_LENGTH_GAP,
)
from speech_to_sub.exceptions import ValidationError
from speech_to_sub.models import TranscriptSegment, TranscriptWord
from speech_to_sub.subtitles.layout import (
    LayoutDiagnostics,
    TimedLine,
    TimedSentence,
    build_timed_sentences,
    prepare_timed_words,
    visible_character_count,
)

_TIMING_MARGIN = 0.002
_MAX_BOUNDARY_DRIFT = 1.0
_MIN_RENDERED_CUE_DURATION = 0.001
_SENTENCE_PAIR_MAX_GAP = 0.5
_GOOD_BOUNDARY_PUNCTUATION = (",", ";", ":", "—")
_COORDINATING_STARTS = frozenset({"а", "да", "и", "или", "либо", "но"})
_WEAK_END_WORDS = frozenset(
    {
        "без",
        "в",
        "для",
        "до",
        "за",
        "из",
        "и",
        "к",
        "на",
        "над",
        "о",
        "об",
        "от",
        "по",
        "под",
        "при",
        "про",
        "с",
        "со",
        "у",
    }
)


@dataclass(frozen=True, slots=True)
class Cue:
    """Одна пронумерованная реплика SRT."""

    index: int
    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class CueBuildResult:
    """Реплики вместе с диагностикой подготовки исходного текста."""

    cues: tuple[Cue, ...]
    diagnostics: LayoutDiagnostics


@dataclass(frozen=True, slots=True)
class _RawCue:
    start: float
    end: float
    text: str
    words: tuple[TranscriptWord, ...] = ()


@dataclass(frozen=True, slots=True)
class _TextPartitionState:
    lines: tuple[TimedLine, ...]
    syntax_penalty: int
    preferred_width_overflow: int
    square_characters: int


@dataclass(frozen=True, slots=True)
class _ScheduledTiming:
    cues: tuple[_RawCue, ...]
    adjusted_boundaries: int
    max_boundary_drift_ms: int
    timing_anomaly_adjustments: int


@dataclass(frozen=True, slots=True)
class _CueGroupingState:
    groups: tuple[tuple[TimedLine, ...], ...]
    orphan_penalty: int
    sentence_boundary_penalty: int
    single_position_penalty: int


def build_cues(
    segments: Sequence[TranscriptSegment],
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE,
    *,
    max_lines: int = 2,
    line_length_gap: int = DEFAULT_LINE_LENGTH_GAP,
    max_cps: float = DEFAULT_MAX_CPS,
    min_duration: float = 0.8,
    max_duration: float = 7.0,
    pause_threshold: float = 0.8,
    audio_duration: float | None = None,
) -> list[Cue]:
    """Строит непересекающиеся cues из единого потока слов."""
    result = build_cues_with_diagnostics(
        segments,
        max_chars_per_line=max_chars_per_line,
        max_lines=max_lines,
        line_length_gap=line_length_gap,
        max_cps=max_cps,
        min_duration=min_duration,
        max_duration=max_duration,
        pause_threshold=pause_threshold,
        audio_duration=audio_duration,
    )
    return list(result.cues)


def build_cues_with_diagnostics(
    segments: Sequence[TranscriptSegment],
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE,
    *,
    max_lines: int = 2,
    line_length_gap: int = DEFAULT_LINE_LENGTH_GAP,
    max_cps: float = DEFAULT_MAX_CPS,
    min_duration: float = 0.8,
    max_duration: float = 7.0,
    pause_threshold: float = 0.8,
    audio_duration: float | None = None,
) -> CueBuildResult:
    """Строит cues и возвращает диагностику reconciliation без изменения сегментов."""
    _validate_settings(
        max_chars_per_line,
        max_lines,
        line_length_gap,
        max_cps,
        min_duration,
        max_duration,
        pause_threshold,
        audio_duration,
    )
    normalized_segments = _normalize_input_segments(segments, audio_duration)
    prepared = prepare_timed_words(normalized_segments)
    if not prepared.words:
        raise ValidationError("Невозможно построить SRT: распознанный текст пуст")

    hard_chars_per_line = max_chars_per_line + line_length_gap
    normalized_words, anchor_adjustments = _normalize_stretched_word_anchors(
        prepared.words,
        min_duration=min_duration,
        max_duration=max_duration,
        max_cps=max_cps,
    )
    layout_words = _expand_oversized_words(
        normalized_words,
        hard_chars_per_line,
    )
    sentences = build_timed_sentences(layout_words, pause_threshold)
    timed_lines = _pack_timed_sentences(
        sentences,
        max_chars_per_line=max_chars_per_line,
        hard_chars_per_line=hard_chars_per_line,
    )
    timing = _schedule_timing(
        timed_lines,
        max_lines=max_lines,
        min_duration=min_duration,
        max_duration=max_duration,
        max_cps=max_cps,
        pause_threshold=pause_threshold,
        audio_duration=audio_duration,
    )
    cues = _canonicalize_cues(
        tuple(
            Cue(index=index, start=cue.start, end=cue.end, text=cue.text)
            for index, cue in enumerate(timing.cues, start=1)
        )
    )
    diagnostics = replace(
        prepared.diagnostics,
        adjusted_boundaries=timing.adjusted_boundaries,
        max_boundary_drift_ms=timing.max_boundary_drift_ms,
        timing_anomaly_adjustments=(
            anchor_adjustments + timing.timing_anomaly_adjustments
        ),
        **_presentation_diagnostics(
            cues,
            hard_chars_per_line=hard_chars_per_line,
            max_cps=max_cps,
            max_duration=max_duration,
        ),
    )
    return CueBuildResult(cues=cues, diagnostics=diagnostics)


def build_srt(
    segments: Sequence[TranscriptSegment],
    *,
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE,
    max_lines: int = 2,
    line_length_gap: int = DEFAULT_LINE_LENGTH_GAP,
    max_cps: float = DEFAULT_MAX_CPS,
    min_duration: float = 0.8,
    max_duration: float = 7.0,
    pause_threshold: float = 0.8,
    audio_duration: float | None = None,
) -> str:
    """Строит и проверяет готовый текст SRT в памяти."""
    result = build_cues_with_diagnostics(
        segments,
        max_chars_per_line=max_chars_per_line,
        max_lines=max_lines,
        line_length_gap=line_length_gap,
        max_cps=max_cps,
        min_duration=min_duration,
        max_duration=max_duration,
        pause_threshold=pause_threshold,
        audio_duration=audio_duration,
    )
    from speech_to_sub.subtitles.validator import validate_cues

    validate_cues(
        result.cues,
        audio_duration=audio_duration,
        max_chars_per_line=(
            max_chars_per_line
            if result.diagnostics.line_length_target_exceeded_lines == 0
            else None
        ),
        max_lines=max_lines,
        line_length_gap=line_length_gap,
        max_cps=None,
    )
    return render_srt(result.cues)


def render_srt(cues: Sequence[Cue]) -> str:
    """Форматирует cues как SRT с LF и завершающей пустой строкой."""
    timestamps = _quantize_cue_timestamps(cues)
    blocks = [
        f"{cue.index}\n{_format_milliseconds(start)} --> "
        f"{_format_milliseconds(end)}\n{cue.text}"
        for cue, (start, end) in zip(cues, timestamps, strict=True)
    ]
    return "\n\n".join(blocks) + "\n" if blocks else ""


def format_timestamp(seconds: float) -> str:
    """Форматирует секунды как HH:MM:SS,mmm с округлением до миллисекунд."""
    return _format_milliseconds(_rounded_milliseconds(seconds))


def _rounded_milliseconds(seconds: float) -> int:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValidationError(
            "Временная метка SRT должна быть конечной и неотрицательной"
        )
    integral_seconds = math.floor(seconds)
    milliseconds = int(round((seconds - integral_seconds) * 1000))
    if milliseconds == 1000:
        integral_seconds += 1
        milliseconds = 0
    return integral_seconds * 1000 + milliseconds


def _format_milliseconds(total_milliseconds: int) -> str:
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def _quantize_cue_timestamps(
    cues: Sequence[Cue],
) -> tuple[tuple[int, int], ...]:
    """Квантует общую шкалу, сохраняя положительность и непересечение в SRT."""
    result: list[tuple[int, int]] = []
    previous_end = 0
    for cue in cues:
        start = max(previous_end, _rounded_milliseconds(cue.start))
        end = max(start + 1, _rounded_milliseconds(cue.end))
        result.append((start, end))
        previous_end = end
    return tuple(result)


def _canonicalize_cues(cues: Sequence[Cue]) -> tuple[Cue, ...]:
    """Возвращает те же реплики на точной миллисекундной шкале SRT."""
    return tuple(
        replace(cue, start=start / 1000, end=end / 1000)
        for cue, (start, end) in zip(
            cues,
            _quantize_cue_timestamps(cues),
            strict=True,
        )
    )


def _pack_timed_sentences(
    sentences: Sequence[TimedSentence],
    *,
    max_chars_per_line: int,
    hard_chars_per_line: int,
) -> tuple[TimedLine, ...]:
    """Упаковывает предложения в смысловые строки без анализа времени показа."""
    result: list[TimedLine] = []
    for sentence_index, sentence in enumerate(sentences):
        try:
            result.extend(
                _pack_timed_sentence(
                    sentence,
                    sentence_index=sentence_index,
                    max_chars_per_line=max_chars_per_line,
                    hard_chars_per_line=hard_chars_per_line,
                )
            )
        except ValidationError as error:
            raise ValidationError(
                f"{error} (предложение {sentence_index + 1}: "
                f"{sentence.start:.3f}–{sentence.end:.3f})"
            ) from error
    return tuple(result)


def _pack_timed_sentence(
    sentence: TimedSentence,
    *,
    sentence_index: int,
    max_chars_per_line: int,
    hard_chars_per_line: int,
) -> tuple[TimedLine, ...]:
    words = sentence.words
    states: list[_TextPartitionState | None] = [None] * (len(words) + 1)
    states[0] = _TextPartitionState((), 0, 0, 0)
    for start in range(len(words)):
        state = states[start]
        if state is None:
            continue
        _extend_text_partitions(
            states,
            words,
            start=start,
            state=state,
            sentence_index=sentence_index,
            max_chars_per_line=max_chars_per_line,
            hard_chars_per_line=hard_chars_per_line,
        )
    if states[-1] is None:
        raise ValidationError("Предложение невозможно разбить без повреждения слов")
    return states[-1].lines


def _extend_text_partitions(
    states: list[_TextPartitionState | None],
    words: Sequence[TranscriptWord],
    *,
    start: int,
    state: _TextPartitionState,
    sentence_index: int,
    max_chars_per_line: int,
    hard_chars_per_line: int,
) -> None:
    for end in range(start + 1, len(words) + 1):
        line = _text_candidate_line(
            words[start:end],
            hard_chars_per_line=hard_chars_per_line,
            sentence_index=sentence_index,
            sentence_end=end == len(words),
        )
        if line is None:
            break
        following = words[end] if end < len(words) else None
        candidate = _advance_text_partition(state, line, following, max_chars_per_line)
        current = states[end]
        if current is None or _text_partition_rank(candidate) < _text_partition_rank(
            current
        ):
            states[end] = candidate


def _text_candidate_line(
    words: Sequence[TranscriptWord],
    *,
    hard_chars_per_line: int,
    sentence_index: int,
    sentence_end: bool,
) -> TimedLine | None:
    text = _join_word_texts(words)
    oversized_single_word = len(words) == 1 and len(text) > hard_chars_per_line
    if len(text) > hard_chars_per_line and not oversized_single_word:
        return None
    return TimedLine(
        start=words[0].start,
        end=max(word.end for word in words),
        text=text,
        words=tuple(words),
        sentence_index=sentence_index,
        sentence_end=sentence_end,
    )


def _advance_text_partition(
    state: _TextPartitionState,
    line: TimedLine,
    following: TranscriptWord | None,
    preferred_chars_per_line: int,
) -> _TextPartitionState:
    syntax_penalty = _text_layout_penalty(line, following)
    overflow = max(0, len(line.text) - preferred_chars_per_line)
    characters = len(line.text)
    return _TextPartitionState(
        lines=(*state.lines, line),
        syntax_penalty=state.syntax_penalty + syntax_penalty,
        preferred_width_overflow=state.preferred_width_overflow + overflow,
        square_characters=state.square_characters + characters * characters,
    )


def _text_partition_rank(state: _TextPartitionState) -> tuple[int, ...]:
    return (
        len(state.lines),
        state.syntax_penalty,
        state.preferred_width_overflow,
        state.square_characters,
    )


def _text_layout_penalty(line: TimedLine, following: TranscriptWord | None) -> int:
    words = line.text.split()
    if len(words) == 1:
        penalty = 20
    elif len(words) == 2:
        penalty = 6
    else:
        penalty = 0
    if following is None:
        return penalty

    final_word = _plain_token(words[-1])
    next_word = _plain_token(following.text)
    if final_word in _WEAK_END_WORDS:
        return penalty + 6
    if line.text.rstrip().endswith(_GOOD_BOUNDARY_PUNCTUATION):
        return penalty
    if next_word in _COORDINATING_STARTS:
        return penalty + 1
    return penalty + 3


def _plain_token(text: str) -> str:
    return text.casefold().strip(".,!?…;:—-\"'«»“”’()[]{}")


def _expand_oversized_words(
    words: Sequence[TranscriptWord],
    capacity: int,
) -> tuple[TranscriptWord, ...]:
    result: list[TranscriptWord] = []
    for word in words:
        result.extend(_split_word_by_capacity(word, capacity))
    return tuple(result)


def _normalize_stretched_word_anchors(
    words: Sequence[TranscriptWord],
    *,
    min_duration: float,
    max_duration: float,
    max_cps: float,
) -> tuple[tuple[TranscriptWord, ...], int]:
    """Отбрасывает ложную начальную тишину у растянутых меток одиночных слов."""
    result: list[TranscriptWord] = []
    adjustments = 0
    for index, word in enumerate(words):
        if (
            word.end - word.start <= max_duration + 1e-9
            or _safe_duration_chunks(word.text, word.end - word.start, max_duration)
            is not None
        ):
            result.append(word)
            continue
        duration = min(
            max_duration,
            _required_duration(word.text, min_duration, max_cps),
        )
        following = words[index + 1] if index + 1 < len(words) else None
        anchor_end = word.end
        if (
            following is not None
            and word.start + duration <= following.start < anchor_end
        ):
            anchor_end = following.start
        result.append(
            TranscriptWord(
                _finite_start_before(anchor_end, duration),
                anchor_end,
                word.text,
                word.probability,
            )
        )
        adjustments += 1
    return tuple(result), adjustments


def _is_vocalization(text: str) -> bool:
    return text.count("-") >= 2


def _retime_word_run(words: Sequence[TranscriptWord]) -> tuple[TranscriptWord, ...]:
    weights = _timing_weights(tuple(word.text for word in words))
    total_weight = sum(weights)
    run_start = words[0].start
    run_end = words[-1].end
    elapsed = 0
    result: list[TranscriptWord] = []
    for index, (word, weight) in enumerate(zip(words, weights, strict=True)):
        start = run_start + (run_end - run_start) * elapsed / total_weight
        elapsed += weight
        end = (
            run_end
            if index == len(words) - 1
            else run_start + (run_end - run_start) * elapsed / total_weight
        )
        result.append(TranscriptWord(start, end, word.text, word.probability))
    return tuple(result)


def _split_word_by_duration(
    word: TranscriptWord,
    max_duration: float,
) -> tuple[TranscriptWord, ...]:
    duration = word.end - word.start
    if duration <= max_duration + 1e-9:
        return (word,)
    chunks = _safe_duration_chunks(word.text, duration, max_duration)
    if chunks is None:
        # У одиночного обычного слова нет корректной внутренней границы. Такой
        # интервал является аномалией alignment; сохраняем слово целиком и
        # отбрасываем только приписанную ему начальную тишину.
        return (
            TranscriptWord(
                _finite_start_before(word.end, max_duration),
                word.end,
                word.text,
                word.probability,
            ),
        )
    return _timed_text_chunks(word, chunks)


def _safe_duration_chunks(
    text: str,
    duration: float,
    max_duration: float,
) -> tuple[str, ...] | None:
    cleaned = " ".join(text.split())
    if not cleaned:
        return None
    maximum_span = len(cleaned) * max_duration / duration
    boundaries = tuple(
        index + 1
        for index, character in enumerate(cleaned)
        if character == "-" or character.isspace()
    )
    chunks: list[str] = []
    start = 0
    while len(cleaned) - start > maximum_span + 1e-9:
        eligible = [
            boundary
            for boundary in boundaries
            if start < boundary < len(cleaned)
            and boundary - start <= maximum_span + 1e-9
        ]
        if not eligible:
            return None
        end = max(eligible)
        chunk = cleaned[start:end].strip()
        if not chunk:
            return None
        chunks.append(chunk)
        start = end
    tail = cleaned[start:].strip()
    if not tail:
        return None
    chunks.append(tail)
    return tuple(chunks) if len(chunks) > 1 else None


def _split_word_by_capacity(
    word: TranscriptWord,
    capacity: int,
) -> tuple[TranscriptWord, ...]:
    if visible_character_count(word.text) <= capacity:
        return (word,)
    return _timed_text_chunks(word, _split_text_by_capacity(word.text, capacity))


def _timed_text_chunks(
    word: TranscriptWord,
    chunks: Sequence[str],
) -> tuple[TranscriptWord, ...]:
    weights = _timing_weights(chunks)
    total_weight = sum(weights)
    elapsed = 0
    result: list[TranscriptWord] = []
    for index, (chunk, weight) in enumerate(zip(chunks, weights, strict=True)):
        start = word.start + (word.end - word.start) * elapsed / total_weight
        elapsed += weight
        end = (
            word.end
            if index == len(chunks) - 1
            else word.start + (word.end - word.start) * elapsed / total_weight
        )
        result.append(TranscriptWord(start, end, chunk, word.probability))
    return tuple(result)


def _timing_weights(chunks: Sequence[str]) -> list[int]:
    weights: list[int] = []
    for index, chunk in enumerate(chunks):
        separator = int(index < len(chunks) - 1 and not chunk.rstrip().endswith("-"))
        weights.append(max(1, visible_character_count(chunk) + separator))
    return weights


def _finite_start_before(end: float, duration: float) -> float:
    start = max(0.0, end - duration)
    if start < end:
        return start
    return max(0.0, math.nextafter(end, -math.inf))


def _split_text_by_capacity(text: str, capacity: int) -> tuple[str, ...]:
    remaining = text.strip()
    chunks: list[str] = []
    while len(remaining) > capacity:
        split_at = remaining.rfind(" ", 1, capacity + 1)
        if split_at < 1:
            hyphen_at = remaining.rfind("-", 1, capacity)
            if hyphen_at < 1:
                return (text.strip(),)
            split_at = hyphen_at + 1
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return tuple(chunks)


def _normalize_timing_anomalies(
    lines: Sequence[TimedLine],
    max_duration: float,
) -> tuple[tuple[TimedLine, ...], int]:
    redistributed, adjustments = _redistribute_vocalization_lines(lines, max_duration)
    result: list[TimedLine] = []
    for line in redistributed:
        if line.end - line.start <= max_duration + 1e-9:
            result.append(line)
            continue
        if len(line.words) != 1:
            result.append(line)
            continue
        chunks = _split_word_by_duration(line.words[0], max_duration)
        if len(chunks) == 1:
            word = chunks[0]
            result.append(
                TimedLine(
                    line.text,
                    (word,),
                    word.start,
                    word.end,
                    line.sentence_index,
                    line.sentence_end,
                )
            )
        else:
            result.extend(
                TimedLine(
                    word.text,
                    (word,),
                    word.start,
                    word.end,
                    line.sentence_index,
                    True,
                )
                for word in chunks
            )
        adjustments += 1
    return tuple(result), adjustments


def _redistribute_vocalization_lines(
    lines: Sequence[TimedLine],
    max_duration: float,
) -> tuple[tuple[TimedLine, ...], int]:
    result = list(lines)
    adjustments = 0
    index = 0
    while index < len(result):
        word = _single_line_word(result[index])
        if (
            word is None
            or word.end - word.start <= max_duration
            or not _is_vocalization(word.text)
        ):
            index += 1
            continue
        run_end = _vocalization_line_run_end(result, index)
        if run_end == index + 1:
            index += 1
            continue
        normalized = _retime_word_run(
            tuple(_single_line_word(line) for line in result[index:run_end])  # type: ignore[arg-type]
        )
        for line_index, normalized_word in zip(
            range(index, run_end),
            normalized,
            strict=True,
        ):
            line = result[line_index]
            result[line_index] = TimedLine(
                line.text,
                (normalized_word,),
                normalized_word.start,
                normalized_word.end,
                line.sentence_index,
                line.sentence_end,
            )
        adjustments += 1
        index = run_end
    return tuple(result), adjustments


def _vocalization_line_run_end(lines: Sequence[TimedLine], start: int) -> int:
    index = start + 1
    previous = _single_line_word(lines[start])
    while previous is not None and index < len(lines):
        current = _single_line_word(lines[index])
        if (
            current is None
            or not _is_vocalization(current.text)
            or current.end - current.start > 0.05
            or current.start - previous.end > 0.05
        ):
            break
        previous = current
        index += 1
    return index


def _single_line_word(line: TimedLine) -> TranscriptWord | None:
    return line.words[0] if len(line.words) == 1 else None


def _group_timed_lines(
    lines: Sequence[TimedLine],
    *,
    max_lines: int,
    pause_threshold: float,
) -> tuple[_RawCue, ...]:
    """Глобально группирует строки, не оставляя короткий хвост справа."""
    states: list[_CueGroupingState | None] = [None] * (len(lines) + 1)
    states[0] = _CueGroupingState((), 0, 0, 0)
    for start in range(len(lines)):
        _expand_cue_grouping_state(
            states,
            lines,
            start=start,
            max_lines=max_lines,
            pause_threshold=pause_threshold,
        )
    final = states[-1]
    if final is None:
        # Одна строка всегда допустима, поэтому сюда можно попасть только при
        # внутренней ошибке алгоритма, а не из-за содержимого распознавания.
        return tuple(_cue_from_lines((line,)) for line in lines)
    return tuple(_cue_from_lines(group) for group in final.groups)


def _expand_cue_grouping_state(
    states: list[_CueGroupingState | None],
    lines: Sequence[TimedLine],
    *,
    start: int,
    max_lines: int,
    pause_threshold: float,
) -> None:
    state = states[start]
    if state is None:
        return
    for size in range(1, max_lines + 1):
        end = start + size
        if end > len(lines):
            return
        group = tuple(lines[start:end])
        if not _line_group_is_allowed(group, pause_threshold):
            continue
        candidate = _advance_cue_grouping(state, group, start)
        current = states[end]
        if current is None or _cue_grouping_rank(candidate) < _cue_grouping_rank(
            current
        ):
            states[end] = candidate


def _line_group_is_allowed(
    lines: Sequence[TimedLine],
    pause_threshold: float,
) -> bool:
    if len(lines) < 2:
        return True
    left, right = lines[-2:]
    if left.sentence_index == right.sentence_index:
        return True
    return right.start - left.end < min(pause_threshold, _SENTENCE_PAIR_MAX_GAP)


def _advance_cue_grouping(
    state: _CueGroupingState,
    group: tuple[TimedLine, ...],
    start: int,
) -> _CueGroupingState:
    words = sum(len(line.text.split()) for line in group)
    orphan_penalty = 0
    if len(group) == 1:
        if words == 1:
            orphan_penalty = 10
        elif words == 2:
            orphan_penalty = 3
    crosses_sentence = int(
        len(group) > 1 and group[0].sentence_index != group[-1].sentence_index
    )
    return _CueGroupingState(
        groups=(*state.groups, group),
        orphan_penalty=state.orphan_penalty + orphan_penalty,
        sentence_boundary_penalty=(state.sentence_boundary_penalty + crosses_sentence),
        single_position_penalty=(
            state.single_position_penalty + (start if len(group) == 1 else 0)
        ),
    )


def _cue_grouping_rank(state: _CueGroupingState) -> tuple[int, ...]:
    return (
        len(state.groups),
        state.orphan_penalty,
        state.sentence_boundary_penalty,
        -state.single_position_penalty,
    )


def _cue_from_lines(lines: Sequence[TimedLine]) -> _RawCue:
    return _RawCue(
        start=lines[0].start,
        end=max(line.end for line in lines),
        text="\n".join(line.text for line in lines),
        words=tuple(word for line in lines for word in line.words),
    )


def _schedule_timing(
    lines: Sequence[TimedLine],
    *,
    max_lines: int,
    min_duration: float,
    max_duration: float,
    max_cps: float,
    pause_threshold: float,
    audio_duration: float | None,
) -> _ScheduledTiming:
    normalized, anomaly_adjustments = _normalize_timing_anomalies(lines, max_duration)
    cues = _group_timed_lines(
        normalized,
        max_lines=max_lines,
        pause_threshold=pause_threshold,
    )
    try:
        scheduled, adjusted, max_drift_ms = _schedule_target_timing(
            cues,
            min_duration=min_duration,
            max_duration=max_duration,
            max_cps=max_cps,
            pause_threshold=pause_threshold,
            audio_duration=audio_duration,
        )
    except ValidationError:
        scheduled = _schedule_anchor_fallback(cues, audio_duration)
        if len(scheduled) == len(cues):
            blocks = _timing_block_ranges(cues, pause_threshold)
            adjusted, max_drift_ms = _boundary_diagnostics(cues, scheduled, blocks)
        else:
            adjusted, max_drift_ms = 0, 0
    return _ScheduledTiming(
        scheduled,
        adjusted,
        max_drift_ms,
        anomaly_adjustments,
    )


def _schedule_target_timing(
    cues: Sequence[_RawCue],
    *,
    min_duration: float,
    max_duration: float,
    max_cps: float,
    pause_threshold: float,
    audio_duration: float | None,
) -> tuple[tuple[_RawCue, ...], int, int]:
    """Пытается назначить целевое время; содержимое не зависит от успеха попытки."""
    durations = tuple(
        max(
            min(cue.end - cue.start, max_duration),
            _required_duration(cue.text, min_duration, max_cps),
        )
        for cue in cues
    )
    _validate_scheduled_durations(cues, durations, max_duration, max_cps)
    block_ranges = _timing_block_ranges(cues, pause_threshold)
    ranges = _bounded_start_ranges(
        cues,
        durations,
        block_ranges,
        audio_duration,
    )
    starts = _non_overlapping_starts(
        ranges,
        durations,
        tuple(cue.start for cue in cues),
        max_cps,
    )
    scheduled = tuple(
        _RawCue(start, start + duration, cue.text, cue.words)
        for cue, start, duration in zip(cues, starts, durations, strict=True)
    )
    scheduled = _remove_floating_overlaps(scheduled)
    if not _cue_timing_is_structural(scheduled):
        raise ValidationError("Точность временной шкалы не позволяет назначить реплики")
    adjusted, max_drift_ms = _boundary_diagnostics(cues, scheduled, block_ranges)
    return scheduled, adjusted, max_drift_ms


def _remove_floating_overlaps(cues: Sequence[_RawCue]) -> tuple[_RawCue, ...]:
    """Устраняет машинную погрешность на общей границе соседних реплик."""
    result: list[_RawCue] = []
    previous_end = 0.0
    for cue in cues:
        start = max(previous_end, cue.start)
        end = max(cue.end, start + 1e-9)
        result.append(_RawCue(start, end, cue.text, cue.words))
        previous_end = end
    return tuple(result)


def _schedule_anchor_fallback(
    cues: Sequence[_RawCue],
    audio_duration: float | None,
) -> tuple[_RawCue, ...]:
    """Назначает монотонное время по якорям без ограничений чтения и длительности."""
    if not cues:
        return ()
    natural_end = max(cue.end for cue in cues)
    timeline_end = max(
        _MIN_RENDERED_CUE_DURATION,
        audio_duration if audio_duration is not None else natural_end,
    )
    required_for_all_cues = len(cues) * _MIN_RENDERED_CUE_DURATION
    maximum_cues = (
        len(cues)
        if timeline_end >= required_for_all_cues
        else max(1, int(timeline_end // _MIN_RENDERED_CUE_DURATION))
    )
    compacted = _compact_cues(cues, maximum_cues)
    cue_count = len(compacted)
    minimum_span = cue_count * _MIN_RENDERED_CUE_DURATION
    timeline_start = min(
        max(0.0, compacted[0].start),
        max(0.0, timeline_end - minimum_span),
    )
    final_anchor = min(timeline_end, max(timeline_start, compacted[-1].end))
    final_end = max(final_anchor, timeline_start + minimum_span)
    final_end = min(timeline_end, final_end)

    boundaries = [timeline_start]
    for index in range(cue_count - 1):
        desired = (compacted[index].end + compacted[index + 1].start) / 2
        minimum = boundaries[-1] + _MIN_RENDERED_CUE_DURATION
        remaining = cue_count - index - 1
        maximum = final_end - remaining * _MIN_RENDERED_CUE_DURATION
        boundaries.append(min(max(desired, minimum), maximum))
    boundaries.append(final_end)
    scheduled = tuple(
        _RawCue(
            start=boundaries[index],
            end=boundaries[index + 1],
            text=cue.text,
            words=cue.words,
        )
        for index, cue in enumerate(compacted)
    )
    if _cue_timing_is_structural(scheduled):
        return scheduled
    return _schedule_evenly_from_zero(compacted, timeline_end)


def _schedule_evenly_from_zero(
    cues: Sequence[_RawCue],
    timeline_end: float,
) -> tuple[_RawCue, ...]:
    step = timeline_end / len(cues)
    scheduled = tuple(
        _RawCue(
            start=step * index,
            end=timeline_end if index + 1 == len(cues) else step * (index + 1),
            text=cue.text,
            words=cue.words,
        )
        for index, cue in enumerate(cues)
    )
    if _cue_timing_is_structural(scheduled):
        return scheduled
    return (
        _RawCue(
            start=0.0,
            end=timeline_end,
            text=" ".join(cue.text.replace("\n", " ") for cue in cues),
            words=tuple(word for cue in cues for word in cue.words),
        ),
    )


def _cue_timing_is_structural(cues: Sequence[_RawCue]) -> bool:
    previous_end = 0.0
    for cue in cues:
        if (
            not math.isfinite(cue.start)
            or not math.isfinite(cue.end)
            or cue.start < previous_end
            or cue.end <= cue.start
        ):
            return False
        previous_end = cue.end
    return True


def _compact_cues(
    cues: Sequence[_RawCue],
    maximum_cues: int,
) -> tuple[_RawCue, ...]:
    """Сжимает только экстремально короткую шкалу, не теряя текст и слова."""
    if len(cues) <= maximum_cues:
        return tuple(cues)
    result: list[_RawCue] = []
    for group_index in range(maximum_cues):
        start = group_index * len(cues) // maximum_cues
        end = (group_index + 1) * len(cues) // maximum_cues
        group = cues[start:end]
        result.append(
            _RawCue(
                start=group[0].start,
                end=max(cue.end for cue in group),
                text=" ".join(cue.text.replace("\n", " ") for cue in group),
                words=tuple(word for cue in group for word in cue.words),
            )
        )
    return tuple(result)


def _timing_block_ranges(
    cues: Sequence[_RawCue],
    pause_threshold: float,
) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(len(cues) - 1):
        if cues[index + 1].start - cues[index].end >= pause_threshold:
            ranges.append((start, index))
            start = index + 1
    ranges.append((start, len(cues) - 1))
    return tuple(ranges)


def _bounded_start_ranges(
    cues: Sequence[_RawCue],
    durations: Sequence[float],
    blocks: Sequence[tuple[int, int]],
    audio_duration: float | None,
) -> tuple[tuple[float, float], ...]:
    result: list[tuple[float, float]] = []
    for block_index, (first, last) in enumerate(blocks):
        lower_limit, deadline = _timing_block_limits(
            cues,
            blocks,
            block_index,
            audio_duration,
        )
        result.extend(
            _bounded_block_start_ranges(
                cues,
                durations,
                first=first,
                last=last,
                lower_limit=lower_limit,
                deadline=deadline,
                audio_duration=audio_duration,
            )
        )
    return tuple(result)


def _timing_block_limits(
    cues: Sequence[_RawCue],
    blocks: Sequence[tuple[int, int]],
    block_index: int,
    audio_duration: float | None,
) -> tuple[float, float | None]:
    first = blocks[block_index][0]
    lower_limit = cues[first - 1].end if first else 0.0
    if block_index + 1 < len(blocks):
        deadline = cues[blocks[block_index + 1][0]].start
    else:
        deadline = audio_duration
    return lower_limit, deadline


def _bounded_block_start_ranges(
    cues: Sequence[_RawCue],
    durations: Sequence[float],
    *,
    first: int,
    last: int,
    lower_limit: float,
    deadline: float | None,
    audio_duration: float | None,
) -> tuple[tuple[float, float], ...]:
    result: list[tuple[float, float]] = []
    for index in range(first, last + 1):
        lower, upper = _safe_start_range(
            cues[index],
            durations[index],
            index=index,
            audio_duration=audio_duration,
        )
        if index == first:
            lower = max(lower, lower_limit)
        if index == last and deadline is not None:
            upper = min(upper, deadline - durations[index])
        if upper + 1e-9 < lower:
            raise ValidationError("Требуемый сдвиг границы реплики превышает 1 секунду")
        result.append((lower, upper))
    return tuple(result)


def _safe_start_range(
    cue: _RawCue,
    duration: float,
    *,
    index: int,
    audio_duration: float | None,
) -> tuple[float, float]:
    try:
        return _start_range(
            cue,
            duration,
            audio_duration,
            start_drift=_MAX_BOUNDARY_DRIFT,
            end_drift=_MAX_BOUNDARY_DRIFT,
        )
    except ValidationError as error:
        raise ValidationError(
            f"{error}: реплика {index + 1} "
            f"({cue.start:.3f}–{cue.end:.3f}, показ {duration:.3f} с)"
        ) from error


def _boundary_diagnostics(
    raw: Sequence[_RawCue],
    scheduled: Sequence[_RawCue],
    blocks: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    drifts: list[float] = []
    for first, last in blocks:
        for index in range(first, last):
            drifts.extend(
                _internal_boundary_drifts(
                    raw[index],
                    raw[index + 1],
                    scheduled[index],
                    scheduled[index + 1],
                )
            )
    adjusted = tuple(drift for drift in drifts if drift > 1e-9)
    maximum = max(adjusted, default=0.0)
    return len(adjusted), _duration_milliseconds(maximum)


def _internal_boundary_drifts(
    raw_left: _RawCue,
    raw_right: _RawCue,
    scheduled_left: _RawCue,
    scheduled_right: _RawCue,
) -> tuple[float, float]:
    """Возвращает два слота сдвига для внутренней границы."""
    lower_anchor = min(raw_left.end, raw_right.start)
    upper_anchor = max(raw_left.end, raw_right.start)
    end_drift = _distance_from_interval(
        scheduled_left.end,
        lower_anchor,
        upper_anchor,
    )
    start_drift = _distance_from_interval(
        scheduled_right.start,
        lower_anchor,
        upper_anchor,
    )
    if math.isclose(
        scheduled_left.end,
        scheduled_right.start,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        return max(end_drift, start_drift), 0.0
    return end_drift, start_drift


def _distance_from_interval(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower - value
    if value > upper:
        return value - upper
    return 0.0


def _required_duration(text: str, min_duration: float, max_cps: float) -> float:
    return max(
        min_duration + _TIMING_MARGIN,
        visible_character_count(text) / max_cps + _TIMING_MARGIN,
    )


def _validate_scheduled_durations(
    cues: Sequence[_RawCue],
    durations: Sequence[float],
    max_duration: float,
    max_cps: float,
) -> None:
    for index, (cue, duration) in enumerate(zip(cues, durations, strict=True), start=1):
        if duration <= max_duration + 1e-9:
            continue
        raise ValidationError(
            "Текстовая разметка физически не помещается: "
            f"реплика {index} ({cue.start:.3f}–{cue.end:.3f}) требует "
            f"{duration:.3f} с при лимите {max_duration:g} с и {max_cps:g} CPS; "
            "синтаксическая граница не изменена"
        )


def _start_range(
    cue: _RawCue,
    duration: float,
    audio_duration: float | None,
    *,
    start_drift: float,
    end_drift: float,
) -> tuple[float, float]:
    lower = max(0.0, cue.end - duration - end_drift)
    upper = cue.start + start_drift
    if audio_duration is not None:
        upper = min(upper, audio_duration - duration)
    if upper + 1e-9 < lower:
        raise ValidationError(
            "Невозможно выдержать лимит CPS в границах длительности аудио"
        )
    return lower, upper


def _non_overlapping_starts(
    ranges: Sequence[tuple[float, float]],
    durations: Sequence[float],
    preferred_starts: Sequence[float],
    max_cps: float,
) -> tuple[float, ...]:
    latest = [0.0] * len(ranges)
    next_start: float | None = None
    for index in range(len(ranges) - 1, -1, -1):
        lower, upper = ranges[index]
        if next_start is not None:
            upper = min(upper, next_start - durations[index])
        if upper + 1e-9 < lower:
            raise ValidationError(
                f"Невозможно выдержать лимит {max_cps:g} CPS без пересечения "
                f"реплик около позиции {index + 1}"
            )
        latest[index] = upper
        next_start = upper
    result = [0.0] * len(ranges)
    previous_end = 0.0
    for index, ((lower, _), preferred) in enumerate(
        zip(ranges, preferred_starts, strict=True)
    ):
        earliest = max(lower, previous_end)
        if earliest > latest[index] + 1e-9:
            raise ValidationError(
                f"Невозможно выдержать лимит {max_cps:g} CPS без пересечения реплик"
            )
        result[index] = min(max(preferred, earliest), latest[index])
        previous_end = result[index] + durations[index]
    return tuple(result)


def _join_word_texts(words: Sequence[TranscriptWord]) -> str:
    result = ""
    for word in words:
        token = word.text.strip()
        if not token:
            continue
        if not result or result.endswith("-"):
            result += token
        else:
            result += f" {token}"
    return result


def _normalize_input_segments(
    segments: Sequence[TranscriptSegment],
    audio_duration: float | None,
) -> tuple[TranscriptSegment, ...]:
    """Сохраняет распознанный текст даже при повреждённых границах сегмента."""
    content: list[tuple[TranscriptSegment, str]] = []
    for segment in segments:
        text = " ".join(segment.text.split()) or _join_word_texts(segment.words)
        if text:
            content.append((segment, text))
    if not content:
        return ()

    valid_ends = [
        segment.end
        for segment, _ in content
        if _interval_is_valid(segment.start, segment.end)
    ]
    horizon = audio_duration or max(valid_ends, default=float(len(content)))
    horizon = max(horizon, _MIN_RENDERED_CUE_DURATION)
    intervals_are_usable = _segment_intervals_are_usable(content, audio_duration)
    if intervals_are_usable:
        return tuple(replace(segment, text=text) for segment, text in content)

    normalized: list[TranscriptSegment] = []
    for index, (segment, text) in enumerate(content):
        start = horizon * index / len(content)
        end = horizon * (index + 1) / len(content)
        normalized.append(
            TranscriptSegment(
                start=start,
                end=max(end, start + horizon / (len(content) * 2)),
                text=text,
                words=(),
                segment_id=segment.segment_id,
            )
        )
    return tuple(normalized)


def _segment_intervals_are_usable(
    content: Sequence[tuple[TranscriptSegment, str]],
    audio_duration: float | None,
) -> bool:
    for segment, _ in content:
        if not _interval_is_valid(segment.start, segment.end):
            return False
        if audio_duration is not None and segment.end > audio_duration + 1e-9:
            return False
    return True


def _presentation_diagnostics(
    cues: Sequence[Cue],
    *,
    hard_chars_per_line: int,
    max_cps: float,
    max_duration: float,
) -> dict[str, int | float]:
    rendered_timestamps = _quantize_cue_timestamps(cues)
    durations_ms = tuple(end - start for start, end in rendered_timestamps)
    actual_cps = tuple(
        visible_character_count(cue.text) * 1000 / duration_ms
        for cue, duration_ms in zip(cues, durations_ms, strict=True)
    )
    line_lengths = tuple(len(line) for cue in cues for line in cue.text.splitlines())
    return {
        "reading_speed_target_exceeded_cues": sum(
            value > max_cps
            and not math.isclose(value, max_cps, rel_tol=1e-9, abs_tol=1e-9)
            for value in actual_cps
        ),
        "max_actual_cps": round(max(actual_cps, default=0.0), 6),
        "duration_target_exceeded_cues": sum(
            duration_ms > _duration_milliseconds(max_duration)
            for duration_ms in durations_ms
        ),
        "max_actual_duration_ms": max(durations_ms, default=0),
        "line_length_target_exceeded_lines": sum(
            length > hard_chars_per_line for length in line_lengths
        ),
        "max_actual_line_length": max(line_lengths, default=0),
    }


def _interval_is_valid(start: float, end: float) -> bool:
    return math.isfinite(start) and math.isfinite(end) and start >= 0 and end > start


def _duration_milliseconds(duration: float) -> int:
    milliseconds = duration * 1000
    if math.isfinite(milliseconds):
        return int(round(milliseconds))
    return int(duration) * 1000


def _validate_settings(
    max_chars_per_line: int,
    max_lines: int,
    line_length_gap: int,
    max_cps: float,
    min_duration: float,
    max_duration: float,
    pause_threshold: float,
    audio_duration: float | None,
) -> None:
    if max_chars_per_line < 1 or max_lines not in (1, 2):
        raise ValidationError("SRT допускает одну или две строки положительной длины")
    if (
        isinstance(line_length_gap, bool)
        or not isinstance(line_length_gap, int)
        or line_length_gap < 0
        or line_length_gap > MAX_LINE_LENGTH_GAP
    ):
        raise ValidationError(
            f"Допуск длины строки должен быть от 0 до {MAX_LINE_LENGTH_GAP} символов"
        )
    if not math.isfinite(max_cps) or max_cps <= 0:
        raise ValidationError("Лимит CPS должен быть положительным конечным числом")
    if min_duration <= 0 or max_duration <= 0 or min_duration > max_duration:
        raise ValidationError("Некорректный диапазон длительности SRT-реплики")
    if pause_threshold < 0:
        raise ValidationError("Порог паузы не может быть отрицательным")
    if audio_duration is not None and (
        not math.isfinite(audio_duration) or audio_duration <= 0
    ):
        raise ValidationError("Длительность аудио должна быть положительной")
