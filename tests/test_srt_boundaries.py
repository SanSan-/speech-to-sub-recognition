from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

import pytest

from speech_to_sub.models import TranscriptSegment, TranscriptWord
from speech_to_sub.subtitles.builder import (
    Cue,
    CueBuildResult,
    build_cues_with_diagnostics,
    build_srt,
    render_srt,
)
from speech_to_sub.subtitles.layout import visible_character_count
from speech_to_sub.subtitles.validator import validate_cues, validate_srt


TARGET_CPS = 17.0
MAX_DURATION = 7.0
BASE_WIDTH = 42
WIDTH_GAP = 8
PUNCTUATION = (".", "?", "!", "…", "?!")
WIDTHS = (41, 42, 43, 49, 50, 51)


@dataclass(frozen=True, slots=True)
class GeneratedCase:
    rate: int
    gap: float
    sentence_count: int
    index: int

    @property
    def test_id(self) -> str:
        return (
            f"rate-{self.rate}-gap-{str(self.gap).replace('.', 'p')}-"
            f"sentences-{self.sentence_count}"
        )


GENERATED_CASES = tuple(
    GeneratedCase(rate, gap, count, index)
    for index, (rate, gap, count) in enumerate(
        (rate, gap, count)
        for rate in (5, 17, 25, 40, 80)
        for gap in (0.0, 0.001, 0.799, 0.8, 0.801, 1.3, 2.6)
        for count in range(1, 9)
    )
)


def test_generated_boundary_matrix_definition_is_complete() -> None:
    assert len(GENERATED_CASES) == 280
    assert {case.rate for case in GENERATED_CASES} == {5, 17, 25, 40, 80}
    assert {case.gap for case in GENERATED_CASES} == {
        0.0,
        0.001,
        0.799,
        0.8,
        0.801,
        1.3,
        2.6,
    }
    assert {case.sentence_count for case in GENERATED_CASES} == set(range(1, 9))
    assert {(case.index // 4) % 4 + 1 for case in GENERATED_CASES} == {1, 2, 3, 4}


@pytest.mark.parametrize(
    "case",
    GENERATED_CASES,
    ids=[case.test_id for case in GENERATED_CASES],
)
def test_generated_dense_boundary_matrix_never_loses_valid_asr_text(
    case: GeneratedCase,
) -> None:
    segments, expected, audio_duration = _generated_segments(case)
    max_lines = 1 if case.index % 5 == 0 else 2

    result = build_cues_with_diagnostics(
        segments,
        max_chars_per_line=BASE_WIDTH,
        max_lines=max_lines,
        line_length_gap=WIDTH_GAP,
        max_cps=TARGET_CPS,
        min_duration=0.8,
        max_duration=MAX_DURATION,
        pause_threshold=0.8,
        audio_duration=audio_duration,
    )

    _assert_lossless_result(
        result,
        expected=expected,
        audio_duration=audio_duration,
        max_lines=max_lines,
    )


def test_joins_short_tail_across_pause() -> None:
    text = "И люди по-разному реагируют."
    words = (
        TranscriptWord(389.560, 389.800, "И"),
        TranscriptWord(389.800, 390.200, "люди"),
        TranscriptWord(390.200, 390.920, "по-разному"),
        TranscriptWord(392.220, 393.022, "реагируют."),
    )

    result = build_cues_with_diagnostics(
        (TranscriptSegment(389.560, 393.022, text, words=words),),
        audio_duration=393.022,
    )

    assert [cue.text for cue in result.cues] == [text]
    _assert_lossless_result(result, expected=text, audio_duration=393.022)


def test_keeps_orphan_prefix_with_its_continuation() -> None:
    text = (
        "Он имеет сосредотачивать на игрушке свой взгляд "
        "и также очень интересуется нашим лицом."
    )
    words = [TranscriptWord(56.990, 57.792, "Он")]
    words.extend(_proportional_words(" ".join(text.split()[1:]), 59.760, 66.760))

    result = build_cues_with_diagnostics(
        (TranscriptSegment(56.990, 66.760, text, words=tuple(words)),),
        audio_duration=66.760,
    )

    assert all(cue.text.strip() != "Он" for cue in result.cues)
    assert result.cues[0].text.startswith("Он имеет")
    _assert_lossless_result(result, expected=text, audio_duration=66.760)


def test_keeps_unfinished_phrase_across_long_pause() -> None:
    text = "А она на границе у меня,"
    words = (
        TranscriptWord(450.450, 450.750, "А"),
        TranscriptWord(450.750, 451.150, "она"),
        TranscriptWord(451.150, 451.610, "на"),
        TranscriptWord(454.200, 454.900, "границе"),
        TranscriptWord(454.900, 455.300, "у"),
        TranscriptWord(455.300, 456.020, "меня,"),
    )

    result = build_cues_with_diagnostics(
        (TranscriptSegment(450.450, 456.020, text, words=words),),
        audio_duration=456.020,
    )

    assert [cue.text for cue in result.cues] == [text]
    _assert_lossless_result(result, expected=text, audio_duration=456.020)


def test_rebalances_single_word_sentence_tail_to_the_right() -> None:
    text = (
        "И как раз на второй стадии происходит вот это обвинение врачей, "
        "родственников, судьбу."
    )
    segment = _segment(501.820, 510.962, text)

    result = build_cues_with_diagnostics((segment,), audio_duration=510.962)

    lines = [line for cue in result.cues for line in cue.text.splitlines()]
    assert "судьбу." not in lines
    assert lines[-1].endswith("родственников, судьбу.")
    _assert_lossless_result(result, expected=text, audio_duration=510.962)


def test_dense_completed_sentences_share_two_line_cues() -> None:
    text = "Раз. Два. Три. Четыре."
    words = (
        TranscriptWord(0.0, 0.6, "Раз."),
        TranscriptWord(0.6, 1.2, "Два."),
        TranscriptWord(1.2, 1.8, "Три."),
        TranscriptWord(1.8, 2.4, "Четыре."),
    )

    result = build_cues_with_diagnostics(
        (TranscriptSegment(0.0, 2.4, text, words=words),),
        audio_duration=2.4,
    )

    assert [cue.text for cue in result.cues] == ["Раз.\nДва.", "Три.\nЧетыре."]
    _assert_lossless_result(result, expected=text, audio_duration=2.4)


def test_exact_pause_threshold_separates_completed_sentences() -> None:
    words = (
        TranscriptWord(0.0, 0.5, "Первое."),
        TranscriptWord(1.3, 1.8, "Второе."),
    )
    result = build_cues_with_diagnostics(
        (TranscriptSegment(0.0, 1.8, "Первое. Второе.", words=words),),
        max_lines=2,
        pause_threshold=0.8,
        audio_duration=2.0,
    )

    assert [cue.text for cue in result.cues] == ["Первое.", "Второе."]


@pytest.mark.parametrize("max_lines", [1, 2])
def test_max_lines_changes_only_cue_grouping(max_lines: int) -> None:
    text = "Первое предложение. Второе предложение. Третье предложение."
    result = build_cues_with_diagnostics(
        (_segment(0.0, 2.0, text),),
        max_lines=max_lines,
        audio_duration=2.0,
    )

    _assert_lossless_result(
        result,
        expected=text,
        audio_duration=2.0,
        max_lines=max_lines,
    )


def test_unattainable_reading_speed_is_diagnostic() -> None:
    text = "Очень много текста произнесено почти мгновенно."
    result = build_cues_with_diagnostics(
        (_segment(0.0, 0.05, text),),
        max_cps=3.0,
        audio_duration=0.05,
    )

    assert result.diagnostics.reading_speed_target_exceeded_cues > 0
    _assert_lossless_result(
        result,
        expected=text,
        audio_duration=0.05,
        max_cps=3.0,
    )


def test_long_speech_anchor_is_diagnostic() -> None:
    text = "Одна смысловая строка."
    result = build_cues_with_diagnostics(
        (_segment(0.0, 20.0, text),),
        max_duration=7.0,
        audio_duration=20.0,
    )

    assert result.diagnostics.duration_target_exceeded_cues == 1
    _assert_lossless_result(result, expected=text, audio_duration=20.0)


def test_unbreakable_long_token_is_preserved_and_diagnosed() -> None:
    text = "сверхдлинноесловобезразрешённыхграницпереносакотороенельзярезать"
    result = build_cues_with_diagnostics(
        (_segment(0.0, 1.0, text),),
        audio_duration=1.0,
    )

    assert [cue.text for cue in result.cues] == [text]
    assert result.diagnostics.line_length_target_exceeded_lines == 1
    _assert_lossless_result(result, expected=text, audio_duration=1.0)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (math.nan, 0.5),
        (0.0, math.inf),
        (-1.0, 0.5),
        (0.5, 0.5),
        (0.8, 0.2),
    ],
)
def test_invalid_word_anchor_falls_back_to_segment_text(
    start: float, end: float
) -> None:
    text = "Текст сохранён."
    segment = TranscriptSegment(
        0.0,
        1.0,
        text,
        words=(TranscriptWord(start, end, text),),
    )

    result = build_cues_with_diagnostics((segment,), audio_duration=1.0)

    assert result.diagnostics.synthetic_timing_segments == 1
    _assert_lossless_result(result, expected=text, audio_duration=1.0)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (math.nan, 1.0),
        (0.0, math.inf),
        (-1.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
    ],
)
def test_invalid_segment_anchor_uses_deterministic_timeline(
    start: float,
    end: float,
) -> None:
    text = "Сегмент сохранён."
    result = build_cues_with_diagnostics(
        (TranscriptSegment(start, end, text),),
        audio_duration=2.0,
    )

    _assert_lossless_result(result, expected=text, audio_duration=2.0)


@pytest.mark.parametrize("text", ["(", "—", "...", "«»", "?!"])
def test_standalone_punctuation_is_not_lost(text: str) -> None:
    result = build_cues_with_diagnostics(
        (TranscriptSegment(0.0, 1.0, text),),
        audio_duration=1.0,
    )

    _assert_lossless_result(result, expected=text, audio_duration=1.0)


@pytest.mark.parametrize(
    "text",
    [
        "— —",
        "— (",
        "— «",
        "( —",
        "« — (",
        "a — )",
        "a — ...",
        "a — ,",
        "a — — )",
    ],
)
def test_standalone_decoration_sequence_preserves_count_and_order(text: str) -> None:
    result = build_cues_with_diagnostics(
        (TranscriptSegment(0.0, 1.0, text),),
        audio_duration=1.0,
    )

    _assert_lossless_result(result, expected=text, audio_duration=1.0)


@pytest.mark.parametrize("text", ["(", "[", "«"])
def test_standalone_opening_punctuation_with_word_anchor_is_not_lost(text: str) -> None:
    result = build_cues_with_diagnostics(
        (
            TranscriptSegment(
                0.0,
                1.0,
                text,
                words=(TranscriptWord(0.0, 1.0, text),),
            ),
        ),
        audio_duration=1.0,
    )

    _assert_lossless_result(result, expected=text, audio_duration=1.0)


def test_mixed_valid_and_invalid_segment_anchors_preserve_source_order() -> None:
    segments = (
        TranscriptSegment(9.0, 10.0, "Первое."),
        TranscriptSegment(math.nan, math.nan, "Второе."),
    )

    result = build_cues_with_diagnostics(segments, audio_duration=10.0)

    _assert_lossless_result(
        result,
        expected="Первое. Второе.",
        audio_duration=10.0,
    )


def test_overlapping_segments_preserve_lexical_order() -> None:
    first = "Первое очень длинное предложение продолжается до самого конца."
    second = "Второе."
    segments = (
        TranscriptSegment(
            0.0, 10.0, first, words=_proportional_words(first, 0.0, 10.0)
        ),
        TranscriptSegment(
            5.0, 6.0, second, words=_proportional_words(second, 5.0, 6.0)
        ),
    )

    result = build_cues_with_diagnostics(segments, audio_duration=10.0)

    _assert_lossless_result(
        result,
        expected=f"{first} {second}",
        audio_duration=10.0,
    )


@pytest.mark.parametrize(
    "words",
    [
        (
            TranscriptWord(0.0, 3.0, "Первое"),
            TranscriptWord(0.0, 1.0, "второе"),
            TranscriptWord(0.0, 2.0, "третье."),
        ),
        (
            TranscriptWord(1.0, 2.0, "Первое"),
            TranscriptWord(0.0, 1.0, "второе"),
            TranscriptWord(2.0, 3.0, "третье."),
        ),
        (
            TranscriptWord(0.0, 2.0, "Первое"),
            TranscriptWord(1.0, 2.5, "второе"),
            TranscriptWord(2.5, 3.0, "третье."),
        ),
    ],
)
def test_conflicting_word_anchors_fall_back_to_segment_text(
    words: tuple[TranscriptWord, ...],
) -> None:
    text = "Первое второе третье."
    result = build_cues_with_diagnostics(
        (TranscriptSegment(0.0, 3.0, text, words=words),),
        audio_duration=3.0,
    )

    assert result.diagnostics.synthetic_timing_segments == 1
    _assert_lossless_result(result, expected=text, audio_duration=3.0)


def test_segment_text_wins_when_alignment_words_disagree() -> None:
    text = "Исходное готовое предложение."
    words = (
        TranscriptWord(0.0, 0.5, "Совсем"),
        TranscriptWord(0.5, 1.0, "другие"),
        TranscriptWord(1.0, 1.5, "слова."),
    )

    result = build_cues_with_diagnostics(
        (TranscriptSegment(0.0, 1.5, text, words=words),),
        audio_duration=1.5,
    )

    assert result.diagnostics.synthetic_timing_segments == 1
    _assert_lossless_result(result, expected=text, audio_duration=1.5)


@pytest.mark.parametrize("audio_duration", [0.001, 0.002, 0.003, 0.004, 0.005])
def test_millisecond_timeline_round_trip_never_collapses(audio_duration: float) -> None:
    text = "Раз. Два. Три."
    result = build_cues_with_diagnostics(
        (TranscriptSegment(0.0, audio_duration, text),),
        audio_duration=audio_duration,
    )

    _assert_lossless_result(result, expected=text, audio_duration=audio_duration)


def test_submillisecond_audio_uses_minimum_srt_time_quantum() -> None:
    text = "Текст."
    result = build_cues_with_diagnostics(
        (TranscriptSegment(0.0, 0.0005, text),),
        audio_duration=0.0005,
    )

    assert [(cue.start, cue.end, cue.text) for cue in result.cues] == [
        (0.0, 0.001, text)
    ]
    parsed = validate_srt(render_srt(result.cues), audio_duration=0.0005)
    assert [cue.text for cue in parsed] == [text]


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (0.0015, 0.0025),
        (0.0025, 0.00251),
        (0.00049, 0.00051),
        (1.2345, 1.2346),
    ],
)
def test_millisecond_quantization_never_collapses_rendered_cue(
    start: float,
    end: float,
) -> None:
    content = build_srt(
        (TranscriptSegment(start, end, "Текст."),),
        audio_duration=end,
    )

    cues = validate_srt(content, audio_duration=end)
    assert len(cues) == 1
    assert cues[0].end > cues[0].start


@pytest.mark.parametrize(
    ("start", "end", "expected_violations"),
    [
        (0.00051, 0.05949, 1),
        (0.00049, 0.05851, 0),
    ],
)
def test_presentation_diagnostics_follow_rendered_milliseconds(
    start: float,
    end: float,
    expected_violations: int,
) -> None:
    segment = TranscriptSegment(start, end, "x")
    result = build_cues_with_diagnostics(
        (segment,),
        max_cps=17,
        min_duration=0.001,
        max_duration=1.0,
        audio_duration=end,
    )
    content = render_srt(result.cues)
    rendered_cue = validate_srt(content, audio_duration=end)[0]

    assert result.diagnostics.reading_speed_target_exceeded_cues == expected_violations
    assert (result.cues[0].start, result.cues[0].end) == (
        rendered_cue.start,
        rendered_cue.end,
    )
    validate_srt(
        content,
        audio_duration=end,
        max_cps=17 if expected_violations == 0 else None,
    )


def test_validator_accepts_cps_equal_to_limit_after_srt_round_trip() -> None:
    content = "1\n06:49:05,711 --> 06:49:05,725\n12345678901234\n"

    cues = validate_srt(content, max_cps=1000)

    assert len(cues) == 1


def test_srt_timestamp_round_trip_keeps_canonical_float_values() -> None:
    cue = Cue(index=1, start=1.235, end=1.296, text="Текст.")

    parsed = validate_srt(render_srt((cue,)))

    assert parsed == (cue,)


def test_extreme_finite_timeline_does_not_overflow_fallback_scheduler() -> None:
    text = "Текст."
    result = build_cues_with_diagnostics(
        (TranscriptSegment(0.0, 1.0, text),),
        max_cps=1e-308,
        audio_duration=1e306,
    )

    assert [cue.text for cue in result.cues] == [text]
    assert result.diagnostics.reading_speed_target_exceeded_cues == 1
    assert math.isfinite(result.cues[0].start)
    assert math.isfinite(result.cues[0].end)


def test_extreme_finite_anchor_keeps_positive_interval() -> None:
    end = float(2**53 + 2)
    result = build_cues_with_diagnostics(
        (TranscriptSegment(0.0, end, "Текст."),),
        audio_duration=end,
    )

    assert result.cues[0].start < result.cues[0].end <= end
    assert result.diagnostics.max_actual_duration_ms > 0


def test_extreme_finite_timestamp_renders_without_overflow() -> None:
    end = 1e306
    start = math.nextafter(end, 0.0)

    content = build_srt(
        (TranscriptSegment(start, end, "Текст."),),
        audio_duration=end,
    )

    assert " --> " in content
    assert content.endswith("Текст.\n")


def test_multiple_cues_near_float_precision_limit_use_safe_timeline() -> None:
    end = float(2**53 + 100)
    segments = tuple(
        TranscriptSegment(end - 2, end, f"Фраза {index}.") for index in range(3)
    )

    result = build_cues_with_diagnostics(
        segments,
        max_cps=1e-308,
        audio_duration=end,
    )

    _assert_lossless_result(
        result,
        expected=" ".join(segment.text for segment in segments),
        audio_duration=end,
        max_cps=1e-308,
    )


def test_extreme_boundary_drift_diagnostic_does_not_overflow() -> None:
    end = 1e306
    start = math.nextafter(end, 0.0)
    result = build_cues_with_diagnostics(
        (TranscriptSegment(start, end, "alpha beta gamma."),),
        max_chars_per_line=1,
        max_lines=1,
        line_length_gap=0,
        max_cps=math.ulp(0.0),
        min_duration=math.ulp(0.0),
        max_duration=math.ulp(0.0),
        pause_threshold=math.ulp(0.0),
        audio_duration=end,
    )

    assert result.cues
    assert result.diagnostics.max_boundary_drift_ms >= 0


def test_many_short_segments_never_exhaust_timeline() -> None:
    segments = tuple(
        TranscriptSegment(index / 100, (index + 1) / 100, f"Фраза {index}.")
        for index in range(120)
    )
    expected = " ".join(segment.text for segment in segments)
    result = build_cues_with_diagnostics(segments, audio_duration=1.2)

    _assert_lossless_result(result, expected=expected, audio_duration=1.2)


def test_more_than_position_316_never_exhausts_dense_timeline() -> None:
    sentence_count = 700
    text = " ".join("Фраза." for _ in range(sentence_count))
    result = build_cues_with_diagnostics(
        (_segment(0.0, 28.0, text),),
        audio_duration=28.0,
    )

    assert len(result.cues) > 316
    assert result.diagnostics.reading_speed_target_exceeded_cues > 0
    _assert_lossless_result(result, expected=text, audio_duration=28.0)


def _generated_segments(
    case: GeneratedCase,
) -> tuple[tuple[TranscriptSegment, ...], str, float]:
    cursor = (0.0, 0.001, 0.999)[case.index % 3]
    all_words: list[TranscriptWord] = []
    sentence_ranges: list[tuple[int, int]] = []
    sentences: list[str] = []
    for sentence_index in range(case.sentence_count):
        width = WIDTHS[(case.index + sentence_index) % len(WIDTHS)]
        punctuation = PUNCTUATION[(case.index + sentence_index) % len(PUNCTUATION)]
        sentence = _sentence_with_exact_width(width, sentence_index + 1, punctuation)
        duration = (
            math.ceil(visible_character_count(sentence) / case.rate * 1000) / 1000
        )
        words = _proportional_words(sentence, cursor, cursor + duration)
        range_start = len(all_words)
        all_words.extend(words)
        sentence_ranges.append((range_start, len(all_words)))
        sentences.append(sentence)
        cursor += duration + case.gap

    last_end = all_words[-1].end
    padding = (0.0, 0.001, 0.010)[case.index % 3]
    audio_duration = math.ceil((last_end + padding) * 1000) / 1000
    cuts = _segment_cuts(
        sentence_ranges,
        word_count=len(all_words),
        mode=case.index % 4,
        orphan_words=(case.index // 4) % 4 + 1,
    )
    return (
        _segments_from_word_cuts(tuple(all_words), cuts),
        " ".join(sentences),
        audio_duration,
    )


def _sentence_with_exact_width(width: int, index: int, punctuation: str) -> str:
    parts = [f"Фраза{index}"]
    while len(" ".join((*parts, "слово"))) + len(punctuation) <= width:
        parts.append("слово")
    remainder = width - len(" ".join(parts)) - len(punctuation)
    if remainder == 1:
        parts[-1] += "я"
    elif remainder > 1:
        parts.append("я" * (remainder - 1))
    result = " ".join(parts) + punctuation
    assert len(result) == width
    return result


def _segment_cuts(
    sentence_ranges: list[tuple[int, int]],
    *,
    word_count: int,
    mode: int,
    orphan_words: int,
) -> tuple[int, ...]:
    boundaries = [end for _, end in sentence_ranges[:-1]]
    if not boundaries and word_count > 1 and mode != 0:
        cut = min(orphan_words, word_count - 1)
        boundaries = [cut if mode == 1 else word_count - cut]
    elif mode == 0:
        boundaries = []
    elif mode == 2:
        boundaries = [
            min(boundary + orphan_words, sentence_ranges[index + 1][1] - 1)
            for index, boundary in enumerate(boundaries)
        ]
    elif mode == 3:
        boundaries = [
            max(sentence_ranges[index][0] + 1, boundary - orphan_words)
            for index, boundary in enumerate(boundaries)
        ]
    return tuple(sorted({cut for cut in boundaries if 0 < cut < word_count}))


def _segments_from_word_cuts(
    words: tuple[TranscriptWord, ...],
    cuts: tuple[int, ...],
) -> tuple[TranscriptSegment, ...]:
    boundaries = (0, *cuts, len(words))
    result = []
    for segment_id, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        segment_words = words[left:right]
        result.append(
            TranscriptSegment(
                segment_words[0].start,
                segment_words[-1].end,
                " ".join(word.text for word in segment_words),
                words=segment_words,
                segment_id=segment_id,
            )
        )
    return tuple(result)


def _segment(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        start, end, text, words=_proportional_words(text, start, end)
    )


def _proportional_words(
    text: str, start: float, end: float
) -> tuple[TranscriptWord, ...]:
    tokens = text.split()
    weights = [max(1, visible_character_count(token)) for token in tokens]
    total = sum(weights)
    elapsed = 0
    result = []
    for index, (token, weight) in enumerate(zip(tokens, weights, strict=True)):
        word_start = start + (end - start) * elapsed / total
        elapsed += weight
        word_end = (
            end if index == len(tokens) - 1 else start + (end - start) * elapsed / total
        )
        result.append(TranscriptWord(word_start, word_end, token, 0.99))
    return tuple(result)


def _assert_lossless_result(
    result: CueBuildResult,
    *,
    expected: str,
    audio_duration: float,
    max_chars_per_line: int = BASE_WIDTH,
    line_length_gap: int = WIDTH_GAP,
    max_lines: int = 2,
    max_cps: float = TARGET_CPS,
    max_duration: float = MAX_DURATION,
) -> None:
    cues = result.cues
    assert cues
    assert [cue.index for cue in cues] == list(range(1, len(cues) + 1))
    assert _compact(_joined_text(cues)) == _compact(expected)
    previous_end = 0.0
    for cue in cues:
        assert math.isfinite(cue.start) and math.isfinite(cue.end)
        assert 0.0 <= cue.start < cue.end <= audio_duration + 1e-9
        assert previous_end <= cue.start
        assert cue.text.strip()
        assert 1 <= len(cue.text.splitlines()) <= max_lines
        previous_end = cue.end

    rendered = render_srt(cues)
    assert rendered.endswith("\n") and not rendered.startswith("\ufeff")
    parsed = validate_srt(
        rendered,
        audio_duration=audio_duration,
        duration_tolerance=0.0,
    )

    diagnostics = result.diagnostics
    cps_values = [
        visible_character_count(cue.text) / (cue.end - cue.start) for cue in parsed
    ]
    durations = [cue.end - cue.start for cue in parsed]
    line_lengths = [len(line) for cue in cues for line in cue.text.splitlines()]
    assert diagnostics.max_actual_cps == pytest.approx(max(cps_values), abs=1e-3)
    assert diagnostics.max_actual_duration_ms == pytest.approx(
        max(durations) * 1000, abs=1
    )
    assert diagnostics.max_actual_line_length == max(line_lengths)

    strict: dict[str, float | int] = {}
    if diagnostics.reading_speed_target_exceeded_cues == 0:
        assert max(cps_values) <= max_cps or math.isclose(
            max(cps_values),
            max_cps,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        strict["max_cps"] = max_cps
    else:
        assert any(
            value > max_cps
            and not math.isclose(
                value,
                max_cps,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for value in cps_values
        )
    if diagnostics.duration_target_exceeded_cues == 0:
        assert max(durations) <= max_duration + 1e-9
    else:
        assert any(value > max_duration + 1e-9 for value in durations)
    hard_width = max_chars_per_line + line_length_gap
    if diagnostics.line_length_target_exceeded_lines == 0:
        assert max(line_lengths) <= hard_width
        strict.update(
            max_chars_per_line=max_chars_per_line,
            line_length_gap=line_length_gap,
            max_lines=max_lines,
        )
    else:
        assert any(length > hard_width for length in line_lengths)

    validate_cues(
        cues,
        audio_duration=audio_duration,
        duration_tolerance=0.0,
        max_chars_per_line=strict.get("max_chars_per_line"),
        line_length_gap=int(strict.get("line_length_gap", 0)),
        max_lines=int(strict.get("max_lines", max_lines)),
    )
    validate_srt(
        rendered,
        audio_duration=audio_duration,
        duration_tolerance=0.0,
        **strict,
    )
    assert _compact(_joined_text(parsed)) == _compact(expected)
    assert render_srt(parsed) == rendered


def _joined_text(cues: Sequence[Cue]) -> str:
    return " ".join(" ".join(cue.text.splitlines()) for cue in cues)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)
