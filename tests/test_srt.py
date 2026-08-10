from __future__ import annotations

import pytest

from speech_to_sub.exceptions import ValidationError
from speech_to_sub.models import TranscriptSegment, TranscriptWord
from speech_to_sub.subtitles.builder import (
    Cue,
    _normalize_stretched_word_anchors,
    _pack_timed_sentences,
    build_cues,
    build_cues_with_diagnostics,
    build_srt,
    format_timestamp,
)
from speech_to_sub.subtitles.layout import (
    LayoutDiagnostics,
    build_timed_sentences,
    prepare_timed_words,
    visible_character_count,
)
from speech_to_sub.subtitles.validator import parse_srt, validate_cues, validate_srt


def test_format_timestamp_rounds_milliseconds_and_supports_long_audio() -> None:
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(3661.9996) == "01:01:02,000"
    assert format_timestamp(100 * 3600 + 2.003) == "100:00:02,003"
    huge_timestamp = format_timestamp(1e306)
    assert huge_timestamp.count(":") == 2
    assert huge_timestamp.endswith(",000")


def test_builder_preserves_segment_text_when_word_text_disagrees() -> None:
    segment = TranscriptSegment(
        start=0.0,
        end=4.0,
        text="Исходный текст должен сохраниться.",
        words=(
            TranscriptWord(0.0, 0.5, "Hello"),
            TranscriptWord(0.5, 1.0, "world."),
            TranscriptWord(2.0, 2.4, "Second"),
            TranscriptWord(2.4, 3.0, "cue."),
        ),
    )

    cues = build_cues((segment,), audio_duration=4.0)

    assert [cue.text for cue in cues] == ["Исходный текст должен сохраниться."]
    assert cues[0].start == 0.0
    assert cues[0].end <= 4.0


def test_builder_does_not_add_spaces_around_hyphenated_word() -> None:
    segment = TranscriptSegment(
        start=0.0,
        end=1.2,
        text="IP-адрес",
        words=(
            TranscriptWord(0.0, 0.4, "IP"),
            TranscriptWord(0.4, 0.6, "-"),
            TranscriptWord(0.6, 1.2, "адрес"),
        ),
    )

    cues = build_cues((segment,))

    assert cues[0].text == "IP-адрес"


def test_alignment_only_decorations_preserve_count_and_order() -> None:
    segment = TranscriptSegment(
        0.0,
        1.0,
        "",
        words=(
            TranscriptWord(0.0, 0.25, "—"),
            TranscriptWord(0.25, 0.5, "—"),
            TranscriptWord(0.5, 0.75, "("),
            TranscriptWord(0.75, 1.0, "«"),
        ),
    )

    prepared = prepare_timed_words((segment,))

    assert [word.text for word in prepared.words] == ["— — («"]


def test_alignment_pending_decorations_do_not_reorder_closing_punctuation() -> None:
    segment = TranscriptSegment(
        0.0,
        1.0,
        "",
        words=(
            TranscriptWord(0.0, 0.3, "a"),
            TranscriptWord(0.3, 0.6, "—"),
            TranscriptWord(0.6, 1.0, ")"),
        ),
    )

    prepared = prepare_timed_words((segment,))

    assert [word.text for word in prepared.words] == ["a — )"]


def test_builder_uses_segment_timestamps_and_limits_text_to_two_lines() -> None:
    segment = TranscriptSegment(
        start=0.0,
        end=8.0,
        text=(
            "This is a deliberately long subtitle sentence that must be split into readable "
            "pieces and at most two lines per cue."
        ),
    )

    cues = build_cues(
        (segment,),
        max_chars_per_line=24,
        line_length_gap=0,
        audio_duration=8.0,
    )

    assert len(cues) >= 2
    assert all(len(cue.text.splitlines()) <= 2 for cue in cues)
    assert all(max(map(len, cue.text.splitlines())) <= 24 for cue in cues)
    assert all(left.end <= right.start for left, right in zip(cues, cues[1:]))


def test_build_srt_creates_valid_utf8_text_without_bom() -> None:
    content = build_srt(
        (TranscriptSegment(0.0, 1.2, "Привет, world!"),),
        audio_duration=1.2,
    )

    assert content == "1\n00:00:00,000 --> 00:00:01,200\nПривет, world!\n"
    assert not content.startswith("\ufeff")
    assert content.encode("utf-8")[:3] != b"\xef\xbb\xbf"
    assert validate_srt(content, audio_duration=1.2)[0].text == "Привет, world!"


def test_builder_removes_overlaps_between_segments() -> None:
    cues = build_cues(
        (
            TranscriptSegment(0.0, 1.5, "First cue."),
            TranscriptSegment(1.5, 3.0, "Second cue."),
        ),
        max_lines=1,
        audio_duration=3.0,
    )

    assert cues[0].end == cues[1].start
    validate_cues(cues, audio_duration=3.0)


@pytest.mark.parametrize(
    "content, message",
    [
        ("", "пуст"),
        ("\ufeff1\n00:00:00,000 --> 00:00:01,000\nText\n", "BOM"),
        ("1\n00:00:00.000 --> 00:00:01,000\nText\n", "метка"),
        ("2\n00:00:00,000 --> 00:00:01,000\nText\n", "Нумерация"),
    ],
)
def test_validator_rejects_invalid_srt(content: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        validate_srt(content)


def test_validator_rejects_overlap_empty_text_and_duration_overrun() -> None:
    overlapping = (Cue(1, 0.0, 2.0, "One"), Cue(2, 1.0, 3.0, "Two"))
    with pytest.raises(ValidationError, match="пересекается"):
        validate_cues(overlapping)

    empty_text = (Cue(1, 0.0, 1.0, "  "),)
    with pytest.raises(ValidationError, match="не содержит текста"):
        validate_cues(empty_text)

    duration_overrun = (Cue(1, 0.0, 2.0, "One"),)
    with pytest.raises(ValidationError, match="выходит за длительность"):
        validate_cues(duration_overrun, audio_duration=1.0)


def test_parse_srt_accepts_crlf_and_two_text_lines() -> None:
    cues = parse_srt(
        "1\r\n00:00:00,000 --> 00:00:01,500\r\nFirst line\r\nSecond line\r\n"
    )

    assert cues == (Cue(1, 0.0, 1.5, "First line\nSecond line"),)


def test_builder_joins_sentence_across_segment_boundary() -> None:
    segments = (
        TranscriptSegment(
            0.0,
            1.0,
            "Доброго времени",
            words=(
                TranscriptWord(0.0, 0.4, "доброго"),
                TranscriptWord(0.4, 1.0, "времени"),
            ),
        ),
        TranscriptSegment(
            1.05,
            2.5,
            "суток, друзья.",
            words=(
                TranscriptWord(1.05, 1.5, "суток"),
                TranscriptWord(1.6, 2.5, "друзья"),
            ),
        ),
    )

    cues = build_cues(segments, audio_duration=3.0)

    assert len(cues) == 1
    assert cues[0].text == "Доброго времени суток, друзья."


def test_builder_restores_punctuation_split_words_and_integral_hyphen() -> None:
    segment = TranscriptSegment(
        0.0,
        3.2,
        "Что ждали, какую-то семью?",
        words=(
            TranscriptWord(0.0, 0.8, "Чтождали"),
            TranscriptWord(0.9, 1.5, "какую"),
            TranscriptWord(1.5, 1.6, "-"),
            TranscriptWord(1.6, 2.0, "то"),
            TranscriptWord(2.1, 3.2, "семью"),
        ),
    )

    cues = build_cues((segment,), audio_duration=3.5)

    assert len(cues) == 1
    assert cues[0].text == "Что ждали, какую-то семью?"


def test_builder_normalizes_standalone_dash_without_gluing_words() -> None:
    segment = TranscriptSegment(
        0.0,
        2.0,
        "",
        words=(
            TranscriptWord(0.0, 0.6, "слово"),
            TranscriptWord(0.6, 0.7, "-"),
            TranscriptWord(0.7, 1.4, "другое"),
            TranscriptWord(1.4, 2.0, "."),
        ),
    )

    cues = build_cues((segment,), audio_duration=2.5)

    assert cues[0].text == "слово — другое."


def test_builder_retimes_narrow_leading_island_without_mutating_transcript() -> None:
    segment = TranscriptSegment(
        0.0,
        22.76,
        "Дорогие друзья, с вами педагоги Анна Францева и Ирина Воронова.",
        words=(
            TranscriptWord(0.0, 1.1, "Дорогие", 0.817),
            TranscriptWord(17.77, 17.83, "друзья,", 0.990),
            TranscriptWord(18.01, 18.65, "с", 0.996),
            TranscriptWord(18.65, 18.81, "вами", 0.989),
            TranscriptWord(18.81, 19.55, "педагоги", 0.997),
            TranscriptWord(19.55, 19.93, "Анна", 0.999),
            TranscriptWord(19.93, 20.63, "Францева", 0.889),
            TranscriptWord(20.63, 21.33, "и", 0.995),
            TranscriptWord(21.46, 22.12, "Ирина", 0.999),
            TranscriptWord(22.12, 22.76, "Воронова.", 0.913),
        ),
    )

    result = build_cues_with_diagnostics((segment,), audio_duration=23.0)

    assert result.diagnostics.retimed_leading_islands == 1
    assert len(result.cues) == 1
    assert result.cues[0].start == pytest.approx(16.67)
    assert " ".join(result.cues[0].text.splitlines()) == segment.text
    assert segment.words[0].start == 0.0


def test_builder_enforces_line_and_reading_speed_limits() -> None:
    text = "Это длинное предложение с пунктуацией, которое нужно показать читателю целиком."
    segment = TranscriptSegment(2.0, 4.0, text)

    cues = build_cues(
        (segment,),
        max_chars_per_line=24,
        line_length_gap=0,
        max_cps=17,
        audio_duration=8.0,
    )

    assert len(cues) >= 2
    assert all(len(cue.text.splitlines()) <= 2 for cue in cues)
    assert all(max(map(len, cue.text.splitlines())) <= 24 for cue in cues)
    assert all(visible_character_count(cue.text) / (cue.end - cue.start) <= 17 for cue in cues)
    validate_cues(
        cues,
        audio_duration=8.0,
        max_chars_per_line=24,
        max_lines=2,
        max_cps=17,
    )


def test_builder_starts_each_sentence_on_a_new_line() -> None:
    segment = TranscriptSegment(
        0.0,
        3.0,
        "Первое предложение. Второе предложение!",
        words=(
            TranscriptWord(0.0, 0.6, "Первое"),
            TranscriptWord(0.6, 1.2, "предложение"),
            TranscriptWord(1.3, 1.9, "Второе"),
            TranscriptWord(1.9, 3.0, "предложение"),
        ),
    )

    cues = build_cues((segment,), audio_duration=3.5)

    assert [cue.text for cue in cues] == [
        "Первое предложение.\nВторое предложение!"
    ]


def test_builder_does_not_split_unfinished_sentence_at_pause() -> None:
    segment = TranscriptSegment(
        0.0,
        3.0,
        "Да, продолжаем.",
        words=(
            TranscriptWord(0.0, 0.2, "Да,"),
            TranscriptWord(1.2, 2.2, "продолжаем."),
        ),
    )

    cues = build_cues((segment,), audio_duration=3.0)

    assert [cue.text for cue in cues] == ["Да, продолжаем."]


def test_validator_applies_layout_limits_only_when_requested() -> None:
    cue = Cue(1, 0.0, 1.0, "Очень длинная строка")

    validate_cues((cue,))
    with pytest.raises(ValidationError, match="символов в строке"):
        validate_cues((cue,), max_chars_per_line=10)
    with pytest.raises(ValidationError, match="CPS"):
        validate_cues((cue,), max_cps=5)


def test_builder_reports_impossible_reading_window_without_failure() -> None:
    segment = TranscriptSegment(0.0, 0.5, "Очень длинная фраза.")

    result = build_cues_with_diagnostics(
        (segment,),
        max_cps=5,
        audio_duration=0.5,
    )

    assert [cue.text for cue in result.cues] == ["Очень длинная фраза."]
    assert result.diagnostics.reading_speed_target_exceeded_cues == 1
    validate_cues(result.cues, audio_duration=0.5)


def test_builder_keeps_raw_speech_inside_extended_cue() -> None:
    segment = TranscriptSegment(
        5.0,
        8.8,
        "Первая содержательная фраза. Вторая содержательная фраза.",
        words=(
            TranscriptWord(5.0, 6.8, "Первая содержательная фраза."),
            TranscriptWord(7.0, 8.8, "Вторая содержательная фраза."),
        ),
    )

    cues = build_cues((segment,), max_chars_per_line=42, max_cps=17, audio_duration=10.0)

    assert len(cues) == 1
    assert cues[0].text == (
        "Первая содержательная фраза.\nВторая содержательная фраза."
    )
    assert cues[0].start <= 5.0 <= 6.8 <= 7.0 <= 8.8 <= cues[0].end


def test_builder_merges_hyphen_suffix_across_segments() -> None:
    segments = (
        TranscriptSegment(
            0.0,
            0.5,
            "что",
            words=(TranscriptWord(0.0, 0.5, "что"),),
        ),
        TranscriptSegment(
            0.5,
            1.8,
            "-то бывает.",
            words=(
                TranscriptWord(0.5, 0.8, "-то"),
                TranscriptWord(0.9, 1.8, "бывает."),
            ),
        ),
    )

    cues = build_cues(segments, audio_duration=2.0)

    assert cues[0].text == "что-то бывает."


def test_builder_splits_sentences_glued_inside_one_token() -> None:
    segment = TranscriptSegment(
        0.0,
        2.4,
        "Нет звука.Ждём продолжения.",
        words=(
            TranscriptWord(0.0, 1.0, "Нет"),
            TranscriptWord(1.0, 1.4, "звука.Ждём"),
            TranscriptWord(1.4, 2.4, "продолжения."),
        ),
    )

    cues = build_cues((segment,), audio_duration=3.0)

    assert [cue.text for cue in cues] == ["Нет звука.\nЖдём продолжения."]


def test_builder_attaches_separate_closing_quote_to_previous_sentence() -> None:
    segment = TranscriptSegment(
        0.0,
        1.5,
        'Почему? ". Следующая.',
        words=(
            TranscriptWord(0.0, 0.8, "Почему?"),
            TranscriptWord(0.8, 0.801, '".'),
            TranscriptWord(0.801, 1.5, "Следующая."),
        ),
    )

    prepared = prepare_timed_words((segment,))
    cues = build_cues((segment,), audio_duration=2.0)

    assert [word.text for word in prepared.words] == ['Почему?".', "Следующая."]
    assert [cue.text for cue in cues] == ['Почему?".\nСледующая.']


def test_builder_prefers_punctuation_line_break_and_never_splits_word() -> None:
    text = "Довольно длинная смысловая часть, но хвост тоже достаточно длинный."
    segment = TranscriptSegment(0.0, 6.0, text)

    cues = build_cues((segment,), max_chars_per_line=38, audio_duration=6.5)

    assert "\n" in cues[0].text
    assert cues[0].text.splitlines()[0].endswith(",")
    assert " ".join(" ".join(cue.text.splitlines()) for cue in cues) == text


def test_builder_uses_bounded_drift_for_dense_realistic_speech() -> None:
    sentences = (
        (
            "Кто-то раздражается на себя, что он не может делать то, что он предыдет делать.",
            2.17,
            7.19,
        ),
        ("Либо он на себя говорит, почему я родила такого ребёнка?", 7.31, 9.81),
        ("У всех моих подруг здоровая, а я вот такую ребёночку родила.", 9.83, 12.95),
        ("Могут родители ссориться.", 13.47, 15.43),
    )
    words = tuple(
        word
        for text, start, end in sentences
        for word in _proportional_words(text, start, end)
    )
    segments = (
        TranscriptSegment(
            0.0,
            1.0,
            "Вступление.",
            words=(TranscriptWord(0.0, 1.0, "Вступление."),),
        ),
        TranscriptSegment(
            2.17,
            15.43,
            " ".join(text for text, _, _ in sentences),
            words=words,
        ),
    )

    result = build_cues_with_diagnostics(segments, audio_duration=17.25)

    assert result.diagnostics.adjusted_boundaries > 0
    assert 0 < result.diagnostics.max_boundary_drift_ms <= 1000
    assert all(
        visible_character_count(cue.text) / (cue.end - cue.start) <= 17
        for cue in result.cues
    )
    assert all(max(map(len, cue.text.splitlines())) <= 50 for cue in result.cues)
    assert all(left.end <= right.start for left, right in zip(result.cues, result.cues[1:]))


def test_builder_uses_best_effort_beyond_boundary_drift_target() -> None:
    sentence = "Много быстрых слов нужно показать читателю без нарушения строгих ограничений."
    segments = (
        TranscriptSegment(
            0.0,
            4.2,
            sentence,
            words=_proportional_words(sentence, 0.0, 4.2),
        ),
        TranscriptSegment(
            4.2,
            4.3,
            sentence,
            words=_proportional_words(sentence, 4.2, 4.3),
        ),
        TranscriptSegment(
            4.3,
            8.5,
            sentence,
            words=_proportional_words(sentence, 4.3, 8.5),
        ),
    )

    result = build_cues_with_diagnostics(segments, audio_duration=13.0)

    assert len(result.cues) == 3
    assert result.diagnostics.reading_speed_target_exceeded_cues == 3
    validate_cues(result.cues, audio_duration=13.0)


def test_rendered_minimum_duration_survives_millisecond_rounding() -> None:
    content = build_srt(
        (TranscriptSegment(0.0, 0.2, "Да."),),
        min_duration=0.8,
        audio_duration=1.0,
    )

    cue = parse_srt(content)[0]

    assert cue.end - cue.start >= 0.8


def test_diagnostics_ignore_external_display_padding_for_short_cue() -> None:
    result = build_cues_with_diagnostics(
        (TranscriptSegment(0.0, 0.2, "Да."),),
        min_duration=0.8,
        audio_duration=1.0,
    )

    assert len(result.cues) == 1
    assert result.cues[0].start == 0.0
    assert result.cues[0].end == pytest.approx(0.802)
    assert result.cues[0].text == "Да."
    assert result.diagnostics.adjusted_boundaries == 0
    assert result.diagnostics.max_boundary_drift_ms == 0
    assert result.diagnostics.to_dict()["timing_anomaly_adjustments"] == 0


def test_diagnostics_counts_shared_internal_boundary_once() -> None:
    segments = (
        TranscriptSegment(0.0, 0.5, "Первое."),
        TranscriptSegment(0.5, 1.0, "Второе."),
    )

    result = build_cues_with_diagnostics(segments, max_lines=1, audio_duration=2.0)

    assert result.cues[0].end == pytest.approx(result.cues[1].start)
    assert result.diagnostics.adjusted_boundaries == 1
    assert result.diagnostics.max_boundary_drift_ms == 302


def test_layout_diagnostics_always_serializes_timing_anomaly_counter() -> None:
    assert LayoutDiagnostics().to_dict()["timing_anomaly_adjustments"] == 0


def test_validator_counts_line_break_as_one_cps_space() -> None:
    cue = Cue(1, 0.0, 1.0, "1234\n5678")

    validate_cues((cue,), max_cps=9)
    with pytest.raises(ValidationError, match="CPS"):
        validate_cues((cue,), max_cps=8.99)


def test_builder_wraps_abnormal_hyphen_chain_only_at_hyphen() -> None:
    token = "альфа-бета-гамма-дельта-эпсилон-дзета."
    segment = TranscriptSegment(
        0.0,
        3.0,
        token,
        words=(TranscriptWord(0.0, 3.0, token),),
    )

    cue = build_cues(
        (segment,),
        max_chars_per_line=24,
        line_length_gap=0,
        audio_duration=3.5,
    )[0]

    assert max(map(len, cue.text.splitlines())) <= 24
    assert cue.text.replace("\n", "") == token


def test_builder_splits_long_vocalization_only_at_restored_hyphens() -> None:
    text = "Та-да-да-дам."
    segment = TranscriptSegment(
        0.0,
        8.478,
        text,
        words=(TranscriptWord(0.0, 8.478, "Тадададам"),),
    )

    cues = build_cues((segment,), audio_duration=8.5)

    assert len(cues) == 1
    assert cues[0].end - cues[0].start <= 7.0
    assert cues[0].text.splitlines()[0].endswith("-")
    assert cues[0].text.replace("\n", "") == text


def test_builder_retimes_long_ordinary_word_without_splitting_text() -> None:
    segment = TranscriptSegment(
        0.0,
        11.04,
        "очень",
        words=(TranscriptWord(0.0, 11.04, "очень"),),
    )

    cues = build_cues((segment,), audio_duration=11.1)

    assert [cue.text for cue in cues] == ["очень"]
    assert cues[0].end - cues[0].start <= 7.0


@pytest.mark.parametrize(
    ("timings", "stretched_index"),
    [
        (
            (
                (1035.36, 1035.5, "Этот"),
                (1035.5, 1035.67, "диагноз"),
                (1035.68, 1035.681, "их"),
                (1036.0, 1047.04, "очень"),
                (1047.04, 1047.6, "поразил."),
            ),
            3,
        ),
        (
            (
                (907.351, 907.8, "сейчас"),
                (907.8, 908.0, "я"),
                (908.0, 908.3, "вам"),
                (908.3, 909.0, "расскажу"),
                (909.0, 909.3, "одну"),
                (909.3, 910.3, "историю"),
                (910.3, 911.0, "да"),
                (911.03, 939.511, "она"),
                (940.87, 941.5, "умница."),
            ),
            7,
        ),
        (
            (
                (1011.671, 1011.8, "что"),
                (1011.8, 1012.0, "они"),
                (1012.0, 1012.5, "упал"),
                (1012.55, 1021.19, "ваза"),
                (1021.59, 1021.8, "он"),
                (1021.8, 1022.1, "даже"),
                (1022.1, 1022.3, "не"),
                (1022.3, 1022.791, "сдогнулся"),
            ),
            3,
        ),
    ],
    ids=("parakeet-ochen", "qwen-ona", "qwen-vaza"),
)
def test_builder_right_anchors_stretched_single_word_before_sentence_layout(
    timings: tuple[tuple[float, float, str], ...],
    stretched_index: int,
) -> None:
    words = tuple(TranscriptWord(start, end, text) for start, end, text in timings)
    text = " ".join(word.text for word in words)
    normalized, adjustments = _normalize_stretched_word_anchors(
        words,
        min_duration=0.8,
        max_duration=7.0,
        max_cps=17.0,
    )
    segment = TranscriptSegment(words[0].start, words[-1].end, text, words=words)

    result = build_cues_with_diagnostics(
        (segment,),
        audio_duration=words[-1].end + 1.0,
    )

    assert adjustments == 1
    assert normalized[stretched_index].end == words[stretched_index].end
    assert normalized[stretched_index].end - normalized[stretched_index].start == pytest.approx(
        0.802
    )
    assert result.diagnostics.timing_anomaly_adjustments == 1
    assert " ".join(" ".join(cue.text.splitlines()) for cue in result.cues) == text
    assert result.diagnostics.duration_target_exceeded_cues > 0
    validate_cues(result.cues, audio_duration=words[-1].end + 1.0)


def test_builder_redistributes_long_vocalization_with_tiny_tail_intervals() -> None:
    text = (
        "Та-да-да-дам. "
        "Та-да-дам-дам-дам-дам-дам-дам-дам-дам-дам-дам. "
        "Да-дам-дам-дам-дам-дам-да-да-дам-да-да-дам."
    )
    segment = TranscriptSegment(
        0.0,
        8.48,
        text,
        words=(
            TranscriptWord(0.0, 8.478, "Тадададам"),
            TranscriptWord(8.478, 8.479, "Тададамдамдамдамдамдамдамдамдамдам"),
            TranscriptWord(8.479, 8.48, "Дадамдамдамдамдамдададамдададам"),
        ),
    )

    cues = build_cues((segment,), audio_duration=8.6)

    assert all(cue.end - cue.start <= 7.0 for cue in cues)
    assert all(
        visible_character_count(cue.text) / (cue.end - cue.start) <= 17
        for cue in cues
    )
    rendered_text = " ".join(" ".join(cue.text.splitlines()) for cue in cues)
    assert rendered_text.replace("- ", "-") == text


def test_builder_keeps_43_character_sentence_on_one_line_with_default_gap() -> None:
    text = "Да, мы почему хотим поговорить на эту тему?"
    segment = TranscriptSegment(
        63.78,
        67.9,
        text,
        words=(
            TranscriptWord(63.78, 64.28, "Да,"),
            TranscriptWord(64.44, 65.3, "мы"),
            TranscriptWord(65.3, 65.72, "почему"),
            TranscriptWord(65.72, 66.2, "хотим"),
            TranscriptWord(66.2, 67.12, "поговорить"),
            TranscriptWord(67.12, 67.22, "на"),
            TranscriptWord(67.22, 67.38, "эту"),
            TranscriptWord(67.38, 67.9, "тему?"),
        ),
    )

    adaptive = build_cues((segment,), audio_duration=68.5)
    strict = build_cues((segment,), line_length_gap=0, audio_duration=68.5)

    assert adaptive[0].text == text
    assert len(adaptive[0].text) == 43
    assert "\n" not in adaptive[0].text
    assert strict[0].text.splitlines()[0] != "Да,"
    assert max(map(len, strict[0].text.splitlines())) <= 42


def test_builder_avoids_single_word_tail_inside_sentence() -> None:
    text = "понимать, что это нужно просто пережить и потом перейти к помощи к ребёнку."
    timings = (
        (102.8, 102.98, "понимать,"),
        (103.0, 103.58, "что"),
        (103.58, 103.84, "это"),
        (103.84, 104.08, "нужно"),
        (104.08, 104.54, "просто"),
        (104.54, 105.48, "пережить"),
        (105.48, 106.08, "и"),
        (106.08, 106.38, "потом"),
        (106.38, 107.02, "перейти"),
        (107.02, 107.14, "к"),
        (107.14, 107.68, "помощи"),
        (107.68, 108.02, "к"),
        (108.53, 109.81, "ребёнку."),
    )
    segment = TranscriptSegment(
        102.8,
        109.81,
        text,
        words=tuple(TranscriptWord(start, end, word) for start, end, word in timings),
    )

    cues = build_cues((segment,), audio_duration=110.0)

    assert len(cues) == 1
    assert cues[0].text == (
        "понимать, что это нужно просто пережить\n"
        "и потом перейти к помощи к ребёнку."
    )
    assert " ".join(" ".join(cue.text.splitlines()) for cue in cues) == text
    assert all(visible_character_count(cue.text) / (cue.end - cue.start) <= 17 for cue in cues)


def test_builder_keeps_people_and_different_inside_same_cue() -> None:
    segments = (
        TranscriptSegment(
            164.33,
            167.56,
            "Кто-то может даже раздражаться на ребёнка,",
            words=_proportional_words(
                "Кто-то может даже раздражаться на ребёнка,",
                164.33,
                167.56,
            ),
        ),
        TranscriptSegment(
            167.56,
            170.82,
            "потому что люди разные, и по-разному могут стресс переживать.",
            words=_proportional_words(
                "потому что люди разные, и по-разному могут стресс переживать.",
                167.56,
                170.82,
            ),
        ),
    )

    cues = build_cues(segments, audio_duration=171.5)

    assert [cue.text for cue in cues] == [
        "Кто-то может даже раздражаться на ребёнка,\nпотому что люди разные,",
        "и по-разному могут стресс переживать.",
    ]


def test_text_first_lines_do_not_depend_on_source_duration() -> None:
    text = "Первая длинная смысловая часть, и вторая тоже остаётся целой."
    variants = []
    for end in (4.0, 9.5):
        segment = TranscriptSegment(
            0.0,
            end,
            text,
            words=_proportional_words(text, 0.0, end),
        )
        prepared = prepare_timed_words((segment,))
        sentences = build_timed_sentences(prepared.words, pause_threshold=0.8)
        variants.append(
            _pack_timed_sentences(
                sentences,
                max_chars_per_line=42,
                hard_chars_per_line=50,
            )
        )

    assert [[line.text for line in lines] for lines in variants] == [
        ["Первая длинная смысловая часть,", "и вторая тоже остаётся целой."],
        ["Первая длинная смысловая часть,", "и вторая тоже остаётся целой."],
    ]
    assert len(build_cues((TranscriptSegment(0.0, 4.0, text),), audio_duration=5.0)) == 1
    assert len(build_cues((TranscriptSegment(0.0, 9.5, text),), audio_duration=10.0)) == 1


def test_scheduler_reports_long_semantic_line_without_failure() -> None:
    text = "Одна смысловая строка."
    segment = TranscriptSegment(
        0.0,
        10.0,
        text,
        words=_proportional_words(text, 0.0, 10.0),
    )

    result = build_cues_with_diagnostics((segment,), audio_duration=11.0)

    assert [cue.text for cue in result.cues] == [text]
    assert result.diagnostics.duration_target_exceeded_cues == 1
    validate_cues(result.cues, audio_duration=11.0)


@pytest.mark.parametrize("line_length_gap", [-1, 21, True, "8"])
def test_builder_rejects_invalid_line_length_gap(line_length_gap: object) -> None:
    segment = TranscriptSegment(0.0, 1.0, "Текст.")

    with pytest.raises(ValidationError, match="Допуск длины строки"):
        build_cues(
            (segment,),
            line_length_gap=line_length_gap,  # type: ignore[arg-type]
            audio_duration=1.0,
        )


def test_validator_accepts_adaptive_width_but_rejects_more_than_hard_limit() -> None:
    validate_cues(
        (Cue(1, 0.0, 3.0, "x" * 43),),
        max_chars_per_line=42,
        line_length_gap=8,
    )
    overlong = (Cue(1, 0.0, 3.0, "x" * 51),)

    with pytest.raises(ValidationError, match="символов в строке"):
        validate_cues(
            overlong,
            max_chars_per_line=42,
            line_length_gap=8,
        )


def _proportional_words(text: str, start: float, end: float) -> tuple[TranscriptWord, ...]:
    tokens = text.split()
    weights = [len(token) for token in tokens]
    total = sum(weights)
    elapsed = 0
    result: list[TranscriptWord] = []
    for index, (token, weight) in enumerate(zip(tokens, weights, strict=True)):
        word_start = start + (end - start) * elapsed / total
        elapsed += weight
        word_end = end if index == len(tokens) - 1 else start + (end - start) * elapsed / total
        result.append(TranscriptWord(word_start, word_end, token, 0.99))
    return tuple(result)
