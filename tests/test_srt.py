from __future__ import annotations

import pytest

from speech_to_sub.exceptions import ValidationError
from speech_to_sub.models import TranscriptSegment, TranscriptWord
from speech_to_sub.subtitles.builder import Cue, build_cues, build_srt, format_timestamp
from speech_to_sub.subtitles.validator import parse_srt, validate_cues, validate_srt


def test_format_timestamp_rounds_milliseconds_and_supports_long_audio() -> None:
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(3661.9996) == "01:01:02,000"
    assert format_timestamp(100 * 3600 + 2.003) == "100:00:02,003"


def test_builder_prefers_word_timestamps_and_splits_on_pause() -> None:
    segment = TranscriptSegment(
        start=0.0,
        end=4.0,
        text="Fallback text must not be used",
        words=(
            TranscriptWord(0.0, 0.5, "Hello"),
            TranscriptWord(0.5, 1.0, "world."),
            TranscriptWord(2.0, 2.4, "Second"),
            TranscriptWord(2.4, 3.0, "cue."),
        ),
    )

    cues = build_cues((segment,), audio_duration=4.0)

    assert [cue.text for cue in cues] == ["Hello world.", "Second cue."]
    assert cues[0].start == 0.0
    assert cues[0].end <= cues[1].start
    assert cues[1].end <= 4.0


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


def test_builder_uses_segment_timestamps_and_limits_text_to_two_lines() -> None:
    segment = TranscriptSegment(
        start=0.0,
        end=8.0,
        text=(
            "This is a deliberately long subtitle sentence that must be split into readable "
            "pieces and at most two lines per cue."
        ),
    )

    cues = build_cues((segment,), max_chars_per_line=24, audio_duration=8.0)

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
            TranscriptSegment(0.0, 2.0, "First cue"),
            TranscriptSegment(1.5, 3.0, "Second cue"),
        ),
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
    with pytest.raises(ValidationError, match="пересекается"):
        validate_cues((Cue(1, 0.0, 2.0, "One"), Cue(2, 1.0, 3.0, "Two")))
    with pytest.raises(ValidationError, match="не содержит текста"):
        validate_cues((Cue(1, 0.0, 1.0, "  "),))
    with pytest.raises(ValidationError, match="выходит за длительность"):
        validate_cues((Cue(1, 0.0, 2.0, "One"),), audio_duration=1.0)


def test_parse_srt_accepts_crlf_and_two_text_lines() -> None:
    cues = parse_srt(
        "1\r\n00:00:00,000 --> 00:00:01,500\r\nFirst line\r\nSecond line\r\n"
    )

    assert cues == (Cue(1, 0.0, 1.5, "First line\nSecond line"),)
