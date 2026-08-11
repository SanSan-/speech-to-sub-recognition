from __future__ import annotations

from collections.abc import Callable

import pytest

from speech_to_sub.constants import SubtitleFormat
from speech_to_sub.exceptions import ValidationError
from speech_to_sub.subtitles.builder import Cue, render_srt
from speech_to_sub.subtitles.formats import (
    _build_ass_export_lines,
    parse_ass_dialogs,
    parse_ass_events,
    render_ass,
    render_subtitles,
    render_vtt,
)
from speech_to_sub.subtitles.validator import (
    parse_ass,
    parse_srt,
    parse_vtt,
    validate_ass,
    validate_subtitle_text,
    validate_vtt,
)


def test_srt_renderer_preserves_legacy_bytes() -> None:
    cues = (
        Cue(1, 0.0, 1.234, "Привет, мир!"),
        Cue(2, 1.234, 3.0, "Вторая\nстрока"),
    )
    expected = (
        "1\n00:00:00,000 --> 00:00:01,234\nПривет, мир!\n\n"
        "2\n00:00:01,234 --> 00:00:03,000\nВторая\nстрока\n"
    ).encode("utf-8")

    assert render_subtitles(cues, SubtitleFormat.SRT).encode("utf-8") == expected
    assert render_srt(cues).encode("utf-8") == expected


def test_ass_round_trip_preserves_visible_text_and_blocks_overrides() -> None:
    text = "Привет, {\\i1} <мир> & Unicode — 漢字\nпуть \\N и \\h"
    rendered = render_ass((Cue(1, 0.0, 1.0, text),))

    assert rendered.startswith("[Script Info]\nScriptType: v4.00+\n")
    assert "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,," in rendered
    assert "{\\i1}" not in rendered
    assert parse_ass(rendered)[0].text == text
    assert validate_ass(rendered, audio_duration=1.0)[0].text == text


def test_ass_ported_exporter_preserves_document_fields_and_override_tags() -> None:
    dialogue = (
        "Dialogue: 7,9:59:59.99,10:00:00.01,Стиль верх,Анна,0012,0034,0056,"
        "Banner;5;0;20,{\\an8}{\\i1}Hello,\\Nworld.{\\i0}"
    )
    origins = [
        "[Script Info]",
        "Title: Проверка Unicode",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        dialogue,
        "Comment: 1,0:00:00.00,0:00:01.00,Стиль верх,Анна,0,0,0,,Не менять",
    ]
    dialogue_index = origins.index(dialogue)
    dialogs = parse_ass_dialogs(origins)
    events = parse_ass_events(origins)
    replacement = "{\\an8}{\\i1}Здравствуйте,\\Nмир.{\\i0}"

    exported = _build_ass_export_lines(
        origins,
        dialogs,
        {dialogue_index: replacement},
    )

    assert events[dialogue_index].layer == 7
    assert events[dialogue_index].actor == "Анна"
    assert events[dialogue_index + 1].translatable is False
    assert exported[:dialogue_index] == origins[:dialogue_index]
    assert exported[dialogue_index + 1 :] == origins[dialogue_index + 1 :]
    original_fields = dialogue.split(",", 9)
    exported_fields = exported[dialogue_index].split(",", 9)
    assert exported_fields[:9] == original_fields[:9]
    assert exported_fields[9] == replacement


def test_vtt_round_trip_escapes_markup_and_validates_visible_text() -> None:
    text = "<tag> & уже &amp; Unicode — 漢字\nвторая строка"
    rendered = render_vtt((Cue(1, 0.0, 1.0, text),))

    assert rendered.startswith("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\n")
    assert "&lt;tag&gt; &amp; уже &amp;amp;" in rendered
    assert "<tag>" not in rendered
    assert parse_vtt(rendered)[0].text == text
    assert validate_vtt(rendered, audio_duration=1.0)[0].text == text


@pytest.mark.parametrize(
    "renderer, parser, expected_ends",
    (
        (
            render_srt,
            parse_srt,
            (0.001, 0.002, 0.01),
        ),
        (
            render_vtt,
            parse_vtt,
            (0.001, 0.002, 0.01),
        ),
        (
            render_ass,
            parse_ass,
            (0.01, 0.02, 0.03),
        ),
    ),
)
def test_renderers_quantize_zero_one_and_ten_millisecond_cues(
    renderer: Callable[..., str],
    parser: Callable[[str], tuple[Cue, ...]],
    expected_ends: tuple[float, float, float],
) -> None:
    cues = (
        Cue(1, 0.0, 0.0, "ноль"),
        Cue(2, 0.0, 0.001, "одна"),
        Cue(3, 0.001, 0.01, "десять"),
    )
    parsed = parser(renderer(cues))

    assert tuple(cue.end for cue in parsed) == expected_ends
    assert all(cue.end > cue.start for cue in parsed)
    assert all(
        current.start >= previous.end
        for previous, current in zip(parsed, parsed[1:], strict=False)
    )


def test_ass_validation_fails_when_quantized_cues_exceed_audio_by_more_than_tick() -> None:
    rendered = render_ass(
        (
            Cue(1, 0.0, 0.0, "первая"),
            Cue(2, 0.0, 0.001, "вторая"),
            Cue(3, 0.001, 0.01, "третья"),
        )
    )

    with pytest.raises(ValidationError, match="длительность аудио"):
        validate_subtitle_text(rendered, SubtitleFormat.ASS, duration=0.01)


@pytest.mark.parametrize(
    "output_format, renderer",
    (
        (SubtitleFormat.SRT, render_srt),
        (SubtitleFormat.VTT, render_vtt),
    ),
)
def test_millisecond_formats_allow_only_one_tick_past_audio(
    output_format: SubtitleFormat,
    renderer: Callable[..., str],
) -> None:
    accepted = renderer((Cue(1, 0.0, 1.0, "текст"),))
    rejected = renderer((Cue(1, 0.0, 1.002, "текст"),))

    validate_subtitle_text(accepted, output_format, duration=0.999)
    with pytest.raises(ValidationError, match="длительность аудио"):
        validate_subtitle_text(rejected, output_format, duration=1.0)


def test_ass_validator_rejects_incomplete_header_and_style() -> None:
    damaged = (
        "[Script Info]\nBAD\n[V4+ Styles]\nBAD\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,текст\n"
    )

    with pytest.raises(ValidationError, match="заголовок или стиль"):
        validate_ass(damaged)


@pytest.mark.parametrize(
    "output_format, content",
    (
        (
            SubtitleFormat.SRT,
            "1\n00:00:00,000 --> 00:00:01,000\nТекст\n",
        ),
        (
            SubtitleFormat.VTT,
            "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nТекст\n",
        ),
        (
            SubtitleFormat.ASS,
            render_ass((Cue(1, 0.0, 1.0, "Текст"),)),
        ),
    ),
)
def test_all_validators_reject_utf8_bom(
    output_format: SubtitleFormat,
    content: str,
) -> None:
    with pytest.raises(ValidationError, match="BOM"):
        validate_subtitle_text("\ufeff" + content, output_format)


def test_vtt_validator_rejects_raw_markup_and_unknown_entities() -> None:
    for text in ("<b>опасно</b>", "неизвестная &copy; сущность"):
        content = f"WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\n{text}\n"
        with pytest.raises(ValidationError, match="разметку|сущность"):
            validate_vtt(content)


def test_ass_validator_rejects_unescaped_override_block() -> None:
    rendered = render_ass((Cue(1, 0.0, 1.0, "текст"),)).replace(
        "текст",
        "{\\i1}текст",
    )

    with pytest.raises(ValidationError, match="фигурную скобку"):
        validate_ass(rendered)
