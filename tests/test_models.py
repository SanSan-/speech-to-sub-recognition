from __future__ import annotations

from pathlib import Path

import pytest

from speech_to_sub import __version__
from speech_to_sub.constants import SubtitleFormat
from speech_to_sub.exceptions import ValidationError
from speech_to_sub.models import (
    FileResult,
    ProcessingSettings,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


def test_unknown_backend_without_explicit_model_is_domain_error() -> None:
    with pytest.raises(ValidationError, match="Неизвестный ASR backend 'unknown'"):
        ProcessingSettings.from_mapping({"backend": "unknown"})


def test_package_version_matches_v160_milestone() -> None:
    assert __version__ == "1.6.0"


def test_processing_settings_round_trip_subtitle_layout_limits() -> None:
    settings = ProcessingSettings.from_mapping(
        {"max_chars_per_line": 40, "line_length_gap": 6, "max_cps": 16.5}
    )

    assert settings.max_chars_per_line == 40
    assert settings.line_length_gap == 6
    assert settings.max_cps == 16.5
    assert settings.to_dict()["line_length_gap"] == 6
    assert settings.to_dict()["max_cps"] == 16.5


def test_processing_settings_defaults_to_srt_and_normalizes_output_format() -> None:
    default_settings = ProcessingSettings.from_mapping({})
    ass_settings = ProcessingSettings.from_mapping({"output_format": " ASS "})

    assert default_settings.output_format is SubtitleFormat.SRT
    assert default_settings.to_dict()["output_format"] == "srt"
    assert ass_settings.output_format is SubtitleFormat.ASS
    assert ass_settings.to_dict()["output_format"] == "ass"


@pytest.mark.parametrize("value", (None, 7, True, "", "txt"))
def test_processing_settings_rejects_invalid_output_format(value: object) -> None:
    with pytest.raises(ValidationError, match="Формат субтитров|Неизвестный формат"):
        ProcessingSettings.from_mapping({"output_format": value})


def test_file_result_exposes_generic_output_and_limits_legacy_srt_field() -> None:
    ass_path = Path("sample.ru.ass")
    result = FileResult(
        input_path=Path("sample.mp4"),
        state="done",
        subtitle_path=ass_path,
        output_format=SubtitleFormat.ASS,
    ).to_dict()

    assert result["output_format"] == "ass"
    assert result["subtitle_output"] == str(ass_path)
    assert result["srt_output"] is None


def test_cloud_settings_round_trip_with_ignored_compatibility_model_path() -> None:
    settings = ProcessingSettings.from_mapping(
        {
            "backend": "OPENAI-API",
            "allow_cloud_processing": True,
            "openai_model": "WHISPER-1",
            "auto_download_model": False,
        }
    )

    assert settings.backend == "openai-api"
    assert settings.allow_cloud_processing is True
    assert settings.openai_model == "whisper-1"
    assert settings.auto_download_model is False
    assert settings.to_dict()["model_path"]


def test_transcript_round_trip_from_sidecar_mapping() -> None:
    transcript = Transcript(
        text="Проверка.",
        language="ru",
        duration=1.5,
        segments=(
            TranscriptSegment(
                0.1,
                1.2,
                "Проверка.",
                words=(TranscriptWord(0.1, 1.2, "Проверка.", 0.9),),
                segment_id=7,
            ),
        ),
        model="local-model",
        device="cpu",
        quantized=False,
        metadata={"engine_version": "test"},
    )

    assert Transcript.from_mapping(transcript.to_dict()) == transcript


@pytest.mark.parametrize(
    "mutation",
    (
        {"duration": "1.5"},
        {"segments": {}},
        {"quantized": 0},
        {"metadata": []},
    ),
)
def test_transcript_from_mapping_rejects_malformed_payload(
    mutation: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "text": "Проверка.",
        "language": "ru",
        "duration": 1.5,
        "segments": [
            {
                "id": 0,
                "start": 0.1,
                "end": 1.2,
                "text": "Проверка.",
                "words": [],
            }
        ],
        "model": "local-model",
        "device": "cpu",
        "quantized": False,
        "metadata": {},
    }
    payload.update(mutation)

    with pytest.raises(ValidationError, match="sidecar"):
        Transcript.from_mapping(payload)


def test_transcript_from_mapping_preserves_text_without_segments() -> None:
    transcript = Transcript(
        text="Готовая расшифровка без посегментных меток.",
        language="ru",
        duration=1.5,
        segments=(),
        model="local-model",
        device="cpu",
        quantized=False,
    )

    assert Transcript.from_mapping(transcript.to_dict()) == transcript
