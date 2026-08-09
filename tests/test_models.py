from __future__ import annotations

import pytest

from speech_to_sub import __version__
from speech_to_sub.exceptions import ValidationError
from speech_to_sub.models import (
    ProcessingSettings,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


def test_unknown_backend_without_explicit_model_is_domain_error() -> None:
    with pytest.raises(ValidationError, match="Неизвестный ASR backend 'unknown'"):
        ProcessingSettings.from_mapping({"backend": "unknown"})


def test_package_version_matches_v150_milestone() -> None:
    assert __version__ == "1.5.0"


def test_processing_settings_round_trip_subtitle_layout_limits() -> None:
    settings = ProcessingSettings.from_mapping(
        {"max_chars_per_line": 40, "line_length_gap": 6, "max_cps": 16.5}
    )

    assert settings.max_chars_per_line == 40
    assert settings.line_length_gap == 6
    assert settings.max_cps == 16.5
    assert settings.to_dict()["line_length_gap"] == 6
    assert settings.to_dict()["max_cps"] == 16.5


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
        {"segments": []},
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
