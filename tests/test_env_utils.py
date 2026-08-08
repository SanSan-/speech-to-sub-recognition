from __future__ import annotations

import pytest

from speech_to_sub.constants import (
    DEFAULT_PARAKEET_MODEL_PATH,
    DEFAULT_QWEN_ALIGNER_MODEL_PATH,
    DEFAULT_QWEN_MODEL_PATH,
)
from speech_to_sub.exceptions import ValidationError
from speech_to_sub.models import ProcessingSettings
from speech_to_sub.utils.env_utils import settings_from_environment


_SETTING_VARIABLES = (
    "ASR_BACKEND",
    "ASR_MODEL_PATH",
    "ASR_ALIGNER",
    "ASR_ALIGNER_MODEL_PATH",
    "ASR_WORKER_PYTHON",
    "ASR_ALIGNER_WORKER_PYTHON",
    "ASR_LANGUAGE",
    "ASR_AUDIO_LANGUAGE",
    "ASR_AUDIO_STREAM_INDEX",
    "ASR_DEVICE",
    "ASR_QUANTIZATION",
    "ASR_ALLOW_CPU_FALLBACK",
    "ASR_KEEP_AUDIO",
    "ASR_LONG_FORM_WINDOW_SECONDS",
    "ASR_LONG_FORM_OVERLAP_SECONDS",
    "ASR_VAD_FILTER",
    "ASR_VAD_MIN_SILENCE_MS",
    "ASR_BEAM_SIZE",
    "ASR_CONDITION_ON_PREVIOUS_TEXT",
    "ASR_OUTPUT_DIR",
)


@pytest.fixture(autouse=True)
def clean_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SETTING_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_environment_settings_parse_supported_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_BACKEND", "FASTER-WHISPER")
    monkeypatch.setenv("ASR_LANGUAGE", "RU")
    monkeypatch.setenv("ASR_DEVICE", "CUDA")
    monkeypatch.setenv("ASR_AUDIO_STREAM_INDEX", "2")
    monkeypatch.setenv("ASR_QUANTIZATION", "yes")
    monkeypatch.setenv("ASR_ALLOW_CPU_FALLBACK", "да")
    monkeypatch.setenv("ASR_KEEP_AUDIO", "off")
    monkeypatch.setenv("ASR_LONG_FORM_WINDOW_SECONDS", "240")
    monkeypatch.setenv("ASR_LONG_FORM_OVERLAP_SECONDS", "3")
    monkeypatch.setenv("ASR_VAD_FILTER", "true")
    monkeypatch.setenv("ASR_VAD_MIN_SILENCE_MS", "750")
    monkeypatch.setenv("ASR_BEAM_SIZE", "3")
    monkeypatch.setenv("ASR_CONDITION_ON_PREVIOUS_TEXT", "false")
    monkeypatch.setenv("ASR_WORKER_PYTHON", r"D:\Runtime\parakeet\python.exe")
    monkeypatch.setenv("ASR_ALIGNER_WORKER_PYTHON", r"D:\Runtime\qwen\python.exe")

    settings = settings_from_environment()

    assert settings.backend == "faster-whisper"
    assert settings.language == "ru"
    assert settings.device == "cuda"
    assert settings.audio_stream_index == 2
    assert settings.quantization_enabled is True
    assert settings.allow_cpu_fallback is True
    assert settings.keep_audio is False
    assert settings.long_form_window_seconds == 240
    assert settings.long_form_overlap_seconds == 3
    assert settings.vad_filter is True
    assert settings.vad_min_silence_ms == 750
    assert settings.beam_size == 3
    assert settings.condition_on_previous_text is False
    assert str(settings.worker_python_path).endswith(r"parakeet\python.exe")
    assert str(settings.aligner_worker_python_path).endswith(r"qwen\python.exe")


def test_environment_and_mapping_choose_backend_specific_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASR_BACKEND", "qwen3-asr")
    monkeypatch.setenv("ASR_ALIGNER", "qwen3-forced-aligner")

    settings = settings_from_environment()

    assert settings.model_path == DEFAULT_QWEN_MODEL_PATH
    assert settings.aligner_model_path == DEFAULT_QWEN_ALIGNER_MODEL_PATH
    assert ProcessingSettings.from_mapping({"backend": "parakeet-tdt-v3"}).model_path == (
        DEFAULT_PARAKEET_MODEL_PATH
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ASR_BACKEND", "unknown"),
        ("ASR_LANGUAGE", "de"),
        ("ASR_DEVICE", "gpu"),
        ("ASR_AUDIO_STREAM_INDEX", "not-an-int"),
        ("ASR_AUDIO_STREAM_INDEX", "-1"),
        ("ASR_QUANTIZATION", "sometimes"),
        ("ASR_ALLOW_CPU_FALLBACK", "2"),
        ("ASR_KEEP_AUDIO", ""),
        ("ASR_LONG_FORM_WINDOW_SECONDS", "0"),
        ("ASR_LONG_FORM_OVERLAP_SECONDS", "-1"),
        ("ASR_VAD_FILTER", "maybe"),
        ("ASR_VAD_MIN_SILENCE_MS", "none"),
        ("ASR_BEAM_SIZE", "0"),
        ("ASR_CONDITION_ON_PREVIOUS_TEXT", "maybe"),
    ],
)
def test_environment_settings_reject_invalid_values(
    name: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError, match=name):
        settings_from_environment()


def test_faster_whisper_environment_uses_local_ct2_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASR_BACKEND", "faster-whisper")

    settings = settings_from_environment()

    assert settings.model_path.name == "whisper-large-v3-ct2"
