from __future__ import annotations

import pytest

from speech_to_sub.constants import (
    DEFAULT_BACKEND_MODEL_PATHS,
    DEFAULT_LINE_LENGTH_GAP,
    DEFAULT_MAX_CHARS_PER_LINE,
    DEFAULT_MAX_CPS,
    DEFAULT_PARAKEET_MODEL_PATH,
    DEFAULT_QWEN_ALIGNER_MODEL_PATH,
    DEFAULT_QWEN_MODEL_PATH,
    MODELS_DIR,
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
    "ASR_MAX_CHARS_PER_LINE",
    "ASR_LINE_LENGTH_GAP",
    "ASR_MAX_CPS",
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
    monkeypatch.setenv("ASR_MAX_CHARS_PER_LINE", "40")
    monkeypatch.setenv("ASR_LINE_LENGTH_GAP", "6")
    monkeypatch.setenv("ASR_MAX_CPS", "16.5")
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
    assert settings.max_chars_per_line == 40
    assert settings.line_length_gap == 6
    assert settings.max_cps == 16.5
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


def test_default_model_paths_use_portable_repository_directory() -> None:
    assert DEFAULT_BACKEND_MODEL_PATHS == {
        "transformers": MODELS_DIR / "whisper-large-v3",
        "faster-whisper": MODELS_DIR / "whisper-large-v3-ct2",
        "parakeet-tdt-v3": MODELS_DIR / "parakeet-tdt-0.6b-v3",
        "qwen3-asr": MODELS_DIR / "Qwen3-ASR-0.6B",
    }
    assert DEFAULT_QWEN_ALIGNER_MODEL_PATH == MODELS_DIR / "Qwen3-ForcedAligner-0.6B"


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
        ("ASR_MAX_CHARS_PER_LINE", "0"),
        ("ASR_LINE_LENGTH_GAP", "-1"),
        ("ASR_LINE_LENGTH_GAP", "21"),
        ("ASR_MAX_CPS", "not-a-float"),
        ("ASR_MAX_CPS", "nan"),
        ("ASR_MAX_CPS", "0"),
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


def test_environment_uses_subtitle_layout_defaults() -> None:
    settings = settings_from_environment()

    assert settings.max_chars_per_line == DEFAULT_MAX_CHARS_PER_LINE
    assert settings.line_length_gap == DEFAULT_LINE_LENGTH_GAP
    assert settings.max_cps == DEFAULT_MAX_CPS
