from __future__ import annotations

from pathlib import Path

import pytest

from speech_to_sub.asr import transformers_whisper
from speech_to_sub.asr.transformers_whisper import (
    TransformersWhisperBackend,
    normalize_pipeline_output,
)
from speech_to_sub.exceptions import AsrModelError
from speech_to_sub.models import ProcessingSettings


def test_normalize_pipeline_output_preserves_words_and_clamps_timestamps() -> None:
    transcript = normalize_pipeline_output(
        {
            "text": " Привет, мир! ",
            "language": "ru",
            "chunks": [
                {"text": " Привет", "timestamp": (0.0, 0.4)},
                {"text": ",", "timestamp": (0.2, 0.6)},
                {"text": " мир!", "timestamp": (None, None)},
                {"text": "", "timestamp": (0.7, 0.8)},
                {"text": "ignored", "timestamp": "invalid"},
                {"text": " beyond", "timestamp": (1.5, 1.8)},
            ],
        },
        duration=1.5,
        language="auto",
        model="local-model",
        device="cuda",
        quantized=True,
    )

    assert transcript.text == "Привет, мир!"
    assert transcript.language == "ru"
    assert transcript.model == "local-model"
    assert transcript.device == "cuda"
    assert transcript.quantized is True
    assert transcript.metadata == {
        "runtime": "transformers",
        "engine_version": "unknown",
        "compute_type": "int8",
        "word_timestamps": True,
    }
    assert len(transcript.segments) == 1
    assert [(word.start, word.end, word.text) for word in transcript.segments[0].words] == [
        (0.0, 0.4, " Привет"),
        (0.4, 0.6, ","),
        (0.6, 1.5, " мир!"),
    ]


def test_normalize_pipeline_output_accepts_single_item_and_falls_back_to_segment() -> None:
    transcript = normalize_pipeline_output(
        [{"text": "English speech"}],
        duration=3.25,
        language="en",
        model="local-model",
        device="cpu",
        quantized=False,
    )

    assert transcript.language == "en"
    assert transcript.segments[0].start == 0.0
    assert transcript.segments[0].end == 3.25
    assert transcript.segments[0].words == ()
    assert transcript.metadata["word_timestamps"] is False


def test_normalize_pipeline_output_reads_detected_language_from_word_chunks() -> None:
    transcript = normalize_pipeline_output(
        {
            "text": " Русская речь ",
            "chunks": [
                {"text": " Русская", "timestamp": (0.0, 0.5), "language": "russian"},
                {"text": " речь", "timestamp": (0.5, 1.0), "language": "russian"},
            ],
        },
        duration=1.0,
        language="auto",
        model="local-model",
        device="cuda",
        quantized=True,
    )

    assert transcript.language == "ru"


def test_whisper_language_token_is_normalized_to_code() -> None:
    assert transformers_whisper._normalize_language_label("<|en|>", None) == "en"
    assert transformers_whisper._normalize_language_label("Russian", None) == "ru"
    assert transformers_whisper._normalize_language_label(None, "auto") == "auto"


@pytest.mark.parametrize(
    "output, message",
    [
        ([], "пакетный"),
        ([{"text": "one"}, {"text": "two"}], "пакетный"),
        ("plain text", "неизвестного формата"),
        ({"text": "   "}, "не вернула текст"),
    ],
)
def test_normalize_pipeline_output_rejects_invalid_contract(
    output: object,
    message: str,
) -> None:
    with pytest.raises(AsrModelError, match=message):
        normalize_pipeline_output(
            output,
            duration=1.0,
            language="en",
            model="local-model",
            device="cpu",
            quantized=False,
        )


def test_default_backend_is_singleton_and_unloads_without_model_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleared: list[bool] = []
    backend = transformers_whisper.get_default_backend()
    monkeypatch.setattr(transformers_whisper, "clear_model_memory", lambda: cleared.append(True))
    monkeypatch.setattr(
        transformers_whisper,
        "_import_transformers_components",
        lambda: pytest.fail("Transformers не должен импортироваться при выгрузке"),
    )
    backend._pipeline = object()
    backend._model = object()
    backend._processor = object()
    backend._load_key = (
        str(Path("model")),
        "transformers",
        "cuda",
        True,
        False,
    )
    backend._device_label = "cuda"
    backend._compute_type = "int8"
    backend._quantized = True

    assert transformers_whisper.get_default_backend() is backend
    transformers_whisper.unload_default_backend()

    assert backend._pipeline is None
    assert backend._model is None
    assert backend._processor is None
    assert backend._load_key is None
    assert backend._device_label == "unloaded"
    assert backend._compute_type == "unloaded"
    assert backend._quantized is False
    assert cleared == [True]


def test_runtime_signature_requires_matching_model_settings(tmp_path: Path) -> None:
    backend = TransformersWhisperBackend()
    settings = ProcessingSettings(
        backend="transformers",
        model_path=tmp_path / "model",
        device="auto",
        quantization_enabled=True,
        allow_cpu_fallback=True,
    )
    backend._pipeline = object()
    backend._load_key = (
        str((tmp_path / "model").resolve()).casefold(),
        "transformers",
        "auto",
        True,
        True,
    )
    backend._device_label = "cpu"
    backend._compute_type = "float32"
    backend._quantized = False

    assert backend.runtime_signature(settings) == {
        "backend": "transformers",
        "engine_version": backend._engine_version,
        "device": "cpu",
        "compute_type": "float32",
        "quantized": False,
    }
    assert backend.runtime_signature(
        ProcessingSettings(
            model_path=tmp_path / "model",
            device="cpu",
            quantization_enabled=False,
            allow_cpu_fallback=True,
        )
    ) is None


def test_expected_runtime_signature_contains_engine_and_compute_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = TransformersWhisperBackend()
    backend._engine_version = "test-transformers-1"
    device = type("Device", (), {"type": "cpu"})()
    monkeypatch.setattr(
        "speech_to_sub.utils.model_utils.resolve_device_and_quantization",
        lambda *_args: (object(), device, object(), None, False),
    )

    signature = backend.expected_runtime_signature(
        ProcessingSettings(device="cpu", quantization_enabled=True)
    )

    assert signature == {
        "backend": "transformers",
        "engine_version": "test-transformers-1",
        "device": "cpu",
        "compute_type": "float32",
        "quantized": False,
    }


def test_cuda_inference_error_retries_once_on_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = TransformersWhisperBackend()
    requested_key = ("local-model", "transformers", "cuda", True, True)
    cuda_calls: list[dict[str, object]] = []
    cpu_calls: list[dict[str, object]] = []

    def cuda_pipeline(_audio: object, **kwargs: object) -> dict[str, object]:
        cuda_calls.append(kwargs)
        raise RuntimeError("CUDA out of memory")

    def cpu_pipeline(_audio: object, **kwargs: object) -> dict[str, object]:
        cpu_calls.append(kwargs)
        return {
            "text": "CPU result",
            "chunks": [{"text": " CPU", "timestamp": (0.0, 0.4)}],
        }

    def load_cpu(model_path: Path, device_name: str, quantization: bool) -> None:
        assert model_path == tmp_path / "model"
        assert device_name == "cpu"
        assert quantization is False
        backend._pipeline = cpu_pipeline
        backend._device_label = "cpu"
        backend._compute_type = "float32"
        backend._quantized = False

    backend._pipeline = cuda_pipeline
    backend._device_label = "cuda"
    backend._quantized = True
    backend._load_key = requested_key
    monkeypatch.setattr(backend, "_ensure_loaded", lambda _settings: None)
    monkeypatch.setattr(backend, "_load_once", load_cpu)
    monkeypatch.setattr(
        transformers_whisper,
        "validate_model_path",
        lambda path: Path(path),
    )
    monkeypatch.setattr(transformers_whisper, "clear_model_memory", lambda: None)
    monkeypatch.setattr(transformers_whisper, "decode_audio_float32", lambda *_args, **_kwargs: object())

    transcript = backend.transcribe(
        tmp_path / "audio.flac",
        ProcessingSettings(
            model_path=tmp_path / "model",
            device="cuda",
            allow_cpu_fallback=True,
        ),
        duration=1.0,
    )

    assert len(cuda_calls) == 1
    assert len(cpu_calls) == 1
    assert cuda_calls[0]["return_language"] is False
    assert cpu_calls[0]["return_language"] is False
    assert transcript.text == "CPU result"
    assert transcript.device == "cpu"
    assert transcript.quantized is False
    assert backend._load_key == requested_key


def test_cuda_inference_error_is_not_retried_without_permission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = TransformersWhisperBackend()

    def cuda_pipeline(_audio: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("CUDA failure")

    backend._pipeline = cuda_pipeline
    backend._device_label = "cuda"
    monkeypatch.setattr(backend, "_ensure_loaded", lambda _settings: None)
    monkeypatch.setattr(transformers_whisper, "decode_audio_float32", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        backend,
        "_load_once",
        lambda *_args: pytest.fail("CPU не должен загружаться без разрешения"),
    )
    settings = ProcessingSettings(
        model_path=tmp_path / "model",
        device="cuda",
        allow_cpu_fallback=False,
    )

    with pytest.raises(AsrModelError, match="CUDA failure"):
        backend.transcribe(
            tmp_path / "audio.flac",
            settings,
            duration=1.0,
        )
