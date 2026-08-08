from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from speech_to_sub.asr import faster_whisper
from speech_to_sub.asr.faster_whisper import FasterWhisperBackend
from speech_to_sub.exceptions import ProcessingCancelled
from speech_to_sub.media.audio_windows import AudioWindow
from speech_to_sub.models import ProcessingSettings


class _FakeCTranslate2:
    @staticmethod
    def contains_model(_path: str) -> bool:
        return True

    @staticmethod
    def get_cuda_device_count() -> int:
        return 1

    @staticmethod
    def get_supported_compute_types(_device: str) -> set[str]:
        return {"float16", "int8_float16", "float32", "int8"}


def _model_dir(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    for name in ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json"):
        (model / name).write_bytes(b"test")
    return model


def test_expected_runtime_uses_cuda_compute_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(faster_whisper, "_import_ctranslate2", lambda: _FakeCTranslate2)
    backend = FasterWhisperBackend()

    quantized = backend.expected_runtime_signature(
        ProcessingSettings(model_path=_model_dir(tmp_path), device="auto")
    )
    fp16 = backend.expected_runtime_signature(
        ProcessingSettings(
            model_path=tmp_path / "model",
            device="cuda",
            quantization_enabled=False,
        )
    )

    assert quantized["compute_type"] == "int8_float16"
    assert quantized["quantized"] is True
    assert fp16["compute_type"] == "float16"
    assert fp16["quantized"] is False


def test_runtime_signature_requires_matching_load_key(tmp_path: Path) -> None:
    backend = FasterWhisperBackend()
    settings = ProcessingSettings(
        model_path=tmp_path / "model",
        device="auto",
        allow_cpu_fallback=True,
    )
    backend._model = object()
    backend._load_key = (
        str((tmp_path / "model").resolve()).casefold(),
        "auto",
        True,
        True,
    )
    backend._device_label = "cpu"
    backend._compute_type = "int8"

    signature = backend.runtime_signature(settings)

    assert signature is not None
    assert signature["backend"] == "faster-whisper"
    assert signature["device"] == "cpu"
    assert signature["compute_type"] == "int8"
    assert backend.runtime_signature(
        ProcessingSettings(model_path=tmp_path / "model", device="cpu")
    ) is None


def test_transcribe_offsets_windows_and_deduplicates_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _model_dir(tmp_path)
    settings = ProcessingSettings(
        model_path=model_path,
        language="en",
        device="cpu",
        long_form_window_seconds=3,
        long_form_overlap_seconds=1,
    )
    backend = FasterWhisperBackend()
    backend._load_key = (
        str(model_path.resolve()).casefold(),
        "cpu",
        True,
        False,
    )
    backend._device_label = "cpu"
    backend._compute_type = "int8"
    calls: list[dict[str, Any]] = []

    class FakeModel:
        def transcribe(self, _samples: np.ndarray, **kwargs: Any) -> tuple[Any, Any]:
            calls.append(kwargs)
            if len(calls) == 1:
                words = [
                    SimpleNamespace(start=0.2, end=0.8, word=" Hello", probability=0.9),
                    SimpleNamespace(start=2.5, end=2.7, word=" boundary", probability=0.8),
                ]
            else:
                words = [
                    SimpleNamespace(start=0.5, end=0.7, word=" boundary", probability=0.8),
                    SimpleNamespace(start=1.0, end=1.4, word=" end", probability=0.95),
                ]
            segments = [SimpleNamespace(start=words[0].start, end=words[-1].end, text="", words=words)]
            return iter(segments), SimpleNamespace(language="en")

    backend._model = FakeModel()
    windows = (
        AudioWindow(0.0, 3.0, np.zeros(48_000, dtype=np.float32), False),
        AudioWindow(2.0, 2.0, np.zeros(32_000, dtype=np.float32), True),
    )
    monkeypatch.setattr(faster_whisper, "iter_audio_windows", lambda *_args, **_kwargs: iter(windows))
    progress: list[int] = []

    transcript = backend.transcribe(
        tmp_path / "audio.flac",
        settings,
        duration=4.0,
        progress_callback=progress.append,
    )

    assert transcript.text == "Hello boundary end"
    assert [(word.start, word.end, word.text.strip()) for segment in transcript.segments for word in segment.words] == [
        (0.2, 0.8, "Hello"),
        (2.5, 2.7, "boundary"),
        (3.0, 3.4, "end"),
    ]
    assert progress[0] == 0
    assert progress[-1] == 100
    assert all(call["word_timestamps"] is True for call in calls)
    assert all(call["vad_filter"] is True for call in calls)


def test_overlap_segment_is_not_restored_after_all_words_are_filtered() -> None:
    raw_segment = SimpleNamespace(
        start=0.0,
        end=0.4,
        text="duplicate",
        words=(SimpleNamespace(start=0.0, end=0.4, word=" duplicate"),),
    )

    accepted = faster_whisper._normalize_window_segments(
        (raw_segment,),
        AudioWindow(
            offset=8.0,
            duration=4.0,
            samples=np.zeros(64_000, dtype=np.float32),
            is_final=True,
        ),
        overlap_seconds=2.0,
        audio_duration=12.0,
        first_segment_id=0,
    )

    assert accepted == []


def test_cancellation_between_windows_is_not_treated_as_cuda_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _model_dir(tmp_path)
    settings = ProcessingSettings(
        model_path=model_path,
        language="en",
        device="cuda",
        allow_cpu_fallback=True,
        long_form_window_seconds=3,
        long_form_overlap_seconds=1,
    )
    backend = FasterWhisperBackend()
    backend._load_key = (
        str(model_path.resolve()).casefold(),
        "cuda",
        True,
        True,
    )
    backend._device_label = "cuda"
    backend._compute_type = "int8_float16"
    calls = 0

    class FakeModel:
        def transcribe(self, _samples: np.ndarray, **_kwargs: Any) -> tuple[Any, Any]:
            nonlocal calls
            calls += 1
            word = SimpleNamespace(start=0.2, end=0.8, word=" text", probability=0.9)
            segment = SimpleNamespace(start=0.2, end=0.8, text=" text", words=(word,))
            return iter((segment,)), SimpleNamespace(language="en")

    backend._model = FakeModel()
    windows = (
        AudioWindow(0.0, 3.0, np.zeros(48_000, dtype=np.float32), False),
        AudioWindow(2.0, 2.0, np.zeros(32_000, dtype=np.float32), True),
    )
    monkeypatch.setattr(faster_whisper, "iter_audio_windows", lambda *_args, **_kwargs: iter(windows))

    def cancel_after_first_window(progress: int) -> None:
        if progress > 0:
            raise ProcessingCancelled("cancel")

    with pytest.raises(ProcessingCancelled, match="cancel"):
        backend.transcribe(
            tmp_path / "audio.flac",
            settings,
            duration=4.0,
            progress_callback=cancel_after_first_window,
        )

    assert calls == 1
    assert backend._device_label == "cuda"
