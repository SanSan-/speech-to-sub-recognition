from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from speech_to_sub.alignment import qwen_forced
from speech_to_sub.alignment.qwen_forced import QwenForcedAlignerAdapter
from speech_to_sub.alignment.registry import aligner_names, get_aligner
from speech_to_sub.asr import parakeet_tdt
from speech_to_sub.exceptions import AsrModelError, ProcessingCancelled, ValidationError
from speech_to_sub.models import (
    ProcessingSettings,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


class _FakeClient:
    def __init__(self, *, with_words: bool = True) -> None:
        self.with_words = with_words
        self.requests: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        self.shutdown_calls = 0

    def request(
        self,
        command: str,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.requests.append((command, payload, kwargs))
        runtime = {
            "backend": "qwen3-forced-aligner",
            "engine_version": "0.0.6",
            "device": "cuda",
            "compute_type": "bfloat16",
            "quantized": False,
        }
        if command == "preflight":
            return runtime
        callback = kwargs.get("progress_callback")
        if callback:
            callback(50)
            callback(100)
        segments = []
        for index, source in enumerate(payload["segments"]):
            words = []
            if self.with_words:
                words = [
                    {
                        "text": source["text"],
                        "start": source["start"] + 0.1,
                        "end": source["end"] - 0.1,
                    }
                ]
            segments.append({"index": index, **source, "words": words})
        return {"segments": segments, "runtime": runtime}

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _checkpoint(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_bytes(b"{}")
    (root / "model.safetensors").write_bytes(b"weights")
    return root


def _settings(tmp_path: Path, *, backend: str = "faster-whisper") -> ProcessingSettings:
    worker_python = tmp_path / "asr-python.exe"
    worker_python.write_bytes(b"")
    aligner_worker_python = tmp_path / "aligner-python.exe"
    aligner_worker_python.write_bytes(b"")
    return ProcessingSettings(
        backend=backend,
        model_path=tmp_path / "asr",
        aligner="qwen3-forced-aligner",
        aligner_model_path=_checkpoint(tmp_path / "aligner"),
        worker_python_path=worker_python,
        aligner_worker_python_path=aligner_worker_python,
        language="ru",
    )


def _transcript() -> Transcript:
    return Transcript(
        text="первый второй",
        language="ru",
        duration=2.0,
        segments=(
            TranscriptSegment(0.0, 1.0, "первый", segment_id=0),
            TranscriptSegment(1.0, 2.0, "второй", segment_id=1),
        ),
        model="asr-model",
        device="cuda",
        quantized=False,
        metadata={
            "runtime": "faster-whisper",
            "engine_version": "1.2.3",
            "compute_type": "float16",
            "segment_languages": ("ru", "ru"),
        },
    )


def test_alignment_registry_exposes_independent_stage() -> None:
    assert aligner_names() == ("none", "qwen3-forced-aligner")
    assert get_aligner("NONE").requires_exclusive_runtime is False
    assert get_aligner("qwen3-forced-aligner").requires_exclusive_runtime is True
    with pytest.raises(ValidationError, match="Неизвестный aligner"):
        get_aligner("unknown")


def test_qwen_aligner_aligns_original_audio_and_preserves_asr_runtime(tmp_path: Path) -> None:
    client = _FakeClient()
    settings = _settings(tmp_path, backend="faster-whisper")
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    adapter = QwenForcedAlignerAdapter(
        lambda _python, _module: client  # type: ignore[arg-type]
    )
    progress: list[int] = []

    aligned = adapter.align(audio, _transcript(), settings, 2.0, progress.append)

    assert aligned.model == "asr-model"
    assert aligned.device == "cuda"
    assert aligned.metadata["runtime"] == "faster-whisper"
    assert aligned.metadata["aligner"] == "qwen3-forced-aligner"
    assert aligned.metadata["alignment_segment_count"] == 2
    assert aligned.metadata["alignment_max_segment_seconds"] == 180.0
    assert aligned.metadata["alignment_runtime"]["backend"] == "qwen3-forced-aligner"
    assert [word.text for segment in aligned.segments for word in segment.words] == [
        "первый",
        "второй",
    ]
    align_payload = next(payload for command, payload, _ in client.requests if command == "align")
    assert align_payload["audio_path"] == str(audio.resolve())
    assert [segment["start"] for segment in align_payload["segments"]] == [0.0, 1.0]
    assert [segment["language"] for segment in align_payload["segments"]] == ["ru", "ru"]
    assert progress == [0, 5, 50, 95, 100]
    assert adapter.runtime_signature(settings, aligned) == aligned.metadata["alignment_runtime"]
    adapter.unload()
    assert client.shutdown_calls == 1


def test_qwen_aligner_splits_transformers_like_338_second_segment(tmp_path: Path) -> None:
    text = "one two three four five six seven eight"
    transcript = Transcript(
        text=text,
        language="en",
        duration=338.0,
        segments=(TranscriptSegment(0.0, 338.0, text),),
        model="transformers-whisper",
        device="cpu",
        quantized=False,
    )
    audio = tmp_path / "transformers-long.flac"
    audio.write_bytes(b"audio")
    client = _FakeClient()
    adapter = QwenForcedAlignerAdapter(
        lambda _python, _module: client  # type: ignore[arg-type]
    )
    progress: list[int] = []

    aligned = adapter.align(
        audio,
        transcript,
        _settings(tmp_path, backend="transformers"),
        338.0,
        progress.append,
    )

    align_payload = next(payload for command, payload, _ in client.requests if command == "align")
    pieces = align_payload["segments"]
    assert len(pieces) == 2
    assert [(piece["start"], piece["end"]) for piece in pieces] == [
        (0.0, 169.0),
        (169.0, 338.0),
    ]
    assert " ".join(piece["text"] for piece in pieces) == text
    assert all(piece["end"] - piece["start"] <= 180.0 for piece in pieces)
    assert aligned.metadata["segment_languages"] == ("en", "en")
    assert [segment.segment_id for segment in aligned.segments] == [0, 1]
    assert progress == [0, 5, 50, 95, 100]


def test_qwen_aligner_splits_parakeet_like_300_second_segment_at_pauses(
    tmp_path: Path,
) -> None:
    words = (
        TranscriptWord(0.0, 80.0, "первый"),
        TranscriptWord(90.0, 170.0, "второй"),
        TranscriptWord(200.0, 250.0, "третий"),
        TranscriptWord(260.0, 299.0, "четвёртый"),
    )
    transcript = Transcript(
        text="первый второй третий четвёртый",
        language="ru",
        duration=300.0,
        segments=(
            TranscriptSegment(
                0.0,
                300.0,
                "первый второй третий четвёртый",
                words,
            ),
        ),
        model="parakeet-tdt-v3",
        device="cuda",
        quantized=False,
        metadata={"segment_languages": ("ru",)},
    )
    audio = tmp_path / "parakeet-long.flac"
    audio.write_bytes(b"audio")
    client = _FakeClient()
    adapter = QwenForcedAlignerAdapter(
        lambda _python, _module: client  # type: ignore[arg-type]
    )

    aligned = adapter.align(
        audio,
        transcript,
        _settings(tmp_path, backend="parakeet-tdt-v3"),
        300.0,
    )

    align_payload = next(payload for command, payload, _ in client.requests if command == "align")
    pieces = align_payload["segments"]
    assert [(piece["start"], piece["end"], piece["text"]) for piece in pieces] == [
        (0.0, 170.0, "первый второй"),
        (200.0, 299.0, "третий четвёртый"),
    ]
    assert pieces[0]["end"] < pieces[1]["start"]
    assert all(piece["end"] - piece["start"] <= 180.0 for piece in pieces)
    assert aligned.metadata["alignment_segment_count"] == 2


def test_parakeet_and_qwen_aligner_use_independent_worker_python_paths(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, backend="parakeet-tdt-v3")

    assert settings.worker_python_path is not None
    assert settings.aligner_worker_python_path is not None
    assert parakeet_tdt._worker_python_path(settings) == settings.worker_python_path.resolve()
    assert qwen_forced._worker_python_path(settings) == (
        settings.aligner_worker_python_path.resolve()
    )


def test_qwen_aligner_preserves_offsets_for_long_audio_segments(tmp_path: Path) -> None:
    transcript = Transcript(
        text="first second",
        language="en",
        duration=338.0,
        segments=(
            TranscriptSegment(0.0, 175.0, "first", segment_id=0),
            TranscriptSegment(175.0, 338.0, "second", segment_id=1),
        ),
        model="qwen-asr",
        device="cuda",
        quantized=False,
        metadata={"segment_languages": ("en", "en")},
    )
    audio = tmp_path / "long.flac"
    audio.write_bytes(b"audio")
    client = _FakeClient()
    adapter = QwenForcedAlignerAdapter(
        lambda _python, _module: client  # type: ignore[arg-type]
    )

    aligned = adapter.align(audio, transcript, _settings(tmp_path), 338.0)

    assert aligned.segments[0].words[0].start == pytest.approx(0.1)
    assert aligned.segments[1].words[0].start == pytest.approx(175.1)
    assert aligned.segments[1].words[0].end == pytest.approx(337.9)


def test_qwen_aligner_rejects_missing_words(tmp_path: Path) -> None:
    client = _FakeClient(with_words=False)
    settings = _settings(tmp_path)
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    adapter = QwenForcedAlignerAdapter(
        lambda _python, _module: client  # type: ignore[arg-type]
    )
    transcript = _transcript()

    with pytest.raises(AsrModelError, match="не вернул слова"):
        adapter.align(audio, transcript, settings, 2.0)


def test_qwen_aligner_honors_cancel_before_worker_load(tmp_path: Path) -> None:
    created = False

    def factory(_python: Path, _module: str) -> _FakeClient:
        nonlocal created
        created = True
        return _FakeClient()

    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    adapter = QwenForcedAlignerAdapter(factory)  # type: ignore[arg-type]
    transcript = _transcript()
    settings = _settings(tmp_path)

    def cancel_check() -> bool:
        return True

    with pytest.raises(ProcessingCancelled, match="отменено"):
        adapter.align(
            audio,
            transcript,
            settings,
            2.0,
            cancel_check=cancel_check,
        )
    assert created is False


def test_qwen_aligner_honors_cancel_during_long_segment_split(tmp_path: Path) -> None:
    created = False

    def factory(_python: Path, _module: str) -> _FakeClient:
        nonlocal created
        created = True
        return _FakeClient()

    transcript = Transcript(
        text="один два три",
        language="ru",
        duration=300.0,
        segments=(
            TranscriptSegment(
                0.0,
                300.0,
                "один два три",
                (
                    TranscriptWord(0.0, 80.0, "один"),
                    TranscriptWord(100.0, 180.0, "два"),
                    TranscriptWord(220.0, 299.0, "три"),
                ),
            ),
        ),
        model="parakeet-tdt-v3",
        device="cuda",
        quantized=False,
    )
    audio = tmp_path / "cancel-long.flac"
    audio.write_bytes(b"audio")
    checks = 0

    def cancelled_during_split() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 4

    adapter = QwenForcedAlignerAdapter(factory)  # type: ignore[arg-type]
    settings = _settings(tmp_path)
    with pytest.raises(ProcessingCancelled, match="отменено"):
        adapter.align(
            audio,
            transcript,
            settings,
            300.0,
            cancel_check=cancelled_during_split,
        )
    assert checks == 4
    assert created is False
