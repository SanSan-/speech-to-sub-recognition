from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from speech_to_sub.asr.parakeet_tdt import (
    ParakeetTdtBackend,
    _Window,
    _group_subword_tokens,
    _merge_window_segments,
    _tokens_to_words,
)
from speech_to_sub.exceptions import ProcessingCancelled
from speech_to_sub.models import ProcessingSettings, TranscriptSegment, TranscriptWord


RUNTIME = {
    "backend": "parakeet-tdt-v3",
    "engine_version": "5.14.1",
    "device": "cuda",
    "compute_type": "float16",
    "quantized": False,
}


class _FakeWorker:
    def __init__(self, python_path: Path, module: str) -> None:
        self.python_path = python_path
        self.module = module
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.shutdown_called = False

    def request(
        self,
        command: str,
        payload: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        self.requests.append((command, payload))
        if command == "preflight":
            return dict(RUNTIME)
        if command == "transcribe-window":
            return {
                "text": "Hello world.",
                "language": "English",
                "tokens": [
                    {"token": "H", "start": 0.1, "end": 0.2},
                    {"token": "ello", "start": 0.2, "end": 0.4},
                    {"token": " world", "start": 0.5, "end": 0.8},
                    {"token": ".", "start": 0.8, "end": 0.8},
                ],
                "runtime": dict(RUNTIME),
            }
        if command == "shutdown":
            return {"shutdown": True}
        raise AssertionError(command)

    def shutdown(self) -> None:
        self.shutdown_called = True


def _checkpoint(path: Path) -> Path:
    path.mkdir()
    for filename in ("config.json", "processor_config.json", "tokenizer.json"):
        (path / filename).write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


def test_subword_tokens_are_grouped_into_words_and_punctuation() -> None:
    grouped = _group_subword_tokens(
        (
            {"token": "W", "start": 0.1, "end": 0.2},
            {"token": "ord", "start": 0.2, "end": 0.4},
            {"token": " two", "start": 0.5, "end": 0.8},
            {"token": ".", "start": 0.8, "end": 0.8},
        )
    )

    assert grouped == [("Word", 0.1, 0.4), ("two.", 0.5, 0.8)]


def test_overlap_commit_keeps_each_word_once() -> None:
    tokens = (
        {"token": " before", "start": 0.1, "end": 0.5},
        {"token": " keep", "start": 1.2, "end": 1.6},
    )

    words = _tokens_to_words(
        tokens,
        _Window(offset=28.0, duration=30.0, is_final=False),
        overlap_seconds=2.0,
        audio_duration=90.0,
    )

    assert [word.text for word in words] == ["keep"]
    assert words[0].start == pytest.approx(29.2)


def test_window_merge_removes_duplicate_and_clamps_other_overlap() -> None:
    segments = (
        TranscriptSegment(
            start=10.0,
            end=11.0,
            text="same",
            words=(TranscriptWord(start=10.0, end=11.0, text="same"),),
            segment_id=4,
        ),
        TranscriptSegment(
            start=10.8,
            end=12.0,
            text="same next",
            words=(
                TranscriptWord(start=10.8, end=11.2, text="Same"),
                TranscriptWord(start=10.9, end=12.0, text="next", probability=0.8),
            ),
            segment_id=5,
        ),
    )

    merged = _merge_window_segments(segments)

    assert [word.text for segment in merged for word in segment.words] == ["same", "next"]
    assert merged[1].words[0].start == pytest.approx(11.0)
    assert merged[1].words[0].probability == pytest.approx(0.8)
    assert [segment.segment_id for segment in merged] == [0, 1]


def test_backend_normalizes_worker_response_and_reuses_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _checkpoint(tmp_path / "model")
    python_path = tmp_path / "python.exe"
    python_path.write_bytes(b"")
    audio_path = tmp_path / "audio.flac"
    audio_path.write_bytes(b"audio")
    clients: list[_FakeWorker] = []

    def factory(path: Path, module: str) -> _FakeWorker:
        client = _FakeWorker(path, module)
        clients.append(client)
        return client

    monkeypatch.setenv("PARAKEET_ASR_PYTHON", str(python_path))
    backend = ParakeetTdtBackend(worker_factory=factory)  # type: ignore[arg-type]
    settings = ProcessingSettings(
        backend="parakeet-tdt-v3",
        model_path=model_path,
        language="auto",
        device="auto",
    )

    transcript = backend.transcribe(audio_path, settings, duration=8.0)

    assert transcript.text == "Hello world."
    assert transcript.language == "en"
    assert [word.text for word in transcript.segments[0].words] == ["Hello", "world."]
    assert backend.runtime_signature(settings) == RUNTIME
    assert len(clients) == 1
    assert [command for command, _payload in clients[0].requests] == [
        "preflight",
        "transcribe-window",
    ]
    backend.unload()
    assert clients[0].shutdown_called is True


def test_backend_propagates_cooperative_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _checkpoint(tmp_path / "model")
    python_path = tmp_path / "python.exe"
    python_path.write_bytes(b"")
    audio_path = tmp_path / "audio.flac"
    audio_path.write_bytes(b"audio")
    monkeypatch.setenv("PARAKEET_ASR_PYTHON", str(python_path))
    backend = ParakeetTdtBackend(worker_factory=_FakeWorker)  # type: ignore[arg-type]
    settings = ProcessingSettings(
        backend="parakeet-tdt-v3",
        model_path=model_path,
        long_form_window_seconds=30,
        long_form_overlap_seconds=2,
    )

    def cancel_after_window(progress: int) -> None:
        if progress > 0:
            raise ProcessingCancelled("Отмена теста")

    with pytest.raises(ProcessingCancelled, match="Отмена теста"):
        backend.transcribe(
            audio_path,
            settings,
            duration=65.0,
            progress_callback=cancel_after_window,
        )
