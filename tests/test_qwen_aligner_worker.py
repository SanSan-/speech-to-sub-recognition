from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from speech_to_sub.workers import qwen_aligner_worker


def _checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "aligner"
    path.mkdir()
    (path / "config.json").write_bytes(b"{}")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


def test_worker_aligns_segments_with_global_offsets_and_reuses_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _checkpoint(tmp_path)
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    load_calls: list[tuple[str, dict[str, Any]]] = []
    align_calls: list[dict[str, Any]] = []

    class FakeCuda:
        is_available = staticmethod(lambda: True)
        is_bf16_supported = staticmethod(lambda: True)
        empty_cache = staticmethod(lambda: None)

    fake_torch = SimpleNamespace(
        cuda=FakeCuda(),
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
    )

    class FakeLoadedModel:
        def align(self, **kwargs: Any) -> list[Any]:
            align_calls.append(kwargs)
            return [
                SimpleNamespace(
                    items=(SimpleNamespace(text=kwargs["text"], start_time=0.1, end_time=0.8),)
                )
            ]

    class FakeAligner:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> FakeLoadedModel:
            load_calls.append((path, kwargs))
            return FakeLoadedModel()

    def normalize(_path: str) -> list[list[float]]:
        return [[0.0] * 48_000]

    monkeypatch.setattr(
        qwen_aligner_worker,
        "_import_qwen_aligner_runtime",
        lambda: (fake_torch, FakeAligner, normalize, "0.0.6"),
    )
    runtime = qwen_aligner_worker.QwenAlignerWorkerRuntime()
    payload = {
        "audio_path": str(audio),
        "model_path": str(model_path),
        "device": "auto",
        "segments": [
            {"text": "one", "language": "en", "start": 0.0, "end": 1.0},
            {"text": "two", "language": "en", "start": 1.0, "end": 2.0},
        ],
    }
    progress: list[int] = []

    assert runtime.preflight(payload)["backend"] == "qwen3-forced-aligner"
    first = runtime.align(payload, progress.append)
    runtime.align(payload)

    assert first["segments"][0]["words"] == [
        {"text": "one", "start": 0.1, "end": 0.8}
    ]
    assert first["segments"][1]["words"] == [
        {"text": "two", "start": 1.1, "end": 1.8}
    ]
    assert progress == [50, 100]
    assert len(load_calls) == 1
    assert load_calls[0][1]["local_files_only"] is True
    assert [call["language"] for call in align_calls] == ["English"] * 4
    assert all(len(call["audio"][0]) == 16_000 for call in align_calls)


def test_worker_rejects_segment_over_180_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _checkpoint(tmp_path)
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")

    class FakeCuda:
        is_available = staticmethod(lambda: False)
        empty_cache = staticmethod(lambda: None)

    fake_torch = SimpleNamespace(
        cuda=FakeCuda(),
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
    )

    class SizedWaveform:
        def __len__(self) -> int:
            return 181 * 16_000

    monkeypatch.setattr(
        qwen_aligner_worker,
        "_import_qwen_aligner_runtime",
        lambda: (fake_torch, object(), lambda _path: [SizedWaveform()], "0.0.6"),
    )
    runtime = qwen_aligner_worker.QwenAlignerWorkerRuntime()

    with pytest.raises(ValueError, match="180"):
        runtime.align(
            {
                "audio_path": str(audio),
                "model_path": str(model_path),
                "device": "cpu",
                "segments": [
                    {"text": "too long", "language": "en", "start": 0, "end": 181}
                ],
            }
        )
