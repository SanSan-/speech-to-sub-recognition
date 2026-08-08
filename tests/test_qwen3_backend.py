from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from speech_to_sub.asr.external_worker import ExternalWorkerError
from speech_to_sub.asr.qwen3 import Qwen3AsrBackend
from speech_to_sub.exceptions import ValidationError
from speech_to_sub.models import ProcessingSettings
from speech_to_sub.workers import qwen_worker


class _FakeClient:
    def __init__(self, *, fail_cuda: bool = False) -> None:
        self.fail_cuda = fail_cuda
        self.requests: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        self.shutdown_calls = 0

    def request(
        self,
        command: str,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.requests.append((command, payload, kwargs))
        device = "cpu" if payload.get("device") == "cpu" else "cuda"
        runtime = {
            "backend": "qwen3-asr",
            "engine_version": "0.0.6",
            "device": device,
            "compute_type": "float32" if device == "cpu" else "bfloat16",
            "quantized": False,
        }
        if command == "preflight":
            return runtime
        if self.fail_cuda and device == "cuda":
            raise ExternalWorkerError("CUDA OOM", error_type="OutOfMemoryError")
        progress_callback = kwargs.get("progress_callback")
        if progress_callback:
            progress_callback(50)
            progress_callback(100)
        return {
            "text": "Hello world.",
            "language": "English",
            "chunks": [
                {"text": "Hello ", "language": "English", "start": 0.0, "end": 1.0},
                {"text": "world.", "language": "English", "start": 1.0, "end": 2.0},
            ],
            "runtime": runtime,
        }

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _checkpoint(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir()
    (path / "config.json").write_bytes(b"{}")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


def _settings(tmp_path: Path, *, allow_cpu_fallback: bool = False) -> ProcessingSettings:
    worker_python = tmp_path / "python.exe"
    worker_python.write_bytes(b"")
    return ProcessingSettings(
        backend="qwen3-asr",
        model_path=_checkpoint(tmp_path, "asr"),
        aligner="qwen3-forced-aligner",
        aligner_model_path=tmp_path / "unused-aligner",
        worker_python_path=worker_python,
        language="auto",
        device="auto",
        allow_cpu_fallback=allow_cpu_fallback,
    )


def test_backend_returns_only_asr_chunks_and_progress(tmp_path: Path) -> None:
    client = _FakeClient()
    settings = _settings(tmp_path)
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    backend = Qwen3AsrBackend(lambda _python, _module: client)  # type: ignore[arg-type]
    progress: list[int] = []

    expected = backend.expected_runtime_signature(settings)
    transcript = backend.transcribe(audio, settings, 2.0, progress.append)

    assert expected["device"] == "cuda"
    assert transcript.language == "en"
    assert transcript.metadata["aligner"] == "none"
    assert transcript.metadata["word_timestamps"] is False
    assert transcript.metadata["segment_languages"] == ("en", "en")
    assert [(item.start, item.end, item.words) for item in transcript.segments] == [
        (0.0, 1.0, ()),
        (1.0, 2.0, ()),
    ]
    transcribe_payload = next(payload for command, payload, _ in client.requests if command == "transcribe")
    assert "aligner_path" not in transcribe_payload
    assert progress == [0, 5, 50, 95, 100]
    assert backend.runtime_signature(settings) is not None
    backend.unload()
    assert client.shutdown_calls == 1


def test_preflight_rejects_checkpoint_without_weights_before_worker(tmp_path: Path) -> None:
    model_path = tmp_path / "asr"
    model_path.mkdir()
    (model_path / "config.json").write_bytes(b"{}")
    worker_python = tmp_path / "python.exe"
    worker_python.write_bytes(b"")
    settings = ProcessingSettings(model_path=model_path, worker_python_path=worker_python)
    created = False

    def factory(_python: Path, _module: str) -> _FakeClient:
        nonlocal created
        created = True
        return _FakeClient()

    backend = Qwen3AsrBackend(factory)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="отсутствуют локальные веса"):
        backend.preflight(settings)
    assert created is False


def test_preflight_rejects_missing_indexed_shard(tmp_path: Path) -> None:
    model_path = tmp_path / "asr"
    model_path.mkdir()
    (model_path / "config.json").write_bytes(b"{}")
    (model_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"layer":"missing-00001.safetensors"}}',
        encoding="utf-8",
    )
    worker_python = tmp_path / "python.exe"
    worker_python.write_bytes(b"")
    settings = ProcessingSettings(model_path=model_path, worker_python_path=worker_python)
    backend = Qwen3AsrBackend()

    with pytest.raises(ValidationError, match="отсутствуют шарды"):
        backend.preflight(settings)


def test_cuda_error_restarts_worker_and_retries_on_cpu(tmp_path: Path) -> None:
    settings = _settings(tmp_path, allow_cpu_fallback=True)
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    clients = [_FakeClient(fail_cuda=True), _FakeClient()]

    def factory(_python: Path, _module: str) -> _FakeClient:
        return clients.pop(0)

    first, second = clients
    transcript = Qwen3AsrBackend(factory).transcribe(audio, settings, 2.0)  # type: ignore[arg-type]

    assert transcript.device == "cpu"
    assert first.shutdown_calls == 1
    assert any(
        command == "preflight" and payload["device"] == "cpu"
        for command, payload, _kwargs in second.requests
    )


def test_worker_splits_audio_and_never_loads_forced_aligner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _checkpoint(tmp_path, "worker-asr")
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    load_calls: list[tuple[str, dict[str, Any]]] = []
    transcribe_calls: list[dict[str, Any]] = []

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
        def transcribe(self, **kwargs: Any) -> list[Any]:
            transcribe_calls.append(kwargs)
            number = len(transcribe_calls)
            return [SimpleNamespace(text=f"part {number}", language="English")]

    class FakeQwenModel:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> FakeLoadedModel:
            load_calls.append((path, kwargs))
            return FakeLoadedModel()

    def normalize(_path: str) -> list[list[float]]:
        return [[0.0] * 32_000]

    def split(**_kwargs: Any) -> list[tuple[list[float], float]]:
        return [([0.0] * 16_000, 0.0), ([0.0] * 16_000, 1.0)]

    monkeypatch.setattr(
        qwen_worker,
        "_import_qwen_runtime",
        lambda: (fake_torch, FakeQwenModel, normalize, split, "0.0.6"),
    )
    runtime = qwen_worker.QwenWorkerRuntime()
    payload = {
        "audio_path": str(audio),
        "model_path": str(model_path),
        "language": "en",
        "device": "auto",
    }
    progress: list[int] = []

    assert runtime.preflight(payload)["compute_type"] == "bfloat16"
    first = runtime.transcribe(payload, progress.append)
    second = runtime.transcribe(payload)

    assert first["chunks"] == [
        {"text": "part 1", "language": "English", "start": 0.0, "end": 1.0},
        {"text": "part 2", "language": "English", "start": 1.0, "end": 2.0},
    ]
    assert first["text"] == "part 1 part 2"
    assert len(second["chunks"]) == 2
    assert progress == [50, 100]
    assert len(load_calls) == 1
    assert "forced_aligner" not in load_calls[0][1]
    assert all(call["return_time_stamps"] is False for call in transcribe_calls)


def test_worker_rejects_incompatible_qwen_package_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _checkpoint(tmp_path, "version-asr")
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(
        qwen_worker,
        "_import_qwen_runtime",
        lambda: (fake_torch, object(), object(), object(), "0.0.5"),
    )
    runtime = qwen_worker.QwenWorkerRuntime()
    payload = {"model_path": str(model_path), "device": "cpu"}

    with pytest.raises(RuntimeError, match="qwen-asr==0.0.6"):
        runtime.preflight(payload)
