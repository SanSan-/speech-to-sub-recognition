from __future__ import annotations

import queue
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from speech_to_sub.asr import external_worker
from speech_to_sub.asr.external_worker import (
    ExternalWorkerError,
    PersistentNdjsonWorker,
    decode_frame,
    encode_frame,
)
from speech_to_sub.exceptions import ProcessingCancelled


class _BlockingStream:
    def __init__(self) -> None:
        self._lines: queue.Queue[str | None] = queue.Queue()

    def push(self, line: str) -> None:
        self._lines.put(line)

    def close(self) -> None:
        self._lines.put(None)

    def __iter__(self) -> _BlockingStream:
        return self

    def __next__(self) -> str:
        line = self._lines.get(timeout=1.0)
        if line is None:
            raise StopIteration
        return line


class _FakeStdin:
    def __init__(self, process: _FakeProcess) -> None:
        self._process = process

    def write(self, value: str) -> int:
        self._process.receive(value)
        return len(value)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeProcess:
    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.stdout = _BlockingStream()
        self.stderr = _BlockingStream()
        self.stdin = _FakeStdin(self)
        self.returncode: int | None = None
        self.requests: list[dict[str, Any]] = []
        self.terminate_calls = 0

    def receive(self, line: str) -> None:
        request = decode_frame(line)
        assert request is not None
        self.requests.append(request)
        command = request["command"]
        if self.mode == "timeout" and command != "shutdown":
            return
        self.stdout.push("сторонняя диагностическая строка\n")
        if self.mode == "progress" and command != "shutdown":
            self.stdout.push(
                encode_frame({"id": request["id"], "event": "progress", "progress": 42})
            )
        if self.mode == "error" and command != "shutdown":
            response = {
                "id": request["id"],
                "ok": False,
                "error": {"type": "RuntimeError", "message": "тестовая ошибка"},
            }
        else:
            response = {
                "id": request["id"],
                "ok": True,
                "result": {"echo": request["payload"], "command": command},
            }
        self.stdout.push(encode_frame(response))
        if command == "shutdown":
            self.returncode = 0
            self.stdout.close()
            self.stderr.close()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15
        self.stdout.close()
        self.stderr.close()

    def kill(self) -> None:
        self.returncode = -9
        self.stdout.close()
        self.stderr.close()


def _install_fake_process(
    monkeypatch: pytest.MonkeyPatch,
    process: _FakeProcess,
) -> list[list[str]]:
    launches: list[list[str]] = []

    def fake_popen(args: list[str], **_kwargs: Any) -> _FakeProcess:
        launches.append(args)
        return process

    monkeypatch.setattr(external_worker.subprocess, "Popen", fake_popen)
    return launches


def test_protocol_frame_ignores_foreign_and_malformed_stdout() -> None:
    payload = {"id": "один", "ok": True, "result": {"text": "Привет"}}

    assert decode_frame("обычный вывод\n") is None
    assert decode_frame(f"{external_worker.PROTOCOL_PREFIX}{{broken\n") is None
    assert decode_frame(encode_frame(payload)) == payload


def test_worker_is_persistent_and_serializes_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    launches = _install_fake_process(monkeypatch, process)
    client = PersistentNdjsonWorker(tmp_path / "python.exe", "worker.module")

    first = client.request("one", {"value": 1})
    second = client.request("two", {"value": 2})
    client.shutdown()

    assert first == {"echo": {"value": 1}, "command": "one"}
    assert second == {"echo": {"value": 2}, "command": "two"}
    assert len(launches) == 1
    assert [request["command"] for request in process.requests] == [
        "one",
        "two",
        "shutdown",
    ]


def test_worker_error_preserves_remote_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess("error")
    _install_fake_process(monkeypatch, process)
    client = PersistentNdjsonWorker(tmp_path / "python.exe", "worker.module")

    with pytest.raises(ExternalWorkerError, match="RuntimeError: тестовая ошибка") as error:
        client.request("transcribe")

    assert error.value.error_type == "RuntimeError"
    client.shutdown()


def test_worker_forwards_progress_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess("progress")
    _install_fake_process(monkeypatch, process)
    client = PersistentNdjsonWorker(tmp_path / "python.exe", "worker.module")
    progress: list[int] = []

    result = client.request("transcribe", progress_callback=progress.append)

    assert result["command"] == "transcribe"
    assert progress == [42]
    client.shutdown()


def test_cancel_check_terminates_blocking_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess("timeout")
    _install_fake_process(monkeypatch, process)
    client = PersistentNdjsonWorker(tmp_path / "python.exe", "worker.module")
    checks = 0

    def cancel_check() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    with pytest.raises(ProcessingCancelled, match="отменена во время"):
        client.request("transcribe", cancel_check=cancel_check)

    assert process.terminate_calls == 1
    assert client.is_running is False


def test_cancel_terminates_windows_venv_descendant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess("timeout")
    process.pid = 12345  # type: ignore[attr-defined]
    child = SimpleNamespace(terminate_calls=0, kill_calls=0)

    def terminate() -> None:
        child.terminate_calls += 1

    def kill() -> None:
        child.kill_calls += 1

    child.terminate = terminate
    child.kill = kill
    fake_parent = SimpleNamespace(children=lambda recursive: [child])
    fake_psutil = SimpleNamespace(
        Error=RuntimeError,
        Process=lambda _pid: fake_parent,
        wait_procs=lambda _items, timeout: ([], [child]),
    )
    monkeypatch.setattr(external_worker, "psutil", fake_psutil)
    _install_fake_process(monkeypatch, process)
    client = PersistentNdjsonWorker(tmp_path / "python.exe", "worker.module")
    checks = 0

    def cancel_check() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    with pytest.raises(ProcessingCancelled):
        client.request("transcribe", cancel_check=cancel_check)

    assert child.terminate_calls == 1
    assert child.kill_calls == 1


def test_timeout_terminates_stuck_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess("timeout")
    _install_fake_process(monkeypatch, process)
    client = PersistentNdjsonWorker(
        tmp_path / "python.exe",
        "worker.module",
        timeout_seconds=0.02,
    )

    with pytest.raises(ExternalWorkerError, match="не ответил") as error:
        client.request("transcribe")

    assert error.value.error_type == "TimeoutError"
    assert process.terminate_calls == 1
    assert client.is_running is False
