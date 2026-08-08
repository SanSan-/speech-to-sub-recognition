from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any

import pytest

from speech_to_sub.asr.external_worker import decode_frame, encode_frame
from speech_to_sub.workers import common


def test_worker_loop_preserves_request_id_for_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.unload_calls = 0

        def preflight(self, _payload: Mapping[str, Any]) -> dict[str, Any]:
            return {"ready": True}

        def transcribe(
            self,
            payload: Mapping[str, Any],
            progress_callback: common.WorkerProgressCallback | None,
        ) -> dict[str, Any]:
            assert progress_callback is not None
            progress_callback(42)
            return {"text": str(payload["text"])}

        def unload(self) -> dict[str, Any]:
            self.unload_calls += 1
            return {"unloaded": True}

    runtime = FakeRuntime()
    action_id = "action-id"

    def handler(
        current_runtime: FakeRuntime,
        request: Mapping[str, Any],
        progress_callback: common.WorkerProgressCallback | None = None,
    ) -> tuple[dict[str, Any], bool]:
        return common.handle_worker_request(
            current_runtime,
            request,
            progress_callback,
            action_name="transcribe",
            worker_name="Тестовый",
        )

    source = "сторонний stdout\n" + "".join(
        (
            encode_frame({"id": "ping-id", "command": "ping"}),
            encode_frame(
                {
                    "id": action_id,
                    "command": "transcribe",
                    "payload": {"text": "готово"},
                }
            ),
            encode_frame({"id": "shutdown-id", "command": "shutdown"}),
        )
    )
    stdout = io.StringIO()
    monkeypatch.setattr(common.sys, "stdin", io.StringIO(source))
    monkeypatch.setattr(common.sys, "stdout", stdout)
    monkeypatch.setattr(common.sys, "stderr", io.StringIO())

    assert common.run_worker_loop(runtime, handler) == 0

    frames = [decode_frame(line) for line in stdout.getvalue().splitlines()]
    assert [frame["id"] for frame in frames if frame is not None] == [
        "ping-id",
        action_id,
        action_id,
        "shutdown-id",
    ]
    assert frames[1] == {
        "id": action_id,
        "event": "progress",
        "progress": 42,
    }
    assert frames[2] == {
        "id": action_id,
        "ok": True,
        "result": {"text": "готово"},
    }
    assert runtime.unload_calls == 2
