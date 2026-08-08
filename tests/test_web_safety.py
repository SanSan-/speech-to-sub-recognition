from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from speech_to_sub.exceptions import ValidationError
from speech_to_sub.utils.env_utils import validate_loopback_host
from speech_to_sub.web import app as web_app
from speech_to_sub.web.jobs import BatchJob, _build_terminal_event


@pytest.mark.parametrize("host", ["127.0.0.1", "127.1.2.3", "localhost", "::1", "[::1]"])
def test_loopback_hosts_are_allowed(host: str) -> None:
    assert validate_loopback_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.test"])
def test_non_loopback_hosts_are_rejected(host: str) -> None:
    with pytest.raises(ValidationError, match="loopback"):
        validate_loopback_host(host)


def test_unhandled_worker_error_cannot_report_ok_for_cached_items() -> None:
    job = BatchJob(
        job_id="job",
        paths=("cached.mp4",),
        source_paths=("cached.mp4",),
        settings={},
        items=[{"path": "cached.mp4", "state": "cached", "cached": True}],
    )

    event = _build_terminal_event(job, "worker failed", {})

    assert event["status"] == "partial"
    assert event["error"] == "worker failed"


def test_build_items_rejects_expanded_batch_over_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OversizedService:
        @staticmethod
        def build_items(_paths: list[str], _settings: dict[str, Any]) -> list[dict[str, Any]]:
            return [{"path": f"sample-{index}.mp4"} for index in range(3)]

    monkeypatch.setattr(web_app, "service_api", OversizedService())
    monkeypatch.setattr(web_app, "MAX_BATCH_PATHS", 2)

    with pytest.raises(web_app.HTTPException, match="не более 2"):
        web_app._build_items(["folder"], {})


def test_folder_limit_is_checked_before_service_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class RecordingService:
        @staticmethod
        def build_items(paths: list[str], _settings: dict[str, Any]) -> list[dict[str, Any]]:
            calls.append(paths)
            return []

    monkeypatch.setattr(web_app, "service_api", RecordingService())
    monkeypatch.setattr(
        web_app,
        "collect_media_paths",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            web_app.PickerError("За один запуск можно выбрать не более 2 файлов.")
        ),
    )

    with pytest.raises(web_app.HTTPException, match="не более 2"):
        web_app._build_items([str(tmp_path)], {"recursive": True})

    assert calls == []
