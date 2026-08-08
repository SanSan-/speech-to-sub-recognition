from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from speech_to_sub.asr import registry as asr_registry
from speech_to_sub.web import app as web_app
from speech_to_sub.web.__main__ import _read_port
from speech_to_sub.web.jobs import JobNotFoundError, JobRegistry
from speech_to_sub.web.picker import PickSelection, collect_media_paths


class FakeService:
    """Лёгкая замена batch service без FFmpeg и модели."""

    def __init__(
        self,
        processor: Callable[[list[str], dict[str, Any], Any, Any], list[dict[str, Any]]]
        | None = None,
    ) -> None:
        self.processor = processor or self._successful_processor
        self.build_calls: list[tuple[list[str], dict[str, Any]]] = []
        self.preflight_settings: list[dict[str, Any]] = []
        self.unloaded = False

    def get_preflight_status(self, settings: dict[str, Any]) -> dict[str, Any]:
        self.preflight_settings.append(dict(settings))
        return {
            "status": "ok",
            "ffmpeg": {"path": "ffmpeg", "available": True},
            "ffprobe": {"path": "ffprobe", "available": True},
            "backend": {
                "id": settings["backend"],
                "available": True,
                "error": None,
            },
            "model": {
                "path": settings["model_path"],
                "available": True,
                "error": None,
            },
        }

    def build_items(
        self,
        paths: list[str],
        settings: dict[str, Any],
    ) -> list[dict[str, Any]]:
        self.build_calls.append((list(paths), dict(settings)))
        return [
            {
                "path": path,
                "name": Path(path).name,
                "state": "queued",
                "progress": 0,
                "cached": False,
                "skipped": False,
            }
            for path in paths
        ]

    def process_paths(
        self,
        paths: list[str],
        settings: dict[str, Any],
        emit_event: Any,
        log: Any,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        self.cancel_check = cancel_check
        return self.processor(paths, settings, emit_event, log)

    def unload(self) -> None:
        self.unloaded = True

    @staticmethod
    def _successful_processor(
        paths: list[str],
        settings: dict[str, Any],
        emit_event: Any,
        log: Any,
    ) -> list[dict[str, Any]]:
        del settings
        results = []
        for path in paths:
            log.info("Тестовое распознавание: %s", Path(path).name)
            emit_event(
                {
                    "type": "file",
                    "path": path,
                    "state": "done",
                    "stage": "Готово",
                    "progress": 100,
                    "srt_output": str(Path(path).with_suffix(".srt")),
                }
            )
            results.append(
                {
                    "path": path,
                    "name": Path(path).name,
                    "state": "done",
                    "srt_output": str(Path(path).with_suffix(".srt")),
                }
            )
        return results


@pytest.fixture(autouse=True)
def clean_job_registry(monkeypatch: pytest.MonkeyPatch) -> Any:
    registry = JobRegistry()
    monkeypatch.setattr(web_app, "job_registry", registry)
    yield
    snapshot = registry.current_snapshot()
    if snapshot.get("active"):
        job = registry.get(str(snapshot["job_id"]))
        assert job.finished.wait(timeout=3)
    registry.reset_for_tests()


@pytest.fixture
def fake_service(monkeypatch: pytest.MonkeyPatch) -> FakeService:
    service = FakeService()
    monkeypatch.setattr(web_app, "service_api", service)
    return service


def test_static_page_and_config_have_no_secret_fields(fake_service: FakeService) -> None:
    del fake_service
    client = TestClient(web_app.app)

    index = client.get("/")
    config = client.get("/api/ui-config")
    health = client.get("/api/health")

    assert index.status_code == 200
    assert "Speech to Sub" in index.text
    assert client.get("/static/app.js").status_code == 200
    assert config.status_code == 200
    assert health.status_code == 200
    assert health.json()["service"] == "speech-to-sub-recognition"
    assert health.json()["version"] == "1.4.0"
    assert client.get("/openapi.json").json()["info"]["version"] == "1.4.0"
    assert health.json()["backend"]["id"] == "faster-whisper"
    defaults = config.json()["defaults"]
    assert defaults["backend"] == "faster-whisper"
    assert defaults["device"] in {"auto", "cuda", "cpu"}
    assert defaults["model_path"]
    assert {item["value"] for item in config.json()["backends"]} == {
        "transformers",
        "faster-whisper",
        "parakeet-tdt-v3",
        "qwen3-asr",
    }
    assert {item["value"] for item in config.json()["aligners"]} == {
        "none",
        "qwen3-forced-aligner",
    }
    qwen_aligner = next(
        item
        for item in config.json()["aligners"]
        if item["value"] == "qwen3-forced-aligner"
    )
    assert set(qwen_aligner["compatible_backends"]) == {
        "transformers",
        "faster-whisper",
        "parakeet-tdt-v3",
        "qwen3-asr",
    }
    backend_paths = {
        item["value"]: item["model_path"] for item in config.json()["backends"]
    }
    assert backend_paths["transformers"].endswith("whisper-large-v3")
    assert backend_paths["faster-whisper"].endswith("whisper-large-v3-ct2")
    assert defaults["long_form_window_seconds"] == 300
    assert defaults["vad_filter"] is True
    assert 'id="backend"' in index.text
    serialized = json.dumps(
        {"config": config.json(), "health": health.json()},
        ensure_ascii=False,
    ).casefold()
    assert "api_key" not in serialized
    assert "password" not in serialized
    assert "token" not in serialized


@pytest.mark.parametrize("value", ["1", "7862", "65535"])
def test_python_web_entrypoints_accept_valid_ports(value: str) -> None:
    assert _read_port(value) == int(value)
    assert web_app._parse_web_port(value) == int(value)


@pytest.mark.parametrize("value", ["0", "65536", "не-число"])
def test_python_web_entrypoints_reject_invalid_ports(value: str) -> None:
    with pytest.raises(ValueError, match="WEB_PORT"):
        _read_port(value)
    with pytest.raises(ValueError, match="WEB_PORT"):
        web_app._parse_web_port(value)


def test_pick_uses_backend_dialog_and_builds_cards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_service: FakeService,
) -> None:
    media_path = tmp_path / "Лекция 01.mp4"
    received: dict[str, Any] = {}

    def fake_picker(kind: str, recursive: bool = False) -> PickSelection:
        received.update(kind=kind, recursive=recursive)
        return PickSelection(mode="files", paths=(media_path,))

    monkeypatch.setattr(web_app, "pick_paths", fake_picker)
    client = TestClient(web_app.app)

    response = client.post(
        "/api/pick",
        json={"kind": "file", "settings": {"recursive": True}},
    )

    assert response.status_code == 200
    assert received == {"kind": "file", "recursive": True}
    assert response.json()["items"][0]["path"] == str(media_path)
    assert fake_service.build_calls[0][0] == [str(media_path)]


def test_pick_and_refresh_are_serialized_on_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_service: FakeService,
) -> None:
    picked_path = tmp_path / "выбранный.mp4"
    refreshed_path = tmp_path / "обновлённый.mkv"
    picker_entered = threading.Event()
    release_picker = threading.Event()
    refresh_build_entered = threading.Event()
    responses: dict[str, int] = {}

    def blocking_picker(kind: str, recursive: bool = False) -> PickSelection:
        del kind, recursive
        picker_entered.set()
        assert release_picker.wait(timeout=3)
        return PickSelection(mode="files", paths=(picked_path,))

    original_build_items = fake_service.build_items

    def tracked_build_items(paths: list[str], settings: dict[str, Any]) -> list[dict[str, Any]]:
        if paths == [str(refreshed_path.resolve(strict=False))]:
            refresh_build_entered.set()
        return original_build_items(paths, settings)

    monkeypatch.setattr(web_app, "pick_paths", blocking_picker)
    monkeypatch.setattr(fake_service, "build_items", tracked_build_items)

    def call_pick() -> None:
        with TestClient(web_app.app) as client:
            responses["pick"] = client.post(
                "/api/pick",
                json={"kind": "file", "settings": {}},
            ).status_code

    def call_refresh() -> None:
        with TestClient(web_app.app) as client:
            responses["refresh"] = client.post(
                "/api/refresh",
                json={"paths": [str(refreshed_path)], "settings": {}},
            ).status_code

    pick_thread = threading.Thread(target=call_pick, daemon=True)
    refresh_thread = threading.Thread(target=call_refresh, daemon=True)
    pick_thread.start()
    assert picker_entered.wait(timeout=2)
    refresh_thread.start()
    try:
        assert not refresh_build_entered.wait(timeout=0.2)
    finally:
        release_picker.set()
    pick_thread.join(timeout=3)
    refresh_thread.join(timeout=3)

    assert not pick_thread.is_alive()
    assert not refresh_thread.is_alive()
    assert responses == {"pick": 200, "refresh": 200}
    assert refresh_build_entered.is_set()


def test_state_changing_api_rejects_cross_origin_and_non_loopback_host(
    fake_service: FakeService,
) -> None:
    payload = {"paths": [r"D:\Media\lesson.mp4"], "settings": {}}
    client = TestClient(web_app.app)

    assert client.post("/api/refresh", json=payload).status_code == 200
    assert client.post(
        "/api/refresh",
        json=payload,
        headers={"Origin": "https://attacker.example"},
    ).status_code == 403
    assert client.post(
        "/api/refresh",
        json=payload,
        headers={"Host": "attacker.example"},
    ).status_code == 403

    local_client = TestClient(web_app.app, base_url="http://127.0.0.1:7862")
    assert local_client.post(
        "/api/refresh",
        json=payload,
        headers={"Origin": "http://127.0.0.1:7862"},
    ).status_code == 200
    assert local_client.post(
        "/api/refresh",
        json=payload,
        headers={"Origin": "http://127.0.0.1:9999"},
    ).status_code == 403
    assert fake_service.build_calls


def test_state_changing_api_rejects_non_loopback_client_address(
    fake_service: FakeService,
) -> None:
    del fake_service
    client = TestClient(
        web_app.app,
        base_url="http://127.0.0.1:7862",
        client=("203.0.113.10", 50_000),
    )

    response = client.post("/api/jobs/missing-job/cancel")

    assert response.status_code == 403


def test_local_api_rejects_remote_get_and_dns_rebinding_host(
    fake_service: FakeService,
) -> None:
    del fake_service
    remote_client = TestClient(
        web_app.app,
        base_url="http://127.0.0.1:7862",
        client=("203.0.113.10", 50_000),
    )
    rebinding_client = TestClient(
        web_app.app,
        base_url="http://attacker.example:7862",
        client=("127.0.0.1", 50_000),
    )

    assert remote_client.get("/api/health").status_code == 403
    assert rebinding_client.get("/api/jobs").status_code == 403


def test_settings_reject_unknown_secret_field(fake_service: FakeService) -> None:
    del fake_service
    client = TestClient(web_app.app)

    response = client.post(
        "/api/refresh",
        json={
            "paths": [r"D:\Media\lesson.mp4"],
            "settings": {"openai_api_key": "не-должен-приниматься"},
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("backend", "model_suffix"),
    (
        ("transformers", "whisper-large-v3"),
        ("faster-whisper", "whisper-large-v3-ct2"),
        ("parakeet-tdt-v3", "parakeet-tdt-0.6b-v3"),
        ("qwen3-asr", "Qwen3-ASR-0.6B"),
    ),
)
def test_web_settings_forward_selected_backend(
    fake_service: FakeService,
    backend: str,
    model_suffix: str,
) -> None:
    client = TestClient(web_app.app)

    response = client.post(
        "/api/refresh",
        json={
            "paths": [r"D:\Media\lesson.mp4"],
            "settings": {"backend": backend},
        },
    )

    assert response.status_code == 200
    assert fake_service.build_calls[0][1]["backend"] == backend
    assert fake_service.build_calls[0][1]["model_path"].endswith(model_suffix)


def test_web_settings_forward_independent_qwen_aligner_without_secret_fields(
    fake_service: FakeService,
) -> None:
    client = TestClient(web_app.app)
    response = client.post(
        "/api/refresh",
        json={
            "paths": [r"D:\Media\lesson.mp4"],
            "settings": {
                "backend": "faster-whisper",
                "model_path": r"D:\Models\whisper-large-v3-ct2",
                "aligner": "qwen3-forced-aligner",
                "aligner_model_path": r"D:\Models\Qwen3-ForcedAligner-0.6B",
                "worker_python_path": r"D:\Runtime\parakeet\python.exe",
                "aligner_worker_python_path": r"D:\Runtime\qwen\python.exe",
            },
        },
    )

    assert response.status_code == 200
    forwarded = fake_service.build_calls[0][1]
    assert forwarded["backend"] == "faster-whisper"
    assert forwarded["aligner"] == "qwen3-forced-aligner"
    assert forwarded["worker_python_path"].endswith("parakeet\\python.exe")
    assert forwarded["aligner_worker_python_path"].endswith("qwen\\python.exe")
    serialized = json.dumps(forwarded, ensure_ascii=False).casefold()
    assert "api_key" not in serialized
    assert "token" not in serialized


@pytest.mark.parametrize(
    "backend",
    ("transformers", "faster-whisper", "parakeet-tdt-v3", "qwen3-asr"),
)
def test_web_transcribe_smoke_accepts_every_registered_backend(
    tmp_path: Path,
    fake_service: FakeService,
    backend: str,
) -> None:
    client = TestClient(web_app.app)
    response = client.post(
        "/api/transcribe",
        json={
            "paths": [str(tmp_path / f"{backend}.mp4")],
            "settings": {"backend": backend},
        },
    )

    assert response.status_code == 200
    job = web_app.job_registry.get(response.json()["job_id"])
    assert job.finished.wait(timeout=3)
    snapshot = job.snapshot(active=False)
    assert snapshot["status"] == "ok"
    assert snapshot["settings"]["backend"] == backend
    assert fake_service.build_calls[0][1]["backend"] == backend


def test_web_settings_reject_unknown_backend(fake_service: FakeService) -> None:
    del fake_service
    client = TestClient(web_app.app)

    response = client.post(
        "/api/refresh",
        json={
            "paths": [r"D:\Media\lesson.mp4"],
            "settings": {"backend": "unknown"},
        },
    )

    assert response.status_code == 422


def test_folder_collection_filters_media_and_respects_recursive_mode(tmp_path: Path) -> None:
    nested = tmp_path / "Вложенный каталог"
    nested.mkdir()
    top_video = tmp_path / "Лекция.mp4"
    nested_audio = nested / "Дорожка.FLAC"
    ignored = tmp_path / "заметки.txt"
    top_video.write_bytes(b"video")
    nested_audio.write_bytes(b"audio")
    ignored.write_text("не медиа", encoding="utf-8")

    assert collect_media_paths(tmp_path, recursive=False) == (top_video,)
    expected = tuple(sorted((top_video, nested_audio), key=lambda path: str(path).casefold()))
    assert collect_media_paths(tmp_path, recursive=True) == expected


def test_partial_job_stream_and_terminal_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_path = str(tmp_path / "first.mp4")
    second_path = str(tmp_path / "second.wav")

    def partial_processor(paths, settings, emit_event, log):
        del settings
        log.info("Проверка пакетного журнала.")
        emit_event(
            {
                "type": "file",
                "path": paths[0],
                "state": "transcribing",
                "stage": "ASR",
                "progress": 42,
            }
        )
        results = [
            {"path": paths[0], "state": "done", "srt_output": f"{paths[0]}.srt"},
            {"path": paths[1], "state": "error", "error": "Тестовая ошибка файла."},
        ]
        return results

    monkeypatch.setattr(web_app, "service_api", FakeService(partial_processor))
    client = TestClient(web_app.app)
    started = client.post(
        "/api/transcribe",
        json={"paths": [first_path, second_path], "settings": {}},
    )

    assert started.status_code == 200
    job_id = started.json()["job_id"]
    response = client.get(f"/api/stream/{job_id}")
    events = _parse_sse_events(response.text)
    done = events[-1]["payload"]

    assert response.status_code == 200
    assert events[0]["payload"]["type"] == "job"
    assert any(event["payload"].get("stage") == "ASR" for event in events)
    assert any(
        "Проверка пакетного журнала" in event["payload"].get("message", "")
        for event in events
    )
    assert done == {
        "type": "done",
        "status": "partial",
        "total": 2,
        "done": 2,
        "cached": 0,
        "skipped": 0,
        "failed": 1,
        "cancelled": 0,
    }

    snapshot = client.get("/api/active-job").json()
    assert snapshot["active"] is False
    assert snapshot["terminal"] is True
    assert snapshot["job_id"] == job_id
    assert snapshot["status"] == "partial"
    assert snapshot["failed"] == 1
    assert snapshot["latest_event_id"] == events[-1]["id"]

    last_event_id = events[-1]["id"]
    reconnect = client.get(
        f"/api/stream/{job_id}",
        headers={"Last-Event-ID": str(last_event_id)},
    )
    reconnect_events = _parse_sse_events(reconnect.text)
    assert reconnect.status_code == 200
    assert reconnect_events == []

    cursor_reconnect = client.get(
        f"/api/stream/{job_id}",
        params={"cursor": events[-2]["id"]},
    )
    cursor_events = _parse_sse_events(cursor_reconnect.text)
    assert [event["payload"]["type"] for event in cursor_events] == ["done"]


def test_second_job_and_unload_are_blocked_while_worker_is_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_processor(paths, settings, emit_event, log):
        del settings, emit_event, log
        entered.set()
        assert release.wait(timeout=3)
        return [{"path": paths[0], "state": "done"}]

    service = FakeService(blocking_processor)
    monkeypatch.setattr(web_app, "service_api", service)
    client = TestClient(web_app.app)
    payload = {"paths": [str(tmp_path / "sample.mp4")], "settings": {}}

    first = client.post("/api/transcribe", json=payload)
    assert first.status_code == 200
    assert entered.wait(timeout=2)
    try:
        second = client.post("/api/transcribe", json=payload)
        unload = client.post("/api/unload", json={})
        assert second.status_code == 409
        assert unload.status_code == 409
        assert len(service.build_calls) == 1
    finally:
        release.set()

    job = web_app.job_registry.get(first.json()["job_id"])
    assert job.finished.wait(timeout=3)


def test_unload_delegates_to_lazy_service(fake_service: FakeService) -> None:
    client = TestClient(web_app.app)

    response = client.post("/api/unload", json={})

    assert response.status_code == 200
    assert fake_service.unloaded is True


def test_service_adapter_unloads_all_registry_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(asr_registry, "unload_backends", lambda: calls.append(True))

    web_app.ServiceAdapter.unload()

    assert calls == [True]


def test_start_reservation_blocks_parallel_probe_before_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingBuildService(FakeService):
        def build_items(self, paths, settings):
            self.build_calls.append((list(paths), dict(settings)))
            entered.set()
            assert release.wait(timeout=3)
            return FakeService.build_items(self, paths, settings)

    service = BlockingBuildService()
    monkeypatch.setattr(web_app, "service_api", service)
    payload = {"paths": [str(tmp_path / "sample.mp4")], "settings": {}}
    first_response: dict[str, Any] = {}

    def start_first_request() -> None:
        first_response["value"] = TestClient(web_app.app).post(
            "/api/transcribe",
            json=payload,
        )

    thread = threading.Thread(target=start_first_request)
    thread.start()
    assert entered.wait(timeout=2)
    try:
        second = TestClient(web_app.app).post("/api/transcribe", json=payload)
        unload = TestClient(web_app.app).post("/api/unload", json={})
        assert second.status_code == 409
        assert unload.status_code == 409
        assert len(service.build_calls) == 1
    finally:
        release.set()
        thread.join(timeout=3)

    assert not thread.is_alive()
    assert first_response["value"].status_code == 200


def test_unload_reservation_blocks_start_before_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingUnloadService(FakeService):
        def unload(self) -> None:
            entered.set()
            assert release.wait(timeout=3)
            self.unloaded = True

    service = BlockingUnloadService()
    monkeypatch.setattr(web_app, "service_api", service)
    unload_response: dict[str, Any] = {}

    def unload_in_background() -> None:
        unload_response["value"] = TestClient(web_app.app).post("/api/unload", json={})

    thread = threading.Thread(target=unload_in_background)
    thread.start()
    assert entered.wait(timeout=2)
    try:
        started = TestClient(web_app.app).post(
            "/api/transcribe",
            json={"paths": [str(tmp_path / "sample.mp4")], "settings": {}},
        )
        assert started.status_code == 409
        assert service.build_calls == []
    finally:
        release.set()
        thread.join(timeout=3)

    assert not thread.is_alive()
    assert unload_response["value"].status_code == 200


def test_active_snapshot_cursor_skips_already_applied_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    media_path = str(tmp_path / "sample.mp4")

    def blocking_processor(paths, settings, emit_event, log):
        del settings, log
        emit_event({"type": "file", "path": paths[0], "state": "transcribing"})
        entered.set()
        assert release.wait(timeout=3)
        return [{"path": paths[0], "state": "done"}]

    monkeypatch.setattr(web_app, "service_api", FakeService(blocking_processor))
    client = TestClient(web_app.app)
    started = client.post(
        "/api/transcribe",
        json={"paths": [media_path], "settings": {"language": "ru"}},
    )
    assert started.status_code == 200
    assert entered.wait(timeout=2)
    snapshot = client.get("/api/active-job").json()
    cursor = snapshot["latest_event_id"]
    assert snapshot["active"] is True
    assert snapshot["source_paths"] == [media_path]
    assert snapshot["settings"]["language"] == "ru"

    release.set()
    response = client.get(
        f"/api/stream/{started.json()['job_id']}",
        params={"cursor": cursor},
    )
    events = _parse_sse_events(response.text)
    assert events[-1]["payload"]["type"] == "done"
    assert all(event["id"] > cursor for event in events)


def test_folder_source_is_reexpanded_for_refresh_and_transcribe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    top_media = tmp_path / "top.mp4"
    nested_media = nested / "nested.wav"
    top_media.write_bytes(b"video")
    nested_media.write_bytes(b"audio")

    class FolderAwareService(FakeService):
        def build_items(self, paths, settings):
            expanded: list[str] = []
            for raw_path in paths:
                path = Path(raw_path)
                if path.is_dir():
                    expanded.extend(
                        str(value)
                        for value in collect_media_paths(
                            path,
                            recursive=bool(settings.get("recursive")),
                        )
                    )
                else:
                    expanded.append(str(path))
            return super().build_items(expanded, settings)

    service = FolderAwareService()
    monkeypatch.setattr(web_app, "service_api", service)
    client = TestClient(web_app.app)
    source_path = str(tmp_path)

    flat = client.post(
        "/api/refresh",
        json={"paths": [source_path], "settings": {"recursive": False}},
    )
    recursive = client.post(
        "/api/refresh",
        json={"paths": [source_path], "settings": {"recursive": True}},
    )
    started = client.post(
        "/api/transcribe",
        json={"paths": [source_path], "settings": {"recursive": True}},
    )

    assert flat.status_code == 200
    assert len(flat.json()["items"]) == 1
    assert recursive.status_code == 200
    assert len(recursive.json()["items"]) == 2
    assert started.status_code == 200
    assert len(started.json()["items"]) == 2
    job_id = started.json()["job_id"]
    assert client.get(f"/api/stream/{job_id}").status_code == 200
    assert client.get("/api/active-job").json()["source_paths"] == [source_path]


def test_jobs_api_cancel_and_retry_only_unsuccessful_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    cancel_seen = threading.Event()
    release = threading.Event()
    paths = [
        str(tmp_path / "готово.mp4"),
        str(tmp_path / "отмена №2.mkv"),
        str(tmp_path / "отмена №3.wav"),
    ]

    def cancellable_processor(selected_paths, settings, emit_event, log):
        del settings, log
        emit_event(
            {
                "type": "file",
                "path": selected_paths[0],
                "state": "done",
                "progress": 100,
            }
        )
        entered.set()
        while not service.cancel_check():
            time.sleep(0.005)
        cancel_seen.set()
        assert release.wait(timeout=3)
        return []

    service = FakeService(cancellable_processor)
    monkeypatch.setattr(web_app, "service_api", service)
    client = TestClient(web_app.app)
    started = client.post(
        "/api/transcribe",
        json={
            "paths": paths,
            "settings": {"language": "ru", "device": "cpu"},
        },
    )
    assert started.status_code == 200
    assert entered.wait(timeout=2)
    job_id = started.json()["job_id"]

    first_cancel = client.post(f"/api/jobs/{job_id}/cancel")
    assert first_cancel.status_code == 200
    assert first_cancel.json()["status"] == "cancelling"
    assert cancel_seen.wait(timeout=2)
    second_cancel = client.post(f"/api/jobs/{job_id}/cancel")
    assert second_cancel.status_code == 200
    release.set()
    job = web_app.job_registry.get(job_id)
    assert job.finished.wait(timeout=3)

    detail = client.get(f"/api/jobs/{job_id}")
    history = client.get("/api/jobs", params={"limit": 1})
    assert detail.status_code == 200
    assert detail.json()["status"] == "partial"
    assert detail.json()["cancelled"] == 2
    assert detail.json()["settings"]["language"] == "ru"
    assert detail.json()["source_paths"] == paths
    assert detail.json()["events"][-1]["event"]["type"] == "done"
    assert history.status_code == 200
    assert history.json()["jobs"][0]["job_id"] == job_id
    assert "items" not in history.json()["jobs"][0]
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409

    retried_paths: list[str] = []

    def retry_processor(selected_paths, settings, emit_event, log):
        del settings, emit_event, log
        retried_paths.extend(selected_paths)
        return [{"path": path, "state": "done"} for path in selected_paths]

    service.processor = retry_processor
    retried_response = client.post(f"/api/jobs/{job_id}/retry")
    assert retried_response.status_code == 200
    assert retried_response.json()["retry_of"] == job_id
    retried_id = retried_response.json()["job_id"]
    retried = web_app.job_registry.get(retried_id)
    assert retried.finished.wait(timeout=3)
    retried_detail = client.get(f"/api/jobs/{retried_id}").json()
    assert retried_paths == paths[1:]
    assert paths[0] not in retried_paths
    assert retried_detail["status"] == "ok"
    assert retried_detail["retry_of"] == job_id
    assert len(service.build_calls) == 2
    assert client.get("/api/jobs/missing-job").status_code == 404
    assert client.post("/api/jobs/missing-job/cancel").status_code == 404
    assert client.post("/api/jobs/missing-job/retry").status_code == 404


def test_job_registry_is_bounded_and_prunes_terminal_snapshot_by_ttl() -> None:
    registry = JobRegistry(max_jobs=1, ttl_seconds=1)

    def processor(paths, settings, emit_event, log, *, cancel_check=None):
        del settings, emit_event, log, cancel_check
        return [{"path": paths[0], "state": "done"}]

    def start(path: str):
        return registry.start(
            paths=[path],
            settings={},
            items=[{"path": path, "name": Path(path).name, "state": "queued"}],
            processor=processor,
        )

    first = start(r"D:\Media\first.mp4")
    assert first.finished.wait(timeout=2)
    assert registry.current_snapshot()["terminal"] is True

    second = start(r"D:\Media\second.mp4")
    assert second.finished.wait(timeout=2)
    with pytest.raises(JobNotFoundError):
        registry.get(first.job_id)

    second.completed_at = time.time() - 2
    assert registry.current_snapshot() == {"active": False, "terminal": False}
    with pytest.raises(JobNotFoundError):
        registry.get(second.job_id)


def _parse_sse_events(body: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current_id: int | None = None
    for line in body.splitlines():
        if line.startswith("id: "):
            current_id = int(line.removeprefix("id: "))
        elif line.startswith("data: "):
            events.append(
                {
                    "id": current_id,
                    "payload": json.loads(line.removeprefix("data: ")),
                }
            )
    return events
