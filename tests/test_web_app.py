from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from speech_to_sub import service as service_module
from speech_to_sub.asr import registry as asr_registry
from speech_to_sub.models import AudioStreamInfo, MediaProbe
from speech_to_sub.web import app as web_app
from speech_to_sub.web import __main__ as web_main
from speech_to_sub.web.__main__ import _read_port, _read_reload
from speech_to_sub.web.jobs import JobNotFoundError, JobRegistry
from speech_to_sub.web.picker import (
    PickSelection,
    collect_media_paths,
    filter_media_paths,
)


class FakeService:
    """Лёгкая замена batch service без FFmpeg и модели."""

    def __init__(
        self,
        processor: Callable[[list[str], dict[str, Any], Any, Any], list[dict[str, Any]]]
        | None = None,
    ) -> None:
        self.processor = processor or self._successful_processor
        self.build_calls: list[tuple[list[str], dict[str, Any]]] = []
        self.pending_calls: list[tuple[list[str], dict[str, Any]]] = []
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

    def build_pending_items(
        self,
        paths: list[str],
        settings: dict[str, Any],
    ) -> list[dict[str, Any]]:
        self.pending_calls.append((list(paths), dict(settings)))
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
    web_app.preparation_registry.reset_for_tests()
    yield
    snapshot = registry.current_snapshot()
    if snapshot.get("active"):
        job = registry.get(str(snapshot["job_id"]))
        assert job.finished.wait(timeout=3)
    registry.reset_for_tests()
    web_app.preparation_registry.reset_for_tests()


@pytest.fixture
def fake_service(monkeypatch: pytest.MonkeyPatch) -> FakeService:
    service = FakeService()
    monkeypatch.setattr(web_app, "service_api", service)
    return service


def test_testclient_without_runtime_marker_does_not_initialize_file_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    monkeypatch.delenv(web_app.WEB_RUNTIME_LOGGING_ENV, raising=False)
    monkeypatch.setattr(web_app, "setup_web_logging", lambda: calls.append(True))

    with TestClient(web_app.app) as client:
        assert client.get("/api/health").status_code == 200

    assert calls == []


def test_runtime_marker_initializes_file_logging_in_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    monkeypatch.setenv(web_app.WEB_RUNTIME_LOGGING_ENV, "1")
    monkeypatch.setattr(web_app, "setup_web_logging", lambda: calls.append(True))

    with TestClient(web_app.app) as client:
        assert client.get("/api/health").status_code == 200

    assert calls == [True]


def test_static_page_and_config_have_no_secret_fields(
    fake_service: FakeService,
) -> None:
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
    assert health.json()["version"] == "1.5.3"
    assert client.get("/openapi.json").json()["info"]["version"] == "1.5.3"
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
        "openai-api",
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
        "openai-api",
    }
    backend_paths = {
        item["value"]: item["model_path"] for item in config.json()["backends"]
    }
    assert backend_paths["transformers"].endswith("whisper-large-v3")
    assert backend_paths["faster-whisper"].endswith("whisper-large-v3-ct2")
    assert backend_paths["openai-api"] == ""
    assert defaults["auto_download_model"] is True
    assert defaults["allow_cloud_processing"] is False
    assert defaults["openai_model"] == "whisper-1"
    assert isinstance(config.json()["openai_configured"], bool)
    assert defaults["long_form_window_seconds"] == 300
    assert defaults["vad_filter"] is True
    assert defaults["max_chars_per_line"] == 42
    assert defaults["line_length_gap"] == 8
    assert defaults["max_cps"] == 17.0
    assert 'id="backend"' in index.text
    assert 'id="maxCharsPerLine"' in index.text
    assert 'id="lineLengthGap"' in index.text
    assert 'id="maxCps"' in index.text
    assert 'id="allowCloudProcessing"' in index.text
    serialized = json.dumps(
        {"config": config.json(), "health": health.json()},
        ensure_ascii=False,
    ).casefold()
    assert "api_key" not in serialized
    assert "password" not in serialized
    assert "token" not in serialized


def test_refresh_and_transcribe_document_internal_errors_in_openapi() -> None:
    paths = TestClient(web_app.app).get("/openapi.json").json()["paths"]

    for route in ("/api/refresh", "/api/transcribe"):
        assert paths[route]["post"]["responses"]["500"] == {
            "description": "Внутренняя ошибка локального сервиса."
        }


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


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_python_web_entrypoint_accepts_enabled_reload(value: str) -> None:
    assert _read_reload(value) is True


@pytest.mark.parametrize("value", [None, "0", "false", "No", "off"])
def test_python_web_entrypoint_accepts_disabled_reload(value: str | None) -> None:
    assert _read_reload(value) is False


def test_python_web_entrypoint_rejects_invalid_reload() -> None:
    with pytest.raises(ValueError, match="WEB_RELOAD"):
        _read_reload("иногда")


@pytest.mark.parametrize(("reload_value", "setup_calls"), [("0", 1), ("1", 0)])
def test_python_web_entrypoint_has_one_file_log_owner_with_reload(
    reload_value: str,
    setup_calls: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    uvicorn_options: dict[str, Any] = {}
    monkeypatch.setenv("WEB_HOST", "127.0.0.1")
    monkeypatch.setenv("WEB_PORT", "7862")
    monkeypatch.setenv("WEB_RELOAD", reload_value)
    monkeypatch.delenv(web_app.WEB_RUNTIME_LOGGING_ENV, raising=False)
    monkeypatch.setattr(web_main, "load_environment", lambda: None)
    monkeypatch.setattr(
        web_main,
        "setup_web_logging",
        lambda: calls.append(True) or web_main.logging.getLogger("test.web.launcher"),
    )
    monkeypatch.setattr(
        web_main.uvicorn,
        "run",
        lambda *_args, **kwargs: uvicorn_options.update(kwargs),
    )

    web_main.main()

    assert len(calls) == setup_calls
    assert uvicorn_options["reload"] is (reload_value == "1")
    assert web_app.WEB_RUNTIME_LOGGING_ENV not in os.environ


def test_python_web_entrypoint_restores_existing_runtime_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(web_app.WEB_RUNTIME_LOGGING_ENV, "предыдущее-значение")
    monkeypatch.setenv("WEB_RELOAD", "1")
    monkeypatch.setattr(web_main, "load_environment", lambda: None)
    monkeypatch.setattr(web_main.uvicorn, "run", lambda *_args, **_kwargs: None)

    web_main.main()

    assert os.environ[web_app.WEB_RUNTIME_LOGGING_ENV] == "предыдущее-значение"


def test_powershell_launcher_delegates_web_log_to_python() -> None:
    launcher = (web_app.PROJECT_ROOT / "run_web.ps1").read_text(encoding="utf-8")
    compact_launcher = " ".join(launcher.split())

    assert "AppendAllText" not in launcher
    assert "Rotate-RunWebLog" not in launcher
    assert '"speech_to_sub.web"' in launcher
    assert '$runtimeMarkerName = "SPEECH_TO_SUB_WEB_RUNTIME"' in launcher
    assert "$previousRuntimeMarker" in launcher
    assert "try {" in launcher
    assert "} finally {" in launcher
    assert (
        'SetEnvironmentVariable( $runtimeMarkerName, $previousRuntimeMarker, "Process" )'
        in compact_launcher
    )
    assert 'SetEnvironmentVariable($runtimeMarkerName, $null, "Process")' in launcher


def test_fake_e2e_server_removes_inherited_runtime_marker_before_lifespan(
    tmp_path: Path,
) -> None:
    marker = web_app.WEB_RUNTIME_LOGGING_ENV
    database = (
        web_app.PROJECT_ROOT
        / "tests"
        / "e2e"
        / ".artifacts"
        / f"marker-test-{uuid.uuid4().hex}.sqlite3"
    )
    environment = os.environ.copy()
    environment[marker] = "1"
    environment["WEB_JOB_DB"] = str(database)
    environment["CHECK_LOG_DIR"] = str(tmp_path / "isolated-logs")
    script = (
        "import os, runpy\n"
        "from pathlib import Path\n"
        "from fastapi.testclient import TestClient\n"
        "from speech_to_sub.utils import logging_utils\n"
        "logging_utils.LOGS_DIR = Path(os.environ['CHECK_LOG_DIR'])\n"
        "logging_utils._web_file_handler = None\n"
        "scope = runpy.run_path('tests/e2e/fake_web_server.py', "
        "run_name='e2e_marker_test')\n"
        f"assert os.environ.get({marker!r}) is None\n"
        "with TestClient(scope['web_app'].app) as client:\n"
        "    assert client.get('/api/health').status_code == 200\n"
        "assert not (Path(os.environ['CHECK_LOG_DIR']) / "
        "'speech_to_sub-web.log').exists()\n"
        "scope['web_app'].job_registry.close()\n"
        "database = Path(os.environ['WEB_JOB_DB'])\n"
        "for path in (database, Path(f'{database}-wal'), Path(f'{database}-shm')):\n"
        "    path.unlink(missing_ok=True)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=web_app.PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_pick_uses_backend_dialog_and_builds_cards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_service: FakeService,
) -> None:
    media_path = tmp_path / "Лекция 01.mp4"
    received: dict[str, Any] = {}

    def fake_picker(
        kind: str,
        recursive: bool = False,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> PickSelection:
        if progress_callback:
            progress_callback({"phase": "collecting", "discovered": 1})
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
    assert response.json()["recursive"] is True
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

    def blocking_picker(
        kind: str,
        recursive: bool = False,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> PickSelection:
        del kind, recursive, progress_callback
        picker_entered.set()
        assert release_picker.wait(timeout=3)
        return PickSelection(mode="files", paths=(picked_path,))

    original_build_items = fake_service.build_items

    def tracked_build_items(
        paths: list[str], settings: dict[str, Any]
    ) -> list[dict[str, Any]]:
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


def test_preparation_status_exposes_live_progress_and_terminal_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    media_path = str(tmp_path / "длинная запись.mp4")

    class ProgressService(FakeService):
        def build_items_with_progress(self, paths, settings, progress_callback):
            progress_callback(
                {
                    "phase": "probing",
                    "discovered": 80,
                    "processed": 31,
                    "total": 80,
                    "message": "Проверены аудиопотоки: 31 из 80.",
                }
            )
            entered.set()
            assert release.wait(timeout=3)
            return FakeService.build_items(self, paths, settings)

    monkeypatch.setattr(web_app, "service_api", ProgressService())
    response_holder: dict[str, Any] = {}

    def refresh_in_background() -> None:
        response_holder["response"] = TestClient(web_app.app).post(
            "/api/refresh",
            json={"paths": [media_path], "settings": {}},
        )

    baseline = TestClient(web_app.app).get("/api/preparation-status").json()
    thread = threading.Thread(target=refresh_in_background, daemon=True)
    thread.start()
    assert entered.wait(timeout=2)
    running = TestClient(web_app.app).get("/api/preparation-status").json()
    try:
        assert running["generation"] > baseline["generation"]
        assert running["operation"] == "refresh"
        assert running["status"] == "running"
        assert running["phase"] == "probing"
        assert running["active"] is True
        assert running["discovered"] == 80
        assert running["processed"] == 31
        assert running["total"] == 80
        assert running["error"] is None
        assert any("31 из 80" in entry["message"] for entry in running["logs"])
    finally:
        release.set()
        thread.join(timeout=3)

    assert not thread.is_alive()
    assert response_holder["response"].status_code == 200
    done = TestClient(web_app.app).get("/api/preparation-status").json()
    assert done["status"] == "done"
    assert done["phase"] == "done"
    assert done["active"] is False
    assert done["finished_at"] is not None


def test_preparation_status_keeps_refresh_error_until_next_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenService(FakeService):
        @staticmethod
        def build_items_with_progress(_paths, _settings, progress_callback):
            progress_callback(
                {
                    "phase": "probing",
                    "discovered": 4,
                    "processed": 1,
                    "total": 4,
                }
            )
            raise RuntimeError("авария проверки")

    monkeypatch.setattr(web_app, "service_api", BrokenService())
    client = TestClient(web_app.app)

    response = client.post(
        "/api/refresh",
        json={"paths": [r"D:\Media\lesson.mp4"], "settings": {}},
    )
    first = client.get("/api/preparation-status").json()
    second = client.get("/api/preparation-status").json()

    assert response.status_code == 400
    assert first == second
    assert first["status"] == "error"
    assert first["phase"] == "error"
    assert first["active"] is False
    assert "авария проверки" in first["error"]
    assert first["processed"] == 1


def test_state_changing_api_rejects_cross_origin_and_non_loopback_host(
    fake_service: FakeService,
) -> None:
    payload = {"paths": [r"D:\Media\lesson.mp4"], "settings": {}}
    client = TestClient(web_app.app)

    assert client.post("/api/refresh", json=payload).status_code == 200
    assert (
        client.post(
            "/api/refresh",
            json=payload,
            headers={"Origin": "https://attacker.example"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/refresh",
            json=payload,
            headers={"Host": "attacker.example"},
        ).status_code
        == 403
    )

    local_client = TestClient(web_app.app, base_url="http://127.0.0.1:7862")
    assert (
        local_client.post(
            "/api/refresh",
            json=payload,
            headers={"Origin": "http://127.0.0.1:7862"},
        ).status_code
        == 200
    )
    assert (
        local_client.post(
            "/api/refresh",
            json=payload,
            headers={"Origin": "http://127.0.0.1:9999"},
        ).status_code
        == 403
    )

    production_client = TestClient(
        web_app.app,
        base_url="http://127.0.0.1:7862",
        client=("127.0.0.1", 50_000),
    )
    assert production_client.post("/api/refresh", json=payload).status_code == 403
    assert (
        production_client.post(
            "/api/refresh",
            json=payload,
            headers={"Origin": "http://127.0.0.1:7862"},
        ).status_code
        == 200
    )
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


def test_ui_config_reports_only_openai_key_availability(
    monkeypatch: pytest.MonkeyPatch,
    fake_service: FakeService,
) -> None:
    del fake_service
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-secret-marker")
    client = TestClient(web_app.app)

    payload = client.get("/api/ui-config").json()

    assert payload["openai_configured"] is True
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "unit-test-secret-marker" not in serialized
    assert "OPENAI_API_KEY" not in serialized


def test_cloud_transcription_requires_fresh_consent_and_forwards_no_key(
    tmp_path: Path,
    fake_service: FakeService,
) -> None:
    client = TestClient(web_app.app)
    path = str(tmp_path / "облачное распознавание.mp4")
    settings = {
        "backend": "openai-api",
        "openai_model": "whisper-1",
    }

    rejected = client.post(
        "/api/transcribe",
        json={"paths": [path], "settings": settings},
    )
    invalid_consent = client.post(
        "/api/transcribe",
        json={
            "paths": [path],
            "settings": {**settings, "allow_cloud_processing": "true"},
        },
    )
    accepted = client.post(
        "/api/transcribe",
        json={
            "paths": [path],
            "settings": {**settings, "allow_cloud_processing": True},
        },
    )

    assert rejected.status_code == 400
    assert "явное согласие" in rejected.json()["detail"]
    assert invalid_consent.status_code == 422
    assert accepted.status_code == 200
    job = web_app.job_registry.get(accepted.json()["job_id"])
    assert job.finished.wait(timeout=3)
    forwarded = fake_service.pending_calls[0][1]
    assert forwarded["backend"] == "openai-api"
    assert forwarded["allow_cloud_processing"] is True
    assert forwarded["openai_model"] == "whisper-1"
    serialized = json.dumps(forwarded, ensure_ascii=False).casefold()
    assert "api_key" not in serialized
    assert "token" not in serialized


def test_web_rejects_unapproved_openai_model(fake_service: FakeService) -> None:
    del fake_service
    client = TestClient(web_app.app)

    response = client.post(
        "/api/refresh",
        json={
            "paths": [r"D:\Media\lesson.mp4"],
            "settings": {
                "backend": "openai-api",
                "openai_model": "gpt-4o-transcribe",
            },
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


def test_web_settings_forward_subtitle_layout_limits(fake_service: FakeService) -> None:
    client = TestClient(web_app.app)

    response = client.post(
        "/api/refresh",
        json={
            "paths": [r"D:\Media\lesson.mp4"],
            "settings": {
                "max_chars_per_line": 40,
                "line_length_gap": 6,
                "max_cps": 16.5,
            },
        },
    )

    assert response.status_code == 200
    forwarded = fake_service.build_calls[0][1]
    assert forwarded["max_chars_per_line"] == 40
    assert forwarded["line_length_gap"] == 6
    assert forwarded["max_cps"] == 16.5


@pytest.mark.parametrize(
    "settings",
    (
        {"max_chars_per_line": 19},
        {"line_length_gap": -1},
        {"line_length_gap": 21},
        {"line_length_gap": True},
        {"max_cps": 4.9},
        {"max_cps": 60.1},
    ),
)
def test_web_settings_reject_invalid_subtitle_layout_limits(
    fake_service: FakeService,
    settings: dict[str, float | int],
) -> None:
    del fake_service
    client = TestClient(web_app.app)

    response = client.post(
        "/api/refresh",
        json={"paths": [r"D:\Media\lesson.mp4"], "settings": settings},
    )

    assert response.status_code == 422


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
    assert fake_service.pending_calls[0][1]["backend"] == backend


def test_web_transcribe_forwards_force_to_the_background_job(
    tmp_path: Path,
    fake_service: FakeService,
) -> None:
    client = TestClient(web_app.app)
    response = client.post(
        "/api/transcribe",
        json={
            "paths": [str(tmp_path / "force.mp4")],
            "settings": {"force": True},
        },
    )

    assert response.status_code == 200
    job = web_app.job_registry.get(response.json()["job_id"])
    assert job.finished.wait(timeout=3)
    assert fake_service.pending_calls[0][1]["force"] is True
    assert job.snapshot(active=False)["settings"]["force"] is True


@pytest.mark.parametrize("invalid_force", ["true", 1])
def test_web_rejects_non_boolean_force(
    fake_service: FakeService,
    invalid_force: Any,
) -> None:
    client = TestClient(web_app.app)
    response = client.post(
        "/api/transcribe",
        json={
            "paths": [r"D:\Media\force.mp4"],
            "settings": {"force": invalid_force},
        },
    )

    assert response.status_code == 422
    assert fake_service.build_calls == []
    assert fake_service.pending_calls == []


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


def test_folder_collection_filters_media_and_respects_recursive_mode(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "Вложенный каталог"
    nested.mkdir()
    top_video = tmp_path / "Лекция.mp4"
    nested_audio = nested / "Дорожка.FLAC"
    ignored = tmp_path / "заметки.txt"
    top_video.write_bytes(b"video")
    nested_audio.write_bytes(b"audio")
    ignored.write_text("не медиа", encoding="utf-8")

    assert collect_media_paths(tmp_path, recursive=False) == (top_video,)
    expected = tuple(
        sorted((top_video, nested_audio), key=lambda path: str(path).casefold())
    )
    assert collect_media_paths(tmp_path, recursive=True) == expected


def test_folder_pick_forces_recursive_collection_for_legacy_false_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_service: FakeService,
) -> None:
    nested = tmp_path / "вложенный"
    nested.mkdir()
    top_media = tmp_path / "верх.mp4"
    nested_media = nested / "низ.wav"
    top_media.write_bytes(b"video")
    nested_media.write_bytes(b"audio")
    received: dict[str, Any] = {}

    def folder_picker(
        kind: str,
        recursive: bool = False,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> PickSelection:
        received.update(kind=kind, recursive=recursive)
        paths = collect_media_paths(
            tmp_path,
            recursive=recursive,
            progress_callback=progress_callback,
        )
        return PickSelection(mode="folder", paths=paths, folder=tmp_path)

    monkeypatch.setattr(web_app, "pick_paths", folder_picker)

    response = TestClient(web_app.app).post(
        "/api/pick",
        json={"kind": "folder", "settings": {"recursive": False}},
    )

    assert response.status_code == 200
    assert received == {"kind": "folder", "recursive": True}
    assert response.json()["recursive"] is True
    assert response.json()["path"] == str(tmp_path)
    assert {item["path"] for item in response.json()["items"]} == {
        str(top_media),
        str(nested_media),
    }
    assert len(fake_service.build_calls) == 1


def test_pick_and_refresh_return_error_card_without_losing_valid_neighbor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = tmp_path / "01-valid.mp4"
    empty = tmp_path / "02-empty.mp4"
    valid.write_bytes(b"media")
    empty.write_bytes(b"")

    def fake_probe(path: str | Path, **_kwargs: object) -> MediaProbe:
        resolved = Path(path)
        return MediaProbe(
            path=resolved,
            duration=1.0,
            streams=(AudioStreamInfo(ordinal=0, index=0, codec_name="aac"),),
            format_name="test",
        )

    def fake_picker(
        kind: str,
        recursive: bool = False,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> PickSelection:
        del kind, recursive, progress_callback
        return PickSelection(mode="files", paths=(valid, empty))

    monkeypatch.setattr(service_module, "probe_media", fake_probe)
    monkeypatch.setattr(web_app, "service_api", web_app.ServiceAdapter())
    monkeypatch.setattr(web_app, "pick_paths", fake_picker)
    client = TestClient(web_app.app)

    picked = client.post("/api/pick", json={"kind": "file", "settings": {}})
    refreshed = client.post(
        "/api/refresh",
        json={"paths": [str(valid), str(empty)], "settings": {}},
    )

    assert picked.status_code == 200
    assert refreshed.status_code == 200
    for response in (picked, refreshed):
        by_name = {item["name"]: item for item in response.json()["items"]}
        assert by_name[valid.name]["state"] == "idle"
        assert by_name[empty.name]["state"] == "error"
        assert "пуст" in by_name[empty.name]["error"]


def test_file_dialog_disappearance_reaches_pick_as_error_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disappeared = tmp_path / "исчезнувший.mp4"
    disappeared.write_bytes(b"media")

    def disappearing_picker(
        kind: str,
        recursive: bool = False,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> PickSelection:
        del kind, recursive, progress_callback
        disappeared.unlink()
        return PickSelection(
            mode="files",
            paths=filter_media_paths([disappeared]),
        )

    monkeypatch.setattr(web_app, "service_api", web_app.ServiceAdapter())
    monkeypatch.setattr(web_app, "pick_paths", disappearing_picker)

    response = TestClient(web_app.app).post(
        "/api/pick",
        json={"kind": "file", "settings": {}},
    )

    assert response.status_code == 200
    assert response.json()["paths"] == [str(disappeared)]
    assert len(response.json()["items"]) == 1
    item = response.json()["items"][0]
    assert item["path"] == str(disappeared)
    assert item["state"] == "error"
    assert "не найден" in item["error"]
    assert item["srt_output"] is None


def test_transcribe_rejects_explicit_generated_asr_flac(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = tmp_path / "lecture.ru.ASR.FLAC"
    generated.write_bytes(b"generated")
    monkeypatch.setattr(web_app, "service_api", web_app.ServiceAdapter())
    client = TestClient(web_app.app)

    response = client.post(
        "/api/transcribe",
        json={"paths": [str(generated)], "settings": {}},
    )

    assert response.status_code == 400
    assert "Не найдено поддерживаемых медиафайлов" in response.json()["detail"]
    preparation = client.get("/api/preparation-status").json()
    assert preparation["status"] == "error"
    assert preparation["active"] is False


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
        assert len(service.pending_calls) == 1
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


def test_start_reservation_blocks_parallel_queue_preparation_before_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingBuildService(FakeService):
        def build_pending_items(self, paths, settings):
            self.pending_calls.append((list(paths), dict(settings)))
            entered.set()
            assert release.wait(timeout=3)
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
    preparation_before_conflict = (
        TestClient(web_app.app).get("/api/preparation-status").json()
    )
    try:
        second = TestClient(web_app.app).post("/api/transcribe", json=payload)
        preparation_after_conflict = (
            TestClient(web_app.app).get("/api/preparation-status").json()
        )
        unload = TestClient(web_app.app).post("/api/unload", json={})
        assert second.status_code == 409
        assert preparation_after_conflict == preparation_before_conflict
        assert unload.status_code == 409
        assert len(service.pending_calls) == 1
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
        assert service.pending_calls == []
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


def test_folder_source_is_always_reexpanded_recursively_for_refresh_and_transcribe(
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
    assert len(flat.json()["items"]) == 2
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
    assert len(service.pending_calls) == 1
    assert len(service.build_calls) == 1
    assert client.get("/api/jobs/missing-job").status_code == 404
    assert client.post("/api/jobs/missing-job/cancel").status_code == 404
    assert client.post("/api/jobs/missing-job/retry").status_code == 404


def test_cloud_retry_requires_new_consent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def processor(selected_paths, settings, emit_event, log):
        nonlocal calls
        del settings, emit_event, log
        calls += 1
        state = "error" if calls == 1 else "done"
        return [{"path": path, "state": state} for path in selected_paths]

    service = FakeService(processor)
    monkeypatch.setattr(web_app, "service_api", service)
    client = TestClient(web_app.app)
    started = client.post(
        "/api/transcribe",
        json={
            "paths": [str(tmp_path / "повтор.mp4")],
            "settings": {
                "backend": "openai-api",
                "openai_model": "whisper-1",
                "allow_cloud_processing": True,
            },
        },
    )
    assert started.status_code == 200
    source_job = web_app.job_registry.get(started.json()["job_id"])
    assert source_job.finished.wait(timeout=3)

    rejected = client.post(f"/api/jobs/{source_job.job_id}/retry")
    accepted = client.post(
        f"/api/jobs/{source_job.job_id}/retry",
        json={"allow_cloud_processing": True},
    )

    assert rejected.status_code == 400
    assert accepted.status_code == 200
    retry_job = web_app.job_registry.get(accepted.json()["job_id"])
    assert retry_job.finished.wait(timeout=3)
    assert calls == 2


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
