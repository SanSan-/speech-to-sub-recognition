from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from speech_to_sub.web.job_store import SQLiteJobStore
from speech_to_sub.web.jobs import (
    JobRegistry,
    _normalize_snapshot_for_read,
    _prepare_retry_item,
)
from speech_to_sub.utils.cache import build_source_fingerprint


class RecordingStore(SQLiteJobStore):
    """Фиксирует строгую initial-запись до входа фонового worker."""

    def __init__(self, path: Path) -> None:
        self.initial_saved = threading.Event()
        super().__init__(path)

    def save(
        self,
        snapshot: Mapping[str, Any],
        events: Iterable[tuple[int, Mapping[str, Any]]] = (),
    ) -> None:
        normalized_events = tuple(events)
        super().save(snapshot, normalized_events)
        if not normalized_events and snapshot.get("latest_event_id") == 0:
            self.initial_saved.set()


class RuntimeFailingStore(SQLiteJobStore):
    """Имитирует отказ SQLite после успешной initial-записи."""

    def save(
        self,
        snapshot: Mapping[str, Any],
        events: Iterable[tuple[int, Mapping[str, Any]]] = (),
    ) -> None:
        normalized_events = tuple(events)
        if normalized_events:
            raise RuntimeError("Контролируемый отказ SQLite")
        super().save(snapshot, normalized_events)


def test_registry_persists_cancel_restart_retry_and_sse(tmp_path: Path) -> None:
    database = tmp_path / "очередь.sqlite3"
    store = RecordingStore(database)
    registry = JobRegistry(store=store)
    entered = threading.Event()
    cancel_seen = threading.Event()
    release_cancelled_worker = threading.Event()
    paths = [
        r"D:\Видео\готово.mp4",
        r"D:\Видео\отмена №2.mkv",
        r"D:\Видео\отмена №3.wav",
    ]
    settings = {"language": "ru", "device": "cpu", "output_format": "ass"}

    def cancellable_processor(
        selected_paths,
        selected_settings,
        emit_event,
        log: logging.Logger,
        *,
        cancel_check=None,
    ):
        assert store.initial_saved.is_set()
        assert selected_settings == settings
        assert callable(cancel_check)
        emit_event(
            {
                "type": "file",
                "path": selected_paths[0],
                "state": "done",
                "stage": "Готово",
                "progress": 100,
            }
        )
        log.info("КОНТРОЛЬ-SSE")
        entered.set()
        while not cancel_check():
            time.sleep(0.005)
        cancel_seen.set()
        assert release_cancelled_worker.wait(timeout=3)
        return []

    job = registry.start(
        paths=paths,
        settings=settings,
        items=[{"path": path, "state": "queued"} for path in paths],
        processor=cancellable_processor,
    )
    assert entered.wait(timeout=2)
    persisted_active = store.load(job.job_id)
    assert persisted_active is not None
    assert persisted_active["active"] is True
    assert persisted_active["source_paths"] == paths

    first_cancel = registry.cancel(job.job_id)
    assert cancel_seen.wait(timeout=2)
    second_cancel = registry.cancel(job.job_id)
    assert first_cancel["cancel_requested"] is True
    assert second_cancel["cancel_requested"] is True
    release_cancelled_worker.set()
    assert job.finished.wait(timeout=3)

    cancelled = registry.snapshot(job.job_id)
    assert cancelled["status"] == "partial"
    assert cancelled["cancelled"] == 2
    assert [item["state"] for item in cancelled["items"]] == [
        "done",
        "cancelled",
        "cancelled",
    ]
    cancelling_events = [
        entry
        for entry in cancelled["events"]
        if entry["event"].get("type") == "job"
        and entry["event"].get("status") == "cancelling"
    ]
    assert len(cancelling_events) == 1
    registry.close()

    restarted = JobRegistry(store=SQLiteJobStore(database))
    restored = restarted.snapshot(job.job_id)
    assert restored["status"] == "partial"
    assert restored["settings"] == settings
    assert restored["source_paths"] == paths
    assert "".join(restarted.iter_sse(job.job_id)).count("КОНТРОЛЬ-SSE") == 1
    terminal_cursor = int(restored["latest_event_id"])
    replay = "".join(restarted.iter_sse(job.job_id, after_event_id=terminal_cursor))
    assert _sse_payloads(replay) == []
    assert (
        _sse_payloads(
            "".join(restarted.iter_sse(job.job_id, after_event_id=terminal_cursor + 10))
        )
        == []
    )

    retried_paths: list[str] = []
    retried_settings: dict[str, Any] = {}

    def retry_processor(
        selected_paths,
        selected_settings,
        emit_event,
        log,
        *,
        cancel_check=None,
    ):
        del emit_event, log
        assert callable(cancel_check)
        retried_paths.extend(selected_paths)
        retried_settings.update(selected_settings)
        return [
            {"path": selected_paths[0], "state": "done"},
            {
                "path": selected_paths[1],
                "state": "error",
                "error": "Контролируемая ошибка повтора",
            },
        ]

    reservation = restarted.reserve_start()
    retried = restarted.retry(
        job.job_id,
        retry_processor,
        reservation_token=reservation,
    )
    assert retried.finished.wait(timeout=3)
    retry_snapshot = restarted.snapshot(retried.job_id)
    assert retried_paths == paths[1:]
    assert paths[0] not in retried_paths
    assert retried_settings == settings
    assert retry_snapshot["retry_of"] == job.job_id
    assert retry_snapshot["status"] == "partial"
    summaries = restarted.list_snapshots(limit=2)
    assert [summary["job_id"] for summary in summaries] == [
        retried.job_id,
        job.job_id,
    ]
    assert all(
        "items" not in summary and "events" not in summary for summary in summaries
    )
    restarted.close()


def test_retry_item_preserves_output_format_and_clears_old_artifact_paths() -> None:
    prepared = _prepare_retry_item(
        {
            "path": r"D:\Видео\лекция.mp4",
            "state": "error",
            "output_format": "vtt",
            "subtitle_output": r"D:\Видео\лекция.ru.vtt",
            "srt_output": r"D:\Видео\старый.ru.srt",
            "sidecar_output": r"D:\Видео\лекция.ru.vtt.asr.json",
            "audio_output": r"D:\Видео\лекция.ru.asr.flac",
            "outputs": {"subtitle": r"D:\Видео\лекция.ru.vtt"},
            "error": "Контролируемая ошибка",
        }
    )

    assert prepared["output_format"] == "vtt"
    assert prepared["state"] == "queued"
    assert prepared["stage"] == "Повторный запуск"
    for key in (
        "subtitle_output",
        "srt_output",
        "sidecar_output",
        "audio_output",
        "outputs",
        "error",
    ):
        assert key not in prepared


def test_retry_canonicalizes_persisted_legacy_settings_to_srt(tmp_path: Path) -> None:
    database = tmp_path / "legacy-jobs.sqlite3"
    registry = JobRegistry(store=SQLiteJobStore(database))
    path = r"D:\Видео\старый выпуск.mp4"
    srt_output = r"D:\Видео\старый выпуск.ru.srt"

    def failed_processor(paths, settings, emit_event, log, *, cancel_check=None):
        del settings, emit_event, log, cancel_check
        return [
            {
                "path": paths[0],
                "state": "error",
                "error": "Контрольная ошибка",
                "srt_output": srt_output,
            }
        ]

    source = registry.start(
        paths=[path],
        settings={"language": "ru"},
        items=[{"path": path, "state": "queued"}],
        processor=failed_processor,
    )
    assert source.finished.wait(timeout=3)
    assert "output_format" not in registry.snapshot(source.job_id)["settings"]
    registry.close()

    store = SQLiteJobStore(database)
    raw_legacy = store.load(source.job_id)
    assert raw_legacy is not None
    assert "output_format" not in raw_legacy["settings"]
    assert "output_format" not in raw_legacy["items"][0]
    assert "subtitle_output" not in raw_legacy["items"][0]

    restarted = JobRegistry(store=store)
    restored = restarted.snapshot(source.job_id)
    assert restored["settings"]["output_format"] == "srt"
    assert restored["items"][0]["output_format"] == "srt"
    assert restored["items"][0]["subtitle_output"] == srt_output
    raw_after_read = store.load(source.job_id)
    assert raw_after_read is not None
    assert "output_format" not in raw_after_read["settings"]
    assert "subtitle_output" not in raw_after_read["items"][0]

    retried = restarted.retry(source.job_id, failed_processor)
    assert retried.settings["output_format"] == "srt"
    assert retried.snapshot(active=True)["settings"]["output_format"] == "srt"
    assert retried.finished.wait(timeout=3)
    assert restarted.snapshot(retried.job_id)["settings"]["output_format"] == "srt"
    restarted.close()


def test_snapshot_normalizer_preserves_generic_ass_and_vtt_outputs() -> None:
    for output_format in ("ass", "vtt"):
        subtitle_output = rf"D:\Видео\лекция.ru.{output_format}"
        source = {
            "settings": {},
            "items": [
                {
                    "path": r"D:\Видео\лекция.mp4",
                    "subtitle_output": subtitle_output,
                    "srt_output": r"D:\Видео\старый путь.ru.srt",
                }
            ],
        }

        normalized = _normalize_snapshot_for_read(source)

        assert normalized["settings"]["output_format"] == "srt"
        assert normalized["items"][0]["output_format"] == output_format
        assert normalized["items"][0]["subtitle_output"] == subtitle_output
        assert "output_format" not in source["settings"]
        assert "output_format" not in source["items"][0]


def test_retry_rebuilds_and_persists_changed_source_fingerprint(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    source = tmp_path / "лекция.mp4"
    source.write_bytes(b"old")
    old_fingerprint = build_source_fingerprint(source)
    registry = JobRegistry(store=SQLiteJobStore(database))

    def failed_processor(paths, settings, emit_event, log, *, cancel_check=None):
        del settings, emit_event, log, cancel_check
        return [{"path": paths[0], "state": "error", "error": "Ошибка"}]

    failed = registry.start(
        paths=[str(source)],
        settings={},
        items=[
            {
                "path": str(source),
                "state": "queued",
                "source_fingerprint": old_fingerprint,
            }
        ],
        processor=failed_processor,
    )
    assert failed.finished.wait(timeout=3)
    source.write_bytes(b"new-content")

    def build_items(paths, settings):
        del settings
        return [
            {
                "path": path,
                "state": "error",
                "source_fingerprint": build_source_fingerprint(Path(path)),
            }
            for path in paths
        ]

    def successful_processor(paths, settings, emit_event, log, *, cancel_check=None):
        del settings, emit_event, log, cancel_check
        return [{"path": paths[0], "state": "done"}]

    retried = registry.retry(
        failed.job_id,
        successful_processor,
        item_builder=build_items,
    )
    assert retried.finished.wait(timeout=3)
    expected = build_source_fingerprint(source)
    snapshot = registry.snapshot(retried.job_id)
    persisted = registry._store.load(retried.job_id)
    registry.close()

    assert expected != old_fingerprint
    assert snapshot["items"][0]["source_fingerprint"] == expected
    assert persisted is not None
    assert persisted["items"][0]["source_fingerprint"] == expected


def test_restart_prunes_persisted_terminal_jobs_by_registry_ttl(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    now = time.time()
    store = SQLiteJobStore(database)
    for job_id, completed_at in (
        ("expired-job", now - 120),
        ("recent-job", now - 10),
    ):
        store.save(
            {
                "job_id": job_id,
                "status": "ok",
                "active": False,
                "terminal": True,
                "created_at": completed_at - 5,
                "updated_at": completed_at,
                "completed_at": completed_at,
                "items": [{"path": f"{job_id}.mp4", "state": "done"}],
            },
            [(1, {"type": "done", "status": "ok"})],
        )
    store.close()

    restarted_store = SQLiteJobStore(database)
    registry = JobRegistry(ttl_seconds=60, store=restarted_store)
    assert restarted_store.load("expired-job") is None
    assert registry.snapshot("recent-job")["status"] == "ok"
    assert [item["job_id"] for item in registry.list_snapshots()] == ["recent-job"]
    registry.close()

    final_store = SQLiteJobStore(database)
    final_registry = JobRegistry(ttl_seconds=60, store=final_store)
    assert final_store.load("expired-job") is None
    assert final_registry.snapshot("recent-job")["terminal"] is True
    final_registry.close()


def test_runtime_store_error_does_not_block_terminal_or_restart_recovery(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    registry = JobRegistry(store=RuntimeFailingStore(database))
    path = r"D:\Видео\лекция.mp4"

    def processor(paths, settings, emit_event, log, *, cancel_check=None):
        del settings, emit_event, log
        assert callable(cancel_check)
        return [{"path": paths[0], "state": "done"}]

    job = registry.start(
        paths=[path],
        settings={},
        items=[{"path": path, "state": "queued"}],
        processor=processor,
    )
    assert job.finished.wait(timeout=3)
    live_snapshot = registry.snapshot(job.job_id)
    assert live_snapshot["status"] == "ok"
    assert live_snapshot["persistence_error"] == "Контролируемый отказ SQLite"
    assert _sse_payloads("".join(registry.iter_sse(job.job_id)))[-1]["type"] == "done"
    registry.close()

    restarted = JobRegistry(store=SQLiteJobStore(database))
    recovered = restarted.snapshot(job.job_id)
    assert recovered["status"] == "interrupted"
    assert recovered["items"][0]["state"] == "error"
    restarted.close()


def test_registry_preserves_probe_for_active_terminal_and_restarted_job(
    tmp_path: Path,
) -> None:
    database = tmp_path / "duration.sqlite3"
    registry = JobRegistry(store=SQLiteJobStore(database))
    path = r"D:\Видео\длительность.mp4"
    probe = {
        "path": path,
        "duration": 3661.0,
        "format_name": "mov,mp4",
        "streams": [{"ordinal": 0, "index": 1, "codec_name": "aac"}],
    }
    active = threading.Event()
    release = threading.Event()

    def processor(paths, settings, emit_event, log, *, cancel_check=None):
        del settings, log
        assert callable(cancel_check)
        emit_event(
            {
                "type": "file",
                "path": paths[0],
                "state": "transcribing",
                "stage": "Распознавание",
                "progress": 42,
                "probe": probe,
            }
        )
        active.set()
        assert release.wait(timeout=3)
        emit_event(
            {
                "type": "file",
                "path": paths[0],
                "state": "error",
                "stage": "Ошибка",
                "progress": 100,
                "error": "Контролируемая ошибка после ffprobe",
            }
        )
        return [
            {
                "path": paths[0],
                "state": "error",
                "error": "Контролируемая ошибка после ffprobe",
                "probe": probe,
            }
        ]

    job = registry.start(
        paths=[path],
        settings={},
        items=[{"path": path, "state": "queued"}],
        processor=processor,
    )
    assert active.wait(timeout=3)
    active_snapshot = registry.snapshot(job.job_id)
    assert active_snapshot["items"][0]["state"] == "transcribing"
    assert active_snapshot["items"][0]["probe"]["duration"] == 3661.0

    release.set()
    assert job.finished.wait(timeout=3)
    terminal_snapshot = registry.snapshot(job.job_id)
    assert terminal_snapshot["items"][0]["state"] == "error"
    assert terminal_snapshot["items"][0]["probe"]["duration"] == 3661.0
    registry.close()

    restarted = JobRegistry(store=SQLiteJobStore(database))
    restored = restarted.snapshot(job.job_id)
    assert restored["items"][0]["state"] == "error"
    assert restored["items"][0]["probe"]["duration"] == 3661.0
    restarted.close()


def test_job_logger_preserves_sse_and_propagates_once_with_traceback() -> None:
    application_logger = logging.getLogger("speech_to_sub")
    saved_handlers = list(application_logger.handlers)
    saved_level = application_logger.level
    saved_propagate = application_logger.propagate
    records: list[logging.LogRecord] = []

    class RecordingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    capture = RecordingHandler()
    application_logger.handlers = [capture]
    application_logger.setLevel(logging.INFO)
    application_logger.propagate = False
    registry = JobRegistry()
    marker = "КОНТРОЛЬ-ФАЙЛОВОГО-ЖУРНАЛА"

    def failing_processor(paths, settings, emit_event, log, *, cancel_check=None):
        del paths, settings, emit_event, cancel_check
        logging.getLogger("speech_to_sub.service").info(marker)
        log.info(marker)
        raise RuntimeError("контролируемая авария worker")

    try:
        job = registry.start(
            paths=[r"D:\Видео\лекция.mp4"],
            settings={},
            items=[{"path": r"D:\Видео\лекция.mp4", "state": "queued"}],
            processor=failing_processor,
        )
        assert job.finished.wait(timeout=3)
        messages = [record.getMessage() for record in records]
        assert messages.count("Запущена пакетная обработка: файлов 1.") == 1
        assert messages.count(marker) == 1
        assert (
            sum(
                "Пакетная обработка завершилась с ошибкой" in value
                for value in messages
            )
            == 1
        )
        assert any(record.exc_info is not None for record in records)

        sse = "".join(registry.iter_sse(job.job_id))
        assert sse.count("Запущена пакетная обработка: файлов 1.") == 1
        assert sse.count(marker) == 1
        assert sse.count("Пакетная обработка завершилась с ошибкой") == 1
        assert "контролируемая авария worker" in sse
    finally:
        registry.reset_for_tests()
        application_logger.handlers = saved_handlers
        application_logger.setLevel(saved_level)
        application_logger.propagate = saved_propagate


def _sse_payloads(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
