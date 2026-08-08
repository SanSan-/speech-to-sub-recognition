from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from speech_to_sub.web.job_store import SQLiteJobStore
from speech_to_sub.web.jobs import JobRegistry
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
    settings = {"language": "ru", "device": "cpu"}

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
    replay = "".join(
        restarted.iter_sse(job.job_id, after_event_id=terminal_cursor)
    )
    assert _sse_payloads(replay) == []
    assert _sse_payloads(
        "".join(restarted.iter_sse(job.job_id, after_event_id=terminal_cursor + 10))
    ) == []

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
    assert all("items" not in summary and "events" not in summary for summary in summaries)
    restarted.close()


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


def _sse_payloads(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
