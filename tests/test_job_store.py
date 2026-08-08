from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from speech_to_sub.web.job_store import RECOVERY_ERROR, SCHEMA_VERSION, SQLiteJobStore


def test_schema_migration_is_idempotent_and_enables_sqlite_safety(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE job_store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO job_store_meta(key, value) VALUES('schema_version', '0')"
    )
    connection.commit()
    connection.close()

    store = SQLiteJobStore(database, busy_timeout_ms=1_234)
    assert store._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert store._connection.execute("PRAGMA busy_timeout").fetchone()[0] == 1_234
    store.close()
    reopened = SQLiteJobStore(database, busy_timeout_ms=1_234)
    reopened.close()

    connection = sqlite3.connect(database)
    version = connection.execute(
        "SELECT value FROM job_store_meta WHERE key = 'schema_version'"
    ).fetchone()[0]
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    encoding = connection.execute("PRAGMA encoding").fetchone()[0]
    connection.close()

    assert int(version) == SCHEMA_VERSION
    assert {"jobs", "job_items", "job_events"} <= tables
    assert journal_mode == "wal"
    assert encoding == "UTF-8"


def test_roundtrip_unicode_snapshot_items_events_and_restart(tmp_path: Path) -> None:
    database = tmp_path / "очередь.sqlite3"
    snapshot = {
        "job_id": "задача-№1",
        "status": "partial",
        "active": False,
        "terminal": True,
        "created_at": datetime(2026, 8, 8, 9, 30, tzinfo=timezone.utc),
        "updated_at": "2026-08-08T12:31:00+03:00",
        "completed_at": 1_754_645_460,
        "latest_event_id": 3,
        "source_paths": [r"D:\Видео\Лекция №1.mkv"],
        "settings": {
            "language": "ru",
            "device": "cpu",
            "model_path": r"D:\Модели\whisper-large-v3",
        },
        "logs": ["Старт", "Файл готов"],
        "items": [
            {
                "path": r"D:\Видео\Лекция №1.mkv",
                "name": "Лекция №1.mkv",
                "state": "done",
                "progress": 100,
                "outputs": {"srt": r"D:\Видео\Лекция №1.srt"},
            },
            {
                "path": r"D:\Видео\Ошибка №2.mp4",
                "name": "Ошибка №2.mp4",
                "state": "error",
                "error": "Нет аудиодорожки",
            },
        ],
    }
    events = [
        (1, {"type": "job", "status": "running", "total": 2}),
        (
            2,
            {
                "type": "file",
                "path": r"D:\Видео\Лекция №1.mkv",
                "state": "done",
            },
        ),
        (3, {"type": "done", "status": "partial", "failed": 1}),
    ]

    store = SQLiteJobStore(database)
    store.save(snapshot, events)
    store.close()

    reopened = SQLiteJobStore(database)
    loaded = reopened.load("задача-№1")
    listed = reopened.list()
    reopened.close()

    assert loaded is not None
    assert loaded["source_paths"] == snapshot["source_paths"]
    assert loaded["settings"] == snapshot["settings"]
    assert loaded["items"] == snapshot["items"]
    assert [entry["id"] for entry in loaded["events"]] == [1, 2, 3]
    assert [entry["event"] for entry in loaded["events"]] == [event for _, event in events]
    assert loaded["created_at"] == "2026-08-08T09:30:00.000000Z"
    assert loaded["updated_at"] == "2026-08-08T09:31:00.000000Z"
    assert loaded["completed_at"].endswith("Z")
    assert loaded["terminal_event"] == events[-1][1]
    assert listed == [loaded]


def test_restart_recovery_persists_terminal_and_failed_item(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    snapshot = {
        "job_id": "active-job",
        "status": "running",
        "active": True,
        "terminal": False,
        "latest_event_id": 2,
        "source_paths": [r"D:\Видео\пачка"],
        "settings": {"language": "ru", "device": "cpu"},
        "logs": ["Задача запущена"],
        "items": [
            {"path": "готово.mp4", "state": "done", "progress": 100},
            {"path": "в работе.mkv", "state": "transcribing", "progress": 42},
        ],
    }
    terminal_snapshot = {
        "job_id": "terminal-job",
        "status": "ok",
        "active": False,
        "terminal": True,
        "items": [{"path": "архив.wav", "state": "done", "progress": 100}],
    }

    store = SQLiteJobStore(database)
    store.save(snapshot, [(1, {"type": "job"}), (2, {"type": "file"})])
    store.save(terminal_snapshot, [(1, {"type": "done", "status": "ok"})])
    store.close()

    restarted = SQLiteJobStore(database)
    assert restarted.recover_interrupted() == ["active-job"]
    assert restarted.recover_interrupted() == []
    recovered = restarted.load("active-job")
    untouched = restarted.load("terminal-job")
    restarted.close()

    assert recovered is not None
    assert recovered["active"] is False
    assert recovered["terminal"] is True
    assert recovered["status"] == "interrupted"
    assert [item["state"] for item in recovered["items"]] == ["done", "error"]
    assert recovered["items"][1]["stage"] == "Прервано"
    assert recovered["items"][1]["error"] == RECOVERY_ERROR
    assert recovered["latest_event_id"] == 3
    assert recovered["terminal_event"]["type"] == "done"
    assert recovered["terminal_event"]["status"] == "interrupted"
    assert recovered["events"][-1]["event"] == recovered["terminal_event"]
    assert recovered["completed_at"].endswith("Z")
    assert untouched is not None and untouched["status"] == "ok"

    final_restart = SQLiteJobStore(database)
    assert final_restart.load("active-job") == recovered
    final_restart.close()


def test_bounded_pruning_keeps_latest_jobs_and_events(tmp_path: Path) -> None:
    store = SQLiteJobStore(
        tmp_path / "jobs.sqlite3",
        max_jobs=2,
        max_events_per_job=2,
    )
    for index in range(1, 4):
        snapshot = {
            "job_id": f"job-{index}",
            "status": "ok",
            "active": False,
            "terminal": True,
            "updated_at": f"2026-08-08T09:0{index}:00Z",
            "items": [{"path": f"file-{index}.mp4", "state": "done"}],
        }
        events = [
            (event_id, {"type": "log", "message": f"событие-{event_id}"})
            for event_id in range(1, 5)
        ]
        store.save(snapshot, events)

    listed = store.list()
    newest = store.load("job-3")
    assert [job["job_id"] for job in listed] == ["job-3", "job-2"]
    assert store.load("job-1") is None
    assert newest is not None
    assert [entry["id"] for entry in newest["events"]] == [3, 4]
    assert newest["latest_event_id"] == 4
    assert store.delete("job-2") is True
    assert store.delete("job-2") is False
    assert [job["job_id"] for job in store.list()] == ["job-3"]
    store.close()

    connection = sqlite3.connect(tmp_path / "jobs.sqlite3")
    child_rows = connection.execute(
        "SELECT COUNT(*) FROM job_items WHERE job_id = 'job-2'"
    ).fetchone()[0]
    connection.close()
    assert child_rows == 0
