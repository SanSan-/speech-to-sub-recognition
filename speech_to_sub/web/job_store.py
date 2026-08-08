from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1
DEFAULT_MAX_JOBS = 16
DEFAULT_MAX_EVENTS_PER_JOB = 2_000
DEFAULT_BUSY_TIMEOUT_MS = 5_000
RECOVERY_ERROR = "Задача была прервана перезапуском приложения."

TERMINAL_ITEM_STATES = frozenset(
    {"cached", "skipped", "done", "error", "cancelled", "interrupted"}
)
SUCCESS_ITEM_STATES = frozenset({"cached", "skipped", "done"})


class SQLiteJobStore:
    """Потокобезопасное SQLite-хранилище snapshot-ов пакетных задач."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_jobs: int = DEFAULT_MAX_JOBS,
        max_events_per_job: int = DEFAULT_MAX_EVENTS_PER_JOB,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        if max_jobs < 1:
            raise ValueError("max_jobs должен быть положительным.")
        if max_events_per_job < 1:
            raise ValueError("max_events_per_job должен быть положительным.")
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms не может быть отрицательным.")

        self.path = Path(path).expanduser()
        self.max_jobs = max_jobs
        self.max_events_per_job = max_events_per_job
        self.busy_timeout_ms = busy_timeout_ms
        self._lock = threading.RLock()
        self._closed = False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(self.path),
            timeout=busy_timeout_ms / 1_000,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._configure_connection()
            self._initialize_schema()
        except BaseException:
            if not self._closed:
                self._connection.close()
                self._closed = True
            raise

    def save(
        self,
        snapshot: Mapping[str, Any],
        events: Iterable[tuple[int, Mapping[str, Any]]] = (),
    ) -> None:
        """Атомарно сохраняет snapshot, его элементы и новые события."""
        normalized = _json_mapping(snapshot, "Snapshot задачи")
        job_id = str(normalized.get("job_id") or "").strip()
        if not job_id:
            raise ValueError("Snapshot задачи должен содержать job_id.")
        items = _normalize_items(normalized.pop("items", []))
        normalized.pop("events", None)
        normalized_events = _normalize_events(events)
        saved_at = _utc_now()

        with self._lock, self._transaction():
            existing = self._connection.execute(
                "SELECT created_at, completed_at, latest_event_id FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            supplied_created_at = normalized.pop("created_at", None)
            created_at = _utc_timestamp(
                existing["created_at"] if existing else supplied_created_at,
                saved_at,
            )
            updated_at = _utc_timestamp(normalized.pop("updated_at", None), saved_at)
            terminal = bool(normalized.pop("terminal", False))
            active = bool(normalized.pop("active", False)) and not terminal
            status = str(normalized.pop("status", "queued"))
            completed_at = self._completed_at(
                normalized.pop("completed_at", None),
                terminal=terminal,
                fallback=(existing["completed_at"] if existing else None) or updated_at,
            )
            requested_cursor = _nonnegative_int(
                normalized.pop("latest_event_id", 0),
                "latest_event_id",
            )
            previous_cursor = int(existing["latest_event_id"]) if existing else 0
            latest_event_id = max(
                requested_cursor,
                previous_cursor,
                max((event_id for event_id, _ in normalized_events), default=0),
            )

            self._upsert_job(
                job_id=job_id,
                status=status,
                active=active,
                terminal=terminal,
                created_at=created_at,
                updated_at=updated_at,
                completed_at=completed_at,
                latest_event_id=latest_event_id,
                snapshot=normalized,
            )
            self._replace_items(job_id, items)
            self._upsert_events(job_id, normalized_events, saved_at)
            self._prune_events(job_id)
            self._prune_jobs()

    def load(self, job_id: str) -> dict[str, Any] | None:
        """Загружает полный snapshot с элементами и ограниченной историей событий."""
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            return self._load_row(row) if row is not None else None

    def list(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Возвращает последние snapshot-ы от новых к старым."""
        if limit is not None and limit < 0:
            raise ValueError("limit не может быть отрицательным.")
        with self._lock:
            self._ensure_open()
            if limit == 0:
                return []
            sql = "SELECT * FROM jobs ORDER BY updated_at DESC, job_id DESC"
            parameters: tuple[int, ...] = ()
            if limit is not None:
                sql += " LIMIT ?"
                parameters = (limit,)
            rows = self._connection.execute(sql, parameters).fetchall()
            return [self._load_row(row) for row in rows]

    def recover_interrupted(self, error: str = RECOVERY_ERROR) -> list[str]:
        """Терминализирует все незавершённые задачи после рестарта процесса."""
        message = str(error).strip()
        if not message:
            raise ValueError("Текст ошибки восстановления не может быть пустым.")
        recovered: list[str] = []
        completed_at = _utc_now()

        with self._lock, self._transaction():
            rows = self._connection.execute(
                "SELECT * FROM jobs WHERE terminal = 0 ORDER BY created_at, job_id"
            ).fetchall()
            for row in rows:
                job_id = str(row["job_id"])
                items = self._load_items(job_id)
                failed_items = [_fail_interrupted_item(item, message) for item in items]
                self._replace_items(job_id, failed_items)
                counts = _count_items(failed_items)
                event_id = self._next_event_id(job_id, int(row["latest_event_id"]))
                terminal_event = {
                    "type": "done",
                    "status": "interrupted",
                    "reason": "interrupted",
                    "error": message,
                    "total": counts["total"],
                    "done": counts["done"],
                    "cached": counts["cached"],
                    "skipped": counts["skipped"],
                    "failed": counts["failed"],
                    "cancelled": counts["cancelled"],
                }
                snapshot = _json_object(str(row["snapshot_json"]), "Snapshot задачи")
                snapshot["error"] = message
                snapshot["terminal_event"] = terminal_event
                logs = snapshot.get("logs")
                if isinstance(logs, list):
                    snapshot["logs"] = [*logs, message]
                self._upsert_job(
                    job_id=job_id,
                    status="interrupted",
                    active=False,
                    terminal=True,
                    created_at=str(row["created_at"]),
                    updated_at=completed_at,
                    completed_at=completed_at,
                    latest_event_id=event_id,
                    snapshot=snapshot,
                )
                self._upsert_events(job_id, [(event_id, terminal_event)], completed_at)
                self._prune_events(job_id)
                recovered.append(job_id)
            self._prune_jobs()
        return recovered

    def delete(self, job_id: str) -> bool:
        """Удаляет задачу вместе с элементами и событиями через foreign key cascade."""
        with self._lock, self._transaction():
            cursor = self._connection.execute(
                "DELETE FROM jobs WHERE job_id = ?",
                (job_id,),
            )
            return cursor.rowcount > 0

    def delete_terminal_completed_before(self, cutoff: float | datetime) -> list[str]:
        """Удаляет terminal-задачи, завершённые строго раньше UTC cutoff."""
        cutoff_utc = _utc_timestamp(cutoff, _utc_now())
        with self._lock, self._transaction():
            rows = self._connection.execute(
                """
                SELECT job_id FROM jobs
                WHERE active = 0 AND terminal = 1
                    AND completed_at IS NOT NULL AND completed_at < ?
                ORDER BY completed_at, job_id
                """,
                (cutoff_utc,),
            ).fetchall()
            job_ids = [str(row["job_id"]) for row in rows]
            self._connection.executemany(
                "DELETE FROM jobs WHERE job_id = ?",
                [(job_id,) for job_id in job_ids],
            )
            return job_ids

    def close(self) -> None:
        """Идемпотентно закрывает соединение с БД."""
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def _configure_connection(self) -> None:
        self._connection.execute("PRAGMA encoding = 'UTF-8'")
        journal_mode = str(
            self._connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        ).casefold()
        if journal_mode != "wal":
            self._connection.close()
            self._closed = True
            raise RuntimeError("SQLite не смог включить WAL для хранилища задач.")
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        self._connection.execute("PRAGMA synchronous = NORMAL")

    def _initialize_schema(self) -> None:
        with self._lock, self._transaction():
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS job_store_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            row = self._connection.execute(
                "SELECT value FROM job_store_meta WHERE key = 'schema_version'"
            ).fetchone()
            version = _schema_version(row["value"] if row else "0")
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Версия схемы SQLite {version} новее поддерживаемой {SCHEMA_VERSION}."
                )
            if version <= 1:
                self._migrate_to_v1()
                version = 1
            self._connection.execute(
                """
                INSERT INTO job_store_meta(key, value) VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(version),),
            )

    def _migrate_to_v1(self) -> None:
        statements = (
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                active INTEGER NOT NULL CHECK(active IN (0, 1)),
                terminal INTEGER NOT NULL CHECK(terminal IN (0, 1)),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                latest_event_id INTEGER NOT NULL DEFAULT 0 CHECK(latest_event_id >= 0),
                snapshot_json TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS jobs_updated_at_idx
                ON jobs(updated_at DESC, job_id DESC)
            """,
            """
            CREATE TABLE IF NOT EXISTS job_items (
                job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                position INTEGER NOT NULL CHECK(position >= 0),
                path TEXT NOT NULL,
                state TEXT NOT NULL,
                item_json TEXT NOT NULL,
                PRIMARY KEY(job_id, position)
            )
            """,
            "CREATE INDEX IF NOT EXISTS job_items_path_idx ON job_items(job_id, path)",
            """
            CREATE TABLE IF NOT EXISTS job_events (
                job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                event_id INTEGER NOT NULL CHECK(event_id > 0),
                created_at TEXT NOT NULL,
                event_json TEXT NOT NULL,
                PRIMARY KEY(job_id, event_id)
            )
            """,
        )
        for statement in statements:
            self._connection.execute(statement)

    def _upsert_job(
        self,
        *,
        job_id: str,
        status: str,
        active: bool,
        terminal: bool,
        created_at: str,
        updated_at: str,
        completed_at: str | None,
        latest_event_id: int,
        snapshot: Mapping[str, Any],
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO jobs(
                job_id, status, active, terminal, created_at, updated_at,
                completed_at, latest_event_id, snapshot_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                status = excluded.status,
                active = excluded.active,
                terminal = excluded.terminal,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                completed_at = excluded.completed_at,
                latest_event_id = excluded.latest_event_id,
                snapshot_json = excluded.snapshot_json
            """,
            (
                job_id,
                status,
                int(active),
                int(terminal),
                created_at,
                updated_at,
                completed_at,
                latest_event_id,
                _json_text(snapshot),
            ),
        )

    def _replace_items(self, job_id: str, items: list[dict[str, Any]]) -> None:
        self._connection.execute("DELETE FROM job_items WHERE job_id = ?", (job_id,))
        self._connection.executemany(
            """
            INSERT INTO job_items(job_id, position, path, state, item_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    job_id,
                    position,
                    str(item.get("path") or ""),
                    str(item.get("state") or "queued"),
                    _json_text(item),
                )
                for position, item in enumerate(items)
            ],
        )

    def _upsert_events(
        self,
        job_id: str,
        events: list[tuple[int, dict[str, Any]]],
        created_at: str,
    ) -> None:
        self._connection.executemany(
            """
            INSERT INTO job_events(job_id, event_id, created_at, event_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(job_id, event_id) DO UPDATE SET event_json = excluded.event_json
            """,
            [
                (job_id, event_id, created_at, _json_text(event))
                for event_id, event in events
            ],
        )

    def _load_row(self, row: sqlite3.Row) -> dict[str, Any]:
        job_id = str(row["job_id"])
        snapshot = _json_object(str(row["snapshot_json"]), "Snapshot задачи")
        items = self._load_items(job_id)
        events = self._load_events(job_id)
        counts = _count_items(items)
        snapshot.update(
            {
                "job_id": job_id,
                "status": str(row["status"]),
                "active": bool(row["active"]),
                "terminal": bool(row["terminal"]),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
                "completed_at": row["completed_at"],
                "latest_event_id": int(row["latest_event_id"]),
                "items": items,
                "events": events,
                "total": max(_optional_count(snapshot.get("total")), counts["total"]),
                "done": counts["done"],
                "cached": counts["cached"],
                "skipped": counts["skipped"],
                "failed": counts["failed"],
                "cancelled": counts["cancelled"],
            }
        )
        if snapshot["terminal"] and "terminal_event" not in snapshot:
            terminal_event = next(
                (
                    entry["event"]
                    for entry in reversed(events)
                    if entry["event"].get("type") == "done"
                ),
                None,
            )
            if terminal_event is not None:
                snapshot["terminal_event"] = terminal_event
        return snapshot

    def _load_items(self, job_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT item_json FROM job_items WHERE job_id = ? ORDER BY position",
            (job_id,),
        ).fetchall()
        return [_json_object(str(row["item_json"]), "Элемент задачи") for row in rows]

    def _load_events(self, job_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT event_id, created_at, event_json
            FROM job_events WHERE job_id = ? ORDER BY event_id
            """,
            (job_id,),
        ).fetchall()
        return [
            {
                "id": int(row["event_id"]),
                "created_at": str(row["created_at"]),
                "event": _json_object(str(row["event_json"]), "Событие задачи"),
            }
            for row in rows
        ]

    def _next_event_id(self, job_id: str, latest_event_id: int) -> int:
        row = self._connection.execute(
            "SELECT MAX(event_id) AS event_id FROM job_events WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        return max(latest_event_id, int(row["event_id"] or 0)) + 1

    def _prune_events(self, job_id: str) -> None:
        self._connection.execute(
            """
            DELETE FROM job_events
            WHERE job_id = ? AND event_id NOT IN (
                SELECT event_id FROM job_events
                WHERE job_id = ? ORDER BY event_id DESC LIMIT ?
            )
            """,
            (job_id, job_id, self.max_events_per_job),
        )

    def _prune_jobs(self) -> None:
        count = int(self._connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])
        excess = count - self.max_jobs
        if excess <= 0:
            return
        rows = self._connection.execute(
            """
            SELECT job_id FROM jobs
            WHERE active = 0
            ORDER BY updated_at, job_id
            LIMIT ?
            """,
            (excess,),
        ).fetchall()
        self._connection.executemany(
            "DELETE FROM jobs WHERE job_id = ?",
            [(str(row["job_id"]),) for row in rows],
        )

    @staticmethod
    def _completed_at(value: Any, *, terminal: bool, fallback: str) -> str | None:
        if not terminal:
            return None
        return _utc_timestamp(value, fallback)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._ensure_open()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._connection.rollback()
            raise
        else:
            try:
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SQLite-хранилище задач уже закрыто.")


def _normalize_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("Поле items должно быть списком.")
    return [_json_mapping(item, "Элемент задачи") for item in value]


def _normalize_events(
    events: Iterable[tuple[int, Mapping[str, Any]]],
) -> list[tuple[int, dict[str, Any]]]:
    normalized: list[tuple[int, dict[str, Any]]] = []
    seen: set[int] = set()
    for event_id, event in events:
        normalized_id = _positive_int(event_id, "event_id")
        if normalized_id in seen:
            raise ValueError(f"Повторный event_id: {normalized_id}")
        seen.add(normalized_id)
        normalized.append((normalized_id, _json_mapping(event, "Событие задачи")))
    return normalized


def _fail_interrupted_item(item: dict[str, Any], error: str) -> dict[str, Any]:
    if str(item.get("state") or "queued") in TERMINAL_ITEM_STATES:
        return item
    failed = dict(item)
    failed.update(
        {
            "state": "error",
            "stage": "Прервано",
            "progress": 100,
            "error": error,
        }
    )
    return failed


def _count_items(items: list[dict[str, Any]]) -> dict[str, int]:
    states = [str(item.get("state") or "queued") for item in items]
    done = sum(state in TERMINAL_ITEM_STATES for state in states)
    return {
        "total": len(states),
        "done": done,
        "cached": states.count("cached"),
        "skipped": states.count("skipped"),
        "failed": states.count("error"),
        "cancelled": sum(state in {"cancelled", "interrupted"} for state in states),
        "succeeded": sum(state in SUCCESS_ITEM_STATES for state in states),
    }


def _json_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} должен быть отображением.")
    return _json_object(_json_text(dict(value)), label)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _json_object(value: str, label: str) -> dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} в SQLite должен быть JSON-объектом.")
    return decoded


def _utc_now() -> str:
    return _format_utc(datetime.now(timezone.utc))


def _utc_timestamp(value: Any, fallback: str) -> str:
    if value is None:
        value = fallback
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        timestamp = datetime.fromtimestamp(value, timezone.utc)
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError("UTC timestamp не может быть пустым.")
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        try:
            timestamp = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"Некорректный UTC timestamp: {value}") from exc
    else:
        raise TypeError("UTC timestamp должен быть строкой, datetime или Unix-временем.")
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return _format_utc(timestamp.astimezone(timezone.utc))


def _format_utc(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _schema_version(value: Any) -> int:
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Некорректная версия схемы SQLite: {value}") from exc
    if version < 0:
        raise RuntimeError(f"Некорректная версия схемы SQLite: {value}")
    return version


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} должен быть целым числом.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} должен быть целым числом.") from exc
    if normalized < 0:
        raise ValueError(f"{label} не может быть отрицательным.")
    return normalized


def _positive_int(value: Any, label: str) -> int:
    normalized = _nonnegative_int(value, label)
    if normalized == 0:
        raise ValueError(f"{label} должен быть положительным.")
    return normalized


def _optional_count(value: Any) -> int:
    if value is None:
        return 0
    return _nonnegative_int(value, "Счётчик задачи")
