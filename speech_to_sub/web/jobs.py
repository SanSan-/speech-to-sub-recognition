from __future__ import annotations

import copy
import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

from speech_to_sub.web.job_store import SQLiteJobStore

LOG_HISTORY_LIMIT = 250
EVENT_HISTORY_LIMIT = 2_000
DEFAULT_MAX_JOBS = 16
DEFAULT_JOB_TTL_SECONDS = 60 * 60
SSE_KEEP_ALIVE_SECONDS = 1.0

CANCELLED_FILE_STATES = frozenset({"cancelled", "interrupted"})
TERMINAL_FILE_STATES = frozenset({"cached", "skipped", "done", "error"}) | (
    CANCELLED_FILE_STATES
)
RETRYABLE_FILE_STATES = frozenset({"error"}) | CANCELLED_FILE_STATES
CANCELLED_ERROR = "Обработка отменена пользователем."

Event = dict[str, Any]
EventEmitter = Callable[[Event], None]
CancelCheck = Callable[[], bool]
PersistSnapshot = Callable[[dict[str, Any], Iterable[tuple[int, Event]]], None]


class ProcessPaths(Protocol):
    """Контракт batch service с кооперативной проверкой отмены."""

    def __call__(
        self,
        paths: list[str],
        settings: dict[str, Any],
        emit_event: EventEmitter,
        log: logging.Logger,
        *,
        cancel_check: CancelCheck | None = None,
    ) -> list[dict[str, Any]]: ...


class BuildItems(Protocol):
    """Контракт повторной проверки карточек перед retry."""

    def __call__(
        self,
        paths: list[str],
        settings: dict[str, Any],
    ) -> list[dict[str, Any]]: ...


class JobBusyError(RuntimeError):
    """Попытка запустить вторую тяжёлую задачу."""


class JobNotFoundError(LookupError):
    """Запрошенная задача отсутствует в ограниченном реестре."""


class JobStateError(RuntimeError):
    """Операция несовместима с текущим состоянием задачи."""


class QueueLogHandler(logging.Handler):
    """Преобразует записи журнала в SSE-события задачи."""

    def __init__(self, emit_event: EventEmitter) -> None:
        super().__init__()
        self._emit_event = emit_event

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._emit_event({"type": "log", "message": self.format(record)})
        except Exception:
            self.handleError(record)


@dataclass
class BatchJob:
    """Ограниченное состояние одной пакетной задачи в памяти процесса."""

    job_id: str
    paths: tuple[str, ...]
    settings: dict[str, Any]
    items: list[dict[str, Any]]
    source_paths: tuple[str, ...] = ()
    retry_of: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    status: str = "queued"
    total: int = 0
    thread: threading.Thread | None = None
    terminal_event: Event | None = None
    terminal_event_id: int | None = None
    log_lines: deque[str] = field(
        default_factory=lambda: deque(maxlen=LOG_HISTORY_LIMIT)
    )
    event_history: deque[tuple[int, Event]] = field(
        default_factory=lambda: deque(maxlen=EVENT_HISTORY_LIMIT)
    )
    next_event_id: int = 1
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    persist_snapshot: PersistSnapshot | None = field(default=None, repr=False)
    persistence_error: str | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    condition: threading.Condition = field(init=False)
    finished: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        if not self.source_paths:
            self.source_paths = self.paths
        self.total = self.total or len(self.paths)
        self.condition = threading.Condition(self.lock)
        self.items = _prepare_initial_items(self.paths, self.items)

    @property
    def completed(self) -> bool:
        return self.finished.is_set()

    def emit(self, event: Mapping[str, Any]) -> int:
        """Сохраняет событие, обновляет snapshot и будит подписчиков."""
        normalized = _normalize_event(event)
        with self.condition:
            event_id = self.next_event_id
            self.next_event_id += 1
            self.updated_at = time.time()
            self._record_event(normalized)
            self.event_history.append((event_id, normalized))
            if normalized.get("type") == "done":
                self.terminal_event = copy.deepcopy(normalized)
                self.terminal_event_id = event_id
                self.status = str(normalized.get("status") or "error")
                self.completed_at = self.updated_at
                self.finished.set()
            if self.persist_snapshot is not None:
                snapshot = self._snapshot_locked(active=not self.completed)
                try:
                    self.persist_snapshot(snapshot, ((event_id, normalized),))
                except Exception as exc:
                    self.persistence_error = str(exc) or exc.__class__.__name__
            self.condition.notify_all()
            return event_id

    def snapshot(self, active: bool) -> dict[str, Any]:
        """Возвращает непротиворечивое состояние для восстановления UI."""
        with self.lock:
            return self._snapshot_locked(active)

    def request_cancel(self) -> bool:
        """Атомарно отмечает активную задачу для кооперативной отмены."""
        with self.lock:
            if self.completed:
                return False
            first_request = not self.cancel_requested.is_set()
            self.cancel_requested.set()
            if first_request:
                self.emit(
                    {"type": "job", "status": "cancelling", "total": self.total}
                )
            return True

    def events_after(self, event_id: int) -> tuple[list[tuple[int, Event]], bool, Event | None]:
        """Возвращает события после курсора и terminal snapshot."""
        with self.lock:
            events = [
                (stored_id, copy.deepcopy(event))
                for stored_id, event in self.event_history
                if stored_id > event_id
            ]
            return events, self.completed, copy.deepcopy(self.terminal_event)

    def wait_for_events(self, event_id: int, timeout: float) -> bool:
        """Ждёт новое событие либо завершение, не удерживая worker."""
        with self.condition:
            if self.completed or any(stored_id > event_id for stored_id, _ in self.event_history):
                return True
            self.condition.wait(timeout=timeout)
            return self.completed or any(
                stored_id > event_id for stored_id, _ in self.event_history
            )

    def _record_event(self, event: Event) -> None:
        event_type = event.get("type")
        if event_type == "log":
            message = str(event.get("message") or "").rstrip()
            if message:
                self.log_lines.append(message)
            return
        if event_type == "job":
            total = event.get("total")
            if isinstance(total, int) and total >= 0:
                self.total = total
            self.status = str(event.get("status") or "running")
            return
        if event_type == "file":
            _merge_file_event(self.items, event)

    def _snapshot_locked(self, active: bool) -> dict[str, Any]:
        counts = _count_items(self.items, self.total)
        status = self.status
        if active and status == "queued":
            status = "running"
        return {
            "active": active,
            "terminal": self.completed,
            "job_id": self.job_id,
            "status": status,
            "source_paths": list(self.source_paths),
            "settings": copy.deepcopy(self.settings),
            "items": copy.deepcopy(self.items),
            "logs": list(self.log_lines),
            "latest_event_id": self.next_event_id - 1,
            "created_at": _epoch_to_utc(self.created_at),
            "updated_at": _epoch_to_utc(self.updated_at),
            "completed_at": _epoch_to_utc(self.completed_at),
            "retry_of": self.retry_of,
            "cancel_requested": self.cancel_requested.is_set(),
            "terminal_event": copy.deepcopy(self.terminal_event),
            "persistence_error": self.persistence_error,
            **counts,
        }


class JobRegistry:
    """Потокобезопасный реестр одной активной и нескольких завершённых задач."""

    def __init__(
        self,
        max_jobs: int = DEFAULT_MAX_JOBS,
        ttl_seconds: float = DEFAULT_JOB_TTL_SECONDS,
        *,
        store: SQLiteJobStore | None = None,
    ) -> None:
        self._max_jobs = max(1, max_jobs)
        self._ttl_seconds = max(1.0, ttl_seconds)
        self._jobs: dict[str, BatchJob] = {}
        self._active_job_id: str | None = None
        self._last_job_id: str | None = None
        self._reservation_token: str | None = None
        self._reservation_kind: str | None = None
        self._lock = threading.RLock()
        self._store = store
        if self._store is not None:
            self._store.recover_interrupted()
            self._restore_persisted_jobs()
            with self._lock:
                self._prune_locked(time.time())

    def reserve_start(self) -> str:
        """Атомарно резервирует подготовку и запуск единственного worker."""
        with self._lock:
            self._prune_locked(time.time())
            self._ensure_worker_available_locked("Распознавание уже выполняется.")
            return self._reserve_locked("start")

    def reserve_unload(self) -> str:
        """Атомарно резервирует выгрузку модели относительно запуска worker."""
        with self._lock:
            self._prune_locked(time.time())
            self._ensure_worker_available_locked(
                "Нельзя выгрузить модель во время распознавания."
            )
            return self._reserve_locked("unload")

    def release_reservation(self, token: str) -> None:
        """Освобождает ещё не потреблённую резервацию операции."""
        with self._lock:
            if self._reservation_token == token:
                self._reservation_token = None
                self._reservation_kind = None

    def start(
        self,
        paths: list[str],
        settings: dict[str, Any],
        items: list[dict[str, Any]],
        processor: ProcessPaths,
        *,
        source_paths: list[str] | None = None,
        retry_of: str | None = None,
        reservation_token: str | None = None,
    ) -> BatchJob:
        """Резервирует единственный worker и запускает daemon-поток."""
        with self._lock:
            self._prune_locked(time.time())
            self._validate_start_reservation_locked(reservation_token)
            if self._active_job_id is not None:
                active = self._jobs.get(self._active_job_id)
                if active is not None and not active.completed:
                    raise JobBusyError("Распознавание уже выполняется.")
                self._active_job_id = None

            job = BatchJob(
                job_id=uuid.uuid4().hex,
                paths=tuple(paths),
                settings=copy.deepcopy(settings),
                items=copy.deepcopy(items),
                source_paths=tuple(source_paths or paths),
                retry_of=retry_of,
                persist_snapshot=self._save_snapshot,
            )
            thread = threading.Thread(
                target=self._run_job,
                args=(job, processor),
                name=f"speech-to-sub-{job.job_id[:8]}",
                daemon=True,
            )
            job.thread = thread
            self._jobs[job.job_id] = job
            self._active_job_id = job.job_id
            try:
                self._save_snapshot(job.snapshot(active=True), ())
            except Exception:
                self._jobs.pop(job.job_id, None)
                self._active_job_id = None
                raise
            if reservation_token is not None:
                self._reservation_token = None
                self._reservation_kind = None

        try:
            thread.start()
        except Exception:
            with self._lock:
                self._jobs.pop(job.job_id, None)
                if self._active_job_id == job.job_id:
                    self._active_job_id = None
                if self._store is not None:
                    self._store.delete(job.job_id)
            raise
        return job

    def get(self, job_id: str) -> BatchJob:
        """Возвращает задачу из памяти либо гидратирует persisted snapshot."""
        with self._lock:
            self._prune_locked(time.time())
            job = self._jobs.get(job_id)
            if job is None and self._store is not None:
                snapshot = self._store.load(job_id)
                if snapshot is not None:
                    job = _job_from_snapshot(snapshot, self._save_snapshot)
                    self._jobs[job_id] = job
        if job is None:
            raise JobNotFoundError(f"Задача не найдена: {job_id}")
        return job

    def snapshot(self, job_id: str, *, include_events: bool = True) -> dict[str, Any]:
        """Возвращает конкретную задачу и при необходимости её SSE-историю."""
        with self._lock:
            self._prune_locked(time.time())
            job = self._jobs.get(job_id)
        if job is not None:
            snapshot = job.snapshot(active=not job.completed)
            if include_events:
                snapshot["events"] = [
                    {"id": event_id, "event": copy.deepcopy(event)}
                    for event_id, event in job.event_history
                ]
            return snapshot
        if self._store is not None:
            snapshot = self._store.load(job_id)
            if snapshot is not None:
                if not include_events:
                    snapshot.pop("events", None)
                return snapshot
        raise JobNotFoundError(f"Задача не найдена: {job_id}")

    def list_snapshots(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Возвращает ограниченную persisted-историю задач."""
        with self._lock:
            self._prune_locked(time.time())
            if self._store is not None:
                return [
                    _job_summary(snapshot) for snapshot in self._store.list(limit=limit)
                ]
            jobs = sorted(
                self._jobs.values(),
                key=lambda job: (job.updated_at, job.job_id),
                reverse=True,
            )
            if limit is not None:
                jobs = jobs[:limit]
            return [_job_summary(job.snapshot(active=not job.completed)) for job in jobs]

    def current_snapshot(self) -> dict[str, Any]:
        """Возвращает активную задачу либо последнюю terminal-задачу."""
        with self._lock:
            self._prune_locked(time.time())
            active_id = self._active_job_id
            if active_id is not None:
                active_job = self._jobs.get(active_id)
                if active_job is not None:
                    return active_job.snapshot(active=not active_job.completed)
            last_job = self._jobs.get(self._last_job_id or "")
            if last_job is not None:
                return last_job.snapshot(active=False)
            if self._store is not None:
                snapshots = self._store.list(limit=1)
                if snapshots:
                    snapshot = snapshots[0]
                    snapshot.pop("events", None)
                    return snapshot
            return {"active": False, "terminal": False}

    def has_active_job(self) -> bool:
        """Проверяет занятость тяжёлого worker."""
        with self._lock:
            active = self._jobs.get(self._active_job_id or "")
            return active is not None and not active.completed

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Запрашивает кооперативную отмену текущей задачи."""
        job = self.get(job_id)
        with self._lock:
            if self._active_job_id != job_id or job.completed:
                raise JobStateError("Отменить можно только активную задачу.")
        if not job.request_cancel():
            raise JobStateError("Задача уже завершена.")
        return job.snapshot(active=True)

    def retry(
        self,
        job_id: str,
        processor: ProcessPaths,
        *,
        item_builder: BuildItems | None = None,
        reservation_token: str | None = None,
    ) -> BatchJob:
        """Создаёт новую задачу только для неуспешных элементов исходной."""
        snapshot = self.snapshot(job_id, include_events=False)
        if not snapshot.get("terminal"):
            raise JobStateError("Повторный запуск доступен только для завершённой задачи.")
        failed_items = [
            copy.deepcopy(dict(item))
            for item in snapshot.get("items", [])
            if str(item.get("state") or "") in RETRYABLE_FILE_STATES
        ]
        if not failed_items:
            raise JobStateError("В задаче нет файлов для повторного запуска.")
        paths = [str(item["path"]) for item in failed_items]
        settings = copy.deepcopy(snapshot.get("settings") or {})
        retry_items = _refresh_retry_items(
            failed_items,
            paths,
            settings,
            item_builder,
        )
        return self.start(
            paths=paths,
            settings=settings,
            items=retry_items,
            processor=processor,
            source_paths=paths,
            retry_of=job_id,
            reservation_token=reservation_token,
        )

    def iter_sse(self, job_id: str, after_event_id: int = 0) -> Iterable[str]:
        """Передаёт историю и новые события; после terminal-события закрывает поток."""
        job = self.get(job_id)
        cursor = max(0, after_event_id)
        while True:
            events, completed, terminal = job.events_after(cursor)
            frames, next_cursor, terminal_seen = _format_sse_batch(events)
            yield from frames
            if next_cursor is not None:
                cursor = next_cursor
            if terminal_seen:
                return
            if events:
                continue
            if completed:
                terminal_frame = _terminal_sse_frame(job, terminal, cursor)
                if terminal_frame is not None:
                    yield terminal_frame
                return
            if not job.wait_for_events(cursor, SSE_KEEP_ALIVE_SECONDS):
                yield ": keep-alive\n\n"

    def reset_for_tests(self, *, clear_store: bool = False) -> None:
        """Очищает завершённые тестовые snapshot-ы."""
        with self._lock:
            if self.has_active_job():
                raise RuntimeError("Нельзя очистить реестр во время активной задачи.")
            self._jobs.clear()
            self._active_job_id = None
            self._last_job_id = None
            self._reservation_token = None
            self._reservation_kind = None
            if clear_store and self._store is not None:
                for snapshot in self._store.list():
                    self._store.delete(str(snapshot["job_id"]))

    def close(self) -> None:
        """Закрывает persistent store после остановки приложения или тестового harness."""
        with self._lock:
            if self.has_active_job():
                raise RuntimeError("Нельзя закрыть реестр во время активной задачи.")
            if self._store is not None:
                self._store.close()

    def _run_job(self, job: BatchJob, processor: ProcessPaths) -> None:
        logger = logging.getLogger(f"speech_to_sub.web.job.{job.job_id}")
        logger.setLevel(logging.DEBUG if job.settings.get("verbose") else logging.INFO)
        logger.propagate = False
        handler = QueueLogHandler(job.emit)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        unhandled_error: str | None = None
        terminal_hint: dict[str, Any] = {}

        def emit_from_service(event: Event) -> None:
            if event.get("type") == "done":
                terminal_hint.update(event)
                return
            job.emit(event)

        try:
            job.emit({"type": "job", "status": "running", "total": len(job.paths)})
            logger.info("Запущена пакетная обработка: файлов %d.", len(job.paths))
            results: list[dict[str, Any]] = []
            if not job.cancel_requested.is_set():
                results = processor(
                    list(job.paths),
                    copy.deepcopy(job.settings),
                    emit_from_service,
                    logger,
                    cancel_check=job.cancel_requested.is_set,
                )
            _merge_results(job, results)
            if job.cancel_requested.is_set():
                _mark_unresolved_files(
                    job,
                    CANCELLED_ERROR,
                    state="cancelled",
                    stage="Отменено",
                )
            else:
                _mark_unresolved_files(job, "Обработка завершилась без итогового состояния.")
        except Exception as exc:
            if job.cancel_requested.is_set():
                logger.info("Пакетная обработка остановлена по запросу пользователя.")
                _mark_unresolved_files(
                    job,
                    CANCELLED_ERROR,
                    state="cancelled",
                    stage="Отменено",
                )
            else:
                unhandled_error = str(exc) or exc.__class__.__name__
                logger.exception("Пакетная обработка завершилась с ошибкой: %s", exc)
                _mark_unresolved_files(job, unhandled_error)
        finally:
            terminal = _build_terminal_event(job, unhandled_error, terminal_hint)
            if not job.completed:
                job.emit(terminal)
            logger.removeHandler(handler)
            handler.close()
            self._complete(job)

    def _complete(self, job: BatchJob) -> None:
        with self._lock:
            if self._active_job_id == job.job_id:
                self._active_job_id = None
            self._last_job_id = job.job_id
            self._prune_locked(time.time(), prune_store=False)

    def _save_snapshot(
        self,
        snapshot: dict[str, Any],
        events: Iterable[tuple[int, Event]],
    ) -> None:
        if self._store is not None:
            self._store.save(snapshot, events)

    def _restore_persisted_jobs(self) -> None:
        if self._store is None:
            return
        snapshots = self._store.list()
        for snapshot in snapshots:
            job = _job_from_snapshot(snapshot, self._save_snapshot)
            self._jobs[job.job_id] = job
            if snapshot.get("active") and not job.completed:
                self._active_job_id = job.job_id
        self._last_job_id = next(
            (
                str(snapshot["job_id"])
                for snapshot in snapshots
                if snapshot.get("terminal")
            ),
            None,
        )

    def _ensure_worker_available_locked(self, active_message: str) -> None:
        active = self._jobs.get(self._active_job_id or "")
        if active is not None and not active.completed:
            raise JobBusyError(active_message)
        if self._active_job_id is not None:
            self._active_job_id = None

    def _reserve_locked(self, kind: str) -> str:
        if self._reservation_token is not None:
            if self._reservation_kind == "unload":
                message = "Выполняется выгрузка локальной модели."
            else:
                message = "Запуск распознавания уже подготавливается."
            raise JobBusyError(message)
        token = uuid.uuid4().hex
        self._reservation_token = token
        self._reservation_kind = kind
        return token

    def _validate_start_reservation_locked(self, token: str | None) -> None:
        if token is None:
            if self._reservation_token is not None:
                raise JobBusyError("Worker занят другой операцией.")
            return
        if self._reservation_token != token or self._reservation_kind != "start":
            raise JobBusyError("Резервация запуска недействительна.")

    def _prune_locked(self, now: float, *, prune_store: bool = True) -> None:
        expired = {
            job_id
            for job_id, job in self._jobs.items()
            if job.completed
            and job.completed_at is not None
            and now - job.completed_at > self._ttl_seconds
        }
        if prune_store and self._store is not None:
            expired.update(
                self._store.delete_terminal_completed_before(now - self._ttl_seconds)
            )
        for job_id in expired:
            self._jobs.pop(job_id, None)

        if len(self._jobs) > self._max_jobs:
            completed = sorted(
                (job for job in self._jobs.values() if job.completed),
                key=lambda job: job.completed_at or job.created_at,
            )
            while len(self._jobs) > self._max_jobs and completed:
                self._jobs.pop(completed.pop(0).job_id, None)

        if self._last_job_id not in self._jobs:
            terminal_jobs = [job for job in self._jobs.values() if job.completed]
            self._last_job_id = (
                max(terminal_jobs, key=lambda job: job.completed_at or job.created_at).job_id
                if terminal_jobs
                else None
            )


def _prepare_initial_items(
    paths: tuple[str, ...],
    supplied_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_path = {
        str(item.get("path")).casefold(): copy.deepcopy(item)
        for item in supplied_items
        if isinstance(item, dict) and item.get("path")
    }
    items: list[dict[str, Any]] = []
    for path in paths:
        item = by_path.pop(path.casefold(), {"path": path, "name": Path(path).name})
        item["path"] = path
        item.setdefault("name", Path(path).name)
        item.setdefault("state", "cached" if item.get("cached") else "queued")
        item.setdefault("progress", 100 if item["state"] in TERMINAL_FILE_STATES else 0)
        items.append(item)
    return items


def _normalize_event(event: Mapping[str, Any]) -> Event:
    if not isinstance(event, Mapping):
        raise TypeError("Событие задачи должно быть отображением.")
    return json.loads(json.dumps(dict(event), ensure_ascii=False, default=str))


def _merge_file_event(items: list[dict[str, Any]], event: Event) -> None:
    path = str(event.get("path") or "")
    if not path:
        return
    item = next(
        (
            candidate
            for candidate in items
            if str(candidate.get("path") or "").casefold() == path.casefold()
        ),
        None,
    )
    if item is None:
        item = {"path": path, "name": Path(path).name, "state": "queued", "progress": 0}
        items.append(item)
    for key, value in event.items():
        if key not in {"type", "index", "total"}:
            item[key] = copy.deepcopy(value)
    state = str(item.get("state") or "queued")
    if state in TERMINAL_FILE_STATES:
        item["progress"] = 100
    item["cached"] = state == "cached" or bool(item.get("cached"))
    item["skipped"] = state == "skipped" or bool(item.get("skipped"))


def _merge_results(job: BatchJob, results: object) -> None:
    if not isinstance(results, list):
        raise TypeError("Пакетный сервис должен вернуть список результатов.")
    for result in results:
        if not isinstance(result, Mapping):
            raise TypeError("Результат файла должен быть отображением.")
        event = {"type": "file", **dict(result)}
        if event.get("state") in TERMINAL_FILE_STATES:
            event.setdefault("progress", 100)
        job.emit(event)


def _mark_unresolved_files(
    job: BatchJob,
    error: str,
    *,
    state: str = "error",
    stage: str = "Ошибка",
) -> None:
    with job.lock:
        unresolved = [
            str(item.get("path"))
            for item in job.items
            if item.get("path") and item.get("state") not in TERMINAL_FILE_STATES
        ]
    for path in unresolved:
        job.emit(
            {
                "type": "file",
                "path": path,
                "state": state,
                "stage": stage,
                "progress": 100,
                "error": error,
            }
        )


def _count_items(items: list[dict[str, Any]], declared_total: int) -> dict[str, int]:
    states = [str(item.get("state") or "queued") for item in items]
    return {
        "total": max(declared_total, len(items)),
        "done": sum(state in TERMINAL_FILE_STATES for state in states),
        "cached": sum(state == "cached" for state in states),
        "skipped": sum(state == "skipped" for state in states),
        "failed": sum(state == "error" for state in states),
        "cancelled": sum(state in CANCELLED_FILE_STATES for state in states),
    }


def _build_terminal_event(
    job: BatchJob,
    unhandled_error: str | None,
    terminal_hint: Mapping[str, Any],
) -> Event:
    with job.lock:
        counts = _count_items(job.items, job.total)
    succeeded = counts["done"] - counts["failed"] - counts["cancelled"]
    if counts["cancelled"] > 0:
        status = "partial" if succeeded > 0 or counts["failed"] > 0 else "cancelled"
    elif unhandled_error:
        status = "partial" if succeeded > 0 else "error"
    elif counts["failed"] == 0 and counts["done"] == counts["total"]:
        status = "ok"
    elif counts["failed"] > 0 and succeeded > 0:
        status = "partial"
    else:
        status = "error"
    event: Event = {"type": "done", "status": status, **counts}
    error = unhandled_error or terminal_hint.get("error")
    if error:
        event["error"] = str(error)
    return event


def _prepare_retry_item(item: Mapping[str, Any]) -> dict[str, Any]:
    retry_item = copy.deepcopy(dict(item))
    for key in (
        "error",
        "warning",
        "warnings",
        "srt_output",
        "sidecar_output",
        "audio_output",
        "outputs",
    ):
        retry_item.pop(key, None)
    retry_item.update(
        state="queued",
        stage="Повторный запуск",
        progress=0,
        cached=False,
        skipped=False,
    )
    return retry_item


def _refresh_retry_items(
    failed_items: list[dict[str, Any]],
    paths: list[str],
    settings: dict[str, Any],
    item_builder: BuildItems | None,
) -> list[dict[str, Any]]:
    """Повторно проверяет retry-карточки и сохраняет свежий fingerprint источника."""
    if item_builder is None:
        return [_prepare_retry_item(item) for item in failed_items]
    refreshed = item_builder(paths, copy.deepcopy(settings))
    by_path = {
        str(item.get("path") or "").casefold(): item
        for item in refreshed
        if isinstance(item, Mapping) and item.get("path")
    }
    missing = [path for path in paths if path.casefold() not in by_path]
    if missing:
        raise JobStateError(
            "Повторная проверка не вернула карточки всех неуспешных файлов."
        )
    return [_prepare_retry_item(by_path[path.casefold()]) for path in paths]


def _job_from_snapshot(
    snapshot: Mapping[str, Any],
    persist_snapshot: PersistSnapshot | None,
) -> BatchJob:
    items = [copy.deepcopy(dict(item)) for item in snapshot.get("items", [])]
    paths = tuple(str(item.get("path") or "") for item in items if item.get("path"))
    job = BatchJob(
        job_id=str(snapshot["job_id"]),
        paths=paths,
        settings=copy.deepcopy(dict(snapshot.get("settings") or {})),
        items=items,
        source_paths=tuple(str(path) for path in snapshot.get("source_paths", paths)),
        retry_of=str(snapshot["retry_of"]) if snapshot.get("retry_of") else None,
        created_at=_timestamp_to_epoch(snapshot.get("created_at"), time.time()),
        updated_at=_timestamp_to_epoch(snapshot.get("updated_at"), time.time()),
        completed_at=_optional_epoch(snapshot.get("completed_at")),
        status=str(snapshot.get("status") or "error"),
        total=int(snapshot.get("total") or len(paths)),
        persist_snapshot=persist_snapshot,
        persistence_error=(
            str(snapshot["persistence_error"])
            if snapshot.get("persistence_error")
            else None
        ),
    )
    job.log_lines.extend(str(line) for line in snapshot.get("logs", []))
    _restore_event_history(job, snapshot)
    _restore_terminal_state(job, snapshot)
    return job


def _restore_event_history(job: BatchJob, snapshot: Mapping[str, Any]) -> None:
    """Восстанавливает валидные события и следующий номер из snapshot-а."""
    stored_events = snapshot.get("events", [])
    for entry in stored_events:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("event"), Mapping):
            continue
        job.event_history.append((int(entry["id"]), copy.deepcopy(dict(entry["event"]))))
    job.next_event_id = max(
        int(snapshot.get("latest_event_id") or 0) + 1,
        max((event_id for event_id, _ in job.event_history), default=0) + 1,
    )


def _restore_terminal_state(job: BatchJob, snapshot: Mapping[str, Any]) -> None:
    """Восстанавливает terminal-событие и локальные флаги задачи."""
    terminal_entry = next(
        (
            (event_id, event)
            for event_id, event in reversed(job.event_history)
            if event.get("type") == "done"
        ),
        None,
    )
    terminal_event = snapshot.get("terminal_event")
    if isinstance(terminal_event, Mapping):
        job.terminal_event = copy.deepcopy(dict(terminal_event))
    elif terminal_entry is not None:
        job.terminal_event = copy.deepcopy(terminal_entry[1])
    if terminal_entry is not None:
        job.terminal_event_id = terminal_entry[0]
    elif isinstance(terminal_event, Mapping) and snapshot.get("terminal"):
        job.terminal_event_id = int(snapshot.get("latest_event_id") or 0) or None
    if snapshot.get("cancel_requested"):
        job.cancel_requested.set()
    if snapshot.get("terminal"):
        job.finished.set()


def _optional_epoch(value: Any) -> float | None:
    if value is None:
        return None
    return _timestamp_to_epoch(value, time.time())


def _epoch_to_utc(value: float | None) -> str | None:
    if value is None:
        return None
    return (
        datetime.fromtimestamp(value, timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _job_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "job_id",
        "status",
        "active",
        "terminal",
        "created_at",
        "updated_at",
        "completed_at",
        "retry_of",
        "total",
        "done",
        "cached",
        "skipped",
        "failed",
        "cancelled",
    )
    return {key: copy.deepcopy(snapshot.get(key)) for key in keys}


def _timestamp_to_epoch(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            return datetime.fromisoformat(normalized).timestamp()
        except ValueError:
            return fallback
    return fallback


def _format_sse(event_id: int, event: Event) -> str:
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event_id}\ndata: {payload}\n\n"


def _format_sse_batch(
    events: list[tuple[int, Event]],
) -> tuple[list[str], int | None, bool]:
    """Форматирует накопленные события до первого terminal-события."""
    frames: list[str] = []
    last_event_id: int | None = None
    for event_id, event in events:
        last_event_id = event_id
        frames.append(_format_sse(event_id, event))
        if event.get("type") == "done":
            return frames, last_event_id, True
    return frames, last_event_id, False


def _terminal_sse_frame(job: BatchJob, terminal: Event | None, cursor: int) -> str | None:
    """Возвращает отсутствующий terminal-кадр для восстановленного SSE-потока."""
    terminal_id = job.terminal_event_id
    if terminal is None or terminal_id is None or terminal_id <= cursor:
        return None
    return _format_sse(terminal_id, terminal)
