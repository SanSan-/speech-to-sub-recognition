from __future__ import annotations

import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

PREPARATION_LOG_LIMIT = 100
_MESSAGE_LIMIT = 2_000


class PreparationTracker:
    """Хранит ограниченный снимок текущей подготовки выбранных источников."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._generation = 0
        self._log_id = 0
        self._logs: deque[dict[str, Any]] = deque(maxlen=PREPARATION_LOG_LIMIT)
        self._state = self._idle_state()

    def begin(self, operation: str, *, phase: str, message: str) -> str:
        """Начинает новое поколение подготовки и возвращает его непрозрачный id."""
        with self._lock:
            self._generation += 1
            operation_id = uuid.uuid4().hex
            now = _utc_now()
            self._logs.clear()
            self._state = {
                "generation": self._generation,
                "operation_id": operation_id,
                "operation": operation,
                "status": "running",
                "phase": phase,
                "active": True,
                "discovered": 0,
                "processed": 0,
                "total": None,
                "message": _bounded(message),
                "error": None,
                "started_at": now,
                "updated_at": now,
                "finished_at": None,
            }
            self._append_log(message, now=now)
            return operation_id

    def update(
        self,
        operation_id: str,
        *,
        phase: str | None = None,
        discovered: int | None = None,
        processed: int | None = None,
        total: int | None = None,
        message: str | None = None,
    ) -> bool:
        """Обновляет только актуальную операцию; запоздалые события игнорируются."""
        with self._lock:
            if not self._is_current(operation_id):
                return False
            if phase is not None:
                self._state["phase"] = phase
            if discovered is not None:
                self._state["discovered"] = _monotonic_counter(
                    self._state["discovered"], discovered
                )
            if processed is not None:
                self._state["processed"] = _monotonic_counter(
                    self._state["processed"], processed
                )
            if total is not None:
                self._state["total"] = _monotonic_counter(self._state["total"], total)
            now = _utc_now()
            if message:
                self._state["message"] = _bounded(message)
                self._append_log(message, now=now)
            self._state["updated_at"] = now
            return True

    def finish(self, operation_id: str, *, message: str) -> bool:
        """Фиксирует успешное терминальное состояние операции."""
        return self._finish(operation_id, status="done", message=message, error=None)

    def fail(self, operation_id: str, *, message: str) -> bool:
        """Фиксирует ошибку так, чтобы она оставалась видимой после HTTP-ответа."""
        return self._finish(
            operation_id,
            status="error",
            message=message,
            error=_bounded(message),
        )

    def snapshot(self) -> dict[str, Any]:
        """Возвращает независимый JSON-совместимый снимок."""
        with self._lock:
            return {**self._state, "logs": [dict(entry) for entry in self._logs]}

    def reset_for_tests(self) -> None:
        """Возвращает реестр в исходное состояние между тестами."""
        with self._lock:
            self._generation = 0
            self._log_id = 0
            self._logs.clear()
            self._state = self._idle_state()

    def _finish(
        self,
        operation_id: str,
        *,
        status: str,
        message: str,
        error: str | None,
    ) -> bool:
        with self._lock:
            if not self._is_current(operation_id):
                return False
            now = _utc_now()
            self._state.update(
                status=status,
                phase="done" if status == "done" else "error",
                active=False,
                message=_bounded(message),
                error=error,
                updated_at=now,
                finished_at=now,
            )
            self._append_log(message, now=now)
            return True

    def _append_log(self, message: str, *, now: str) -> None:
        self._log_id += 1
        self._logs.append(
            {
                "id": self._log_id,
                "time": now,
                "message": _bounded(message),
            }
        )

    def _is_current(self, operation_id: str) -> bool:
        return self._state.get("operation_id") == operation_id

    def _idle_state(self) -> dict[str, Any]:
        return {
            "generation": self._generation,
            "operation_id": None,
            "operation": None,
            "status": "idle",
            "phase": "idle",
            "active": False,
            "discovered": 0,
            "processed": 0,
            "total": None,
            "message": "Подготовка источников не запущена.",
            "error": None,
            "started_at": None,
            "updated_at": None,
            "finished_at": None,
        }


def _bounded(value: str) -> str:
    text = str(value).strip()
    return text[:_MESSAGE_LIMIT]


def _monotonic_counter(current: object, incoming: int) -> int:
    previous = current if isinstance(current, int) else 0
    return max(previous, max(0, int(incoming)))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["PreparationTracker"]
