"""Постоянный NDJSON-клиент для изолированных ASR runtime."""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, TextIO

from speech_to_sub.exceptions import AsrModelError, ProcessingCancelled

try:
    import psutil
except ImportError:  # Изолированному worker-у psutil не нужен для серверной стороны протокола.
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

PROTOCOL_PREFIX = "@@SPEECH_TO_SUB_WORKER_V1@@"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 3_600.0
_SHUTDOWN_TIMEOUT_SECONDS = 5.0
_DIAGNOSTIC_LINES = 20

CancelCheck = Callable[[], bool]
ProgressCallback = Callable[[int], None]


class ExternalWorkerError(AsrModelError):
    """Ошибка запуска, протокола или выполнения изолированного worker-а."""

    def __init__(self, message: str, *, error_type: str = "WorkerError") -> None:
        super().__init__(message)
        self.error_type = error_type


def encode_frame(payload: Mapping[str, Any]) -> str:
    """Кодирует один JSON-кадр с отличимым от стороннего stdout префиксом."""
    body = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
    return f"{PROTOCOL_PREFIX}{body}\n"


def decode_frame(line: str) -> dict[str, Any] | None:
    """Декодирует только кадры протокола, игнорируя сторонний stdout."""
    normalized = line.rstrip("\r\n")
    if not normalized.startswith(PROTOCOL_PREFIX):
        return None
    try:
        value = json.loads(normalized[len(PROTOCOL_PREFIX) :])
    except json.JSONDecodeError:
        return None
    return dict(value) if isinstance(value, Mapping) else None


class PersistentNdjsonWorker:
    """Сериализует запросы к одному долгоживущему subprocess."""

    def __init__(
        self,
        python_path: Path,
        module: str,
        *,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        cwd: Path | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Таймаут worker-а должен быть положительным.")
        self.python_path = Path(python_path)
        self.module = module
        self.timeout_seconds = float(timeout_seconds)
        self.cwd = cwd or Path(__file__).resolve().parents[2]
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.RLock()
        self._responses: queue.Queue[
            tuple[subprocess.Popen[str], dict[str, Any] | None]
        ] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=_DIAGNOSTIC_LINES)
        self._stdout_tail: deque[str] = deque(maxlen=_DIAGNOSTIC_LINES)

    @property
    def is_running(self) -> bool:
        """Сообщает, жив ли созданный subprocess."""
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def request(
        self,
        command: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> dict[str, Any]:
        """Отправляет один запрос и ждёт ответ с тем же идентификатором."""
        normalized_command = str(command).strip()
        if not normalized_command:
            raise ValueError("Команда worker-а не может быть пустой.")
        timeout = self.timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        if timeout <= 0:
            raise ValueError("Таймаут worker-а должен быть положительным.")
        with self._lock:
            return self._request_locked(
                normalized_command,
                payload or {},
                timeout,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )

    def shutdown(self) -> None:
        """Просит worker завершиться и принудительно закрывает зависший процесс."""
        with self._lock:
            process = self._process
            if process is None:
                return
            if process.poll() is None:
                try:
                    self._request_locked("shutdown", {}, _SHUTDOWN_TIMEOUT_SECONDS)
                    process.wait(timeout=1.0)
                except (ExternalWorkerError, subprocess.TimeoutExpired):
                    logger.debug("Worker не завершился штатно; процесс будет остановлен.")
            self._terminate_locked(process)

    def _request_locked(
        self,
        command: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> dict[str, Any]:
        if cancel_check is not None and cancel_check():
            raise ProcessingCancelled(f"Команда worker-а '{command}' отменена до запуска.")
        process = self._ensure_started_locked()
        request_id = uuid.uuid4().hex
        message = {"id": request_id, "command": command, "payload": dict(payload)}
        try:
            assert process.stdin is not None
            process.stdin.write(encode_frame(message))
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            detail = self._diagnostic_suffix()
            self._terminate_locked(process)
            raise ExternalWorkerError(
                f"Не удалось отправить команду '{command}' в worker.{detail}",
                error_type=type(exc).__name__,
            ) from exc
        return self._wait_for_response(
            process,
            request_id,
            command,
            timeout_seconds,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    def _wait_for_response(
        self,
        process: subprocess.Popen[str],
        request_id: str,
        command: str,
        timeout_seconds: float,
        *,
        progress_callback: ProgressCallback | None,
        cancel_check: CancelCheck | None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            if cancel_check is not None and cancel_check():
                self._terminate_locked(process)
                raise ProcessingCancelled(
                    f"Команда worker-а '{command}' отменена во время выполнения."
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate_locked(process)
                raise ExternalWorkerError(
                    f"Worker не ответил на команду '{command}' за {timeout_seconds:g} с.",
                    error_type="TimeoutError",
                )
            try:
                owner, response = self._responses.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                if process.poll() is not None:
                    return self._raise_terminated(process, command)
                continue
            if owner is not process:
                continue
            if response is None:
                return self._raise_terminated(process, command)
            if str(response.get("id", "")) != request_id:
                logger.warning("Worker вернул ответ с неожиданным идентификатором.")
                continue
            if response.get("event") == "progress":
                if progress_callback is not None:
                    try:
                        progress_callback(_normalize_progress(response.get("progress")))
                    except Exception:
                        self._terminate_locked(process)
                        raise
                continue
            return self._unwrap_response(response, command)

    def _unwrap_response(
        self,
        response: Mapping[str, Any],
        command: str,
    ) -> dict[str, Any]:
        if response.get("ok") is not True:
            error = response.get("error")
            error_data = dict(error) if isinstance(error, Mapping) else {}
            error_type = str(error_data.get("type") or "WorkerError")
            message = str(error_data.get("message") or "неизвестная ошибка")
            raise ExternalWorkerError(
                f"Команда worker-а '{command}' завершилась ошибкой "
                f"{error_type}: {message}",
                error_type=error_type,
            )
        result = response.get("result", {})
        if not isinstance(result, Mapping):
            raise ExternalWorkerError(
                f"Worker вернул некорректный результат команды '{command}'.",
                error_type="ProtocolError",
            )
        return dict(result)

    def _ensure_started_locked(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        if self._process is not None:
            self._terminate_locked(self._process)
        self._stderr_tail.clear()
        self._stdout_tail.clear()
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        creation_flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            process = subprocess.Popen(
                [str(self.python_path), "-u", "-m", self.module],
                cwd=str(self.cwd),
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
            )
        except OSError as exc:
            raise ExternalWorkerError(
                f"Не удалось запустить Python worker-а '{self.python_path}': {exc}",
                error_type=type(exc).__name__,
            ) from exc
        self._process = process
        assert process.stdout is not None and process.stderr is not None
        threading.Thread(
            target=self._read_stdout,
            args=(process, process.stdout),
            name="asr-worker-stdout",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._read_stderr,
            args=(process.stderr,),
            name="asr-worker-stderr",
            daemon=True,
        ).start()
        return process

    def _read_stdout(self, process: subprocess.Popen[str], stream: TextIO) -> None:
        try:
            for line in stream:
                response = decode_frame(line)
                if response is None:
                    normalized = line.rstrip("\r\n")
                    if normalized:
                        self._stdout_tail.append(normalized)
                        logger.debug("Сторонний stdout worker-а: %s", normalized)
                    continue
                self._responses.put((process, response))
        finally:
            self._responses.put((process, None))

    def _read_stderr(self, stream: TextIO) -> None:
        for line in stream:
            normalized = line.rstrip("\r\n")
            if normalized:
                self._stderr_tail.append(normalized)
                logger.debug("stderr worker-а: %s", normalized)

    def _raise_terminated(
        self,
        process: subprocess.Popen[str],
        command: str,
    ) -> dict[str, Any]:
        code = process.poll()
        detail = self._diagnostic_suffix()
        self._terminate_locked(process)
        raise ExternalWorkerError(
            f"Worker завершился до ответа на команду '{command}' "
            f"(код {code}).{detail}",
            error_type="WorkerExited",
        )

    def _diagnostic_suffix(self) -> str:
        if self._stderr_tail:
            return f" Последняя строка stderr: {self._stderr_tail[-1]}"
        if self._stdout_tail:
            return f" Последняя сторонняя строка stdout: {self._stdout_tail[-1]}"
        return ""

    def _terminate_locked(self, process: subprocess.Popen[str]) -> None:
        descendants = _worker_descendants(process) if process.poll() is None else []
        if self._process is process:
            self._process = None
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        _terminate_descendants(descendants)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)


def _worker_descendants(process: subprocess.Popen[str]) -> list[Any]:
    """Фиксирует только дочерние процессы конкретного запущенного worker-а."""
    if psutil is None or not isinstance(getattr(process, "pid", None), int):
        return []
    try:
        return list(psutil.Process(process.pid).children(recursive=True))
    except psutil.Error:
        return []


def _terminate_descendants(processes: list[Any]) -> None:
    """Завершает redirector-child дерево, чтобы отмена освобождала CUDA-веса."""
    if psutil is None or not processes:
        return
    for process in reversed(processes):
        try:
            process.terminate()
        except psutil.Error:
            pass
    _gone, alive = psutil.wait_procs(processes, timeout=2.0)
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            pass


def _normalize_progress(value: Any) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "ExternalWorkerError",
    "PersistentNdjsonWorker",
    "PROTOCOL_PREFIX",
    "decode_frame",
    "encode_frame",
]
