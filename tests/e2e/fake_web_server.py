from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from speech_to_sub.utils.logging_utils import WEB_RUNTIME_LOGGING_ENV  # noqa: E402

os.environ.pop(WEB_RUNTIME_LOGGING_ENV, None)

E2E_ARTIFACTS = Path(__file__).resolve().parent / ".artifacts"
E2E_ARTIFACTS.mkdir(parents=True, exist_ok=True)
E2E_JOB_DB = Path(os.environ.get("WEB_JOB_DB", E2E_ARTIFACTS / "jobs.sqlite3")).resolve(
    strict=False
)
if E2E_ARTIFACTS.resolve() not in E2E_JOB_DB.parents:
    raise RuntimeError("E2E SQLite разрешён только внутри tests/e2e/.artifacts.")
for database_file in (
    E2E_JOB_DB,
    Path(f"{E2E_JOB_DB}-wal"),
    Path(f"{E2E_JOB_DB}-shm"),
):
    database_file.unlink(missing_ok=True)
os.environ["WEB_JOB_DB"] = str(E2E_JOB_DB)

from speech_to_sub.web import app as web_app  # noqa: E402
from speech_to_sub.web.job_store import SQLiteJobStore  # noqa: E402
from speech_to_sub.web.jobs import JobRegistry  # noqa: E402
from speech_to_sub.web.picker import PickSelection  # noqa: E402

FAKE_PATHS = (
    r"D:\E2E\Лекция 01.mp4",
    r"D:\E2E\Разбор алгоритма №2.mkv",
    r"D:\E2E\Ошибка дорожки 03.wav",
)
FOLDER_ROOT = (E2E_ARTIFACTS / "Каталог Юникод").resolve()
FOLDER_PATHS = (
    FOLDER_ROOT / "Лекция верхнего уровня.mp4",
    FOLDER_ROOT / "Вложенная папка" / "Глубокий разбор №4.mkv",
)
for fixture_path in FOLDER_PATHS:
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_bytes(b"speech-to-sub e2e fixture")

ALL_FAKE_PATHS = (*FAKE_PATHS, *(str(path) for path in FOLDER_PATHS))
CONTROL_LOG_LINE = "E2E-КОНТРОЛЬ-SSE"
FOLDER_DISCOVERY_LOG = "Сбор каталога завершён: найдено медиафайлов — 2."


class FakeBatchService:
    """Управляемая заглушка пакетного распознавания для браузерного теста."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.entered = threading.Event()
        self.cancel_seen = threading.Event()
        self.retry_entered = threading.Event()
        self.released = threading.Event()
        self.folder_preparation_entered = threading.Event()
        self.folder_preparation_released = threading.Event()
        self.calls: list[list[str]] = []
        self.build_settings: list[dict[str, Any]] = []
        self.process_settings: list[dict[str, Any]] = []
        self.picker_calls: list[dict[str, Any]] = []
        self._hold_next_folder_build = False
        self._items = {
            path: self._make_item(path, index)
            for index, path in enumerate(ALL_FAKE_PATHS)
        }

    @staticmethod
    def _make_item(path: str, index: int) -> dict[str, Any]:
        return {
            "id": f"e2e-{index + 1}",
            "path": path,
            "name": Path(path).name,
            "state": "queued",
            "stage": "В очереди",
            "progress": 0,
            "probe": {
                "path": path,
                "duration": 60 + index,
                "format_name": Path(path).suffix.casefold().lstrip("."),
                "streams": [{"ordinal": 0, "index": 1, "codec_name": "aac"}],
            },
            "error": None,
            "outputs": None,
        }

    def build_items(
        self,
        paths: list[str],
        settings: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return self._record_build(paths, settings)

    def build_items_with_progress(
        self,
        paths: list[str],
        settings: dict[str, Any],
        progress_callback: Callable[[dict[str, Any]], None],
    ) -> list[dict[str, Any]]:
        with self._lock:
            should_hold = self._hold_next_folder_build and set(paths) == {
                str(path) for path in FOLDER_PATHS
            }
            if should_hold:
                self._hold_next_folder_build = False
        total = len(paths)
        progress_callback(
            {
                "phase": "probing",
                "processed": 0,
                "total": total,
                "message": f"Проверка аудиопотоков: 0 из {total}.",
            }
        )
        if should_hold:
            self.folder_preparation_entered.set()
            if not self.folder_preparation_released.wait(timeout=45):
                raise RuntimeError(
                    "E2E не получил разрешение завершить подготовку папки"
                )
        for processed, _path in enumerate(paths, start=1):
            progress_callback(
                {
                    "phase": "probing",
                    "processed": processed,
                    "total": total,
                    "message": f"Проверка аудиопотоков: {processed} из {total}.",
                }
            )
        return self._record_build(paths, settings)

    def begin_picker(self, kind: str, *, recursive: bool) -> None:
        with self._lock:
            self.picker_calls.append({"kind": kind, "recursive": recursive})
            if kind == "folder":
                self.folder_preparation_entered.clear()
                self.folder_preparation_released.clear()
                self._hold_next_folder_build = True

    def _record_build(
        self,
        paths: list[str],
        settings: dict[str, Any],
    ) -> list[dict[str, Any]]:
        with self._lock:
            self.build_settings.append(copy.deepcopy(settings))
            return [copy.deepcopy(self._items[path]) for path in paths]

    def unload(self) -> None:
        """Заглушка выгрузки модели: локальная модель в E2E не создаётся."""

    def process_paths(
        self,
        paths: list[str],
        settings: dict[str, Any],
        emit_event: Callable[[dict[str, Any]], None],
        log: logging.Logger,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        if settings.get("language") != "ru" or settings.get("device") != "cpu":
            raise RuntimeError("E2E ожидает настройки ru/cpu")
        if cancel_check is None:
            raise RuntimeError("E2E ожидает именованный cancel_check")
        with self._lock:
            self.calls.append(list(paths))
            self.process_settings.append(copy.deepcopy(settings))
            call_number = len(self.calls)

        if call_number == 1:
            return self._run_initial(paths, emit_event, log, cancel_check)
        if call_number == 2:
            return self._run_retry(paths, emit_event)
        raise RuntimeError(f"E2E не ожидает запуск №{call_number}")

    def _run_initial(
        self,
        paths: list[str],
        emit_event: Callable[[dict[str, Any]], None],
        log: logging.Logger,
        cancel_check: Callable[[], bool],
    ) -> list[dict[str, Any]]:
        if paths != list(FAKE_PATHS):
            raise RuntimeError(f"Первый E2E-запуск получил неверные пути: {paths}")

        self._update(
            paths[0],
            state="done",
            stage="Готово",
            progress=100,
            outputs=self._outputs(paths[0]),
        )
        emit_event(self._file_event(paths[0]))
        log.info(CONTROL_LOG_LINE)

        self._update(
            paths[1],
            state="transcribing",
            stage="Локальная ASR",
            progress=42,
        )
        emit_event(self._file_event(paths[1]))
        self.entered.set()

        deadline = time.monotonic() + 45
        while not cancel_check():
            if time.monotonic() >= deadline:
                raise RuntimeError("E2E не получил запрос отмены")
            time.sleep(0.01)
        self.cancel_seen.set()
        return []

    def _run_retry(
        self,
        paths: list[str],
        emit_event: Callable[[dict[str, Any]], None],
    ) -> list[dict[str, Any]]:
        if paths != list(FAKE_PATHS[1:]):
            raise RuntimeError(f"Retry повторно запустил неверные пути: {paths}")
        self._update(
            paths[0],
            state="transcribing",
            stage="Локальная ASR",
            progress=42,
            error=None,
        )
        emit_event(self._file_event(paths[0]))
        self.retry_entered.set()

        if not self.released.wait(timeout=45):
            raise RuntimeError("E2E retry-барьер не был освобождён")

        self._update(
            paths[0],
            state="done",
            stage="Готово",
            progress=100,
            error=None,
            outputs=self._outputs(paths[0]),
        )
        self._update(
            paths[1],
            state="error",
            stage="Ошибка",
            progress=100,
            error="Контролируемая ошибка E2E",
        )
        return [self._result(paths[0]), self._result(paths[1])]

    def state(self) -> dict[str, Any]:
        with self._lock:
            calls = copy.deepcopy(self.calls)
            build_settings = copy.deepcopy(self.build_settings)
            process_settings = copy.deepcopy(self.process_settings)
            picker_calls = copy.deepcopy(self.picker_calls)
        return {
            "entered": self.entered.is_set(),
            "cancel_seen": self.cancel_seen.is_set(),
            "retry_entered": self.retry_entered.is_set(),
            "released": self.released.is_set(),
            "folder_preparation_entered": self.folder_preparation_entered.is_set(),
            "folder_preparation_released": self.folder_preparation_released.is_set(),
            "folder_root": str(FOLDER_ROOT),
            "folder_paths": [str(path) for path in FOLDER_PATHS],
            "calls": calls,
            "build_settings": build_settings,
            "process_settings": process_settings,
            "picker_calls": picker_calls,
        }

    def release(self) -> dict[str, bool]:
        self.released.set()
        return {"released": True}

    def release_folder_preparation(self) -> dict[str, bool]:
        self.folder_preparation_released.set()
        return {"released": True}

    def _update(self, path: str, **changes: Any) -> None:
        with self._lock:
            self._items[path].update(changes)

    def _file_event(self, path: str) -> dict[str, Any]:
        with self._lock:
            item = copy.deepcopy(self._items[path])
        return {"type": "file", **item}

    def _result(self, path: str) -> dict[str, Any]:
        with self._lock:
            item = copy.deepcopy(self._items[path])
        return {
            "path": item["path"],
            "state": item["state"],
            "stage": item["stage"],
            "progress": item["progress"],
            "error": item["error"],
            "outputs": item["outputs"],
            "probe": copy.deepcopy(item["probe"]),
        }

    @staticmethod
    def _outputs(path: str) -> dict[str, str]:
        source = Path(path)
        return {
            "srt": str(source.with_suffix(".srt")),
            "sidecar": str(source.with_suffix(".json")),
        }


fake_service = FakeBatchService()


def fake_picker(
    kind: str,
    recursive: bool = False,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> PickSelection:
    fake_service.begin_picker(kind, recursive=recursive)
    if kind == "folder":
        if progress_callback:
            progress_callback(
                {
                    "phase": "dialog",
                    "message": "Открыт системный диалог выбора каталога.",
                }
            )
            progress_callback(
                {
                    "phase": "collecting",
                    "discovered": len(FOLDER_PATHS),
                    "total": len(FOLDER_PATHS),
                    "message": FOLDER_DISCOVERY_LOG,
                }
            )
        return PickSelection(
            mode="folder",
            paths=FOLDER_PATHS,
            folder=FOLDER_ROOT,
        )
    if kind != "file":
        raise RuntimeError(f"E2E не поддерживает режим выбора: {kind}")
    return PickSelection(
        mode="files",
        paths=tuple(Path(path) for path in FAKE_PATHS),
    )


web_app.service_api = fake_service
web_app.pick_paths = fake_picker


@web_app.app.get("/__e2e__/state")
def e2e_state() -> dict[str, Any]:
    return fake_service.state()


@web_app.app.post("/__e2e__/release")
def e2e_release() -> dict[str, bool]:
    return fake_service.release()


@web_app.app.post("/__e2e__/release-preparation")
def e2e_release_preparation() -> dict[str, bool]:
    return fake_service.release_folder_preparation()


@web_app.app.post("/__e2e__/restart")
def e2e_restart() -> dict[str, Any]:
    """Переоткрывает реестр на той же SQLite, имитируя рестарт процесса."""
    previous = web_app.job_registry
    previous.close()
    web_app.job_registry = JobRegistry(store=SQLiteJobStore(E2E_JOB_DB))
    snapshot = web_app.job_registry.current_snapshot()
    return {
        "job_id": snapshot.get("job_id"),
        "status": snapshot.get("status"),
        "terminal": snapshot.get("terminal", False),
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="strict")
    parser = argparse.ArgumentParser(description="Тестовый web-сервер для Playwright")
    parser.add_argument("--port", type=int, default=17862)
    args = parser.parse_args()
    uvicorn.run(web_app.app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
