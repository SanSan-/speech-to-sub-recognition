from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from speech_to_sub.constants import MAX_BATCH_PATHS, SUPPORTED_MEDIA_EXTENSIONS
from speech_to_sub.utils.io_utils import is_link_or_junction


class PickerError(RuntimeError):
    """Ошибка локального системного диалога выбора."""


@dataclass(frozen=True)
class PickSelection:
    """Нормализованный результат системного диалога."""

    mode: Literal["files", "folder"]
    paths: tuple[Path, ...]
    folder: Path | None = None


ProgressCallback = Callable[[dict[str, Any]], None]


def pick_paths(
    kind: Literal["file", "folder"],
    recursive: bool = False,
    *,
    progress_callback: ProgressCallback | None = None,
) -> PickSelection:
    """Открывает системный диалог и возвращает поддерживаемые локальные медиафайлы."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - зависит от окружения Python
        raise PickerError(f"Не удалось загрузить системный диалог: {exc}") from exc

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if kind == "folder":
            _report_progress(
                progress_callback,
                phase="dialog",
                message="Открыт системный диалог выбора каталога.",
            )
            selected = filedialog.askdirectory(title="Выберите каталог с медиафайлами")
            if not selected:
                return PickSelection(mode="folder", paths=())
            folder = Path(selected).expanduser().resolve()
            mode = "рекурсивный" if recursive else "без вложенных каталогов"
            _report_progress(
                progress_callback,
                phase="collecting",
                message=f"Выбран каталог {folder}. Начат {mode} поиск медиафайлов.",
            )
            return PickSelection(
                mode="folder",
                paths=collect_media_paths(
                    folder,
                    recursive=recursive,
                    progress_callback=progress_callback,
                ),
                folder=folder,
            )
        if kind != "file":
            raise PickerError(f"Неизвестный режим выбора: {kind}")
        _report_progress(
            progress_callback,
            phase="dialog",
            message="Открыт системный диалог выбора медиафайлов.",
        )
        selected_paths = filedialog.askopenfilenames(
            title="Выберите медиафайлы",
            filetypes=[
                ("Медиафайлы", _media_file_pattern()),
                ("Все файлы", "*.*"),
            ],
        )
        paths = tuple(_normalize_selected_path(value) for value in selected_paths)
        return PickSelection(mode="files", paths=filter_media_paths(paths))
    except PickerError:
        raise
    except Exception as exc:  # pragma: no cover - зависит от оконной системы
        raise PickerError(f"Не удалось выбрать медиафайлы: {exc}") from exc
    finally:
        if root is not None:
            root.destroy()


def collect_media_paths(
    folder: Path,
    recursive: bool = False,
    *,
    max_paths: int = MAX_BATCH_PATHS,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, ...]:
    """Собирает поддерживаемые файлы каталога в стабильном порядке."""
    try:
        root = folder.expanduser().resolve(strict=True)
    except OSError as exc:
        raise PickerError(f"Каталог не найден: {folder}") from exc
    if not root.is_dir():
        raise PickerError(f"Каталог не найден: {folder}")
    unique: dict[str, Path] = {}
    scanned = 0
    last_report = time.monotonic()
    directories = [root]
    while directories:
        current = directories.pop()
        ordered = _read_directory_entries(
            current,
            root=root,
            discovered=len(unique),
            progress_callback=progress_callback,
        )
        if ordered is None:
            continue
        nested: list[Path] = []
        for entry in ordered:
            scanned += 1
            nested_path = _collect_directory_entry(
                entry,
                recursive=recursive,
                unique=unique,
                max_paths=max_paths,
                progress_callback=progress_callback,
            )
            if nested_path is not None:
                nested.append(nested_path)
            last_report = _report_scan_progress(
                scanned,
                last_report=last_report,
                discovered=len(unique),
                progress_callback=progress_callback,
            )
        directories.extend(reversed(nested))
    paths = tuple(sorted(unique.values(), key=lambda value: str(value).casefold()))
    _report_progress(
        progress_callback,
        phase="collecting",
        discovered=len(paths),
        total=len(paths),
        message=f"Сбор каталога завершён: найдено медиафайлов — {len(paths)}.",
    )
    return paths


def _read_directory_entries(
    current: Path,
    *,
    root: Path,
    discovered: int,
    progress_callback: ProgressCallback | None,
) -> list[os.DirEntry[str]] | None:
    try:
        with os.scandir(current) as entries:
            return sorted(entries, key=lambda entry: entry.name.casefold())
    except OSError as exc:
        if current == root:
            raise PickerError(f"Не удалось прочитать каталог {root}: {exc}") from exc
        _report_progress(
            progress_callback,
            phase="collecting",
            discovered=discovered,
            message=f"Пропущен недоступный вложенный каталог: {current}.",
        )
        return None


def _collect_directory_entry(
    entry: os.DirEntry[str],
    *,
    recursive: bool,
    unique: dict[str, Path],
    max_paths: int,
    progress_callback: ProgressCallback | None,
) -> Path | None:
    candidate = Path(entry.path)
    try:
        if is_link_or_junction(entry.path):
            return None
        if recursive and entry.is_dir(follow_symlinks=False):
            return Path(entry.path)
        if not entry.is_file(follow_symlinks=False):
            return None
        if not _is_supported_input(candidate):
            return None
        _add_unique_media_candidate(unique, candidate, max_paths=max_paths)
    except OSError as exc:
        _report_progress(
            progress_callback,
            phase="collecting",
            discovered=len(unique),
            message=f"Не удалось проверить элемент каталога {entry.name}: {exc}.",
        )
        if _is_supported_input(candidate):
            _add_unique_media_candidate(unique, candidate, max_paths=max_paths)
    return None


def _add_unique_media_candidate(
    unique: dict[str, Path],
    candidate: Path,
    *,
    max_paths: int,
) -> None:
    unique.setdefault(str(candidate).casefold(), candidate)
    if len(unique) > max_paths:
        raise PickerError(f"За один запуск можно выбрать не более {max_paths} файлов.")


def _report_scan_progress(
    scanned: int,
    *,
    last_report: float,
    discovered: int,
    progress_callback: ProgressCallback | None,
) -> float:
    now = time.monotonic()
    if scanned % 128 != 0 and now - last_report < 0.25:
        return last_report
    _report_progress(
        progress_callback,
        phase="collecting",
        discovered=discovered,
    )
    return now


def filter_media_paths(
    paths: Iterable[str | Path],
    *,
    max_paths: int = MAX_BATCH_PATHS,
) -> tuple[Path, ...]:
    """Фильтрует явный выбор, сохраняя поддерживаемые пути для проверки сервисом."""
    unique: dict[str, Path] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not _is_supported_input(path):
            continue
        unique.setdefault(str(path).casefold(), path)
        if len(unique) > max_paths:
            raise PickerError(
                f"За один запуск можно выбрать не более {max_paths} файлов."
            )
    return tuple(sorted(unique.values(), key=lambda value: str(value).casefold()))


def _normalize_selected_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    try:
        return path.resolve(strict=False)
    except OSError:
        return Path(os.path.abspath(path))


def _media_file_pattern() -> str:
    return " ".join(f"*{extension}" for extension in sorted(SUPPORTED_MEDIA_EXTENSIONS))


def _is_supported_input(path: Path) -> bool:
    return (
        path.suffix.casefold() in SUPPORTED_MEDIA_EXTENSIONS
        and not path.name.casefold().endswith(".asr.flac")
    )


def _report_progress(
    callback: ProgressCallback | None,
    *,
    phase: str,
    discovered: int | None = None,
    total: int | None = None,
    message: str | None = None,
) -> None:
    if callback is None:
        return
    event: dict[str, Any] = {"phase": phase}
    if discovered is not None:
        event["discovered"] = discovered
    if total is not None:
        event["total"] = total
    if message:
        event["message"] = message
    try:
        callback(event)
    except Exception:
        return
