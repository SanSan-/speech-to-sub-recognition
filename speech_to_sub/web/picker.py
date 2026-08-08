from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from speech_to_sub.constants import MAX_BATCH_PATHS, SUPPORTED_MEDIA_EXTENSIONS


class PickerError(RuntimeError):
    """Ошибка локального системного диалога выбора."""


@dataclass(frozen=True)
class PickSelection:
    """Нормализованный результат системного диалога."""

    mode: Literal["files", "folder"]
    paths: tuple[Path, ...]
    folder: Path | None = None


def pick_paths(kind: Literal["file", "folder"], recursive: bool = False) -> PickSelection:
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
            selected = filedialog.askdirectory(title="Выберите каталог с медиафайлами")
            if not selected:
                return PickSelection(mode="folder", paths=())
            folder = Path(selected).expanduser().resolve()
            return PickSelection(
                mode="folder",
                paths=collect_media_paths(folder, recursive=recursive),
                folder=folder,
            )
        if kind != "file":
            raise PickerError(f"Неизвестный режим выбора: {kind}")
        selected_paths = filedialog.askopenfilenames(
            title="Выберите медиафайлы",
            filetypes=[
                ("Медиафайлы", _media_file_pattern()),
                ("Все файлы", "*.*"),
            ],
        )
        paths = tuple(Path(value).expanduser().resolve() for value in selected_paths)
        return PickSelection(mode="files", paths=filter_media_paths(paths))
    except PickerError:
        raise
    except Exception as exc:  # pragma: no cover - зависит от оконной системы
        raise PickerError(f"Не удалось выбрать медиафайлы: {exc}") from exc
    finally:
        if root is not None:
            root.destroy()


def collect_media_paths(folder: Path, recursive: bool = False) -> tuple[Path, ...]:
    """Собирает поддерживаемые файлы каталога в стабильном порядке."""
    if not folder.exists() or not folder.is_dir():
        raise PickerError(f"Каталог не найден: {folder}")
    try:
        candidates = folder.rglob("*") if recursive else folder.iterdir()
        return filter_media_paths(candidates)
    except OSError as exc:
        raise PickerError(f"Не удалось прочитать каталог {folder}: {exc}") from exc


def filter_media_paths(
    paths: Iterable[str | Path],
    *,
    max_paths: int = MAX_BATCH_PATHS,
) -> tuple[Path, ...]:
    """Фильтрует файлы по расширению и удаляет повторы без изменения данных."""
    unique: dict[str, Path] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_MEDIA_EXTENSIONS:
            continue
        unique.setdefault(str(path).casefold(), path)
        if len(unique) > max_paths:
            raise PickerError(f"За один запуск можно выбрать не более {max_paths} файлов.")
    return tuple(sorted(unique.values(), key=lambda value: str(value).casefold()))


def _media_file_pattern() -> str:
    return " ".join(f"*{extension}" for extension in sorted(SUPPORTED_MEDIA_EXTENSIONS))
