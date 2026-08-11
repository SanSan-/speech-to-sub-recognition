from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from speech_to_sub.constants import SUPPORTED_MEDIA_EXTENSIONS
from speech_to_sub.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class MediaDiscoveryFailure:
    """Описывает один источник, который не удалось безопасно подготовить."""

    path: Path
    error: str


def validate_media_file(path: Path) -> Path:
    """Проверяет существование и расширение входного медиафайла."""
    resolved = _resolve_existing(path, "Входной файл")
    if not resolved.is_file():
        raise ValidationError(f"Входной путь не является файлом: {resolved}")
    if resolved.suffix.casefold() not in SUPPORTED_MEDIA_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_MEDIA_EXTENSIONS))
        raise ValidationError(
            f"Неподдерживаемое расширение '{resolved.suffix}'. Поддерживаются: {supported}."
        )
    if resolved.stat().st_size <= 0:
        raise ValidationError(f"Входной файл пуст: {resolved}")
    return resolved


def discover_media(paths: Iterable[Path], recursive: bool) -> list[Path]:
    """Детерминированно собирает поддерживаемые файлы из путей и папок."""
    media_paths, failures = discover_media_resilient(paths, recursive)
    if failures:
        raise ValidationError(failures[0].error)
    return media_paths


def discover_media_resilient(
    paths: Iterable[Path],
    recursive: bool,
) -> tuple[list[Path], list[MediaDiscoveryFailure]]:
    """Собирает доступные медиа и отдельно возвращает ошибки отдельных источников."""
    found: dict[str, Path] = {}
    failures: dict[str, MediaDiscoveryFailure] = {}
    for raw_path in paths:
        _inspect_media_path(
            raw_path,
            recursive=recursive,
            found=found,
            failures=failures,
        )
    for key in found:
        failures.pop(key, None)
    return (
        sorted(found.values(), key=lambda value: str(value).casefold()),
        sorted(failures.values(), key=lambda value: str(value.path).casefold()),
    )


def _inspect_media_path(
    raw_path: Path,
    *,
    recursive: bool,
    found: dict[str, Path],
    failures: dict[str, MediaDiscoveryFailure],
) -> None:
    try:
        expanded = raw_path.expanduser()
        if is_link_or_junction(expanded):
            raise ValidationError(
                f"Символические ссылки и junction не обрабатываются: {expanded}"
            )
        resolved = _resolve_existing(raw_path, "Входной путь")
        if resolved.is_file():
            if not _is_generated_audio(resolved):
                _validate_discovered_file(resolved, found=found, failures=failures)
            return
        if resolved.is_dir():
            _inspect_media_directory(
                resolved,
                recursive=recursive,
                found=found,
                failures=failures,
            )
            return
        raise ValidationError(
            f"Входной путь не является файлом или каталогом: {resolved}"
        )
    except (OSError, ValidationError) as exc:
        _record_discovery_failure(failures, raw_path, exc)


def _inspect_media_directory(
    root: Path,
    *,
    recursive: bool,
    found: dict[str, Path],
    failures: dict[str, MediaDiscoveryFailure],
) -> None:
    directories = [root]
    while directories:
        current = directories.pop()
        ordered = _list_directory_entries(current, failures)
        nested: list[Path] = []
        for entry in ordered:
            _inspect_directory_entry(
                entry,
                recursive=recursive,
                nested=nested,
                found=found,
                failures=failures,
            )
        directories.extend(reversed(nested))


def _list_directory_entries(
    directory: Path,
    failures: dict[str, MediaDiscoveryFailure],
) -> list[os.DirEntry[str]]:
    try:
        with os.scandir(directory) as entries:
            return sorted(entries, key=lambda entry: entry.name.casefold())
    except OSError as exc:
        _record_discovery_failure(failures, directory, exc)
        return []


def _inspect_directory_entry(
    entry: os.DirEntry[str],
    *,
    recursive: bool,
    nested: list[Path],
    found: dict[str, Path],
    failures: dict[str, MediaDiscoveryFailure],
) -> None:
    candidate = Path(entry.path)
    try:
        if is_link_or_junction(candidate):
            return
        if entry.is_dir(follow_symlinks=False):
            if recursive:
                nested.append(candidate)
            return
        if _is_supported_media_entry(entry, candidate):
            _validate_discovered_file(candidate, found=found, failures=failures)
    except (OSError, ValidationError) as exc:
        _record_discovery_failure(failures, candidate, exc)


def _is_supported_media_entry(entry: os.DirEntry[str], path: Path) -> bool:
    return (
        path.suffix.casefold() in SUPPORTED_MEDIA_EXTENSIONS
        and not _is_generated_audio(path)
        and entry.is_file(follow_symlinks=False)
    )


def _validate_discovered_file(
    path: Path,
    *,
    found: dict[str, Path],
    failures: dict[str, MediaDiscoveryFailure],
) -> None:
    try:
        resolved = validate_media_file(path)
    except (OSError, ValidationError) as exc:
        _record_discovery_failure(failures, path, exc)
        return
    found[str(resolved).casefold()] = resolved


def _record_discovery_failure(
    failures: dict[str, MediaDiscoveryFailure],
    path: Path,
    error: BaseException,
) -> None:
    normalized = _path_for_report(path)
    message = str(error).strip() or error.__class__.__name__
    failures.setdefault(
        str(normalized).casefold(),
        MediaDiscoveryFailure(path=normalized, error=message),
    )


def _path_for_report(path: Path) -> Path:
    expanded = path.expanduser()
    try:
        if is_link_or_junction(expanded):
            return Path(os.path.abspath(expanded))
        return expanded.resolve(strict=False)
    except OSError:
        return Path(os.path.abspath(expanded))


def is_link_or_junction(path: str | Path) -> bool:
    """Определяет символическую ссылку или Windows junction без обхода цели."""
    candidate = Path(path)
    if candidate.is_symlink():
        return True
    is_junction = getattr(os.path, "isjunction", None)
    return bool(is_junction and is_junction(candidate))


def validate_model_path(path: Path) -> Path:
    """Проверяет локальный Hugging Face checkpoint без сетевых запросов."""
    resolved = _resolve_existing(path, "Каталог ASR-модели")
    if not resolved.is_dir():
        raise ValidationError(f"Путь модели не является каталогом: {resolved}")
    required = ("config.json", "preprocessor_config.json", "tokenizer_config.json")
    missing = [name for name in required if not (resolved / name).is_file()]
    has_weights = any(
        candidate.is_file()
        for pattern in ("*.safetensors", "pytorch_model*.bin")
        for candidate in resolved.glob(pattern)
    )
    if missing or not has_weights:
        details = list(missing)
        if not has_weights:
            details.append("веса *.safetensors или pytorch_model*.bin")
        raise ValidationError(
            "В каталоге ASR-модели отсутствуют обязательные файлы: "
            + ", ".join(details)
        )
    return resolved


def read_text_utf8(path: Path) -> str:
    """Читает текст как строгий UTF-8."""
    with path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
        return stream.read()


def atomic_write_text_utf8(path: Path, text: str) -> None:
    """Атомарно пишет текст в UTF-8 без BOM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open(
            "w", encoding="utf-8", errors="strict", newline="\n"
        ) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    """Атомарно пишет JSON в UTF-8 без BOM."""
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text_utf8(path, text)


def atomic_copy_file(source: Path, target: Path) -> None:
    """Копирует файл с атомарной публикацией результата."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(target)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Вычисляет SHA-256 файла без чтения целиком в память."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def has_utf8_bom(path: Path) -> bool:
    """Проверяет наличие UTF-8 BOM."""
    with path.open("rb") as stream:
        return stream.read(3) == b"\xef\xbb\xbf"


def _resolve_existing(path: Path, label: str) -> Path:
    try:
        return path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValidationError(f"{label} не найден: {path}") from exc


def _temporary_sibling(path: Path) -> Path:
    suffix = f".{uuid.uuid4().hex}.tmp"
    return path.with_name(f".{path.name}{suffix}")


def _is_generated_audio(path: Path) -> bool:
    return path.name.casefold().endswith(".asr.flac")
