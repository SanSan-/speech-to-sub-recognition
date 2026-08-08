from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable

from speech_to_sub.constants import SUPPORTED_MEDIA_EXTENSIONS
from speech_to_sub.exceptions import ValidationError


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
    found: dict[str, Path] = {}
    for raw_path in paths:
        resolved = _resolve_existing(raw_path, "Входной путь")
        if resolved.is_file():
            media = validate_media_file(resolved)
            found[str(media).casefold()] = media
            continue
        iterator = resolved.rglob("*") if recursive else resolved.iterdir()
        for candidate in iterator:
            if not candidate.is_file():
                continue
            if candidate.suffix.casefold() not in SUPPORTED_MEDIA_EXTENSIONS:
                continue
            if _is_generated_audio(candidate):
                continue
            media = validate_media_file(candidate)
            found[str(media).casefold()] = media
    return sorted(found.values(), key=lambda value: str(value).casefold())


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
            "В каталоге ASR-модели отсутствуют обязательные файлы: " + ", ".join(details)
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
        with temporary.open("w", encoding="utf-8", errors="strict", newline="\n") as stream:
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
