"""Строгая проверка локальных checkpoint-ов без загрузки весов."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from speech_to_sub.exceptions import ValidationError

_WEIGHT_INDEX_NAMES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
_WEIGHT_PATTERNS = ("*.safetensors", "pytorch_model*.bin")


def require_local_checkpoint(
    value: Path | str | Any,
    label: str,
    *,
    error_type: type[Exception] = ValidationError,
) -> Path:
    """Проверяет config, индекс шардов и наличие всех локальных весов."""
    if value is None or not str(value).strip():
        raise error_type(f"Локальный checkpoint {label} не задан.")
    path = Path(str(value or "")).expanduser().resolve()
    if not path.is_dir():
        raise error_type(f"Каталог локальной модели {label} не найден: {path}")
    if not (path / "config.json").is_file():
        raise error_type(f"В checkpoint {label} отсутствует config.json: {path}")
    indexed_files = _indexed_weight_files(path, label, error_type)
    if indexed_files:
        return path
    if not any(any(path.glob(pattern)) for pattern in _WEIGHT_PATTERNS):
        raise error_type(f"В checkpoint {label} отсутствуют локальные веса: {path}")
    return path


def _indexed_weight_files(
    path: Path,
    label: str,
    error_type: type[Exception],
) -> set[str]:
    found: set[str] = set()
    for name in _WEIGHT_INDEX_NAMES:
        index_path = path / name
        if not index_path.is_file():
            continue
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise error_type(f"Некорректный индекс весов {label}: {index_path}") from exc
        weight_map = raw.get("weight_map") if isinstance(raw, Mapping) else None
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise error_type(f"Индекс весов {label} не содержит weight_map: {index_path}")
        referenced = {str(value) for value in weight_map.values() if str(value)}
        targets = {name: (path / name).resolve() for name in referenced}
        escaped = sorted(name for name, target in targets.items() if path not in target.parents)
        if escaped:
            raise error_type(
                f"Индекс весов {label} содержит путь вне checkpoint: {', '.join(escaped)}"
            )
        missing = sorted(name for name, target in targets.items() if not target.is_file())
        if missing:
            raise error_type(
                f"В checkpoint {label} отсутствуют шарды из индекса: {', '.join(missing)}"
            )
        found.update(referenced)
    return found


__all__ = ["require_local_checkpoint"]
