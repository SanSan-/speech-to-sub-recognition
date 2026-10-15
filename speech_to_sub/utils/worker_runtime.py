"""Разрешение и проверка Python выбранных изолированных модельных сред."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from speech_to_sub.constants import BASE_DIR
from speech_to_sub.exceptions import ValidationError
from speech_to_sub.models import ProcessingSettings


@dataclass(frozen=True)
class _WorkerProfile:
    path_setting: str
    environment_names: tuple[str, ...]
    default_relative_path: Path
    runtime_name: str
    setting_hint: str


_QWEN_PYTHON = Path("resources/runtimes/qwen/Scripts/python.exe")
_WORKER_PROFILES = {
    "qwen3-asr": _WorkerProfile(
        "worker_python_path",
        ("QWEN_ASR_PYTHON",),
        _QWEN_PYTHON,
        "Qwen",
        "QWEN_ASR_PYTHON или worker_python_path",
    ),
    "parakeet-tdt-v3": _WorkerProfile(
        "worker_python_path",
        ("PARAKEET_ASR_PYTHON",),
        Path("resources/runtimes/parakeet/Scripts/python.exe"),
        "Parakeet",
        "PARAKEET_ASR_PYTHON или worker_python_path",
    ),
    "qwen3-forced-aligner": _WorkerProfile(
        "aligner_worker_python_path",
        ("ASR_ALIGNER_WORKER_PYTHON", "QWEN_ASR_PYTHON"),
        _QWEN_PYTHON,
        "Qwen",
        "ASR_ALIGNER_WORKER_PYTHON или aligner_worker_python_path",
    ),
}


def resolve_worker_python(settings: ProcessingSettings, profile_id: str) -> Path:
    """Разрешает путь профиля без проверки наличия и запуска интерпретатора."""
    profile = _WORKER_PROFILES[profile_id.strip().casefold()]
    raw_path = getattr(settings, profile.path_setting, None)
    for name in profile.environment_names:
        raw_path = raw_path or os.environ.get(name)
    if raw_path:
        return Path(str(raw_path)).expanduser().resolve()
    return (BASE_DIR / profile.default_relative_path).resolve()


def require_worker_python(settings: ProcessingSettings, profile_id: str) -> Path:
    """Проверяет наличие файла Python, не запуская worker и не читая модель."""
    profile = _WORKER_PROFILES[profile_id.strip().casefold()]
    worker_python = resolve_worker_python(settings, profile_id)
    if not worker_python.is_file():
        raise ValidationError(
            f"Python изолированного {profile.runtime_name} runtime не найден: "
            f"{worker_python}. Укажите {profile.setting_hint}."
        )
    return worker_python


def validate_selected_worker_pythons(settings: ProcessingSettings) -> None:
    """Проверяет Python только выбранных изолированных движка и выравнивателя."""
    for selected in (settings.backend, settings.aligner):
        profile_id = selected.strip().casefold()
        if profile_id in _WORKER_PROFILES:
            require_worker_python(settings, profile_id)
