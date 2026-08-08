"""Локальные ASR backend-ы."""

from speech_to_sub.asr.base import AsrBackend
from speech_to_sub.asr.registry import (
    activate_backend,
    backend_names,
    get_backend,
    unload_backend,
    unload_backends,
)


def get_default_backend() -> AsrBackend:
    """Возвращает общий Transformers backend для обратной совместимости."""
    return get_backend("transformers")


def unload_default_backend() -> None:
    """Выгружает общий Transformers backend для обратной совместимости."""
    unload_backend("transformers")


__all__ = [
    "activate_backend",
    "backend_names",
    "get_backend",
    "get_default_backend",
    "unload_backends",
    "unload_default_backend",
]
