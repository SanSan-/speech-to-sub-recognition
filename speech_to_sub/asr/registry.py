from __future__ import annotations

import importlib
import threading
from collections.abc import Callable
from typing import Any, cast

from speech_to_sub.asr.base import AsrBackend
from speech_to_sub.exceptions import AsrModelError, ValidationError

BackendFactory = Callable[[], AsrBackend]
FactoryReference = str | BackendFactory

_FACTORIES: dict[str, FactoryReference] = {
    "transformers": (
        "speech_to_sub.asr.transformers_whisper:TransformersWhisperBackend"
    ),
    "faster-whisper": "speech_to_sub.asr.faster_whisper:FasterWhisperBackend",
    "parakeet-tdt-v3": "speech_to_sub.asr.parakeet_tdt:ParakeetTdtBackend",
    "qwen3-asr": "speech_to_sub.asr.qwen3:Qwen3AsrBackend",
    "openai-api": "speech_to_sub.asr.openai_api:OpenAiApiBackend",
}
_INSTANCES: dict[str, AsrBackend] = {}
_ACTIVE_BACKEND_ID: str | None = None
_LOCK = threading.RLock()


def backend_names() -> tuple[str, ...]:
    """Возвращает поддерживаемые идентификаторы в стабильном порядке."""
    return tuple(_FACTORIES)


def get_backend(name: str) -> AsrBackend:
    """Лениво создаёт и повторно использует backend без загрузки весов."""
    backend_id = _normalize_backend_id(name)
    with _LOCK:
        existing = _INSTANCES.get(backend_id)
        if existing is not None:
            return existing
        factory = _resolve_factory(backend_id, _FACTORIES[backend_id])
        try:
            backend = factory()
        except Exception as exc:
            raise AsrModelError(
                f"Не удалось инициализировать ASR backend '{backend_id}': {exc}"
            ) from exc
        _validate_backend(backend_id, backend)
        _INSTANCES[backend_id] = backend
        return backend


def activate_backend(name: str) -> AsrBackend:
    """Активирует backend и освобождает модель другого backend-а."""
    global _ACTIVE_BACKEND_ID

    backend_id = _normalize_backend_id(name)
    with _LOCK:
        backend = get_backend(backend_id)
        if _ACTIVE_BACKEND_ID != backend_id:
            previous = _INSTANCES.get(_ACTIVE_BACKEND_ID or "")
            if previous is not None:
                previous.unload()
            _ACTIVE_BACKEND_ID = backend_id
        return backend


def unload_backend(name: str) -> None:
    """Выгружает один уже созданный backend, не создавая новый."""
    global _ACTIVE_BACKEND_ID

    backend_id = _normalize_backend_id(name)
    with _LOCK:
        backend = _INSTANCES.get(backend_id)
        if backend is not None:
            backend.unload()
        if _ACTIVE_BACKEND_ID == backend_id:
            _ACTIVE_BACKEND_ID = None


def unload_backends() -> None:
    """Освобождает модели всех созданных backend-ов."""
    global _ACTIVE_BACKEND_ID

    with _LOCK:
        first_error: Exception | None = None
        for backend in tuple(_INSTANCES.values()):
            try:
                backend.unload()
            except Exception as exc:  # pragma: no cover - защитный барьер выгрузки
                if first_error is None:
                    first_error = exc
        _ACTIVE_BACKEND_ID = None
        if first_error is not None:
            raise AsrModelError(
                f"Не удалось выгрузить все ASR backend-ы: {first_error}"
            ) from first_error


def _normalize_backend_id(name: str) -> str:
    backend_id = str(name).strip().casefold()
    if backend_id not in _FACTORIES:
        variants = ", ".join(backend_names())
        raise ValidationError(
            f"Неизвестный ASR backend '{name}'. Поддерживаются: {variants}."
        )
    return backend_id


def _resolve_factory(backend_id: str, reference: FactoryReference) -> BackendFactory:
    if callable(reference):
        return cast(BackendFactory, reference)
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise AsrModelError(
            f"Некорректная factory-ссылка ASR backend '{backend_id}': {reference}"
        )
    try:
        module = importlib.import_module(module_name)
        factory: Any = getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise AsrModelError(
            f"ASR backend '{backend_id}' недоступен: {exc}"
        ) from exc
    if not callable(factory):
        raise AsrModelError(f"Factory ASR backend '{backend_id}' не является вызываемой.")
    return cast(BackendFactory, factory)


def _validate_backend(backend_id: str, backend: AsrBackend) -> None:
    actual_id = str(getattr(backend, "backend_id", "")).strip().casefold()
    if actual_id != backend_id:
        raise AsrModelError(
            f"Factory '{backend_id}' вернула backend с идентификатором '{actual_id or '?'}'."
        )
    required = (
        "preflight",
        "expected_runtime_signature",
        "runtime_signature",
        "transcribe",
        "unload",
    )
    if any(not callable(getattr(backend, name, None)) for name in required):
        raise AsrModelError(f"ASR backend '{backend_id}' не соблюдает общий контракт.")


__all__ = [
    "activate_backend",
    "backend_names",
    "get_backend",
    "unload_backend",
    "unload_backends",
]
