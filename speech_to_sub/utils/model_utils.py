from __future__ import annotations

import gc
import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from speech_to_sub.exceptions import AsrModelError, ValidationError
from speech_to_sub.models import RuntimeSignature

logger = logging.getLogger(__name__)


def resolve_device_and_quantization(
    requested_device: str,
    quantization_enabled: bool,
) -> tuple[Any, Any, Any | None, bool]:
    """Выбирает torch device, dtype и поддерживаемую 8-битную конфигурацию."""
    torch = import_torch()
    requested = requested_device.strip().casefold()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValidationError("Устройство должно быть auto, cuda или cpu.")
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise AsrModelError("Запрошена CUDA, но torch.cuda.is_available() вернул False.")
    use_cuda = cuda_available and requested != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float16 if use_cuda else torch.float32
    quantization_config = None
    quantized = False
    if use_cuda and quantization_enabled:
        config_class = import_bitsandbytes_config()
        if config_class is None:
            logger.warning("8-битная конфигурация недоступна; используется FP16.")
        else:
            quantization_config = config_class(load_in_8bit=True)
            quantized = True
    return torch, device, dtype, quantization_config, quantized


def resolve_expected_runtime_signature(
    requested_device: str,
    quantization_enabled: bool,
    *,
    backend: str = "transformers",
    engine_version: str | None = None,
) -> RuntimeSignature:
    """Определяет ожидаемый runtime кеша без загрузки весов модели."""
    _torch, device, _dtype, _config, quantized = resolve_device_and_quantization(
        requested_device,
        quantization_enabled,
    )
    device_name = str(getattr(device, "type", device)).casefold()
    return {
        "backend": backend.strip().casefold(),
        "engine_version": engine_version or package_version("transformers"),
        "device": "cuda" if device_name == "cuda" else "cpu",
        "compute_type": _transformers_compute_type(device_name, quantized),
        "quantized": quantized,
    }


def package_version(package: str) -> str:
    """Возвращает установленную версию без импорта тяжёлого runtime."""
    try:
        return version(package)
    except PackageNotFoundError:
        return "unavailable"


def _transformers_compute_type(device: str, quantized: bool) -> str:
    if quantized:
        return "int8"
    return "float16" if device == "cuda" else "float32"


def clear_model_memory() -> None:
    """Освобождает Python- и CUDA-память после выгрузки модели."""
    gc.collect()
    try:
        torch = import_torch()
    except AsrModelError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            ipc_collect()


def import_torch() -> Any:
    """Лениво импортирует torch."""
    try:
        import torch
    except ImportError as exc:
        raise AsrModelError("Для локального распознавания требуется torch.") from exc
    return torch


def import_bitsandbytes_config() -> Any | None:
    """Возвращает BitsAndBytesConfig, если он доступен."""
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        return None
    return BitsAndBytesConfig
