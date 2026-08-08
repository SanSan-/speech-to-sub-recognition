"""Изолированный Transformers worker Parakeet TDT v3."""

from __future__ import annotations

import gc
import importlib.metadata
import os
import site
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from speech_to_sub.asr.external_worker import decode_frame, encode_frame

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_REQUIRED_TRANSFORMERS_VERSION = "5.14.1"
_REQUIRED_TORCH_VERSION = "2.11.0+cu128"
_MAX_NEW_TOKENS = 8_192


class ParakeetWorkerRuntime:
    """Держит Parakeet загруженным между оконными запросами."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._processor: Any | None = None
        self._load_key: tuple[str, str, str] | None = None
        self._modules: tuple[Any, Any, Any, Any, str] | None = None
        self._runtime: dict[str, Any] | None = None

    def preflight(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Проверяет версии, устройство и checkpoint без загрузки весов."""
        self._model_path(payload)
        torch, _model_class, _processor_class, _soundfile, engine_version = self._runtime_modules()
        device, compute_type, _dtype = _resolve_device(torch, str(payload.get("device") or "auto"))
        return _runtime_signature(engine_version, device, compute_type)

    def transcribe_window(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Распознаёт один диапазон нормализованного FLAC."""
        audio_path = Path(str(payload.get("audio_path") or "")).expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Аудиофайл для Parakeet не найден: {audio_path}")
        model_path = self._model_path(payload)
        offset = _non_negative_float(payload.get("offset"), "offset")
        duration = _positive_float(payload.get("duration"), "duration")
        self._ensure_loaded(model_path, str(payload.get("device") or "auto"))
        assert self._model is not None and self._processor is not None and self._runtime is not None
        torch, _model_class, _processor_class, soundfile, _version = self._runtime_modules()
        samples, sample_rate = _read_audio_window(soundfile, audio_path, offset, duration)
        inputs = self._processor(samples, sampling_rate=sample_rate, return_tensors="pt")
        inputs = inputs.to(device=self._model.device, dtype=self._model.dtype)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                return_dict_in_generate=True,
                max_new_tokens=_MAX_NEW_TOKENS,
            )
        decoded, timestamps = self._processor.decode(
            output.sequences,
            durations=output.durations,
            skip_special_tokens=True,
        )
        text = str(decoded[0] if isinstance(decoded, (list, tuple)) else decoded).strip()
        token_items = timestamps[0] if timestamps else []
        return {
            "text": text,
            "language": "",
            "tokens": _serialize_tokens(token_items),
            "runtime": dict(self._runtime),
        }

    def unload(self) -> dict[str, Any]:
        """Освобождает веса и CUDA cache."""
        self._model = None
        self._processor = None
        self._load_key = None
        self._runtime = None
        gc.collect()
        if self._modules is not None:
            torch = self._modules[0]
            if bool(torch.cuda.is_available()):
                torch.cuda.empty_cache()
        return {"unloaded": True}

    def _ensure_loaded(self, model_path: Path, requested_device: str) -> None:
        torch, model_class, processor_class, _soundfile, engine_version = self._runtime_modules()
        device, compute_type, dtype = _resolve_device(torch, requested_device)
        key = (str(model_path).casefold(), device, compute_type)
        if self._model is not None and self._load_key == key:
            return
        if self._model is not None:
            self.unload()
        self._processor = processor_class.from_pretrained(str(model_path), local_files_only=True)
        self._model = model_class.from_pretrained(
            str(model_path), local_files_only=True, dtype=dtype
        ).to(device)
        self._model.eval()
        self._load_key = key
        self._runtime = _runtime_signature(engine_version, device, compute_type)

    def _model_path(self, payload: Mapping[str, Any]) -> Path:
        path = Path(str(payload.get("model_path") or "")).expanduser().resolve()
        required = ("config.json", "processor_config.json", "tokenizer.json")
        if not path.is_dir() or any(not (path / name).is_file() for name in required):
            raise FileNotFoundError(f"Некорректный локальный checkpoint Parakeet: {path}")
        if not any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
            raise FileNotFoundError(f"В checkpoint Parakeet отсутствуют веса: {path}")
        return path

    def _runtime_modules(self) -> tuple[Any, Any, Any, Any, str]:
        if self._modules is None:
            self._modules = _import_runtime()
        return self._modules


def handle_request(
    runtime: ParakeetWorkerRuntime,
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Выполняет один запрос протокола."""
    request_id = str(request.get("id") or "")
    command = str(request.get("command") or "").strip().casefold()
    payload_value = request.get("payload")
    payload = dict(payload_value) if isinstance(payload_value, Mapping) else {}
    try:
        if command == "ping":
            result = {"protocol": 1}
        elif command == "preflight":
            result = runtime.preflight(payload)
        elif command == "transcribe-window":
            result = runtime.transcribe_window(payload)
        elif command == "unload":
            result = runtime.unload()
        elif command == "shutdown":
            result = runtime.unload()
            result["shutdown"] = True
        else:
            raise ValueError(f"Неизвестная команда Parakeet worker-а: {command or '?'}")
        return {"id": request_id, "ok": True, "result": result}, command == "shutdown"
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return {
            "id": request_id,
            "ok": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }, False


def main() -> int:
    """Обслуживает framed NDJSON до shutdown или EOF."""
    for stream, errors in ((sys.stdin, "strict"), (sys.stdout, "strict"), (sys.stderr, "replace")):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors=errors, write_through=stream is not sys.stdin)
    runtime = ParakeetWorkerRuntime()
    for line in sys.stdin:
        request = decode_frame(line)
        if request is None:
            continue
        response, should_stop = handle_request(runtime, request)
        sys.stdout.write(encode_frame(response))
        sys.stdout.flush()
        if should_stop:
            break
    runtime.unload()
    return 0


def _import_runtime() -> tuple[Any, Any, Any, Any, str]:
    project_site = Path.cwd() / ".venv" / "Lib" / "site-packages"
    configured_site = os.environ.get("SPEECH_TO_SUB_MAIN_SITE_PACKAGES")
    site.addsitedir(str(Path(configured_site).resolve() if configured_site else project_site.resolve()))
    try:
        import soundfile
        import torch
        from transformers import AutoModelForTDT, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Parakeet worker требует Transformers 5.14.1 и Torch 2.11.0+cu128 из основной .venv."
        ) from exc
    transformers_version = importlib.metadata.version("transformers")
    if transformers_version != _REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Требуется transformers=={_REQUIRED_TRANSFORMERS_VERSION}, обнаружено {transformers_version}."
        )
    if str(torch.__version__) != _REQUIRED_TORCH_VERSION:
        raise RuntimeError(
            f"Требуется torch=={_REQUIRED_TORCH_VERSION}, обнаружено {torch.__version__}."
        )
    return torch, AutoModelForTDT, AutoProcessor, soundfile, transformers_version


def _resolve_device(torch: Any, requested_device: str) -> tuple[str, str, Any]:
    requested = requested_device.strip().casefold()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("Устройство Parakeet должно быть auto, cuda или cpu.")
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("Запрошена CUDA, но PyTorch worker-а не обнаружил GPU.")
    device = "cuda" if cuda_available and requested != "cpu" else "cpu"
    return (device, "float16", torch.float16) if device == "cuda" else (device, "float32", torch.float32)


def _runtime_signature(engine_version: str, device: str, compute_type: str) -> dict[str, Any]:
    return {
        "backend": "parakeet-tdt-v3",
        "engine_version": engine_version,
        "device": device,
        "compute_type": compute_type,
        "quantized": False,
    }


def _read_audio_window(soundfile: Any, path: Path, offset: float, duration: float) -> tuple[Any, int]:
    with soundfile.SoundFile(str(path)) as audio:
        sample_rate = int(audio.samplerate)
        if audio.channels != 1 or sample_rate != 16_000:
            raise RuntimeError("Parakeet ожидает mono 16 kHz FLAC после общей нормализации.")
        audio.seek(round(offset * sample_rate))
        samples = audio.read(round(duration * sample_rate), dtype="float32", always_2d=False)
    if getattr(samples, "size", 0) == 0:
        raise RuntimeError("Окно Parakeet не содержит аудиосэмплов.")
    return samples, sample_rate


def _serialize_tokens(items: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in items or ():
        if not isinstance(item, Mapping):
            continue
        result.append(
            {
                "token": str(item.get("token") or ""),
                "start": float(item.get("start", 0.0)),
                "end": float(item.get("end", 0.0)),
            }
        )
    return result


def _non_negative_float(value: Any, label: str) -> float:
    result = float(value)
    if result < 0:
        raise ValueError(f"{label} должен быть неотрицательным.")
    return result


def _positive_float(value: Any, label: str) -> float:
    result = float(value)
    if result <= 0:
        raise ValueError(f"{label} должен быть положительным.")
    return result


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ParakeetWorkerRuntime", "handle_request", "main"]
