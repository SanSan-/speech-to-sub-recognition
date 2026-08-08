"""Изолированный persistent worker локального Qwen3-ASR 0.0.6."""

from __future__ import annotations

import gc
import importlib.metadata
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from speech_to_sub.asr.checkpoints import require_local_checkpoint
from speech_to_sub.workers.common import (
    make_worker_request_handler,
    resolve_device,
    run_worker_loop,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_SUPPORTED_LANGUAGES: dict[str, str | None] = {
    "auto": None,
    "en": "English",
    "ru": "Russian",
}
_MAX_NEW_TOKENS = 4_096
# Внутренний splitter ищет тихую границу в пределах +/-5 секунд.
# Target 175 гарантирует фактический фрагмент не длиннее лимита aligner-а 180 секунд.
_ASR_CHUNK_TARGET_SECONDS = 175.0
_SAMPLE_RATE = 16_000
_REQUIRED_QWEN_ASR_VERSION = "0.0.6"
_REQUIRED_TRANSFORMERS_VERSION = "4.57.6"
_REQUIRED_TORCH_VERSION = "2.8.0+cu128"

WorkerProgressCallback = Callable[[int], None]


class QwenWorkerRuntime:
    """Держит только Qwen ASR, не загружая ForcedAligner в тот же процесс."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._load_key: tuple[str, str, str] | None = None
        self._modules: tuple[Any, Any, Any, Any, str] | None = None
        self._runtime: dict[str, Any] | None = None

    def preflight(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Проверяет пакеты, устройство и полный локальный checkpoint."""
        self._request_model_path(payload)
        torch, _model, _normalize, _split, version = self._runtime_modules()
        device, compute_type, _dtype = _resolve_device(
            torch,
            str(payload.get("device") or "auto"),
        )
        return _runtime_signature(version, device, compute_type)

    def transcribe(
        self,
        payload: Mapping[str, Any],
        progress_callback: WorkerProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Распознаёт последовательно неперекрывающиеся фрагменты до 180 секунд."""
        audio_path = Path(str(payload.get("audio_path") or "")).expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Аудиофайл для Qwen3-ASR не найден: {audio_path}")
        model_path = self._request_model_path(payload)
        language = _language_name(str(payload.get("language") or "auto"))
        self._ensure_loaded(model_path, str(payload.get("device") or "auto"))
        chunks = self._audio_chunks(audio_path)
        serialized = self._transcribe_chunks(chunks, language, progress_callback)
        assert self._runtime is not None
        return {
            "text": " ".join(chunk["text"] for chunk in serialized if chunk["text"]),
            "language": _merge_languages(serialized),
            "chunks": serialized,
            "runtime": dict(self._runtime),
        }

    def unload(self) -> dict[str, Any]:
        """Освобождает веса, сохраняя импортированные модули runtime."""
        self._model = None
        self._load_key = None
        self._runtime = None
        gc.collect()
        if self._modules is not None:
            torch = self._modules[0]
            if bool(torch.cuda.is_available()):
                torch.cuda.empty_cache()
        return {"unloaded": True}

    def _audio_chunks(self, audio_path: Path) -> list[tuple[Any, float]]:
        _torch, _model, normalize_audios, split_audio, _version = self._runtime_modules()
        waveforms = normalize_audios(str(audio_path))
        if len(waveforms) != 1:
            raise RuntimeError("Qwen3-ASR ожидал одну нормализованную аудиодорожку.")
        return list(
            split_audio(
                wav=waveforms[0],
                sr=_SAMPLE_RATE,
                max_chunk_sec=_ASR_CHUNK_TARGET_SECONDS,
            )
        )

    def _transcribe_chunks(
        self,
        chunks: list[tuple[Any, float]],
        language: str | None,
        progress_callback: WorkerProgressCallback | None,
    ) -> list[dict[str, Any]]:
        if not chunks:
            raise RuntimeError("Qwen3-ASR не получил аудиофрагментов для распознавания.")
        assert self._model is not None
        result: list[dict[str, Any]] = []
        for index, (waveform, offset) in enumerate(chunks, start=1):
            outputs = self._model.transcribe(
                audio=(waveform, _SAMPLE_RATE),
                language=language,
                return_time_stamps=False,
            )
            if not outputs:
                raise RuntimeError("Qwen3-ASR не вернул результат распознавания.")
            item = outputs[0]
            chunk_duration = len(waveform) / float(_SAMPLE_RATE)
            result.append(
                {
                    "text": str(_value(item, "text", "")).strip(),
                    "language": str(_value(item, "language", "") or ""),
                    "start": float(offset),
                    "end": float(offset) + chunk_duration,
                }
            )
            if progress_callback is not None:
                progress_callback(round(index / len(chunks) * 100))
        return result

    def _ensure_loaded(self, model_path: Path, requested_device: str) -> None:
        torch, model_class, _normalize, _split, version = self._runtime_modules()
        device, compute_type, dtype = _resolve_device(torch, requested_device)
        key = (str(model_path).casefold(), device, compute_type)
        if self._model is not None and self._load_key == key:
            return
        if self._model is not None:
            self.unload()
        self._model = model_class.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map="cuda:0" if device == "cuda" else "cpu",
            local_files_only=True,
            max_inference_batch_size=1,
            max_new_tokens=_MAX_NEW_TOKENS,
        )
        self._load_key = key
        self._runtime = _runtime_signature(version, device, compute_type)

    def _request_model_path(self, payload: Mapping[str, Any]) -> Path:
        return require_local_checkpoint(
            payload.get("model_path"),
            "Qwen3-ASR",
            error_type=FileNotFoundError,
        )

    def _runtime_modules(self) -> tuple[Any, Any, Any, Any, str]:
        if self._modules is None:
            self._modules = _import_qwen_runtime()
            if self._modules[4] != _REQUIRED_QWEN_ASR_VERSION:
                raise RuntimeError(
                    "Изолированный runtime должен содержать "
                    f"qwen-asr=={_REQUIRED_QWEN_ASR_VERSION}, обнаружено "
                    f"{self._modules[4]}."
                )
        return self._modules


handle_request = make_worker_request_handler(action_name="transcribe", worker_name="Qwen")


def main() -> int:
    """Читает framed NDJSON из stdin до команды shutdown или EOF."""
    return run_worker_loop(QwenWorkerRuntime(), handle_request)


def _resolve_device(torch: Any, requested_device: str) -> tuple[str, str, Any]:
    return resolve_device(torch, requested_device, device_name="Qwen3-ASR")


def _runtime_signature(engine_version: str, device: str, compute_type: str) -> dict[str, Any]:
    return {
        "backend": "qwen3-asr",
        "engine_version": engine_version,
        "device": device,
        "compute_type": compute_type,
        "quantized": False,
    }


def _language_name(language: str) -> str | None:
    normalized = language.strip().casefold()
    if normalized not in _SUPPORTED_LANGUAGES:
        variants = ", ".join(_SUPPORTED_LANGUAGES)
        raise ValueError(f"Qwen3-ASR поддерживает языки профиля: {variants}.")
    return _SUPPORTED_LANGUAGES[normalized]


def _merge_languages(chunks: list[dict[str, Any]]) -> str:
    values = [str(chunk.get("language") or "").strip() for chunk in chunks]
    return ",".join(dict.fromkeys(value for value in values if value))


def _value(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _import_qwen_runtime() -> tuple[Any, Any, Any, Any, str]:
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
        from qwen_asr.inference.utils import normalize_audios, split_audio_into_chunks
    except ImportError as exc:
        raise RuntimeError(
            "В изолированном runtime требуется qwen-asr==0.0.6 и совместимый PyTorch."
        ) from exc
    version = _package_version("qwen-asr")
    transformers_version = _package_version("transformers")
    torch_version = str(torch.__version__)
    if transformers_version != _REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Изолированный Qwen runtime должен содержать transformers=="
            f"{_REQUIRED_TRANSFORMERS_VERSION}, обнаружено {transformers_version}."
        )
    if torch_version != _REQUIRED_TORCH_VERSION:
        raise RuntimeError(
            "Изолированный Qwen runtime должен содержать torch=="
            f"{_REQUIRED_TORCH_VERSION}, обнаружено {torch_version}."
        )
    return torch, Qwen3ASRModel, normalize_audios, split_audio_into_chunks, version


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["QwenWorkerRuntime", "handle_request", "main"]
