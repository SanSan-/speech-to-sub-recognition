"""Изолированный persistent worker локального Qwen3 ForcedAligner."""

from __future__ import annotations

import gc
import importlib.metadata
import math
import os
from collections.abc import Callable, Mapping, Sequence
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

_SAMPLE_RATE = 16_000
_MAX_SEGMENT_SECONDS = 180.0
_REQUIRED_QWEN_ASR_VERSION = "0.0.6"
_REQUIRED_TRANSFORMERS_VERSION = "4.57.6"
_REQUIRED_TORCH_VERSION = "2.8.0+cu128"
_SUPPORTED_LANGUAGES = {"en": "English", "ru": "Russian"}

WorkerProgressCallback = Callable[[int], None]


class QwenAlignerWorkerRuntime:
    """Держит только ForcedAligner и выравнивает переданные ASR-сегменты."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._load_key: tuple[str, str, str] | None = None
        self._modules: tuple[Any, Any, Any, str] | None = None
        self._runtime: dict[str, Any] | None = None

    def preflight(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Проверяет пакеты, устройство и полный локальный checkpoint."""
        self._request_model_path(payload)
        torch, _model, _normalize, version = self._runtime_modules()
        device, compute_type, _dtype = _resolve_device(
            torch,
            str(payload.get("device") or "auto"),
        )
        return _runtime_signature(version, device, compute_type)

    def align(
        self,
        payload: Mapping[str, Any],
        progress_callback: WorkerProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Вырезает исходное аудио по сегментам и добавляет глобальные offsets."""
        audio_path = Path(str(payload.get("audio_path") or "")).expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Аудиофайл для ForcedAligner не найден: {audio_path}")
        model_path = self._request_model_path(payload)
        waveform = self._normalized_audio(audio_path)
        segments = _request_segments(payload.get("segments"), len(waveform) / _SAMPLE_RATE)
        self._ensure_loaded(model_path, str(payload.get("device") or "auto"))
        aligned = self._align_segments(waveform, segments, progress_callback)
        assert self._runtime is not None
        return {"segments": aligned, "runtime": dict(self._runtime)}

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

    def _normalized_audio(self, audio_path: Path) -> Any:
        _torch, _model, normalize_audios, _version = self._runtime_modules()
        waveforms = normalize_audios(str(audio_path))
        if len(waveforms) != 1:
            raise RuntimeError("ForcedAligner ожидал одну нормализованную аудиодорожку.")
        return waveforms[0]

    def _align_segments(
        self,
        waveform: Any,
        segments: list[dict[str, Any]],
        progress_callback: WorkerProgressCallback | None,
    ) -> list[dict[str, Any]]:
        assert self._model is not None
        result: list[dict[str, Any]] = []
        for index, segment in enumerate(segments, start=1):
            start_sample = round(segment["start"] * _SAMPLE_RATE)
            end_sample = round(segment["end"] * _SAMPLE_RATE)
            fragment = waveform[start_sample:end_sample]
            outputs = self._model.align(
                audio=(fragment, _SAMPLE_RATE),
                text=segment["text"],
                language=_language_name(segment["language"]),
            )
            if not outputs:
                raise RuntimeError("Qwen3 ForcedAligner не вернул результат.")
            result.append(
                {
                    "index": segment["index"],
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "words": _serialize_words(outputs[0], segment["start"]),
                }
            )
            if progress_callback is not None:
                progress_callback(round(index / len(segments) * 100))
        return result

    def _ensure_loaded(self, model_path: Path, requested_device: str) -> None:
        torch, model_class, _normalize, version = self._runtime_modules()
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
        )
        self._load_key = key
        self._runtime = _runtime_signature(version, device, compute_type)

    def _request_model_path(self, payload: Mapping[str, Any]) -> Path:
        return require_local_checkpoint(
            payload.get("model_path"),
            "Qwen3 ForcedAligner",
            error_type=FileNotFoundError,
        )

    def _runtime_modules(self) -> tuple[Any, Any, Any, str]:
        if self._modules is None:
            self._modules = _import_qwen_aligner_runtime()
            if self._modules[3] != _REQUIRED_QWEN_ASR_VERSION:
                raise RuntimeError(
                    "Изолированный runtime должен содержать "
                    f"qwen-asr=={_REQUIRED_QWEN_ASR_VERSION}, обнаружено "
                    f"{self._modules[3]}."
                )
        return self._modules


handle_request = make_worker_request_handler(
    action_name="align",
    worker_name="ForcedAligner",
)


def main() -> int:
    """Читает framed NDJSON из stdin до команды shutdown или EOF."""
    return run_worker_loop(QwenAlignerWorkerRuntime(), handle_request)


def _request_segments(value: Any, audio_duration: float) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("ForcedAligner не получил список ASR-сегментов.")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Некорректный ASR-сегмент {index}.")
        segment = _request_segment(raw, index, audio_duration)
        if segment["text"]:
            result.append(segment)
    if not result:
        raise ValueError("ForcedAligner не получил текста для выравнивания.")
    return result


def _request_segment(raw: Mapping[str, Any], index: int, duration: float) -> dict[str, Any]:
    text = " ".join(str(raw.get("text") or "").split())
    language = str(raw.get("language") or "").strip().casefold()
    try:
        start = float(raw.get("start", 0.0))
        end = float(raw.get("end", duration))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Некорректные границы ASR-сегмента {index}.") from exc
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end > duration + 0.05:
        raise ValueError(f"ASR-сегмент {index} выходит за границы аудио.")
    end = min(end, duration)
    if end <= start or end - start > _MAX_SEGMENT_SECONDS + 0.05:
        raise ValueError(
            f"ASR-сегмент {index} должен иметь длительность от 0 до {_MAX_SEGMENT_SECONDS:g} с."
        )
    _language_name(language)
    return {"index": index, "text": text, "language": language, "start": start, "end": end}


def _serialize_words(result: Any, offset: float) -> list[dict[str, Any]]:
    raw_items = result.get("items", ()) if isinstance(result, Mapping) else getattr(result, "items", ())
    words: list[dict[str, Any]] = []
    for item in raw_items:
        text = str(_value(item, "text", "")).strip()
        start = float(_value(item, "start_time", 0.0)) + offset
        end = float(_value(item, "end_time", 0.0)) + offset
        if text and math.isfinite(start) and math.isfinite(end):
            words.append({"text": text, "start": start, "end": end})
    if not words:
        raise RuntimeError("Qwen3 ForcedAligner не вернул временные метки слов.")
    return words


def _resolve_device(torch: Any, requested_device: str) -> tuple[str, str, Any]:
    return resolve_device(torch, requested_device, device_name="ForcedAligner")


def _runtime_signature(engine_version: str, device: str, compute_type: str) -> dict[str, Any]:
    return {
        "backend": "qwen3-forced-aligner",
        "engine_version": engine_version,
        "device": device,
        "compute_type": compute_type,
        "quantized": False,
    }


def _language_name(language: str) -> str:
    try:
        return _SUPPORTED_LANGUAGES[language]
    except KeyError as exc:
        raise ValueError("Qwen3 ForcedAligner поддерживает язык en или ru.") from exc


def _value(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _import_qwen_aligner_runtime() -> tuple[Any, Any, Any, str]:
    try:
        import torch
        from qwen_asr import Qwen3ForcedAligner
        from qwen_asr.inference.utils import normalize_audios
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
    return torch, Qwen3ForcedAligner, normalize_audios, version


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["QwenAlignerWorkerRuntime", "handle_request", "main"]
