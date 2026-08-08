from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Mapping, cast

from speech_to_sub.asr.base import CancelCheck, ProgressCallback
from speech_to_sub.exceptions import AsrModelError, ProcessingCancelled
from speech_to_sub.media.ffmpeg import decode_audio_float32
from speech_to_sub.models import (
    ProcessingSettings,
    RuntimeSignature,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)
from speech_to_sub.utils.io_utils import validate_model_path
from speech_to_sub.utils.env_utils import get_command_path
from speech_to_sub.utils.model_utils import (
    clear_model_memory,
    package_version,
    resolve_device_and_quantization,
    resolve_expected_runtime_signature,
)

logger = logging.getLogger(__name__)


class TransformersWhisperBackend:
    """Локальный Whisper backend через Hugging Face Transformers."""

    backend_id = "transformers"

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._model: Any | None = None
        self._processor: Any | None = None
        self._load_key: tuple[str, str, str, bool, bool] | None = None
        self._device_label = "unloaded"
        self._compute_type = "unloaded"
        self._quantized = False
        self._engine_version = package_version("transformers")
        self._lock = threading.RLock()

    def preflight(self, settings: ProcessingSettings) -> Path:
        """Проверяет локальный Hugging Face checkpoint без загрузки весов."""
        return validate_model_path(settings.model_path)

    def expected_runtime_signature(
        self,
        settings: ProcessingSettings,
    ) -> RuntimeSignature:
        """Определяет ожидаемый Transformers runtime без загрузки весов."""
        return resolve_expected_runtime_signature(
            settings.device,
            settings.quantization_enabled,
            backend=self.backend_id,
            engine_version=self._engine_version,
        )

    def transcribe(
        self,
        audio_path: Path,
        settings: ProcessingSettings,
        duration: float,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> Transcript:
        """Распознаёт FLAC/WAV и нормализует ответ pipeline."""
        _raise_if_cancelled(cancel_check)
        if progress_callback:
            progress_callback(0)
        with self._lock:
            self._ensure_loaded(settings)
            assert self._pipeline is not None
            audio_samples = decode_audio_float32(
                audio_path,
                ffmpeg_path=get_command_path("FFMPEG_PATH", "ffmpeg"),
            )
            _raise_if_cancelled(cancel_check)
            language = None if settings.language == "auto" else settings.language
            if language is None:
                language = self._detect_language(audio_samples)
            generation: dict[str, Any] = {"task": "transcribe"}
            if language:
                generation["language"] = language
            try:
                output = self._run_pipeline(
                    audio_samples,
                    settings=settings,
                    generation=generation,
                )
            except Exception as exc:
                output = self._retry_inference_on_cpu(
                    exc,
                    audio_samples=audio_samples,
                    settings=settings,
                    generation=generation,
                )
        _raise_if_cancelled(cancel_check)
        if progress_callback:
            progress_callback(100)
        return normalize_pipeline_output(
            output,
            duration=duration,
            language=language or "auto",
            model=str(settings.model_path),
            device=self._device_label,
            engine_version=self._engine_version,
            compute_type=self._compute_type,
            quantized=self._quantized,
        )

    def unload(self) -> None:
        """Выгружает pipeline, processor и model."""
        with self._lock:
            self._pipeline = None
            self._model = None
            self._processor = None
            self._load_key = None
            self._device_label = "unloaded"
            self._compute_type = "unloaded"
            self._quantized = False
        clear_model_memory()

    def runtime_signature(
        self,
        settings: ProcessingSettings,
    ) -> RuntimeSignature | None:
        """Возвращает runtime только для модели, загруженной с этими настройками."""
        requested_key = _model_load_key(
            settings.model_path.expanduser().resolve(),
            settings,
        )
        with self._lock:
            if self._pipeline is None or self._load_key != requested_key:
                return None
            return {
                "backend": self.backend_id,
                "engine_version": self._engine_version,
                "device": self._device_label.casefold(),
                "compute_type": self._compute_type,
                "quantized": self._quantized,
            }

    def _ensure_loaded(self, settings: ProcessingSettings) -> None:
        model_path = validate_model_path(settings.model_path)
        key = _model_load_key(model_path, settings)
        if self._pipeline is not None and self._load_key == key:
            return
        if self._pipeline is not None:
            self.unload()
        try:
            self._load(model_path, settings, allow_fallback=settings.allow_cpu_fallback)
        except AsrModelError:
            raise
        except Exception as exc:
            raise AsrModelError(f"Не удалось загрузить локальную ASR-модель: {exc}") from exc
        self._load_key = key

    def _retry_inference_on_cpu(
        self,
        cuda_error: Exception,
        *,
        audio_samples: Any,
        settings: ProcessingSettings,
        generation: Mapping[str, Any],
    ) -> Any:
        """Один раз повторяет неудачный CUDA-инференс на CPU, если это разрешено."""
        if not settings.allow_cpu_fallback or self._device_label != "cuda":
            raise AsrModelError(
                f"Локальная ASR-модель завершилась ошибкой: {cuda_error}"
            ) from cuda_error

        logger.warning("Инференс ASR на CUDA не удался, пробую CPU: %s", cuda_error)
        requested_key = self._load_key
        self.unload()
        try:
            model_path = validate_model_path(settings.model_path)
            self._load_once(model_path, "cpu", False)
            self._load_key = requested_key
            return self._run_pipeline(
                audio_samples,
                settings=settings,
                generation=generation,
            )
        except Exception as cpu_error:
            self.unload()
            raise AsrModelError(
                "Инференс ASR завершился ошибкой на CUDA и при разрешённом повторе на CPU."
            ) from cpu_error

    def _run_pipeline(
        self,
        audio_samples: Any,
        *,
        settings: ProcessingSettings,
        generation: Mapping[str, Any],
    ) -> Any:
        assert self._pipeline is not None
        return self._pipeline(
            {"raw": audio_samples, "sampling_rate": 16_000},
            chunk_length_s=settings.chunk_length_seconds,
            stride_length_s=settings.stride_length_seconds,
            return_timestamps="word",
            return_language=settings.language == "auto",
            generate_kwargs=dict(generation),
        )

    def _detect_language(self, audio_samples: Any) -> str | None:
        """Определяет язык по первому окну Whisper до основного generate."""
        assert self._model is not None
        assert self._processor is not None
        try:
            first_window = audio_samples[: 30 * 16_000]
            features = self._processor.feature_extractor(
                first_window,
                sampling_rate=16_000,
                return_tensors="pt",
            )
            input_features = features.input_features.to(
                device=self._model.device,
                dtype=self._model.dtype,
            )
            language_id = int(self._model.detect_language(input_features)[0].item())
            token = self._processor.tokenizer.convert_ids_to_tokens(language_id)
            return _normalize_language_label(token, None)
        except Exception as exc:
            logger.warning("Не удалось определить язык Whisper автоматически: %s", exc)
            return None

    def _load(
        self,
        model_path: Path,
        settings: ProcessingSettings,
        allow_fallback: bool,
    ) -> None:
        try:
            self._load_once(model_path, settings.device, settings.quantization_enabled)
        except Exception as exc:
            if allow_fallback and settings.device.casefold() != "cpu":
                logger.warning("Загрузка ASR на GPU не удалась, пробую CPU: %s", exc)
                self.unload()
                try:
                    self._load_once(model_path, "cpu", False)
                    return
                except Exception as cpu_exc:
                    raise AsrModelError("Не удалось загрузить ASR-модель на CPU.") from cpu_exc
            raise AsrModelError("Не удалось загрузить ASR-модель.") from exc

    def _load_once(self, model_path: Path, device_name: str, quantization: bool) -> None:
        torch, device, dtype, quantization_config, quantized = resolve_device_and_quantization(
            device_name,
            quantization,
        )
        processor_class, model_class, pipeline_factory = _import_transformers_components()
        processor = processor_class.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if getattr(device, "type", "") == "cuda":
            model_kwargs["device_map"] = "auto"
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        model = model_class.from_pretrained(str(model_path), **model_kwargs)
        if getattr(device, "type", "") == "cpu":
            model.to(device)
        model.eval()
        asr_pipeline = pipeline_factory(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=dtype,
            device=None if getattr(device, "type", "") == "cuda" else device,
        )
        self._processor = processor
        self._model = model
        self._pipeline = asr_pipeline
        self._device_label = str(getattr(device, "type", device))
        self._quantized = quantized
        self._compute_type = _compute_type(self._device_label, quantized)
        logger.info(
            "Локальная ASR-модель загружена: device=%s, quantized=%s.",
            self._device_label,
            self._quantized,
        )
        del torch


def _raise_if_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and cancel_check():
        raise ProcessingCancelled("Распознавание Transformers отменено пользователем.")


def _model_load_key(
    model_path: Path,
    settings: ProcessingSettings,
) -> tuple[str, str, str, bool, bool]:
    """Формирует единый ключ загрузки модели и фактического runtime."""
    return (
        str(model_path).casefold(),
        settings.backend.casefold(),
        settings.device.casefold(),
        settings.quantization_enabled,
        settings.allow_cpu_fallback,
    )


def _compute_type(device: str, quantized: bool) -> str:
    if quantized:
        return "int8"
    return "float16" if device.casefold() == "cuda" else "float32"


def normalize_pipeline_output(
    output: Any,
    *,
    duration: float,
    language: str,
    model: str,
    device: str,
    engine_version: str = "unknown",
    compute_type: str | None = None,
    quantized: bool,
) -> Transcript:
    """Преобразует ответ Transformers pipeline в общий контракт."""
    if isinstance(output, list):
        if len(output) != 1 or not isinstance(output[0], Mapping):
            raise AsrModelError("ASR pipeline вернул неожиданный пакетный ответ.")
        output = output[0]
    if not isinstance(output, Mapping):
        raise AsrModelError("ASR pipeline вернул ответ неизвестного формата.")
    text = str(output.get("text", "")).strip()
    if not text:
        raise AsrModelError("ASR-модель не вернула текст.")
    words = _extract_words(output.get("chunks"), duration)
    if words:
        segment = TranscriptSegment(
            start=words[0].start,
            end=words[-1].end,
            text=text,
            words=tuple(words),
            segment_id=0,
        )
        segments = (segment,)
    else:
        segments = (
            TranscriptSegment(
                start=0.0,
                end=max(0.001, duration),
                text=text,
                segment_id=0,
            ),
        )
    detected_language = _extract_language(output, language)
    return Transcript(
        text=text,
        language=detected_language,
        duration=duration,
        segments=segments,
        model=model,
        device=device,
        quantized=quantized,
        metadata={
            "runtime": "transformers",
            "engine_version": engine_version,
            "compute_type": compute_type or _compute_type(device, quantized),
            "word_timestamps": bool(words),
        },
    )


def _extract_words(raw_chunks: Any, duration: float) -> list[TranscriptWord]:
    if not isinstance(raw_chunks, list):
        return []
    words: list[TranscriptWord] = []
    previous_end = 0.0
    for raw in raw_chunks:
        if not isinstance(raw, Mapping):
            continue
        timestamp = raw.get("timestamp")
        if not isinstance(timestamp, (tuple, list)) or len(timestamp) != 2:
            continue
        start = _timestamp_value(timestamp[0], previous_end)
        end = _timestamp_value(timestamp[1], max(start, duration))
        start = max(previous_end, min(start, duration))
        if start >= duration:
            continue
        end = min(duration, max(start + 0.001, min(end, duration)))
        word_text = str(raw.get("text", ""))
        if not word_text.strip():
            continue
        words.append(TranscriptWord(start=start, end=end, text=word_text))
        previous_end = end
    return words


def _timestamp_value(value: Any, fallback: float) -> float:
    try:
        return float(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


def _extract_language(output: Mapping[str, Any], fallback: str) -> str:
    value = output.get("language")
    if not value and isinstance(output.get("chunks"), list):
        for chunk in output["chunks"]:
            if isinstance(chunk, Mapping) and chunk.get("language"):
                value = chunk["language"]
                break
    return _normalize_language_label(value, fallback) or fallback


def _normalize_language_label(value: Any, fallback: str | None) -> str | None:
    normalized = str(value or fallback or "").strip().casefold()
    aliases = {
        "english": "en",
        "<|en|>": "en",
        "russian": "ru",
        "<|ru|>": "ru",
    }
    if normalized.startswith("<|") and normalized.endswith("|>"):
        normalized = normalized[2:-2]
    return aliases.get(normalized, normalized or fallback)


def _import_transformers_components() -> tuple[Any, Any, Any]:
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError as exc:
        raise AsrModelError("Для локального Whisper требуется transformers.") from exc
    return AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline


def get_default_backend() -> TransformersWhisperBackend:
    """Совместимый alias общего Transformers backend из registry."""
    from speech_to_sub.asr.registry import get_backend

    return cast(TransformersWhisperBackend, get_backend("transformers"))


def unload_default_backend() -> None:
    """Выгружает общий Transformers backend, если он уже создан."""
    from speech_to_sub.asr.registry import unload_backend

    unload_backend("transformers")
