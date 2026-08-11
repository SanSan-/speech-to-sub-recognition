from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from speech_to_sub.alignment.registry import aligner_names
from speech_to_sub.asr.registry import backend_names
from speech_to_sub.constants import (
    DEFAULT_ALIGNER,
    DEFAULT_ASR_BACKEND,
    DEFAULT_AUDIO_LANGUAGE,
    DEFAULT_BACKEND_MODEL_PATHS,
    DEFAULT_BEAM_SIZE,
    DEFAULT_CHUNK_LENGTH_SECONDS,
    DEFAULT_LANGUAGE,
    DEFAULT_LINE_LENGTH_GAP,
    DEFAULT_LONG_FORM_OVERLAP_SECONDS,
    DEFAULT_LONG_FORM_WINDOW_SECONDS,
    DEFAULT_MAX_CHARS_PER_LINE,
    DEFAULT_MAX_CPS,
    DEFAULT_MODEL_PATH,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_QWEN_ALIGNER_MODEL_PATH,
    DEFAULT_STRIDE_LENGTH_SECONDS,
    DEFAULT_VAD_MIN_SILENCE_MS,
    MAX_BATCH_PATHS,
    MAX_LINE_LENGTH_GAP,
)


class ApiModel(BaseModel):
    """Общая строгая схема локального HTTP API."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        protected_namespaces=(),
    )


class ProcessingSettingsPayload(ApiModel):
    """Настройки пакетного распознавания, принимаемые веб-интерфейсом."""

    backend: str = DEFAULT_ASR_BACKEND
    model_path: str = Field(default=str(DEFAULT_MODEL_PATH), min_length=1, max_length=4096)
    aligner: str = DEFAULT_ALIGNER
    aligner_model_path: str | None = Field(default=None, max_length=4096)
    worker_python_path: str | None = Field(default=None, max_length=4096)
    aligner_worker_python_path: str | None = Field(default=None, max_length=4096)
    language: Literal["en", "ru", "auto"] = cast(
        Literal["en", "ru", "auto"],
        DEFAULT_LANGUAGE,
    )
    audio_language: str = Field(default=DEFAULT_AUDIO_LANGUAGE, min_length=1, max_length=32)
    audio_stream_index: int | None = Field(default=None, ge=0)
    device: Literal["auto", "cuda", "cpu"] = "auto"
    quantization_enabled: bool = True
    auto_download_model: bool = True
    allow_cpu_fallback: bool = False
    allow_cloud_processing: bool = Field(default=False, strict=True)
    openai_model: Literal["whisper-1"] = DEFAULT_OPENAI_MODEL
    output_format: Literal["srt", "ass", "vtt"] = "srt"
    keep_audio: bool = False
    force: bool = Field(default=False, strict=True)
    recursive: bool = False
    output_dir: str | None = Field(default=None, max_length=4096)
    verbose: bool = False
    max_chars_per_line: int = Field(default=DEFAULT_MAX_CHARS_PER_LINE, ge=20, le=80)
    line_length_gap: int = Field(
        default=DEFAULT_LINE_LENGTH_GAP,
        ge=0,
        le=MAX_LINE_LENGTH_GAP,
        strict=True,
    )
    max_cps: float = Field(default=DEFAULT_MAX_CPS, ge=5, le=60)
    chunk_length_seconds: int = Field(default=DEFAULT_CHUNK_LENGTH_SECONDS, ge=10, le=3600)
    stride_length_seconds: int = Field(default=DEFAULT_STRIDE_LENGTH_SECONDS, ge=0, le=600)
    long_form_window_seconds: int = Field(
        default=DEFAULT_LONG_FORM_WINDOW_SECONDS,
        ge=30,
        le=3600,
    )
    long_form_overlap_seconds: int = Field(
        default=DEFAULT_LONG_FORM_OVERLAP_SECONDS,
        ge=0,
        le=600,
    )
    vad_filter: bool = True
    vad_min_silence_ms: int = Field(default=DEFAULT_VAD_MIN_SILENCE_MS, ge=100, le=10_000)
    beam_size: int = Field(default=DEFAULT_BEAM_SIZE, ge=1, le=20)
    condition_on_previous_text: bool = True

    @field_validator(
        "output_dir",
        "aligner_model_path",
        "worker_python_path",
        "aligner_worker_python_path",
        mode="before",
    )
    @classmethod
    def normalize_optional_path(cls, value: object) -> object:
        """Преобразует пустое поле каталога результата в отсутствие значения."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        """Проверяет backend по единому registry и нормализует идентификатор."""
        backend = value.casefold()
        if backend not in backend_names():
            variants = ", ".join(backend_names())
            raise ValueError(f"ASR backend должен иметь одно из значений: {variants}.")
        return backend

    @field_validator("aligner")
    @classmethod
    def validate_aligner(cls, value: str) -> str:
        """Проверяет aligner по единому registry."""
        aligner = value.casefold()
        if aligner not in aligner_names():
            variants = ", ".join(aligner_names())
            raise ValueError(f"Aligner должен иметь одно из значений: {variants}.")
        return aligner

    @model_validator(mode="after")
    def validate_audio_windows(self) -> ProcessingSettingsPayload:
        """Проверяет, что перекрытие короче основного окна ASR."""
        if (
            "model_path" not in self.model_fields_set
            and self.backend in DEFAULT_BACKEND_MODEL_PATHS
        ):
            self.model_path = str(DEFAULT_BACKEND_MODEL_PATHS[self.backend])
        if (
            self.aligner == "qwen3-forced-aligner"
            and "aligner_model_path" not in self.model_fields_set
        ):
            self.aligner_model_path = str(DEFAULT_QWEN_ALIGNER_MODEL_PATH)
        if self.stride_length_seconds * 2 >= self.chunk_length_seconds:
            raise ValueError("Перекрытие должно быть короче половины окна распознавания.")
        if self.long_form_overlap_seconds >= self.long_form_window_seconds:
            raise ValueError("Long-form overlap должен быть короче основного окна.")
        if self.aligner == "qwen3-forced-aligner":
            if not self.aligner_model_path:
                raise ValueError("Для Qwen3 ForcedAligner требуется путь к модели.")
        return self


class PickRequest(ApiModel):
    """Запрос системного выбора файлов или каталога."""

    kind: Literal["file", "folder"]
    settings: ProcessingSettingsPayload = Field(default_factory=ProcessingSettingsPayload)


class RefreshRequest(ApiModel):
    """Запрос повторного построения карточек выбранных файлов."""

    paths: list[str] = Field(default_factory=list, max_length=MAX_BATCH_PATHS)
    settings: ProcessingSettingsPayload = Field(default_factory=ProcessingSettingsPayload)

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, values: list[str]) -> list[str]:
        return _validate_paths(values)


class TranscribeRequest(ApiModel):
    """Запрос запуска одной фоновой пакетной задачи."""

    paths: list[str] = Field(min_length=1, max_length=MAX_BATCH_PATHS)
    settings: ProcessingSettingsPayload = Field(default_factory=ProcessingSettingsPayload)

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, values: list[str]) -> list[str]:
        return _validate_paths(values)


class RetryRequest(ApiModel):
    """Одноразовые разрешения повторной обработки."""

    allow_cloud_processing: bool = Field(default=False, strict=True)
    force: bool = Field(default=False, strict=True)


def _validate_paths(values: list[str]) -> list[str]:
    """Проверяет непустые пути и сохраняет их исходный порядок."""
    if any(not value.strip() for value in values):
        raise ValueError("Пути к медиафайлам не должны быть пустыми.")
    return values
