from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, TypedDict

from speech_to_sub.constants import (
    DEFAULT_ASR_BACKEND,
    DEFAULT_BACKEND_MODEL_PATHS,
    DEFAULT_ALIGNER,
    DEFAULT_AUDIO_LANGUAGE,
    DEFAULT_CHUNK_LENGTH_SECONDS,
    DEFAULT_LANGUAGE,
    DEFAULT_LONG_FORM_OVERLAP_SECONDS,
    DEFAULT_LONG_FORM_WINDOW_SECONDS,
    DEFAULT_MAX_CHARS_PER_LINE,
    DEFAULT_MODEL_PATH,
    DEFAULT_QWEN_ALIGNER_MODEL_PATH,
    DEFAULT_STRIDE_LENGTH_SECONDS,
    DEFAULT_VAD_MIN_SILENCE_MS,
    DEFAULT_BEAM_SIZE,
)
from speech_to_sub.exceptions import ValidationError


class RuntimeSignature(TypedDict):
    """Стабильная часть фактического ASR runtime для sidecar-кеша."""

    backend: str
    engine_version: str
    device: Literal["cpu", "cuda"]
    compute_type: str
    quantized: bool


@dataclass(frozen=True)
class AudioStreamInfo:
    """Метаданные одного аудиопотока контейнера."""

    ordinal: int
    index: int
    codec_name: str = "unknown"
    sample_rate: int | None = None
    channels: int | None = None
    language: str | None = None
    title: str | None = None
    duration: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MediaProbe:
    """Результат проверки медиафайла через ffprobe."""

    path: Path
    duration: float
    streams: tuple[AudioStreamInfo, ...]
    format_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "duration": self.duration,
            "format_name": self.format_name,
            "streams": [stream.to_dict() for stream in self.streams],
        }


@dataclass(frozen=True)
class TranscriptWord:
    """Слово с временными границами."""

    start: float
    end: float
    text: str
    probability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TranscriptSegment:
    """Сегмент распознавания с необязательными словными метками."""

    start: float
    end: float
    text: str
    words: tuple[TranscriptWord, ...] = ()
    segment_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [word.to_dict() for word in self.words],
        }


@dataclass(frozen=True)
class Transcript:
    """Нормализованный результат ASR backend."""

    text: str
    language: str
    duration: float
    segments: tuple[TranscriptSegment, ...]
    model: str
    device: str
    quantized: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "duration": self.duration,
            "model": self.model,
            "device": self.device,
            "quantized": self.quantized,
            "segments": [segment.to_dict() for segment in self.segments],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProcessingSettings:
    """Общие настройки CLI и веб-пакета."""

    backend: str = DEFAULT_ASR_BACKEND
    model_path: Path = field(default_factory=lambda: DEFAULT_MODEL_PATH)
    aligner: str = DEFAULT_ALIGNER
    aligner_model_path: Path | None = None
    worker_python_path: Path | None = None
    aligner_worker_python_path: Path | None = None
    language: str = DEFAULT_LANGUAGE
    audio_language: str = DEFAULT_AUDIO_LANGUAGE
    audio_stream_index: int | None = None
    device: str = "auto"
    quantization_enabled: bool = True
    allow_cpu_fallback: bool = False
    keep_audio: bool = False
    force: bool = False
    recursive: bool = False
    output_dir: Path | None = None
    verbose: bool = False
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE
    chunk_length_seconds: int = DEFAULT_CHUNK_LENGTH_SECONDS
    stride_length_seconds: int = DEFAULT_STRIDE_LENGTH_SECONDS
    long_form_window_seconds: int = DEFAULT_LONG_FORM_WINDOW_SECONDS
    long_form_overlap_seconds: int = DEFAULT_LONG_FORM_OVERLAP_SECONDS
    vad_filter: bool = True
    vad_min_silence_ms: int = DEFAULT_VAD_MIN_SILENCE_MS
    beam_size: int = DEFAULT_BEAM_SIZE
    condition_on_previous_text: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> ProcessingSettings:
        """Создаёт настройки из JSON-совместимого отображения."""
        if not values:
            return cls()
        data = dict(values)
        _normalize_choice_field(data, "backend")
        _normalize_model_path(data)
        _normalize_choice_field(data, "aligner")
        _normalize_aligner_model_path(data)
        for key in ("worker_python_path", "aligner_worker_python_path", "output_dir"):
            _normalize_optional_path(data, key)
        known = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in data.items() if key in known})

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model_path"] = str(self.model_path)
        data["aligner_model_path"] = (
            str(self.aligner_model_path) if self.aligner_model_path else None
        )
        data["worker_python_path"] = (
            str(self.worker_python_path) if self.worker_python_path else None
        )
        data["aligner_worker_python_path"] = (
            str(self.aligner_worker_python_path) if self.aligner_worker_python_path else None
        )
        data["output_dir"] = str(self.output_dir) if self.output_dir else None
        return data


def _normalize_choice_field(data: dict[str, Any], key: str) -> None:
    value = data.get(key)
    if value:
        data[key] = str(value).strip().casefold()
    else:
        data.pop(key, None)


def _normalize_model_path(data: dict[str, Any]) -> None:
    if data.get("model_path"):
        data["model_path"] = Path(str(data["model_path"]))
        return
    if "backend" not in data:
        data.pop("model_path", None)
        return
    try:
        data["model_path"] = DEFAULT_BACKEND_MODEL_PATHS[data["backend"]]
    except KeyError as exc:
        variants = ", ".join(DEFAULT_BACKEND_MODEL_PATHS)
        raise ValidationError(
            f"Неизвестный ASR backend '{data['backend']}'. Поддерживаются: {variants}."
        ) from exc


def _normalize_aligner_model_path(data: dict[str, Any]) -> None:
    if data.get("aligner_model_path"):
        data["aligner_model_path"] = Path(str(data["aligner_model_path"]))
    elif data.get("aligner") == "qwen3-forced-aligner":
        data["aligner_model_path"] = DEFAULT_QWEN_ALIGNER_MODEL_PATH
    else:
        data["aligner_model_path"] = None


def _normalize_optional_path(data: dict[str, Any], key: str) -> None:
    value = data.get(key)
    data[key] = Path(str(value)) if value else None


@dataclass(frozen=True)
class OutputPaths:
    """Пути артефактов одного входного файла."""

    srt_path: Path
    sidecar_path: Path
    normalized_audio_path: Path


@dataclass(frozen=True)
class FileResult:
    """Итог обработки одного файла."""

    input_path: Path
    state: str
    srt_path: Path | None = None
    sidecar_path: Path | None = None
    audio_path: Path | None = None
    error: str | None = None
    cached: bool = False
    skipped: bool = False
    probe: MediaProbe | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.input_path),
            "name": self.input_path.name,
            "state": self.state,
            "srt_output": str(self.srt_path) if self.srt_path else None,
            "sidecar_output": str(self.sidecar_path) if self.sidecar_path else None,
            "audio_output": str(self.audio_path) if self.audio_path else None,
            "error": self.error,
            "cached": self.cached,
            "skipped": self.skipped,
            "probe": self.probe.to_dict() if self.probe else None,
        }
