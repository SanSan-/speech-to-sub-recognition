from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, TypedDict

from speech_to_sub.constants import (
    CLOUD_ASR_BACKENDS,
    DEFAULT_ASR_BACKEND,
    DEFAULT_ALIGNER,
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
    SubtitleFormat,
)
from speech_to_sub.exceptions import ValidationError


class RuntimeSignature(TypedDict):
    """Стабильная часть фактического ASR runtime для sidecar-кеша."""

    backend: str
    engine_version: str
    device: Literal["cpu", "cuda", "cloud"]
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

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, duration: float) -> TranscriptWord:
        """Безопасно восстанавливает слово из диагностического JSON."""
        start = _finite_float(value.get("start"), "начала слова", minimum=0.0)
        end = _finite_float(value.get("end"), "конца слова", minimum=0.0)
        if end <= start or end > duration + 0.25:
            raise ValidationError("Слово в sidecar содержит некорректный интервал.")
        text = _required_string(value.get("text"), "текст слова")
        probability_raw = value.get("probability")
        probability = (
            None
            if probability_raw is None
            else _finite_float(probability_raw, "вероятность слова")
        )
        return cls(start=start, end=end, text=text, probability=probability)


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

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        duration: float,
    ) -> TranscriptSegment:
        """Безопасно восстанавливает сегмент и его словные метки."""
        start = _finite_float(value.get("start"), "начала сегмента", minimum=0.0)
        end = _finite_float(value.get("end"), "конца сегмента", minimum=0.0)
        if end <= start or end > duration + 0.25:
            raise ValidationError("Сегмент в sidecar содержит некорректный интервал.")
        words_raw = value.get("words", [])
        if not isinstance(words_raw, list) or any(
            not isinstance(word, Mapping) for word in words_raw
        ):
            raise ValidationError("Список слов сегмента в sidecar имеет неверный формат.")
        segment_id = value.get("id")
        if segment_id is not None and (
            isinstance(segment_id, bool) or not isinstance(segment_id, int)
        ):
            raise ValidationError("Идентификатор сегмента в sidecar должен быть целым числом.")
        return cls(
            start=start,
            end=end,
            text=_required_string(value.get("text"), "текст сегмента"),
            words=tuple(
                TranscriptWord.from_mapping(word, duration=duration)
                for word in words_raw
            ),
            segment_id=segment_id,
        )


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

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Transcript:
        """Безопасно восстанавливает распознавание из sidecar без преобразования типов."""
        duration = _finite_float(value.get("duration"), "длительность распознавания")
        if duration <= 0:
            raise ValidationError("Длительность распознавания в sidecar должна быть положительной.")
        segments_raw = value.get("segments")
        if (
            not isinstance(segments_raw, list)
            or any(not isinstance(segment, Mapping) for segment in segments_raw)
        ):
            raise ValidationError("Список сегментов распознавания в sidecar имеет неверный формат.")
        quantized = value.get("quantized")
        if not isinstance(quantized, bool):
            raise ValidationError("Признак квантования в sidecar должен быть логическим.")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValidationError("Метаданные распознавания в sidecar имеют неверный формат.")
        return cls(
            text=_required_string(value.get("text"), "текст распознавания"),
            language=_required_string(value.get("language"), "язык распознавания"),
            duration=duration,
            segments=tuple(
                TranscriptSegment.from_mapping(segment, duration=duration)
                for segment in segments_raw
            ),
            model=_required_string(value.get("model"), "модель распознавания"),
            device=_required_string(value.get("device"), "устройство распознавания"),
            quantized=quantized,
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class ProcessingSettings:
    """Общие настройки CLI и веб-пакета."""

    backend: str = DEFAULT_ASR_BACKEND
    model_path: Path = field(default_factory=lambda: DEFAULT_MODEL_PATH)
    auto_download_model: bool = True
    allow_cloud_processing: bool = False
    openai_model: str = DEFAULT_OPENAI_MODEL
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
    output_format: SubtitleFormat = SubtitleFormat.SRT
    output_dir: Path | None = None
    verbose: bool = False
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE
    line_length_gap: int = DEFAULT_LINE_LENGTH_GAP
    max_cps: float = DEFAULT_MAX_CPS
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
        _normalize_choice_field(data, "openai_model")
        _normalize_choice_field(data, "aligner")
        if "output_format" in data:
            data["output_format"] = _normalize_output_format(data["output_format"])
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
        data["output_format"] = self.output_format.value
        return data


def _normalize_choice_field(data: dict[str, Any], key: str) -> None:
    value = data.get(key)
    if value:
        data[key] = str(value).strip().casefold()
    else:
        data.pop(key, None)


def _normalize_output_format(value: object) -> SubtitleFormat:
    if isinstance(value, SubtitleFormat):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("Формат субтитров должен быть строкой: srt, ass или vtt.")
    try:
        return SubtitleFormat(value.strip().casefold())
    except ValueError as exc:
        raise ValidationError(
            f"Неизвестный формат субтитров '{value}'. Поддерживаются: srt, ass, vtt."
        ) from exc


def _normalize_model_path(data: dict[str, Any]) -> None:
    if data.get("model_path"):
        data["model_path"] = Path(str(data["model_path"]))
        return
    if "backend" not in data:
        data.pop("model_path", None)
        return
    backend = data["backend"]
    if backend in CLOUD_ASR_BACKENDS:
        data["model_path"] = DEFAULT_MODEL_PATH
        return
    try:
        data["model_path"] = DEFAULT_BACKEND_MODEL_PATHS[backend]
    except KeyError as exc:
        variants = ", ".join((*DEFAULT_BACKEND_MODEL_PATHS, *CLOUD_ASR_BACKENDS))
        raise ValidationError(
            f"Неизвестный ASR backend '{backend}'. Поддерживаются: {variants}."
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


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"Поле «{label}» в sidecar должно быть непустой строкой.")
    return value


def _finite_float(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"Поле «{label}» в sidecar должно быть числом.")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValidationError(f"Поле «{label}» в sidecar содержит недопустимое число.")
    return number


@dataclass(frozen=True)
class OutputPaths:
    """Пути артефактов одного входного файла."""

    subtitle_path: Path
    sidecar_path: Path
    normalized_audio_path: Path
    output_format: SubtitleFormat = SubtitleFormat.SRT

    @property
    def srt_path(self) -> Path | None:
        """Возвращает устаревший SRT-путь только для формата SRT."""
        return self.subtitle_path if self.output_format is SubtitleFormat.SRT else None


@dataclass(frozen=True)
class FileResult:
    """Итог обработки одного файла."""

    input_path: Path
    state: str
    subtitle_path: Path | None = None
    output_format: SubtitleFormat = SubtitleFormat.SRT
    sidecar_path: Path | None = None
    audio_path: Path | None = None
    error: str | None = None
    cached: bool = False
    skipped: bool = False
    probe: MediaProbe | None = None

    @property
    def srt_path(self) -> Path | None:
        """Возвращает устаревший SRT-путь только для формата SRT."""
        if self.output_format is not SubtitleFormat.SRT:
            return None
        return self.subtitle_path

    def to_dict(self) -> dict[str, Any]:
        subtitle_output = str(self.subtitle_path) if self.subtitle_path else None
        return {
            "path": str(self.input_path),
            "name": self.input_path.name,
            "state": self.state,
            "output_format": self.output_format.value,
            "subtitle_output": subtitle_output,
            "srt_output": subtitle_output
            if self.output_format is SubtitleFormat.SRT
            else None,
            "sidecar_output": str(self.sidecar_path) if self.sidecar_path else None,
            "audio_output": str(self.audio_path) if self.audio_path else None,
            "error": self.error,
            "cached": self.cached,
            "skipped": self.skipped,
            "probe": self.probe.to_dict() if self.probe else None,
        }
