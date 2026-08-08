"""Оконное декодирование аудио с ограниченным потреблением оперативной памяти."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from speech_to_sub.exceptions import MediaError, ValidationError


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """Один mono float32 фрагмент аудио с абсолютным смещением."""

    offset: float
    duration: float
    samples: np.ndarray
    is_final: bool = False


def iter_audio_windows(
    path: str | Path,
    *,
    window_seconds: float,
    overlap_seconds: float,
    sample_rate: int = 16_000,
    opener: Callable[..., Any] | None = None,
) -> Iterator[AudioWindow]:
    """Декодирует файл через PyAV и выдаёт ограниченные окна вместо полного массива."""
    _validate_window_settings(window_seconds, overlap_seconds, sample_rate)
    av = _import_av()
    open_media = opener or av.open
    source = Path(path).expanduser().resolve()
    try:
        with open_media(str(source), mode="r") as container:
            stream = next(
                (candidate for candidate in container.streams if candidate.type == "audio"),
                None,
            )
            if stream is None:
                raise MediaError(f"В подготовленном аудио нет дорожки: {source}")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
            pcm_chunks = _iter_resampled_pcm(container, stream, resampler)
            yield from iter_pcm_s16_windows(
                pcm_chunks,
                window_seconds=window_seconds,
                overlap_seconds=overlap_seconds,
                sample_rate=sample_rate,
            )
    except (MediaError, ValidationError):
        raise
    except Exception as exc:
        raise MediaError(f"Не удалось оконно декодировать аудио {source}: {exc}") from exc


def iter_pcm_s16_windows(
    chunks: Iterable[bytes],
    *,
    window_seconds: float,
    overlap_seconds: float,
    sample_rate: int = 16_000,
) -> Iterator[AudioWindow]:
    """Разбивает поток little-endian PCM S16LE на перекрывающиеся float32-окна."""
    _validate_window_settings(window_seconds, overlap_seconds, sample_rate)
    window_samples = max(1, round(window_seconds * sample_rate))
    overlap_samples = round(overlap_seconds * sample_rate)
    step_samples = window_samples - overlap_samples
    window_bytes = window_samples * 2
    overlap_bytes = overlap_samples * 2
    step_bytes = step_samples * 2
    buffer = bytearray()
    offset_samples = 0
    full_windows = 0
    pending: AudioWindow | None = None

    for chunk in chunks:
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise ValidationError("PCM chunk должен быть bytes-подобным объектом.")
        buffer.extend(chunk)
        while len(buffer) >= window_bytes:
            current = _window_from_bytes(
                bytes(buffer[:window_bytes]),
                offset_samples,
                sample_rate,
            )
            if pending is not None:
                yield pending
            pending = current
            del buffer[:step_bytes]
            offset_samples += step_samples
            full_windows += 1

    if len(buffer) % 2:
        raise ValidationError("PCM S16LE завершился неполным сэмплом.")
    has_new_tail = full_windows == 0 or len(buffer) > overlap_bytes
    if buffer and has_new_tail:
        current = _window_from_bytes(bytes(buffer), offset_samples, sample_rate)
        if pending is not None:
            yield pending
        pending = current
    if pending is not None:
        yield replace(pending, is_final=True)


def _iter_resampled_pcm(container: Any, stream: Any, resampler: Any) -> Iterator[bytes]:
    for frame in container.decode(stream):
        for converted in _as_frames(resampler.resample(frame)):
            yield _frame_to_s16_bytes(converted)
    for converted in _as_frames(resampler.resample(None)):
        yield _frame_to_s16_bytes(converted)


def _as_frames(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _frame_to_s16_bytes(frame: Any) -> bytes:
    array = np.asarray(frame.to_ndarray())
    if array.dtype != np.int16:
        array = array.astype(np.int16, copy=False)
    return np.ascontiguousarray(array.reshape(-1)).astype("<i2", copy=False).tobytes()


def _window_from_bytes(
    payload: bytes,
    offset_samples: int,
    sample_rate: int,
) -> AudioWindow:
    pcm = np.frombuffer(payload, dtype="<i2")
    samples = pcm.astype(np.float32) / 32_768.0
    return AudioWindow(
        offset=offset_samples / sample_rate,
        duration=len(pcm) / sample_rate,
        samples=samples,
    )


def _validate_window_settings(
    window_seconds: float,
    overlap_seconds: float,
    sample_rate: int,
) -> None:
    if sample_rate < 1:
        raise ValidationError("Частота дискретизации должна быть положительной.")
    if window_seconds <= 0:
        raise ValidationError("Длительность аудиоокна должна быть положительной.")
    if overlap_seconds < 0 or overlap_seconds >= window_seconds:
        raise ValidationError("Перекрытие должно быть неотрицательным и короче аудиоокна.")


def _import_av() -> Any:
    try:
        import av
    except ImportError as exc:
        raise MediaError("Для оконного декодирования требуется PyAV.") from exc
    return av


__all__ = ["AudioWindow", "iter_audio_windows", "iter_pcm_s16_windows"]
