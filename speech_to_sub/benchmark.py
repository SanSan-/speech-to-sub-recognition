"""Метрики и измерение ресурсов для воспроизводимых ASR benchmark-прогонов."""

from __future__ import annotations

import os
import re
import subprocess
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

import psutil

from speech_to_sub.constants import (
    DEFAULT_LINE_LENGTH_GAP,
    DEFAULT_MAX_CHARS_PER_LINE,
    DEFAULT_MAX_CPS,
)
from speech_to_sub.subtitles.validator import validate_srt

_WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", flags=re.UNICODE)


class ResourceSampler(AbstractContextManager["ResourceSampler"]):
    """Сэмплирует peak RSS дерева процесса и общий VRAM без загрузки ML runtime."""

    def __init__(self, interval_seconds: float = 0.25) -> None:
        if interval_seconds <= 0:
            raise ValueError("Интервал измерения ресурсов должен быть положительным.")
        self._interval_seconds = interval_seconds
        self._process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_rss_bytes = self._tree_rss_bytes()
        self.peak_rss_bytes = self.start_rss_bytes
        self.start_vram_mib = _gpu_used_vram_mib()
        self.peak_vram_mib: int | None = self.start_vram_mib
        self.vram_scope: str | None = "total_gpu" if self.start_vram_mib is not None else None

    def __enter__(self) -> ResourceSampler:
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="asr-resource-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_seconds * 4))
        self._sample_once(include_vram=True)

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "start_rss_bytes": self.start_rss_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_rss_delta_bytes": max(0, self.peak_rss_bytes - self.start_rss_bytes),
            "rss_scope": "process_tree",
            "start_vram_mib": self.start_vram_mib,
            "peak_vram_mib": self.peak_vram_mib,
            "peak_vram_delta_mib": (
                max(0, self.peak_vram_mib - self.start_vram_mib)
                if self.peak_vram_mib is not None and self.start_vram_mib is not None
                else None
            ),
            "vram_scope": self.vram_scope,
        }

    def _sample_loop(self) -> None:
        iteration = 0
        while not self._stop.wait(self._interval_seconds):
            iteration += 1
            self._sample_once(include_vram=iteration % 4 == 0)

    def _sample_once(self, *, include_vram: bool) -> None:
        try:
            self.peak_rss_bytes = max(
                self.peak_rss_bytes,
                self._tree_rss_bytes(),
            )
        except (psutil.Error, OSError):
            pass
        if include_vram:
            current_vram = _gpu_used_vram_mib()
            if current_vram is not None:
                self.peak_vram_mib = max(self.peak_vram_mib or 0, current_vram)
                self.vram_scope = "total_gpu"

    def _tree_rss_bytes(self) -> int:
        """Суммирует RSS родителя и живых worker-процессов без двойного счёта."""
        processes = [self._process]
        try:
            processes.extend(self._process.children(recursive=True))
        except (psutil.Error, OSError):
            pass
        total = 0
        seen: set[int] = set()
        for process in processes:
            if process.pid in seen:
                continue
            seen.add(process.pid)
            try:
                total += process.memory_info().rss
            except (psutil.Error, OSError):
                continue
        return total


def analyze_srt(
    content: str,
    *,
    audio_duration: float,
    sidecar: Mapping[str, Any] | None = None,
    max_reading_speed: float = DEFAULT_MAX_CPS,
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE,
    line_length_gap: int = DEFAULT_LINE_LENGTH_GAP,
) -> dict[str, int | float]:
    """Проверяет SRT и возвращает структурные long-form метрики."""
    cues = validate_srt(content, audio_duration=audio_duration)
    cue_durations = [cue.end - cue.start for cue in cues]
    overlap_count = sum(
        current.start < previous.end
        for previous, current in zip(cues, cues[1:])
    )
    visible_chars = [
        len(" ".join(line.strip() for line in cue.text.splitlines()))
        for cue in cues
    ]
    words = _tokens(" ".join(cue.text for cue in cues))
    repeated_5grams = _repeated_ngram_occurrences(words, 5)
    boundary_repeats = _boundary_repeated_ngrams(sidecar or {}, ngram_size=3)
    transcript = (sidecar or {}).get("transcript")
    word_spans = _sidecar_word_spans(transcript) if isinstance(transcript, Mapping) else []
    return {
        "cue_count": len(cues),
        "overlap_count": overlap_count,
        "reading_speed_violations": sum(
            chars / duration > max_reading_speed
            for chars, duration in zip(visible_chars, cue_durations)
        ),
        "short_cue_count": sum(duration < 0.8 for duration in cue_durations),
        "long_cue_count": sum(duration > 7.0 for duration in cue_durations),
        "long_line_count": sum(
            len(line) > max_chars_per_line + line_length_gap
            for cue in cues
            for line in cue.text.splitlines()
        ),
        "repeated_5gram_occurrences": repeated_5grams,
        "boundary_repeated_3gram_occurrences": boundary_repeats,
        "word_timestamp_count": len(word_spans),
        "non_monotonic_word_timestamp_count": sum(
            current[0] < previous[1]
            for previous, current in zip(word_spans, word_spans[1:])
        ),
        "word_timestamps_after_300_seconds": sum(
            start >= 300.0 for start, _end, _text in word_spans
        ),
        "last_word_end_seconds": round(word_spans[-1][1], 3) if word_spans else 0.0,
        "speech_coverage_ratio": round(
            sum(cue_durations) / audio_duration if audio_duration > 0 else 0.0,
            6,
        ),
        "last_cue_end_seconds": cues[-1].end,
        "tail_gap_seconds": round(max(0.0, audio_duration - cues[-1].end), 3),
    }


def _boundary_repeated_ngrams(
    sidecar: Mapping[str, Any],
    *,
    ngram_size: int,
) -> int:
    transcript = sidecar.get("transcript")
    if not isinstance(transcript, Mapping):
        return 0
    metadata = transcript.get("metadata")
    if not isinstance(metadata, Mapping):
        return 0
    try:
        window_seconds = float(metadata["window_seconds"])
        overlap_seconds = float(metadata["overlap_seconds"])
        duration = float(transcript["duration"])
    except (KeyError, TypeError, ValueError):
        return 0
    step = window_seconds - overlap_seconds
    if step <= 0 or duration <= step:
        return 0
    timed_words = _sidecar_words(transcript)
    radius = max(2.0, overlap_seconds * 2.0)
    repeats = 0
    boundary = step
    while boundary < duration:
        local_tokens = [
            token
            for midpoint, token in timed_words
            if boundary - radius <= midpoint <= boundary + radius
        ]
        repeats += _repeated_ngram_occurrences(local_tokens, ngram_size)
        boundary += step
    return repeats


def _sidecar_words(transcript: Mapping[str, Any]) -> list[tuple[float, str]]:
    result: list[tuple[float, str]] = []
    for start, end, text in _sidecar_word_spans(transcript):
        midpoint = (start + end) / 2.0
        result.extend((midpoint, token) for token in _tokens(text))
    return sorted(result)


def _sidecar_word_spans(
    transcript: Mapping[str, Any],
) -> list[tuple[float, float, str]]:
    result: list[tuple[float, float, str]] = []
    for segment in _mapping_sequence(transcript.get("segments")):
        for word in _mapping_sequence(segment.get("words")):
            span = _parse_word_span(word)
            if span is not None:
                result.append(span)
    return result


def _mapping_sequence(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _parse_word_span(word: Mapping[str, Any]) -> tuple[float, float, str] | None:
    try:
        start = float(word["start"])
        end = float(word["end"])
    except (KeyError, TypeError, ValueError):
        return None
    text = str(word.get("text", "")).strip()
    return (start, end, text) if text and end > start else None


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_RE.finditer(text)]


def _repeated_ngram_occurrences(tokens: Sequence[str], size: int) -> int:
    if size < 1 or len(tokens) < size:
        return 0
    counts = Counter(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))
    return sum(count - 1 for count in counts.values() if count > 1)


def _gpu_used_vram_mib() -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    values: list[int] = []
    for line in completed.stdout.splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            continue
    return sum(values) if values else None


__all__ = ["ResourceSampler", "analyze_srt"]
