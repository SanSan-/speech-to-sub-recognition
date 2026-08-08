from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

from speech_to_sub.models import ProcessingSettings, RuntimeSignature, Transcript

ProgressCallback = Callable[[int], None]
CancelCheck = Callable[[], bool]


class AsrBackend(Protocol):
    """Контракт локального движка распознавания."""

    backend_id: str

    def preflight(self, settings: ProcessingSettings) -> Path:
        """Проверяет runtime и локальную модель без загрузки весов."""

    def expected_runtime_signature(
        self,
        settings: ProcessingSettings,
    ) -> RuntimeSignature:
        """Определяет ожидаемый runtime без загрузки весов."""

    def runtime_signature(
        self,
        settings: ProcessingSettings,
    ) -> RuntimeSignature | None:
        """Возвращает runtime только для совпадающего полного load key."""

    def transcribe(
        self,
        audio_path: Path,
        settings: ProcessingSettings,
        duration: float,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> Transcript:
        """Распознаёт подготовленную аудиодорожку."""

    def unload(self) -> None:
        """Освобождает ресурсы модели."""
