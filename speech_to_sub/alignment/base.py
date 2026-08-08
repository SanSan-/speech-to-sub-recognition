from __future__ import annotations

from pathlib import Path
from typing import Protocol

from speech_to_sub.asr.base import CancelCheck, ProgressCallback
from speech_to_sub.models import ProcessingSettings, RuntimeSignature, Transcript


class AlignmentAdapter(Protocol):
    """Контракт optional этапа выравнивания слов."""

    aligner_id: str
    requires_exclusive_runtime: bool

    def preflight(self, settings: ProcessingSettings) -> Path | None:
        """Проверяет локальную модель и совместимость без загрузки весов."""

    def expected_runtime_signature(
        self,
        settings: ProcessingSettings,
    ) -> RuntimeSignature | None:
        """Возвращает ожидаемый runtime aligner-а."""

    def runtime_signature(
        self,
        settings: ProcessingSettings,
        transcript: Transcript | None = None,
    ) -> RuntimeSignature | None:
        """Возвращает фактический runtime для совпадающего результата."""

    def align(
        self,
        audio_path: Path,
        transcript: Transcript,
        settings: ProcessingSettings,
        duration: float,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> Transcript:
        """Выравнивает текст по исходному нормализованному аудио."""

    def unload(self) -> None:
        """Освобождает собственные ресурсы, если они есть."""
