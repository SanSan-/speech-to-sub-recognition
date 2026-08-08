from __future__ import annotations

from pathlib import Path

from speech_to_sub.alignment.base import AlignmentAdapter
from speech_to_sub.asr.base import CancelCheck, ProgressCallback
from speech_to_sub.alignment.qwen_forced import QwenForcedAlignerAdapter
from speech_to_sub.exceptions import ValidationError
from speech_to_sub.models import ProcessingSettings, RuntimeSignature, Transcript


class NoAlignmentAdapter:
    """Явный адаптер отключённого этапа выравнивания."""

    aligner_id = "none"
    requires_exclusive_runtime = False

    def preflight(self, settings: ProcessingSettings) -> None:
        return None

    def expected_runtime_signature(
        self,
        settings: ProcessingSettings,
    ) -> RuntimeSignature | None:
        return None

    def runtime_signature(
        self,
        settings: ProcessingSettings,
        transcript: Transcript | None = None,
    ) -> RuntimeSignature | None:
        return None

    def align(
        self,
        audio_path: Path,
        transcript: Transcript,
        settings: ProcessingSettings,
        duration: float,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> Transcript:
        del audio_path, settings, duration, progress_callback, cancel_check
        return transcript

    def unload(self) -> None:
        return None


_ALIGNERS: dict[str, AlignmentAdapter] = {
    "none": NoAlignmentAdapter(),
    "qwen3-forced-aligner": QwenForcedAlignerAdapter(),
}


def aligner_names() -> tuple[str, ...]:
    """Возвращает стабильный список optional aligner-ов."""
    return tuple(_ALIGNERS)


def get_aligner(name: str) -> AlignmentAdapter:
    """Возвращает зарегистрированный adapter без загрузки весов."""
    aligner_id = str(name or "none").strip().casefold()
    try:
        return _ALIGNERS[aligner_id]
    except KeyError as exc:
        variants = ", ".join(aligner_names())
        raise ValidationError(
            f"Неизвестный aligner '{name}'. Поддерживаются: {variants}."
        ) from exc


def unload_aligners() -> None:
    """Освобождает ресурсы всех созданных aligner-ов."""
    for aligner in _ALIGNERS.values():
        aligner.unload()


__all__ = ["aligner_names", "get_aligner", "unload_aligners"]
