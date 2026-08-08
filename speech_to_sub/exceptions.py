from __future__ import annotations


class SpeechToSubError(RuntimeError):
    """Базовая ошибка конвейера."""


class ValidationError(SpeechToSubError):
    """Ошибка проверки входов, настроек или результатов."""


class MediaError(SpeechToSubError):
    """Ошибка ffprobe или FFmpeg."""


class AsrModelError(SpeechToSubError):
    """Ошибка загрузки или выполнения локальной ASR-модели."""


class CacheError(SpeechToSubError):
    """Ошибка чтения или записи sidecar/cache."""


class ProcessingCancelled(SpeechToSubError):
    """Кооперативная отмена пакетной обработки в безопасной точке."""
