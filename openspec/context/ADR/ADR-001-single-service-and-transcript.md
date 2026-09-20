# ADR-001: Единый сервис и общий Transcript

## Status

Accepted.

## Context

CLI, веб, разные ASR и три формата должны сохранять одинаковые правила обработки.

## Decision

Интерфейсы используют общий сервис; ASR и aligner возвращают единый Transcript, формирователи работают после
распознавания.

## Consequences

Расширение движка не создаёт отдельный путь публикации; формат не меняет распознанный текст.

## Alternatives considered

Отдельные CLI/web конвейеры и форматные ASR потребовали бы повторения кеша и правил файлов.

## Evidence

[Рабочий источник](D:/Projects/-py/speech-to-sub-recognition/speech_to_sub/service.py), [связанный тест или свидетельство](D:/Projects/-py/speech-to-sub-recognition/tests/test_service.py).
Наличие теста не означает успешный прогон.

## Review date

Изменение публичного контракта Transcript или добавление нового интерфейса.
