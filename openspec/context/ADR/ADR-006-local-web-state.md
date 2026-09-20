# ADR-006: Локальное API и сохранённые состояния

## Status

Accepted.

## Context

Долгая пакетная задача должна переживать перезагрузку страницы, а незавершённая работа не должна выглядеть успешной.

## Decision

Loopback/Host/Origin защищают API; SQLite сохраняет задачи и SSE, generation/operation_id отделяют подготовку; после
перезапуска незавершённая задача interrupted.

## Consequences

Можно восстановить карточки и журнал без повторного ASR; это не продолжение с середины файла.

## Alternatives considered

Состояние только в браузере или памяти сервера потеряло бы результат; удалённый сервис потребовал бы отдельной модели
доступа.

## Evidence

[Рабочий источник](D:/Projects/-py/speech-to-sub-recognition/speech_to_sub/web/job_store.py), [связанный тест или свидетельство](D:/Projects/-py/speech-to-sub-recognition/tests/test_web_app.py).
Наличие теста не означает успешный прогон.

## Review date

Появление принятого многопользовательского режима или продолжения ASR.
