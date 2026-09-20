# ADR-003: Два уровня кеша и полный force

## Status

Accepted.

## Context

Пользователь должен получать другой формат без ненужного ASR, но явный force должен гарантировать полный пересчёт.

## Decision

Формат и раскладка отделены от тяжёлого отпечатка; отсутствующий формат при force=false использует Transcript,
force=true повторяет весь тяжёлый конвейер. Полные условия задают
[ART-002](../../specs/recognition-artifacts/spec.md#requirement-art-002-два-уровня-идентичности),
[ART-003](../../specs/recognition-artifacts/spec.md#requirement-art-003-повтор-без-force) и
[ART-004](../../specs/recognition-artifacts/spec.md#requirement-art-004-полный-обход-при-force).

## Consequences

Существующий неподтверждённый результат без force пропускается; смена ASS не повреждает SRT.

## Alternatives considered

Один отпечаток вынуждал бы повторять ASR при каждом формате; лёгкий force противоречил бы явному режиму нового
распознавания.

## Evidence

[Рабочий источник](../../../speech_to_sub/utils/cache.py), [связанный тест или свидетельство](../../../tests/test_service.py).
Наличие теста не означает успешный прогон.

## Review date

Изменение значения force либо схемы sidecar.
