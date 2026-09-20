# Архитектура

CLI и локальный веб используют один прикладной сервис. Общий Transcript отделяет распознавание и выравнивание
от раскладки и формирователей субтитров. Изолированные среды ограничивают конфликт зависимостей тяжёлых движков.
Кеш и публикация принадлежат общему сервису, поэтому интерфейсы используют одинаковые правила.

```mermaid
flowchart LR
    UI[CLI / локальный веб] --> Service[Общий сервис]
    Service --> Media[Подготовка аудио]
    Media --> ASR[Распознавание]
    ASR --> Align[Выравнивание]
    Align --> Transcript[Transcript]
    Service <--> Cache[Кеш результатов]
    Cache --> Transcript
    Transcript --> Layout[Раскладка]
    Layout --> Format[SRT / ASS / WebVTT]
    Format --> Publish[Публикация артефактов]
```

Схема показывает границы компонентов. Условия обхода кеша, пропуска этапов и возврата готового результата заданы в ART.

## Владельцы контрактов

| Область | Единственный источник полных требований |
|---|---|
| Входы, аудиопоток, обход и подготовка | [media-preparation](../specs/media-preparation/spec.md) |
| Движки, модели, устройство, выравнивание и облако | [speech-recognition](../specs/speech-recognition/spec.md) |
| Содержание, временная раскладка и форматная структура | [subtitle-layout](../specs/subtitle-layout/spec.md) |
| Кеш, force, совместимость и публикация результатов | [recognition-artifacts](../specs/recognition-artifacts/spec.md) |
| Локальное API, подготовка, события и история задач | [local-web-workflow](../specs/local-web-workflow/spec.md) |
| Инженерный профиль и безопасность работы агента | [engineering-governance](../specs/engineering-governance/spec.md) |
| Маршрут контекста, очередь задач и восстановление работы | [project-context](../specs/project-context/spec.md) |

Параметры, маршруты, поля и публичные состояния — в
[приложении интерфейсов](../specs/engineering-governance/reference/application-interface.md);
среды и команды — в [проектном приложении](../specs/engineering-governance/reference/project-specific.md).
Связи с кодом и тестами поддерживаются в [примерах](examples/README.md).
[ADR](ADR/README.md) сохраняют причины решений и условия пересмотра.
Граница этих документов закреплена в [ADR-007](ADR/ADR-007-specification-boundaries.md).
