# Тестирование и качество

Критерий точности распознавания — [SUB-007](../specs/subtitle-layout/spec.md#requirement-sub-007-граница-доказанной-точности);
границы обычных и тяжёлых проверок — в
[ENG-003](../specs/engineering-constraints/spec.md#requirement-eng-003-границы-проверок),
команды и параметры включения — в [процедуре проверок](../../README.md#проверки).

| Область проверки | Требования и критерии | Реализация и тесты |
|---|---|---|
| Входы, потоки и подготовка | [MEDIA-001…006](../specs/media-preparation/spec.md) | [Карта примеров](examples/README.md) |
| Модели, устройства, сеть и облако | [ASR-001…011](../specs/speech-recognition/spec.md) | [Карта примеров](examples/README.md) |
| Содержание, временные якоря и синтаксис форматов | [SUB-001…007](../specs/subtitle-layout/spec.md) | [Карта примеров](examples/README.md) |
| Кеш, force и сохранность артефактов | [ART-001…009](../specs/recognition-artifacts/spec.md) | [Проверяемый пример](examples/force-and-formats.md) |
| Веб-процесс, восстановление и браузер | [WEB-001…011](../specs/local-web-workflow/spec.md) | [Карта примеров](examples/README.md) |
| Публичные параметры, маршруты и состояния | [IFACE-001…004](../specs/application-interface/spec.md) | [Карта примеров](examples/README.md) |
| Среды, установка и техническая приёмка | [ENG-001…004](../specs/engineering-constraints/spec.md) | [Команды](../../README.md#подготовка-сред) |

Сценарии требований задают различающие успешные, граничные и отказные случаи; таблица не создаёт отдельный список
критериев. Источник границы браузерного покрытия — [Sonar](../specs/engineering-constraints/spec.md#requirement-eng-004-анализ-sonar-и-граница-покрытия).

Ручной эталон и измерение ошибки текста/времени находятся в
[открытом изменении качества](../changes/establish-labeled-asr-quality-baseline/tasks.md).
