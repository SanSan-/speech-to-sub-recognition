# Тестирование и качество

Каждая проверка относится к своему контракту и охвату. Требования к доказательству —
[ENG-003/004/005](../specs/engineering-governance/spec.md); команды и условия включения тяжёлых проверок —
в [проектном приложении](../specs/engineering-governance/reference/project-specific.md#команды-и-ограничения-проверок).

| Область проверки | Требования и критерии | Реализация и тесты |
|---|---|---|
| Входы, потоки и подготовка | [MEDIA-001…006](../specs/media-preparation/spec.md) | [Карта примеров](examples/README.md) |
| Модели, устройства, сеть и облако | [ASR-001…010](../specs/speech-recognition/spec.md) | [Карта примеров](examples/README.md) |
| Содержание, временные якоря и синтаксис форматов | [SUB-001…007](../specs/subtitle-layout/spec.md) | [Карта примеров](examples/README.md) |
| Кеш, force и сохранность артефактов | [ART-001…009](../specs/recognition-artifacts/spec.md) | [Проверяемый пример](examples/force-and-formats.md) |
| API, состояния, восстановление и браузер | [WEB-001…011](../specs/local-web-workflow/spec.md) | [Карта примеров](examples/README.md) |

Сценарии требований задают различающие успешные, граничные и отказные случаи; таблица не создаёт отдельный список
критериев. Источник границы браузерного покрытия — [Sonar](../specs/engineering-governance/reference/project-specific.md#sonar).

Ручной эталон и измерение ошибки текста/времени находятся в
[открытом изменении качества](../changes/establish-labeled-asr-quality-baseline/tasks.md).
