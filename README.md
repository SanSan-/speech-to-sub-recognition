# speech-to-sub-recognition

Приложение для пакетного распознавания речи в аудио- и видеофайлах с созданием субтитров SRT, ASS или WebVTT.
Работает из командной строки и через локальный веб-интерфейс; поддерживает локальные модели и облачное распознавание.

## Структура проекта

```text
.
├── speech_to_sub/          # Общая обработка, настройки и командный интерфейс
│   ├── asr/                # Движки распознавания речи
│   ├── alignment/          # Выравнивание слов по времени
│   ├── media/              # Подготовка аудио через FFmpeg
│   ├── subtitles/          # Формирование и проверка субтитров
│   ├── workers/            # Изолированные процессы моделей
│   └── web/                # Локальный веб-интерфейс и очередь задач
├── resources/              # Модели, кеш, рабочие файлы и состояние задач
├── requirements-workers/   # Зависимости изолированных сред моделей
├── tests/                  # Модульные, интеграционные и браузерные проверки
├── tools/                  # Инструменты сравнительных измерений
├── docs/benchmarks/        # Методики и результаты измерений
├── openspec/               # Требования и документация проекта
│   ├── context/            # Начало работы, архитектура, словарь и решения
│   ├── specs/              # Предметные требования
│   └── changes/            # Открытые изменения и задачи
└── .github/workflows/      # Сборка выпусков в GitHub Actions
```

## Начало работы

Начните с [контекста проекта](openspec/context/00-start-here.md): он направляет к описанию продукта, архитектуре
и правилам работы.

| Задача                               | Документация                                                                                                                                                         |
|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Разобраться в устройстве приложения  | [Архитектура](openspec/context/03-architecture.md)                                                                                                                   |
| Установить и запустить приложение    | [Подготовка среды и команды](#подготовка-сред)                                     |
| Выбрать модель и настроить обработку | [Распознавание речи](openspec/specs/speech-recognition/spec.md) и [интерфейсы и параметры](openspec/specs/application-interface/spec.md) |
| Настроить формирование субтитров     | [Разметка и форматы субтитров](openspec/specs/subtitle-layout/spec.md)                                                                                               |
| Подготовить изменения                | [Инженерные правила](openspec/context/04-engineering-standards.md) и [проверки качества](openspec/context/06-testing-and-quality.md)                                 |
| Проверить архитектурное решение      | [Принятые решения](openspec/context/ADR/README.md) |
| Посмотреть запланированные работы    | [Открытые изменения](openspec/changes/README.md)                                                                                                                     |

Все команды выполняются из корня проекта. Среды, источники зависимостей и ограничения пакета определены в
[ENG-001…002](openspec/specs/engineering-constraints/spec.md); границы обычных и тяжёлых проверок — в
[ENG-003](openspec/specs/engineering-constraints/spec.md#requirement-eng-003-границы-проверок).

## Подготовка сред

Создайте основную среду и установите зависимости перед выполнением команд, использующих проектный Python:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Для браузерных проверок установите зависимости через `npm ci`.
Для Parakeet/Qwen используется [setup_workers.ps1](setup_workers.ps1) с -Parakeet/-Qwen.
FFmpeg/ffprobe доступны в PATH или по настроенным путям.

## Запуск приложения

CLI запускается проектным Python; укажите путь к медиафайлу:

```powershell
.\.venv\Scripts\python.exe -m speech_to_sub --input "<media-file>"
```

Веб-интерфейс запускается через [run_web.ps1](run_web.ps1) либо проектным Python
`-m speech_to_sub.web`. Параметры и маршруты определены в
[спецификации интерфейсов](openspec/specs/application-interface/spec.md).

## Проверки

После подготовки нужных сред доступны команды:

```powershell
.\.venv\Scripts\python.exe -B -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q speech_to_sub tests
node --check speech_to_sub\web\static\app.js
npm test
openspec validate --all --strict --no-interactive
```

Для тяжёлой интеграционной проверки задайте RUN_LOCAL_ASR_TEST=1, ASR_TEST_MEDIA=`<media-file>`,
ASR_TEST_LANGUAGE=en и запустите проектный Python `-B -m pytest -m integration`.
Создание среды и тяжёлые проверки не входят в обслуживание документации.

## Sonar

Подготовьте проектную среду, sonar-scanner, Node.js, доступ к серверу и параметры локального окружения по
[ENG-004](openspec/specs/engineering-constraints/spec.md#requirement-eng-004-анализ-sonar-и-граница-покрытия).

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .codex\skills\fix-sonar\scripts\sonar_cycle.ps1 -TimeoutSeconds 600 -RequireClean
```

Полная процедура — в [.codex/skills/fix-sonar/SKILL.md](.codex/skills/fix-sonar/SKILL.md).
Параметр -SkipScanner проверяет прежний анализ; критерии приёмки и граница браузерного покрытия заданы в ENG-004.
