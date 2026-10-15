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
├── openspec/               # Требования и документация проекта
│   ├── context/            # Начало работы, архитектура, словарь и решения
│   ├── specs/              # Предметные требования
│   └── changes/            # Открытые изменения и задачи
└── .github/workflows/      # Сборка выпусков в GitHub Actions
```

## Начало работы

Начните с [контекста проекта](openspec/context/00-start-here.md): он направляет к описанию продукта, архитектуре
и правилам работы.

| Задача                               | Документация                                                                                                                             |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| Разобраться в устройстве приложения  | [Архитектура](openspec/context/03-architecture.md)                                                                                       |
| Установить и запустить приложение    | [Подготовка среды и команды](#подготовка-сред)                                                                                           |
| Выбрать модель и настроить обработку | [Распознавание речи](openspec/specs/speech-recognition/spec.md) и [интерфейсы и параметры](openspec/specs/application-interface/spec.md) |
| Настроить формирование субтитров     | [Разметка и форматы субтитров](openspec/specs/subtitle-layout/spec.md)                                                                   |
| Подготовить изменения                | [Инженерные правила](openspec/context/04-engineering-standards.md) и [проверки качества](openspec/context/06-testing-and-quality.md)     |
| Проверить архитектурное решение      | [Принятые решения](openspec/context/ADR/README.md)                                                                                       |
| Посмотреть запланированные работы    | [Открытые изменения](openspec/changes/README.md)                                                                                         |

Все команды выполняются из корня проекта. Среды, источники зависимостей и ограничения пакета определены в
[ENG-001…002](openspec/specs/engineering-constraints/spec.md); границы обычных и тяжёлых проверок — в
[ENG-003](openspec/specs/engineering-constraints/spec.md#requirement-eng-003-границы-проверок).

## Подготовка сред

Создайте основную среду и установите зависимости перед выполнением команд, использующих проектный Python:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Для браузерных проверок нужны Node.js с npm и Chromium, установленный для используемой версии
[Playwright](https://playwright.dev/docs/browsers#install-browsers):

```powershell
npm.cmd ci
npx.cmd playwright install chromium
```

Для Parakeet/Qwen используется [setup_workers.ps1](setup_workers.ps1) с -Parakeet/-Qwen.
FFmpeg/ffprobe доступны в PATH или по настроенным путям.

## Запуск приложения

CLI запускается проектным Python; укажите путь к медиафайлу:

```powershell
.\.venv\Scripts\python.exe -m speech_to_sub --input "<media-file>"
```

Для облачного движка `openai-api` каждый вызов CLI требует нового флага `--allow-cloud-processing`;
сохранённое согласие из окружения или `.env` его не заменяет. Полные условия —
[ASR-008](openspec/specs/speech-recognition/spec.md#requirement-asr-008-явный-облачный-режим).

Веб-интерфейс запускается через [run_web.ps1](run_web.ps1) либо проектным Python
`-m speech_to_sub.web`. Параметры и маршруты определены в
[спецификации интерфейсов](openspec/specs/application-interface/spec.md).

JSON-кеш размером более 64 МиБ исключается из повторного использования. При отказе по размеру существующие
субтитры, JSON и FLAC выбранного формата сохраняются без перезаписи; это относится и к неполному набору.
Если выбранный формат полностью отсутствует, обработка использует следующий пригодный кеш либо выполняет
распознавание, сохраняя соседние результаты. Условия отказа и явного пересчёта через force —
[ART-005/010](openspec/specs/recognition-artifacts/spec.md#requirement-art-010-ограниченное-чтение-json-кеша).

## Проверки

После подготовки нужных сред доступны команды ниже. Для проверки спецификаций дополнительно требуется
установленный OpenSpec CLI.

```powershell
.\.venv\Scripts\python.exe -B -m pytest
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q speech_to_sub tests
node --check speech_to_sub\web\static\app.js
npm.cmd test
openspec.cmd validate --all --strict --no-interactive
```

Для проверки Ruff требуется отдельно установленный в `.venv` Ruff; `requirements.txt` его не содержит:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

Для тяжёлой интеграционной проверки задайте RUN_LOCAL_ASR_TEST=1, ASR_TEST_MEDIA=`<media-file>`,
ASR_TEST_LANGUAGE=en и запустите проектный Python `-B -m pytest -m integration`.

## Sonar

Подготовьте проектную среду, sonar-scanner, Node.js, доступ к серверу и параметры локального окружения по
[ENG-004](openspec/specs/engineering-constraints/spec.md#requirement-eng-004-анализ-sonar-и-граница-покрытия).
При использовании локально установленного сценария `fix-sonar` укажите путь к нему и загрузите его
в Windows PowerShell 5.1 с явно указанной кодировкой UTF-8:

```powershell
$sonarScript = "<sonar-cycle.ps1>"
$sonarCycle = [ScriptBlock]::Create([IO.File]::ReadAllText($sonarScript, [Text.Encoding]::UTF8))
& $sonarCycle -ProjectRoot (Get-Location).Path -TimeoutSeconds 600 -RequireClean
```

Параметр -SkipScanner проверяет прежний анализ; критерии приёмки и граница браузерного покрытия заданы в ENG-004.
