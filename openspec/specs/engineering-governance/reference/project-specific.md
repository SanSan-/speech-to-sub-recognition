# Инженерное приложение speech-to-sub-recognition

Обязательно вместе с [общим профилем](python-common.md). Общие правила разработки, кодировки, сохранности данных и
проверок задаются там; ниже — проектные дополнения и рабочие команды.

## Проектные дополнения

Русские технические тексты перечитываются на необъяснённые термины, скрытые предпосылки, скачки рассуждения и служебные
фразы. Буквальные идентификаторы, команды и названия моделей сохраняются. Комментарии, docstring и журналы — по-русски,
кроме принятого в конкретном файле устойчивого стиля. Ориентир размера функции — 40–50 строк при одной ответственности.

Текст существующих журналов не изменяй без отдельного запроса. Сохранность изменений и ограничения действий с Git —
в пункте 8 общего профиля.

## Среды и зависимости

| Среда | Основа и источник конфигурации |
|---|---|
| Основная | Python 3.14 x64, .venv; [requirements.txt](../../../../requirements.txt) |
| Parakeet | Python 3.14, resources/runtimes/parakeet; [parakeet.txt](../../../../requirements-workers/parakeet.txt) |
| Qwen ASR/aligner | Python 3.11, resources/runtimes/qwen; [qwen.txt](../../../../requirements-workers/qwen.txt) |
| Установка изолированных сред | [setup_workers.ps1](../../../../setup_workers.ps1) |
| Пакет | [pyproject.toml](../../../../pyproject.toml); версия из [__init__.py](../../../../speech_to_sub/__init__.py) |
| Браузер | [package.json](../../../../package.json), package-lock.json, Playwright |

Точные версии зависимостей задаются соответствующим requirements; основной граф не меняется ради несовместимой
изолированной среды. Установку выполняет pip. Зависимости устанавливаются по requirements.txt.
pyproject dependencies=[] и Requires-Dist=0 — известное ограничение: wheel не устанавливает зависимости приложения.
Внешний пакетный реестр требует отдельного контракта. [Workflow сборки](../../../../.github/workflows/python-release-build.yml)
сохраняет артефакты GitHub Release.

## Настройки и компоненты

[Приложение интерфейсов](application-interface.md) — единственный источник перечней CLI-параметров, API-маршрутов,
публичных состояний и порядка разрешения настроек. Значения по умолчанию и версии идентичности определяются
[constants.py](../../../../speech_to_sub/constants.py) и [models.py](../../../../speech_to_sub/models.py);
их текущие числа не копируются в архитектурные обзоры.

При смене backend без явного пути выбирается его собственный каталог. Основная модель — whisper-large-v3-ct2,
Transformers — запасной профиль. Готовность и докачка модели определены в
[ASR-002](../../speech-recognition/spec.md#requirement-asr-002-локальная-готовность) и
[ASR-003](../../speech-recognition/spec.md#requirement-asr-003-разрешённая-докачка),
изоляция — в [ASR-005](../../speech-recognition/spec.md#requirement-asr-005-изолированная-обработка),
последовательное выравнивание — в
[ASR-007](../../speech-recognition/spec.md#requirement-asr-007-независимое-выравнивание),
переиспользование распознавания — в
[ART-003](../../recognition-artifacts/spec.md#requirement-art-003-повтор-без-force).

Новая переменная описывается в интерфейсном приложении и в [.env.example](../../../../.env.example) переносимым значением.
README содержит согласованную пользовательскую справку. Безопасное обращение с локальными значениями — по
[SEC-CONTEXT-002](../spec.md#requirement-sec-context-002-минимальное-чтение-и-передача-данных).

## Документы и доказательства

Размещение требований, задач, решений и сводок задано в
[CTX-001/003/006](../../project-context/spec.md#requirement-ctx-001-единый-проектный-контекст).
[Карта архитектуры](../../../context/03-architecture.md) указывает владельцев предметных контрактов;
[примеры](../../../context/examples/README.md) связывают их с реализацией и проверками.

README — переносимая пользовательская инструкция. CHANGELOG фиксирует значимые реализованные изменения по версиям,
раздел «Не выпущено» отделяет их от сведений о выпуске. docs/benchmarks содержит первичные измерения.
.env.example и sonar-project.properties.example — переносимые примеры без секретов.

В инструкциях используй плейсхолдеры `<repo-root>`, `<media-file>`, `<media-directory>`, `<local-asr-model-path>`,
`<local-aligner-model-path>`, `<local-hf-cache>`, `<sonar-url>`. Локальную установку и проверки располагай ближе к концу
пользовательской инструкции; команды Windows и Linux/macOS/WSL разделяй.

## Команды и ограничения проверок

Все команды выполняются из корня проекта. Выбор охвата и граница доказательства — в пунктах 13–14 общего профиля.

```powershell
.\.venv\Scripts\python.exe -B -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q speech_to_sub tests
node --check speech_to_sub\web\static\app.js
npm test
openspec validate --all --strict --no-interactive
```

Создание среды: py -3.14 -m venv .venv, затем проектный Python -m pip install -r requirements.txt; браузерная установка —
npm ci. Для Parakeet/Qwen используется setup_workers.ps1 с -Parakeet/-Qwen. Запуск CLI — проектный Python -m speech_to_sub;
веб — run_web.ps1 либо -m speech_to_sub.web. FFmpeg/ffprobe доступны в PATH или по настроенным путям.

Тяжёлая проверка: RUN_LOCAL_ASR_TEST=1, ASR_TEST_MEDIA=`<media-file>`, ASR_TEST_LANGUAGE=en и проектный Python
-B -m pytest -m integration. Отказные проверки и результаты размещаются в тестовом каталоге. Условия сети и облака —
по ASR-002/003/008. Создание среды и тяжёлые проверки не входят в обслуживание документации.

## Sonar

Ключ проекта — speech-to-sub-recognition; [пример конфигурации](../../../../sonar-project.properties.example).
Адрес задаётся через SONAR_HOST_URL, токен — через SONAR_TOKEN в локальном окружении.
Нужны проектная среда, sonar-scanner, Node.js и доступный сервер.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .codex\skills\fix-sonar\scripts\sonar_cycle.ps1 -TimeoutSeconds 600 -RequireClean
```

Полная процедура — в [.codex/skills/fix-sonar/SKILL.md](../../../../.codex/skills/fix-sonar/SKILL.md).
Для приёмки нового дерева нужны свежий анализ, завершённая вычислительная задача (CE), чистый шлюз качества,
отсутствие открытых замечаний и замечаний SECURITY, проверка дублирования и порога покрытия.
-SkipScanner проверяет прежний анализ. Датированное доказательство связывает команды, CE ID/status и показатели качества;
пропущенные проверки отмечаются по ENG-004.

Клиентская статика входит в анализ; без LCOV её нет в численном покрытии. Playwright проверяется отдельно.
Включение браузерного покрытия требует сбора LCOV, sonar.javascript.lcov.reportPaths и новой проверки объединённой метрики.

При расхождении API/UI после SUCCESS сначала проверяются API и ключ проекта. Диагностика чужого сервера подчиняется его
инструкциям; состояние замечаний не меняется через SQL.

## Автоматические правила контекста

Полные обязательные правила находятся в [CTX-005…008](../../project-context/spec.md)
и [SEC-CONTEXT-001…004](../spec.md). Методика CTX-009 относится к отдельно запрошенному фоновому аудиту.
