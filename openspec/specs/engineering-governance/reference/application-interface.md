# Действующие интерфейсы и параметры

Нормативное приложение к MEDIA/ASR/SUB/ART/WEB. Здесь задаются параметры, публичные поля и состояния; правила
поведения принадлежат предметным требованиям по ссылкам ниже. Источники деклараций:
[cli.py](../../../../speech_to_sub/cli.py), [models.py](../../../../speech_to_sub/models.py),
[schemas.py](../../../../speech_to_sub/web/schemas.py), [constants.py](../../../../speech_to_sub/constants.py).

## Настройки

Порядок разрешения: значения по умолчанию → локальный .env без замены окружения процесса → явные параметры.
Точные значения по умолчанию задаются constants.py и models.py. Допустимы перечисленные значения, неотрицательный ordinal,
положительные числовые настройки, overlap < window и line_length_gap в пределах 0–20.
Проверка выполняется до первого файла; коды завершения — по
[MEDIA-006](../../media-preparation/spec.md#requirement-media-006-безопасный-запуск-и-настройки).

Локальная готовность, возможность докачки и проверка облачного профиля — по
[ASR-002](../../speech-recognition/spec.md#requirement-asr-002-локальная-готовность),
[ASR-003](../../speech-recognition/spec.md#requirement-asr-003-разрешённая-докачка) и
[ASR-008](../../speech-recognition/spec.md#requirement-asr-008-явный-облачный-режим).
До обработки проверяется наличие worker Python и FFmpeg/ffprobe.

## CLI

| Группа | Параметры |
|---|---|
| Вход | --input с одним или несколькими путями; --recursive; --output-dir |
| Формат и язык | --output-format srt/ass/vtt; --language en/ru/auto; --audio-stream-index как ordinal; --audio-language |
| Движок | --backend; --model-path; --no-auto-download-model |
| Облако | --allow-cloud-processing; --openai-model whisper-1; параметра ключа нет |
| Выравнивание | --aligner none/qwen3-forced-aligner; --aligner-model-path; --worker-python-path; --aligner-worker-python-path |
| Устройство | --device auto/cuda/cpu; --no-quantization; --allow-cpu-fallback |
| Декодирование | --long-form-window-seconds; --long-form-overlap-seconds; --no-vad; --vad-min-silence-ms; --beam-size; --no-condition-on-previous-text |
| Раскладка | --max-chars-per-line; --line-length-gap; --max-cps |
| Результат | --keep-audio; --force; --verbose; --version без обработки |

--output-dir сохраняет относительную структуру; коллизии разрешаются детерминированно.
Обход — по [MEDIA-002](../../media-preparation/spec.md#requirement-media-002-детерминированный-обход),
выбор потока — по [MEDIA-004](../../media-preparation/spec.md#requirement-media-004-выбор-дорожки).
Полный force и сохранение форматных результатов — по [ART-001…006](../../recognition-artifacts/spec.md).

## Локальное API

| Маршрут | Ответ или входные поля | Правила поведения |
|---|---|---|
| GET /api/health | Версия, FFmpeg, локальная готовность | ASR-002 |
| GET /api/ui-config | Безопасные defaults и варианты; признак наличия ключа | WEB-002, ASR-008 |
| GET /api/active-job | Текущая либо последняя задача и журнал | WEB-007 |
| GET /api/preparation-status | generation, operation_id, фаза, счётчики, журнал | WEB-004 |
| GET /api/jobs и /api/jobs/{job_id} | История и сохранённый снимок; неизвестный id — 404 | WEB-007/010 |
| POST /api/pick | kind=file/folder, settings | MEDIA-002, WEB-003/004 |
| POST /api/refresh | paths/settings; пустой список допустим | MEDIA-002, WEB-003/004 |
| POST /api/transcribe | Непустой набор, settings | WEB-002/003 |
| POST /api/jobs/{job_id}/cancel | Запрос cancelling; завершение подтверждает событие done | WEB-008 |
| POST /api/jobs/{job_id}/retry | Новый job_id, retry_of | WEB-008 |
| GET /api/stream/{job_id} | SSE, Last-Event-ID/cursor | WEB-006 |
| POST /api/unload | Освобождение активной модели в допустимом состоянии | WEB-003 |

Ссылки владельцев: [WEB-001…011](../../local-web-workflow/spec.md), [MEDIA-002](../../media-preparation/spec.md),
[ASR-002/008](../../speech-recognition/spec.md). Некорректное значение получает 400 либо стандартный ответ проверки схемы;
конфликт состояния — по WEB-003. Строгие типы определены в WEB-002, секреты — в ASR-011, согласие на облако — в ASR-008.

## События и состояния

Файл: queued → probing → extracting → transcribing → aligning → writing; этапы пропускаются по фактическому пути кеша.
Терминальные состояния: cached, skipped, done, cancelled, interrupted, error. Каждый принятый конечный результат имеет progress=100.
Достоверность снимка и неизвестных величин — по WEB-007.

События: type=job/file/log/done; id — положительное целое внутри задачи, порядок и восстановление — по WEB-006.
Done содержит результат ok/partial/error/cancelled/interrupted и счётчики.
Поля результата: subtitle_output, srt_output; их совместимость — по
[ART-007](../../recognition-artifacts/spec.md#requirement-art-007-совместимость-srt).
MediaProbe сохраняется в снимке по WEB-007.

Подготовка: операции pick/refresh/transcribe; состояния idle/running/done/error; фазы
idle/dialog/collecting/probing/queueing/done/error. Терминальная ошибка остаётся до следующей операции;
актуальность поколения и последствия конфликта — по WEB-003/004.

## Хранение и параметры окружения

Постоянное хранилище — SQLite; основание выбора — [ADR-006](../../../context/ADR/ADR-006-local-web-state.md).
Значения по умолчанию: 16 задач и 2000 событий на задачу. Срок хранения и безопасная очистка — по WEB-010;
область служебных каталогов — job-* с маркером владения.

Полные перечни параметров окружения: [.env.example](../../../../.env.example) и
[env_utils.py](../../../../speech_to_sub/utils/env_utils.py). Изменение конфигурационного контракта синхронизируется
с этим приложением и безопасным примером по ENG-006.
