# Действующие интерфейсы и параметры

Нормативное приложение к MEDIA/ASR/SUB/ART/WEB. компоненты
параметров — [cli.py](../../../../speech_to_sub/cli.py), [models.py](../../../../speech_to_sub/models.py), [schemas.py](../../../../speech_to_sub/web/schemas.py), [constants.py](../../../../speech_to_sub/constants.py).
Новая функция не добавляется изменением этой таблицы без соответствующего change.

## CLI

| Группа        | Параметры и контракт                                                                                                                                   |
|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| Вход          | --input с одним или несколькими путями, --recursive только явно, --output-dir сохраняет относительную структуру; коллизии разрешаются детерминированно |
| Формат и язык | --output-format srt/ass/vtt, --language en/ru/auto, --audio-stream-index как ordinal, --audio-language предпочтение                                    |
| Движок        | --backend, --model-path, --no-auto-download-model                                                                                                      |
| Облако        | --allow-cloud-processing для текущего запуска, --openai-model только whisper-1; ключа CLI-параметром нет                                               |
| Выравнивание  | --aligner none/qwen3-forced-aligner, --aligner-model-path, --worker-python-path, --aligner-worker-python-path                                          |
| Устройство    | --device auto/cuda/cpu, --no-quantization, --allow-cpu-fallback                                                                                        |
| Декодирование | --long-form-window-seconds, --long-form-overlap-seconds, --no-vad, --vad-min-silence-ms, --beam-size, --no-condition-on-previous-text                  |
| Раскладка     | --max-chars-per-line, --line-length-gap (0–20), --max-cps                                                                                              |
| Результат     | --keep-audio, --force, --verbose; --version не запускает обработку                                                                                     |

До первого файла проверяются вход, enums, неотрицательный ordinal, положительные числовые настройки, overlap<window,
локальная готовность или допустимость последующей загрузки, наличие worker Python, FFmpeg/ffprobe и явное облачное
согласие/ключ без запроса API.

## Локальное API

| Маршрут                            | Контракт                                                                   |
|------------------------------------|----------------------------------------------------------------------------|
| GET /api/health                    | Версия, FFmpeg и локальная готовность без весов и сети                     |
| GET /api/ui-config                 | Безопасные defaults и варианты, только признак наличия ключа               |
| GET /api/active-job                | Текущая либо последняя задача и журнал                                     |
| GET /api/preparation-status        | Generation, operation_id, фаза, счётчики и ограниченный журнал             |
| GET /api/jobs и /api/jobs/{job_id} | Ограниченная история и сохранённый снимок; неизвестный id — 404            |
| POST /api/pick                     | kind=file/folder и settings; folder всегда recursive=true                  |
| POST /api/refresh                  | paths/settings; пустой список допустим, каталог рекурсивен                 |
| POST /api/transcribe               | Непустой набор, строгие settings, 409 при занятом обработчике              |
| POST /api/jobs/{job_id}/cancel     | Запрос cancelling; завершение подтверждается done                          |
| POST /api/jobs/{job_id}/retry      | Новый job_id/retry_of только error/cancelled; force требует нового boolean |
| GET /api/stream/{job_id}           | SSE по максимальному курсору; конечное событие закрывает поток             |
| POST /api/unload                   | Освобождение активной модели в допустимом состоянии                        |

Неизвестные поля и секреты отклоняются. Некорректное значение — 400 либо стандартный ответ проверки схемы; конфликт
состояния — 409. Согласие на облако не хранится как постоянная настройка.

## События и состояния

Файл: queued → probing → extracting → transcribing → aligning → writing; промежуточные этапы пропускаются по реальному
пути кеша. Терминальные: cached, skipped, done, cancelled, error. Каждый принятый конечный результат имеет progress=100;
при ошибке до probe длительность остаётся неизвестной.

События type=job/file/log/done имеют положительный монотонный id внутри задачи. Done описывает
ok/partial/error/cancelled/interrupted и счётчики. Поле subtitle_output общее; srt_output равно ему лишь для SRT, иначе
null. Положительный MediaProbe сохраняется в активном/конечном snapshot и SQLite. Повторное подключение восстанавливает
состояние, не запускает ASR.

Подготовка: операции pick/refresh/transcribe, состояния idle/running/done/error, фазы
idle/dialog/collecting/probing/queueing/done/error. Терминальная ошибка остаётся до следующей операции. Конфликт 409 не
меняет поколение.

## Хранение и параметры окружения

Состояние — SQLite, defaults 16 задач и 2000 событий на задачу; завершённые записи также очищаются по сроку.
Work-cleanup ограничена job-* с маркером владения и без блокировки. Полные перечни полей и безопасных имён
окружения: [.env.example](../../../../.env.example), [env_utils.py](../../../../speech_to_sub/utils/env_utils.py).
Секретные значения туда не переносятся; изменение конфигурационного контракта синхронизируется с этой спецификацией и
примером.
