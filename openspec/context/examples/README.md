# Примеры и карта проверки

Команды используют проектный Python из корня. Карта связывает текущие требования с существующими компонентами
и проверками. Наличие сценария в тестах не подтверждает успешный прогон.

| Требования      | Компонент реализации                                                                                                                                                                      | Селектор / проверяемый эффект                                                                                                                                                                                                                                      |
|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MEDIA-001–006   | [service.py](../../../speech_to_sub/service.py), [ffmpeg.py](../../../speech_to_sub/media/ffmpeg.py)                                                                                      | [test_ffmpeg.py](../../../tests/test_ffmpeg.py), [test_cli.py](../../../tests/test_cli.py), [test_picker.py](../../../tests/test_picker.py): потоки, аргументы и пути                                                                                              |
| ASR-001–007     | [registry.py](../../../speech_to_sub/asr/registry.py), [external_worker.py](../../../speech_to_sub/asr/external_worker.py), [huggingface.py](../../../speech_to_sub/utils/huggingface.py) | [test_asr_registry.py](../../../tests/test_asr_registry.py), [test_external_worker.py](../../../tests/test_external_worker.py), [test_huggingface.py](../../../tests/test_huggingface.py)                                                                          |
| ASR-008–010     | [openai_api.py](../../../speech_to_sub/asr/openai_api.py)                                                                                                                                 | [test_openai_api_backend.py](../../../tests/test_openai_api_backend.py): имитация API, части и согласие                                                                                                                                                            |
| ASR-011         | [openai_api.py](../../../speech_to_sub/asr/openai_api.py), [huggingface.py](../../../speech_to_sub/utils/huggingface.py), [schemas.py](../../../speech_to_sub/web/schemas.py)             | [test_web_app.py](../../../tests/test_web_app.py)::test_settings_reject_unknown_secret_field, [test_huggingface.py](../../../tests/test_huggingface.py)::test_repository_error_is_mapped_without_token: секрет не принимается через браузер и не выдаётся в ошибке |
| SUB-001–004/006 | [builder.py](../../../speech_to_sub/subtitles/builder.py), [validator.py](../../../speech_to_sub/subtitles/validator.py)                                                                  | [test_srt_boundaries.py](../../../tests/test_srt_boundaries.py)::test_segment_text_wins_when_alignment_words_disagree, [test_srt.py](../../../tests/test_srt.py): сохранность текста и исходного распознавания, раскладка и структурные отказы                     |
| SUB-005         | [formats.py](../../../speech_to_sub/subtitles/formats.py), [validator.py](../../../speech_to_sub/subtitles/validator.py)                                                                  | [test_subtitle_formats.py](../../../tests/test_subtitle_formats.py): кванты SRT/ASS/WebVTT, экранирование и форматные отказы                                                                                                                                       |
| SUB-007         | [Критерий точности](../../specs/subtitle-layout/spec.md#requirement-sub-007-граница-доказанной-точности)                                                                                  | [Задачи эталонной оценки](../../changes/establish-labeled-asr-quality-baseline/tasks.md): измерение точности — NOT_RUN                                                                                                                                             |
| ART-001–010     | [service.py](../../../speech_to_sub/service.py), [cache.py](../../../speech_to_sub/utils/cache.py)                                                                                        | [force и форматы](force-and-formats.md), [test_io_cache.py](../../../tests/test_io_cache.py)                                                                                                                                                                       |
| WEB-001–011     | [app.py](../../../speech_to_sub/web/app.py), [job_store.py](../../../speech_to_sub/web/job_store.py)                                                                                      | [test_web_app.py](../../../tests/test_web_app.py), [test_job_store.py](../../../tests/test_job_store.py), [batch-reload.spec.mjs](../../../tests/e2e/batch-reload.spec.mjs)                                                                                        |

Проверяемые эффекты: при конфликте текста и якорей сохраняется `segment.text`; запрет Origin исключает запуск;
после ошибки публикации и восстановления резервная копия остаётся доступной.

Примеры узких команд:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_service.py -k "format_switch_reuses_recognition_and_keeps_artifacts_isolated or artifact_commit_keeps_unrestored_backup_after_double_failure"
.\.venv\Scripts\python.exe -B -m pytest tests/test_web_app.py -k state_changing_api_rejects_cross_origin_and_non_loopback_host
```
## Связи контрактов с реализацией

- Ранняя проверка Python по
  [IFACE-001](../../specs/application-interface/spec.md#requirement-iface-001-разрешение-и-проверка-настроек)
  вызывается из [service.py](../../../speech_to_sub/service.py) через общий
  [worker_runtime.py](../../../speech_to_sub/utils/worker_runtime.py). Его используют и три изолированных адаптера.
  [test_worker_runtime.py](../../../tests/test_worker_runtime.py) проверяет приоритеты путей и независимость
  настройки выравнивателя. В [test_service.py](../../../tests/test_service.py) селектор
  `selected_worker_python or unselected_worker_python` проверяет отказ до обработки, сохранность результатов
  при обоих значениях force, отсутствие требований к невыбранным профилям и повтор по совместимому кешу
  при существующем Python. Проверки используют временные файлы и имитации, без запуска интерпретатора и моделей.
- Предварительная проверка и новый локальный запуск различаются по [ASR-002](../../specs/speech-recognition/spec.md#requirement-asr-002-локальная-готовность).
  [ensure_huggingface_model](../../../speech_to_sub/utils/huggingface.py) возвращает готовый каталог до обращения к Hub;
  [test_complete_model_never_touches_hub](../../../tests/test_huggingface.py) проверяет этот запрет через имитации.
- Обработка ключей поставщиков по
  [ASR-011](../../specs/speech-recognition/spec.md#requirement-asr-011-секреты-поставщиков) связана с
  [openai_api.py](../../../speech_to_sub/asr/openai_api.py) и [huggingface.py](../../../speech_to_sub/utils/huggingface.py).
- [ART-003](../../specs/recognition-artifacts/spec.md#requirement-art-003-повтор-без-force) — владелец автономного повтора;
  существующий [пример force и форматов](force-and-formats.md) показывает его отличие от нового тяжёлого запуска.
- Буквальные состояния и поля перечислены в [спецификации интерфейсов](../../specs/application-interface/spec.md);
  эффекты запросов проверяются по WEB, совместимость форматных полей — по ART-007.

- Повтор после interrupted определяется
  [WEB-008](../../specs/local-web-workflow/spec.md#requirement-web-008-отмена-и-повтор);
  [jobs.py](../../../speech_to_sub/web/jobs.py) включает это состояние в RETRYABLE_FILE_STATES.
  Обработчик [retry_job](../../../speech_to_sub/web/app.py) создаёт новую задачу, сохраняя retry_of.

## Граница проверки облачного согласия

[ASR-008](../../specs/speech-recognition/spec.md#requirement-asr-008-явный-облачный-режим) обеспечивается в
[cli.py](../../../speech_to_sub/cli.py): отсутствие флага текущего запуска передаёт `False` при слиянии настроек.
[process_paths](../../../speech_to_sub/service.py) отклоняет облачную операцию до поиска файлов.

[test_cli_requires_fresh_cloud_consent_on_each_invocation](../../../tests/test_cli.py) вызывает настоящий вход CLI,
загрузчик окружения и сервис. Проверяются окружение процесса и искусственный `.env`, обычный запуск и `--force`:
первый вызов с флагом проходит; повтор без флага возвращает 1, не увеличивая счётчики поиска и обработки файлов;
новый вызов с флагом снова разрешён. Обработка файла подменена имитацией, реальные медиа и сеть не используются.
Причина и исправление — в [записи об исправлении](../failure-log.md#новое-облачное-согласие-в-cli).

## Граница чтения JSON-кеша

[ART-010](../../specs/recognition-artifacts/spec.md#requirement-art-010-ограниченное-чтение-json-кеша) задаёт
байтовый предел общего загрузчика [cache.py](../../../speech_to_sub/utils/cache.py). Сервис получает отдельную
причину отказа и применяет сохранность результатов по
[ART-005](../../specs/recognition-artifacts/spec.md#requirement-art-005-существующий-неподтверждённый-результат).

| Сценарий                                          | Селектор и проверяемый эффект                                                                                                                                                                                                                                                                                                                                  |
|---------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Байтовый предел до декодирования                  | [test_io_cache.py](../../../tests/test_io_cache.py): `test_sidecar_production_limit_is_64_mib` закрепляет 67 108 864 байта; `test_sidecar_byte_boundary` проверяет точную границу и превышение на байт; `test_sidecar_limit_counts_utf8_bytes` — многобайтный UTF-8                                                                                            |
| Фактический поток больше предварительного размера | [test_sidecar_growing_stream_is_bounded_before_decoding](../../../tests/test_io_cache.py) запрещает безразмерное чтение и вызов JSON-парсера после превышения; суммарно читается не более предела плюс один байт                                                                                                                                               |
| Полный и неполный наборы результатов SRT/ASS/VTT  | [test_service.py](../../../tests/test_service.py): `test_oversized_cache_preserves_selected_and_neighbor_artifacts` и `test_oversized_neighbor_preserves_selected_audio_without_subtitles` проверяют прежние байты, отсутствие нормализации, распознавания и публикации, а также отсутствие фиктивного пути субтитров                                          |
| Полностью отсутствующий выбранный формат          | [test_service.py](../../../tests/test_service.py): `test_oversized_neighbor_allows_next_valid_cache_for_empty_format` проверяет переход к следующему пригодному кешу без повторного чтения кандидатов; `test_oversized_neighbor_without_valid_cache_allows_new_recognition_and_batch` — новое распознавание и продолжение пакета с сохранением соседних файлов |
| Поздний отказ перед добавлением FLAC              | [test_late_oversized_cache_preserves_results_before_audio_publication](../../../tests/test_service.py) проверяет ошибку файла до публикации, сохранность прежних артефактов, отсутствие ложного `audio_output` и продолжение пакета                                                                                                                            |
| Явный полный пересчёт                             | [test_force_bypasses_oversized_cache_reader_and_preserves_neighbors](../../../tests/test_service.py) запрещает чтение прежнего кеша, проверяет повторную нормализацию и распознавание, замену выбранного набора и сохранность соседних форматов                                                                                                                |
| Прежние допустимые пути                           | [test_sidecar_ordinary_read_failure_is_not_size_rejection](../../../tests/test_io_cache.py) отличает обычный некорректный вход от превышения предела; [test_valid_selected_cache_is_read_once_without_inspecting_oversized_neighbor](../../../tests/test_service.py) проверяет повтор по подтверждённому выбранному кешу                                       |

Граничные проверки используют уменьшенный тестовый предел и отдельную проверку производственного значения.
Сервисные проверки работают с синтетическими файлами и имитациями, без реальных моделей, сети и исчерпания памяти.
Состояние реализации и проверки — в [задачах изменения](../../changes/archive/2026-10-15-bound-json-cache-loading/tasks.md).

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m pytest -p no:cacheprovider -q tests/test_io_cache.py tests/test_service.py
```

## Сохранность цели junction при очистке

Исключение junction по [WEB-010](../../specs/local-web-workflow/spec.md#requirement-web-010-очистка-и-история) выполняется в
[service.py](../../../speech_to_sub/service.py) через общий детектор `is_link_or_junction` из
[io_utils.py](../../../speech_to_sub/utils/io_utils.py).

В [test_service.py](../../../tests/test_service.py) сценарий `test_cleanup_stale_workspaces_preserves_junction_target`
создаёт настоящий NTFS junction на соседний пользовательский каталог внутри временного `WORK_DIR`.
Проверяются сохранность ссылки и её прежней цели, неизменность байтов маркера и файлов, включая вложенный,
и удаление отдельного обычного устаревшего `job-*`. Сценарий выполняется на Windows;
`test_cleanup_stale_workspaces_removes_only_safe_unlocked_job_directories` сохраняет проверки возраста,
активной блокировки, маркера владения и границ очистки.
