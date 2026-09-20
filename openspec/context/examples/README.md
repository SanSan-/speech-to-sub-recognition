# Реальные примеры

Команды используют проектный Python из корня. Наличие сценария в тестах не подтверждает успешный прогон.

| Требования    | Компонент реализации                                                                                                                                                                      | Селектор / проверяемый эффект                                                                                                                                                             |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MEDIA-001–006 | [service.py](../../../speech_to_sub/service.py), [ffmpeg.py](../../../speech_to_sub/media/ffmpeg.py)                                                                                      | [test_ffmpeg.py](../../../tests/test_ffmpeg.py), [test_cli.py](../../../tests/test_cli.py), [test_picker.py](../../../tests/test_picker.py): потоки, аргументы и пути                     |
| ASR-001–007   | [registry.py](../../../speech_to_sub/asr/registry.py), [external_worker.py](../../../speech_to_sub/asr/external_worker.py), [huggingface.py](../../../speech_to_sub/utils/huggingface.py) | [test_asr_registry.py](../../../tests/test_asr_registry.py), [test_external_worker.py](../../../tests/test_external_worker.py), [test_huggingface.py](../../../tests/test_huggingface.py) |
| ASR-008–010   | [openai_api.py](../../../speech_to_sub/asr/openai_api.py)                                                                                                                                 | [test_openai_api_backend.py](../../../tests/test_openai_api_backend.py): имитация API, части и согласие                                                                                   |
| SUB-001–007   | [builder.py](../../../speech_to_sub/subtitles/builder.py)                                                                                                                                 | [test_srt_boundaries.py](../../../tests/test_srt_boundaries.py)::test_segment_text_wins_when_alignment_words_disagree                                                                     |
| ART-001–009   | [service.py](../../../speech_to_sub/service.py), [cache.py](../../../speech_to_sub/utils/cache.py)                                                                                        | [force и форматы](force-and-formats.md), [test_io_cache.py](../../../tests/test_io_cache.py)                                                                                              |
| WEB-001–011   | [app.py](../../../speech_to_sub/web/app.py), [job_store.py](../../../speech_to_sub/web/job_store.py)                                                                                      | [test_web_app.py](../../../tests/test_web_app.py), [test_job_store.py](../../../tests/test_job_store.py), [batch-reload.spec.mjs](../../../tests/e2e/batch-reload.spec.mjs)               |

Предпочтительно: при конфликте текста и якорей сохранить `segment.text`; при запрете Origin доказать отсутствие запуска;
при ошибке публикации проверить backup. Плохо: проверять только наличие исключения или число зелёных тестов без
наблюдаемого эффекта.

Примеры узких команд:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_service.py -k "format_switch_reuses_recognition_and_keeps_artifacts_isolated or artifact_commit_keeps_unrestored_backup_after_double_failure"
.\.venv\Scripts\python.exe -B -m pytest tests/test_web_app.py -k state_changing_api_rejects_cross_origin_and_non_loopback_host
```
## Связи контрактов с реализацией

- Предварительная проверка и новый локальный запуск различаются по [ASR-002](../../specs/speech-recognition/spec.md#requirement-asr-002-локальная-готовность).
  [ensure_huggingface_model](../../../speech_to_sub/utils/huggingface.py) возвращает готовый каталог до обращения к Hub;
  [test_complete_model_never_touches_hub](../../../tests/test_huggingface.py) проверяет этот запрет через имитации.
- Обработка ключей поставщиков по
  [ASR-011](../../specs/speech-recognition/spec.md#requirement-asr-011-секреты-поставщиков) связана с
  [openai_api.py](../../../speech_to_sub/asr/openai_api.py) и [huggingface.py](../../../speech_to_sub/utils/huggingface.py).
- [ART-003](../../specs/recognition-artifacts/spec.md#requirement-art-003-повтор-без-force) — владелец автономного повтора;
  существующий [пример force и форматов](force-and-formats.md) показывает его отличие от нового тяжёлого запуска.
- Буквальные состояния и поля перечислены в [приложении интерфейсов](../../specs/engineering-governance/reference/application-interface.md);
  эффекты запросов проверяются по WEB, совместимость форматных полей — по ART-007.

- Повтор после interrupted определяется
  [WEB-008](../../specs/local-web-workflow/spec.md#requirement-web-008-отмена-и-повтор);
  [jobs.py](../../../speech_to_sub/web/jobs.py) включает это состояние в RETRYABLE_FILE_STATES.
  Обработчик [retry_job](../../../speech_to_sub/web/app.py) создаёт новую задачу, сохраняя retry_of.
