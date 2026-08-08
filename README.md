# speech-to-sub-recognition

Локальный Python-проект для пакетного извлечения аудио из видео, распознавания речи
локальной моделью и создания субтитров SRT.

Основной сценарий:

```text
MP4/audio -> ffprobe -> FFmpeg 16 kHz mono FLAC -> local ASR -> optional aligner -> SRT
```

## Статус проекта

```text
.
├─ speech_to_sub
│  ├─ __main__.py                  # CLI
│  ├─ cli.py                       # аргументы и коды завершения
│  ├─ service.py                   # общий batch-пайплайн
│  ├─ asr                          # локальные ASR backend-ы
│  ├─ alignment                    # optional aligner-ы
│  ├─ workers                      # протокол изолированных ML-worker-ов
│  ├─ media                        # ffprobe и FFmpeg
│  ├─ subtitles                    # SRT builder и validator
│  ├─ utils                        # env, I/O, кеш, логи, GPU
│  └─ web
│     ├─ app.py                    # FastAPI
│     ├─ jobs.py                   # worker, state и SSE
│     ├─ picker.py                 # локальные системные диалоги
│     ├─ schemas.py                # API DTO
│     └─ static                    # HTML/CSS/JS
├─ docs                            # исходные исследования
├─ logs
├─ resources
│  ├─ cache
│  ├─ state                        # SQLite web-задач
│  ├─ runtimes                     # локальные изолированные venv
│  └─ work
├─ requirements-workers            # точные lock-файлы worker runtime
├─ tests
├─ setup_workers.ps1
├─ run_web.ps1
├─ requirements.txt
├─ pyproject.toml
├─ PRD.md
└─ README.md
```

Версия проекта — `1.4.0`; CLI и локальный веб-интерфейс используют общий service layer и четыре локальных ASR backend-а:
faster-whisper, Transformers Whisper, Parakeet TDT v3 и Qwen3-ASR 0.6B. Qwen3 ForcedAligner доступен как optional aligner.
Сквозной путь проверен на русских и английских MKV/MP4, включая long-form записи.

Веб-задачи, файловые состояния и события сохраняются в SQLite. После рестарта незавершённая
задача переводится в явный terminal-статус `interrupted`, а успешные результаты остаются
доступны; поддержаны cancel, retry только неуспешных файлов и SSE reconnect по курсору.

## Основные решения первой версии

- Только локальное распознавание.
- Четыре ASR backend-а и optional aligner выбираются через общие registry и настройки.
- `faster-whisper` + CTranslate2 как основной backend; `transformers` + PyTorch как fallback.
- Parakeet и Qwen работают в изолированных worker-процессах с несовместимыми между собой
  закреплёнными ML-зависимостями.
- Long-form окна декодируются через PyAV последовательно, без единого PCM-массива всей записи.
- FFmpeg и ffprobe как системные зависимости.
- Один активный ASR worker, чтобы модель не дублировалась в VRAM.
- Один service layer для CLI и web.
- SRT строится отдельным компонентом по временным меткам модели.
- Исходные медиа не изменяются.
- Существующие результаты не перезаписываются без явного `force`.
- Набор артефактов защищён межпроцессным lock и публикуется с откатом при частичном сбое.
- Web-state хранится в SQLite; завершённые записи ограничиваются размером/TTL, а устаревшие
  `resources/work/job-*` удаляются только при наличии служебного lock-маркера и отсутствии
  активной блокировки.

## Локальные модели

Основной CTranslate2 checkpoint:

```text
D:\Projects\-ai\+automatic-speech-recognition\whisper-large-v3-ct2
```

Он полностью локально сконвертирован из существующего Hugging Face checkpoint и содержит
`model.bin`, `config.json`, tokenizer, vocabulary и preprocessor config. Обычный запуск не
обращается к сети. Профили: CUDA `int8_float16`/`float16`, CPU `int8`/`float32`.

Fallback Transformers checkpoint:

```text
D:\Projects\-ai\+automatic-speech-recognition\whisper-large-v3
```

Каталог содержит Hugging Face checkpoint `WhisperForConditionalGeneration`, tokenizer,
processor и generation config.

Whisper Large v3 многоязычный. Английский (`en`), русский (`ru`) и автоопределение (`auto`)
поддерживаются одним локальным backend; отдельный язык аудиодорожки задаётся независимо от
языка распознавания.

Parakeet TDT v3:

```text
D:\Projects\-ai\+automatic-speech-recognition\parakeet-tdt-0.6b-v3
```

Он запускается отдельным Python 3.14 worker с `transformers 5.14.1` и PyTorch/CUDA. Нативные
token timestamps нормализуются в общий word/segment-контракт; длинные записи обрабатываются
окнами с overlap.

Qwen3-ASR и ForcedAligner:

```text
D:\Projects\-ai\+automatic-speech-recognition\Qwen3-ASR-0.6B
D:\Projects\-ai\+automatic-speech-recognition\Qwen3-ForcedAligner-0.6B
```

Qwen запускается в изолированном Python 3.11 runtime с `qwen-asr 0.0.6`,
`transformers 4.57.6` и PyTorch/CUDA. ASR и ForcedAligner работают в отдельных persistent
worker-процессах и загружаются последовательно. ASR формирует coarse-сегменты с target
175 секунд и жёсткой границей 180 секунд; самостоятельный ForcedAligner режет аудио по этим
границам, применяет точные offsets и возвращает нормализованные словные временные метки.
Aligner можно сочетать с любым backend-ом: длинные сегменты он детерминированно делит на части
до 180 секунд по словным меткам и паузам, а без них — пропорционально тексту и времени. Все
backend-ы проверяют локальный путь до тяжёлого инференса и не скачивают модели при обычном запуске.

## Поддерживаемые входы

Видео:

- `.mp4`
- `.m4v`
- `.mov`
- `.mkv`
- `.webm`

Аудио:

- `.wav`
- `.flac`
- `.mp3`
- `.m4a`
- `.aac`
- `.ogg`
- `.opus`
- `.mka`

Перед обработкой контейнер проверяется через `ffprobe`. Для файла с несколькими дорожками
можно задать нулевой порядковый номер аудиопотока; иначе приложение ищет предпочитаемый язык
из `ASR_AUDIO_LANGUAGE` и показывает предупреждение при неоднозначности.

## Результаты

Для `lecture.mp4` по умолчанию:

```text
lecture.en.srt
lecture.en.asr.json
```

- `.srt` — проверенные субтитры в UTF-8 без BOM;
- `.asr.json` — fingerprint источника, выбранный поток, ASR/aligner runtime и параметры,
  `started_at`, `finished_at` и исходные временные сегменты для диагностики и повторной сборки SRT;
- нормализованный `.flac` является временным; сохранить его можно отдельной настройкой.

SRT, sidecar и сохраняемый FLAC сначала готовятся во временных соседних файлах, затем
публикуются одним защищённым commit с восстановлением предыдущей версии при ошибке. Пустой
или непрошедший проверку SRT не публикуется.

Если два выбранных файла дают одинаковый целевой путь, к имени сначала добавляется расширение
источника, а при повторной коллизии — детерминированный короткий hash абсолютного пути.

## Требования

- Windows с PowerShell.
- Python 3.14 x64.
- FFmpeg и ffprobe в `PATH` либо их явные пути в `.env`.
- Для штатного GPU-режима — совместимая NVIDIA GPU/CUDA.
- Локальный checkpoint выбранного backend-а.
- Python 3.11 x64 дополнительно для Qwen3-ASR/ForcedAligner.

Проверка внешних команд:

```powershell
ffmpeg -version
ffprobe -version
py -3.14 --version
```

## Установка

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` перенесён из более свежего `ocr-local` и содержит полный закреплённый
набор PyTorch/CUDA, Transformers, FastAPI и тестовых зависимостей. Для V1.2 добавлены точно
закреплённые `faster-whisper 1.2.1`, `CTranslate2 4.8.1`, `PyAV 18.0.0` и
`onnxruntime 1.28.0`; их Windows wheels проверены на Python 3.14.

Parakeet и Qwen нельзя устанавливать в основной venv из-за разных требований к Python,
Transformers и PyTorch. Их точные зависимости находятся в
`requirements-workers/parakeet.txt` и `requirements-workers/qwen.txt`, а изолированные
runtime-ы создаются командой:

```powershell
.\setup_workers.ps1
```

Можно подготовить только один runtime: `.\setup_workers.ps1 -Parakeet` или
`.\setup_workers.ps1 -Qwen`. Скрипт использует `resources/runtimes/parakeet` на Python 3.14
и `resources/runtimes/qwen` на Python 3.11, затем выполняет `pip check`.

Для воспроизводимого browser E2E дополнительно нужен Node.js:

```powershell
npm ci
```

Проверка ML-стека:

```powershell
.\.venv\Scripts\python.exe -B -c "import torch, transformers, faster_whisper, ctranslate2, av; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, faster_whisper.__version__, ctranslate2.__version__, av.__version__)"
```

## Переменные окружения

Рабочий `.env`:

```env
ASR_BACKEND=faster-whisper
ASR_MODEL_PATH=D:\Projects\-ai\+automatic-speech-recognition\whisper-large-v3-ct2
ASR_ALIGNER=none
ASR_ALIGNER_MODEL_PATH=
ASR_WORKER_PYTHON=
ASR_ALIGNER_WORKER_PYTHON=resources\runtimes\qwen\Scripts\python.exe
QWEN_ASR_PYTHON=resources\runtimes\qwen\Scripts\python.exe
PARAKEET_ASR_PYTHON=resources\runtimes\parakeet\Scripts\python.exe
ASR_LANGUAGE=en
ASR_AUDIO_LANGUAGE=eng
ASR_AUDIO_STREAM_INDEX=
ASR_DEVICE=auto
ASR_QUANTIZATION=1
ASR_ALLOW_CPU_FALLBACK=0
ASR_KEEP_AUDIO=0
ASR_LONG_FORM_WINDOW_SECONDS=300
ASR_LONG_FORM_OVERLAP_SECONDS=2
ASR_VAD_FILTER=1
ASR_VAD_MIN_SILENCE_MS=600
ASR_BEAM_SIZE=5
ASR_CONDITION_ON_PREVIOUS_TEXT=1
ASR_OUTPUT_DIR=
FFMPEG_PATH=ffmpeg
FFPROBE_PATH=ffprobe
WEB_HOST=127.0.0.1
WEB_PORT=7862
WEB_JOB_DB=resources\state\jobs.sqlite3
RUN_LOCAL_ASR_TEST=0
ASR_TEST_MEDIA=
ASR_TEST_LANGUAGE=en
RUN_BATCH_ASR_TEST=0
ASR_BATCH_TEST_MEDIA=
RUN_MULTITRACK_ASR_TEST=0
ASR_MULTITRACK_MEDIA=
ASR_MULTITRACK_RU_STREAM_INDEX=0
ASR_MULTITRACK_EN_STREAM_INDEX=1
HF_HOME=D:\Projects\-ai-cache\huggingface
HUGGINGFACE_HUB_CACHE=D:\Projects\-ai-cache\huggingface\hub
```

Значения являются локальными настройками. `ASR_ALIGNER` принимает `none` или
`qwen3-forced-aligner`; второй вариант требует `ASR_ALIGNER_MODEL_PATH` и принимает общий
ASR transcript любого backend-а. `ASR_WORKER_PYTHON` переопределяет Python worker-а выбранного
ASR backend-а, а `ASR_ALIGNER_WORKER_PYTHON` независимо задаёт Python aligner worker-а.
`QWEN_ASR_PYTHON` и `PARAKEET_ASR_PYTHON` остаются backend-специфичными путями по умолчанию.

При `ASR_BACKEND=transformers` укажите
`ASR_MODEL_PATH=D:\Projects\-ai\+automatic-speech-recognition\whisper-large-v3`; для
Parakeet и Qwen используйте пути checkpoint-ов из раздела «Локальные модели».
Long-form окно, overlap, VAD, beam и контекст входят в fingerprint faster-whisper-кеша.
`FFMPEG_PATH` используется для подготовки и декодирования Transformers; faster-whisper
декодирует подготовленный FLAC через PyAV. `WEB_HOST` принимает только loopback-адрес;
внешний bind и порт вне диапазона 1–65535 отклоняются. `WEB_JOB_DB` задаёт SQLite-файл
состояния web-задач. TTL очистки служебных job-workspace сейчас фиксирован кодом и не является
переменной окружения. Секреты не нужны.

## CLI

Один файл:

```powershell
python -m speech_to_sub --input "D:\Media\lecture.mp4"
```

Папка:

```powershell
python -m speech_to_sub --input "D:\Media\Lectures" --recursive
```

Явный faster-whisper long-form профиль:

```powershell
python -m speech_to_sub `
  --input "D:\Media\lecture.mkv" `
  --backend faster-whisper `
  --model-path "D:\Projects\-ai\+automatic-speech-recognition\whisper-large-v3-ct2" `
  --language auto `
  --long-form-window-seconds 300 `
  --long-form-overlap-seconds 2 `
  --vad-min-silence-ms 600
```

Для диагностики циклических повторов доступны `--no-vad` и
`--no-condition-on-previous-text`. Backend и модель можно менять в web UI; при выборе
известного backend путь checkpoint переключается автоматически.

Parakeet TDT v3:

```powershell
python -m speech_to_sub `
  --input "D:\Media\lecture.mkv" `
  --backend parakeet-tdt-v3 `
  --model-path "D:\Projects\-ai\+automatic-speech-recognition\parakeet-tdt-0.6b-v3" `
  --worker-python-path "resources\runtimes\parakeet\Scripts\python.exe" `
  --language ru
```

Qwen3-ASR с ForcedAligner:

```powershell
python -m speech_to_sub `
  --input "D:\Media\lecture.mkv" `
  --backend qwen3-asr `
  --model-path "D:\Projects\-ai\+automatic-speech-recognition\Qwen3-ASR-0.6B" `
  --aligner qwen3-forced-aligner `
  --aligner-model-path "D:\Projects\-ai\+automatic-speech-recognition\Qwen3-ForcedAligner-0.6B" `
  --worker-python-path "resources\runtimes\qwen\Scripts\python.exe" `
  --aligner-worker-python-path "resources\runtimes\qwen\Scripts\python.exe" `
  --language en
```

При сочетании Parakeet с Qwen aligner укажите Parakeet Python через `--worker-python-path`,
а Qwen Python независимо через `--aligner-worker-python-path`.

Явный аудиопоток и CPU fallback:

```powershell
python -m speech_to_sub `
  --input "D:\Media\lecture.mp4" `
  --language en `
  --audio-stream-index 0 `
  --allow-cpu-fallback `
  --verbose
```

## Веб-интерфейс

![web.png](./resources/assets/web.png)

Запуск:

```powershell
.\run_web.ps1
```

или:

```powershell
python -m speech_to_sub.web
```

Адрес по умолчанию: http://127.0.0.1:7862. Разрешены только `127.0.0.0/8`, `localhost` и
`::1`; удалённый режим без аутентификации намеренно отсутствует.

Интерфейс предназначен для локального рабочего стола. Кнопки выбора открывают системные
диалоги на машине backend; большие MP4 не копируются через multipart и не читаются целиком
в память браузера.

Пачка отображает:

- исходный путь и найденные аудиопотоки;
- выбранную дорожку;
- стадии probe/extract/ASR/align/write;
- прогресс и журнал;
- итоговые SRT/sidecar либо понятную ошибку.

Состояние хранится в `WEB_JOB_DB`. UI восстанавливает persisted queue после reload/restart,
поддерживает cancel и retry только `error`/`cancelled` элементов. Успешные результаты при
retry не запускаются повторно. Picker/refresh-запросы сериализуются и отменяют устаревшие
ответы. SSE reconnect использует `Last-Event-ID`/cursor и не повторяет уже принятые события.

## Тесты

Основной набор не загружает реальную модель:

```powershell
.\.venv\Scripts\python.exe -B -m pytest
```

Browser E2E с изолированным fake ASR worker:

```powershell
npm test
```

Тяжёлый локальный тест включается только явно и пишет результаты во временный каталог pytest:

```powershell
$env:RUN_LOCAL_ASR_TEST = "1"
$env:ASR_TEST_MEDIA = "D:\Media\short-sample.mp4"
$env:ASR_TEST_LANGUAGE = "en"
.\.venv\Scripts\python.exe -B -m pytest -m integration
```

`ASR_TEST_MEDIA` лучше указывать на короткий репрезентативный фрагмент. Без флага тяжёлый
тест пропускается и модель не загружается.

Для opt-in проверки реальной пачки `done/done/error` и повторного кеш-прохода:

```powershell
$env:RUN_BATCH_ASR_TEST = "1"
$env:ASR_BATCH_TEST_MEDIA = "D:\Media\short-sample.mp4"
$env:ASR_TEST_LANGUAGE = "en"
.\.venv\Scripts\python.exe -B -m pytest tests\test_integration_batch_asr.py -m integration
```

Для opt-in проверки реального контейнера с двумя дорожками:

```powershell
$env:RUN_MULTITRACK_ASR_TEST = "1"
$env:ASR_MULTITRACK_MEDIA = "D:\Media\multi-track.mp4"
$env:ASR_MULTITRACK_RU_STREAM_INDEX = "0"
$env:ASR_MULTITRACK_EN_STREAM_INDEX = "1"
.\.venv\Scripts\python.exe -B -m pytest tests\test_integration_multitrack_asr.py -m integration
```

### SonarQube и coverage

В репозитории хранится безопасный шаблон `sonar-project.properties.example`. Рабочий файл
локален и не содержит token:

```powershell
Copy-Item -LiteralPath sonar-project.properties.example -Destination sonar-project.properties
$env:SONAR_TOKEN = "<локальный-token>"
```

Рекомендуемый `$fix-sonar` runner сначала запускает pytest-cov, создаёт `coverage.xml`, затем
выполняет свежий `sonar-scanner`, ждёт завершения CE task и получает issues, SECURITY,
duplication и coverage через API:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .codex\skills\fix-sonar\scripts\sonar_cycle.ps1
```

Для финальной проверки с нулём Sonar findings:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .codex\skills\fix-sonar\scripts\sonar_cycle.ps1 -RequireClean
```

`SONAR_HOST_URL` и `SONAR_TOKEN` можно задать в локальном `.env`; значения token не попадают
в properties, логи или `SONAR.md`. `coverage.xml` и `.coverage` являются локальными
артефактами и не добавляются в репозиторий. В SonarQube импортируется строковое и веточное
покрытие Python из pytest-cov. Статические HTML/CSS/JavaScript продолжают анализироваться,
но исключены только из строчной метрики coverage до появления LCOV; основной браузерный
сценарий проверяется Playwright. `tsconfig.sonar.json` ограничивает семантический анализ
JavaScript исходниками frontend и не позволяет scanner-у обходить изолированные ML-runtime-ы.

### Подтверждённый прогон макета

Для проверки были без изменения исходников созданы короткие аудиофрагменты из указанных
пользователем файлов:

| Язык | Контейнер | Результат |
|---|---|---|
| русский | MKV | валидные SRT и sidecar |
| русский | MP4 | валидные SRT и sidecar; тег дорожки `eng` не повлиял на явный `language=ru` |
| английский | MKV | валидные SRT и sidecar |
| английский | MP4 | валидные SRT и sidecar; `auto` определил `en`, повторный запуск взят из кеша без загрузки весов |
| русский + английский | multi-track MP4 | обе дорожки распознаны по явным индексам одним загруженным checkpoint |

На проверочной машине использовались Python 3.14, PyTorch 2.11.0/CUDA 12.8,
Transformers 5.6.2, faster-whisper 1.2.1 и CTranslate2 4.8.1. Числа скорости зависят от GPU
и длины файла и не являются обещанием
производительности. Отдельно проверены штатный `run_web.ps1`, health endpoint, SSE/reload,
ранний `409` для второй задачи и запрет внешнего bind. Исторический быстрый набор V1.1:
`89 passed, 3 skipped`; финальный быстрый набор V1.4 — `185 passed, 3 skipped`.
Три opt-in интеграционные проверки завершились как `3 passed`: одиночное медиа, реальная
пачка `done/done/error` с cache pass и multi-track RU/EN. Live batch
`done/done/error` завершился как `partial`, а второй проход дал
`cached/cached/error`. Playwright-сценарий reload/SSE завершился успешно.

### Подтверждённый long-form benchmark V1.2

На четырёх предоставленных RU/EN MP4/MKV faster-whisper получил RTF `0,132–0,219`, не создал
ни одного граничного дубля и сохранил peak RSS delta в диапазоне `3,42–3,66 ГиБ`, включая
часовой русский MKV. На одинаковом EN MP4 Transformers получил RTF `1,739` против `0,214`,
peak RSS delta `9,61` против `3,43 ГиБ` и VRAM delta `8439` против `3788 МиБ`.

Подробная методика, все показатели и JSON-артефакты находятся в
[docs/benchmarks/v1.2](docs/benchmarks/v1.2/README.md).

### Подтверждённые V1.3 и V1.4

V1.3 проверена unit/TestClient и Playwright-сценарием: SQLite переживает повторное открытие,
незавершённая задача после рестарта получает `interrupted`, cancel охватывает очередь и
безопасные точки текущего backend-а, retry строит новую задачу только для неуспешных файлов.
Source fingerprint сохраняется в файловом snapshot. Очистка затрагивает только устаревшие
служебные `resources/work/job-*` с lock-маркером; пользовательские исходники не удаляются.

V1.4 подтверждена реальными CUDA RU/EN smoke-прогонами Parakeet и Qwen. После устранения
граничных дублей Parakeet обработал русское аудио длительностью 375,520 с при RTF `0,073654`
и английское длительностью 338,152 с при RTF `0,077936`. Самостоятельные Qwen3-ASR и
ForcedAligner обработали английское аудио длительностью 338,152 с при RTF `0,383180`;
два ASR-сегмента до 180 секунд дали 832 монотонные словные метки, включая 44 слова после
300-й секунды. Единый short benchmark содержит
RU/EN JSON для всех четырёх backend-ов. Метрики охватывают RTF, RSS дерева процессов, VRAM,
coverage, повторы и структуру SRT. Без эталонных расшифровок WER/CER и абсолютная ошибка
таймингов не заявляются. Методика и артефакты: [docs/benchmarks/v1.4](docs/benchmarks/v1.4/README.md).

Compatibility spike WhisperX выполнен в изолированном Python 3.11, но production-интеграция
отложена: основной Python 3.14 не поддерживается, а локального воспроизводимого RU aligner,
pyannote pipeline и NLTK-ресурсов нет. Решение и условия возврата описаны в
[whisperx-spike.md](docs/benchmarks/v1.4/whisperx-spike.md); общий alignment registry остаётся
стабильной точкой расширения.

## Ограничения первой версии

- Облачные API не реализуются.
- Диаризация не реализована; forced alignment доступен через Qwen3 ForcedAligner.
- Модель не скачивается автоматически.
- Веб-интерфейс не предназначен для удалённого или многопользовательского доступа.
- Незавершённый тяжёлый инференс не продолжается с середины после рестарта: задача становится
  `interrupted` и может быть явно запущена через retry.
- Архивная копия исходного аудиобитстрима и mux SRT обратно в видео отложены.
- Без эталонных расшифровок benchmark не вычисляет WER/CER и абсолютную timestamp MAE.
- Transformers сохраняет риск единого float32 PCM-массива и на измеренном long-form кейсе
  работает медленнее реального времени; для длинных файлов используйте faster-whisper.
- Изолированные worker-ы требуют заранее созданных runtime-ов и локальных checkpoint-ов;
  автоматическая подготовка при обычном запуске запрещена.
- Qwen3 ForcedAligner требует определённый язык `en`/`ru`; длинные ASR-сегменты он сам
  делит на неперекрывающиеся части до 180 секунд.

Полные требования и критерии приёмки находятся в [PRD.md](PRD.md).
