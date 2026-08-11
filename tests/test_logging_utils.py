from __future__ import annotations

import logging
from pathlib import Path

from speech_to_sub.utils import logging_utils


def test_web_logging_is_utf8_rotating_and_captures_app_and_uvicorn(
    tmp_path: Path,
    monkeypatch,
) -> None:
    logger_names = ("speech_to_sub", "uvicorn", "uvicorn.error", "uvicorn.access")
    saved = {
        name: (
            list(logging.getLogger(name).handlers),
            logging.getLogger(name).level,
            logging.getLogger(name).propagate,
        )
        for name in logger_names
    }
    monkeypatch.setattr(logging_utils, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(logging_utils, "_WEB_LOG_MAX_BYTES", 320)
    monkeypatch.setattr(logging_utils, "_WEB_LOG_BACKUP_COUNT", 2)
    monkeypatch.setattr(logging_utils, "_web_file_handler", None)
    handler = None
    try:
        logging_utils.setup_web_logging()
        handler = logging_utils._web_log_handler()
        assert handler.encoding.casefold().replace("-", "") == "utf8"
        assert handler.maxBytes == 320
        assert handler.backupCount == 2
        for name in logger_names:
            assert handler in logging.getLogger(name).handlers

        app_logger = logging.getLogger("speech_to_sub.web.app")
        job_logger = logging.getLogger("speech_to_sub.web.job.test")
        access_logger = logging.getLogger("uvicorn.access")
        for index in range(20):
            app_logger.info("Подготовка каталога, шаг %s: данные в UTF-8.", index)
        job_logger.info("Задание передало ход обработки в файловый журнал.")
        access_logger.info("Локальный HTTP-запрос завершён.")
        handler.flush()

        paths = sorted(tmp_path.glob("speech_to_sub-web.log*"))
        assert 2 <= len(paths) <= 3
        combined = "".join(path.read_text(encoding="utf-8") for path in paths)
        assert "Подготовка каталога" in combined
        assert combined.count("Задание передало ход обработки") == 1
        assert "Локальный HTTP-запрос" in combined
        assert all(not path.read_bytes().startswith(b"\xef\xbb\xbf") for path in paths)
    finally:
        if handler is not None:
            handler.close()
        for name, (handlers, level, propagate) in saved.items():
            target = logging.getLogger(name)
            target.handlers = handlers
            target.setLevel(level)
            target.propagate = propagate


def test_web_logging_uses_release_rotation_limits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(logging_utils, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(logging_utils, "_web_file_handler", None)

    handler = logging_utils._web_log_handler()
    try:
        assert handler.maxBytes == 5 * 1024 * 1024
        assert handler.backupCount == 3
    finally:
        handler.close()
