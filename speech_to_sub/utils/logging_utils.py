from __future__ import annotations

import logging

from speech_to_sub.constants import LOGS_DIR


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Настраивает консольный и файловый журнал приложения."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("speech_to_sub")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        file_handler = logging.FileHandler(
            LOGS_DIR / "speech_to_sub.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(console)
        logger.addHandler(file_handler)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger

