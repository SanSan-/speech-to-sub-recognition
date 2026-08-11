from __future__ import annotations

import logging
import os

import uvicorn

from speech_to_sub.constants import DEFAULT_WEB_HOST, DEFAULT_WEB_PORT
from speech_to_sub.utils.env_utils import load_environment, validate_loopback_host
from speech_to_sub.utils.logging_utils import (
    setup_web_logging,
    web_runtime_logging_scope,
)


def main() -> None:
    """Запускает локальный веб-интерфейс с единым журналом жизненного цикла."""
    with web_runtime_logging_scope():
        _run_web_service()


def _run_web_service() -> None:
    """Проверяет параметры и удерживает внутренний признак до остановки Uvicorn."""
    load_environment()
    host = validate_loopback_host(os.getenv("WEB_HOST", DEFAULT_WEB_HOST))
    port = _read_port(os.getenv("WEB_PORT"))
    reload_enabled = _read_reload(os.getenv("WEB_RELOAD"))
    logger = logging.getLogger(__name__) if reload_enabled else setup_web_logging()
    process_id = os.getpid()
    if not reload_enabled:
        logger.info("Запрошен запуск локального веб-сервиса, PID=%s.", process_id)
    try:
        if not reload_enabled:
            logger.info(
                "Запускается локальный веб-сервис на %s:%s, reload=%s, PID=%s.",
                host,
                port,
                reload_enabled,
                process_id,
            )
        uvicorn.run(
            "speech_to_sub.web.app:app",
            host=host,
            port=port,
            reload=reload_enabled,
        )
    except Exception:
        if not reload_enabled:
            logger.exception(
                "Локальный веб-сервис аварийно завершён, PID=%s.", process_id
            )
        raise
    finally:
        if not reload_enabled:
            logger.info("Локальный веб-сервис завершён, PID=%s.", process_id)


def _read_port(raw_value: str | None) -> int:
    if not raw_value:
        return DEFAULT_WEB_PORT
    try:
        port = int(raw_value)
    except ValueError as exc:
        raise ValueError("WEB_PORT должен быть целым числом.") from exc
    if not 1 <= port <= 65_535:
        raise ValueError("WEB_PORT должен находиться в диапазоне 1..65535.")
    return port


def _read_reload(raw_value: str | None) -> bool:
    if not raw_value:
        return False
    normalized = raw_value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("WEB_RELOAD должен быть логическим значением.")


if __name__ == "__main__":
    main()
