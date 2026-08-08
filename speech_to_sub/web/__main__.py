from __future__ import annotations

import os

import uvicorn

from speech_to_sub.constants import DEFAULT_WEB_HOST, DEFAULT_WEB_PORT
from speech_to_sub.utils.env_utils import load_environment, validate_loopback_host


def main() -> None:
    """Запускает локальный веб-интерфейс без автоматической перезагрузки."""
    load_environment()
    host = validate_loopback_host(os.getenv("WEB_HOST", DEFAULT_WEB_HOST))
    port = _read_port(os.getenv("WEB_PORT"))
    uvicorn.run(
        "speech_to_sub.web.app:app",
        host=host,
        port=port,
        reload=False,
    )


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


if __name__ == "__main__":
    main()
