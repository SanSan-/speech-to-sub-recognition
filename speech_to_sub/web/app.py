from __future__ import annotations

import importlib
import ipaddress
import logging
import os
import threading
from pathlib import Path
from typing import Annotated, Any, Callable, Mapping
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from speech_to_sub import __version__
from speech_to_sub.alignment.registry import aligner_names
from speech_to_sub.asr.registry import backend_names
from speech_to_sub.constants import (
    DEFAULT_FASTER_WHISPER_MODEL_PATH,
    DEFAULT_PARAKEET_MODEL_PATH,
    DEFAULT_QWEN_ALIGNER_MODEL_PATH,
    DEFAULT_QWEN_MODEL_PATH,
    DEFAULT_TRANSFORMERS_MODEL_PATH,
    MAX_BATCH_PATHS,
    SUPPORTED_MEDIA_EXTENSIONS,
)
from speech_to_sub.models import ProcessingSettings
from speech_to_sub.utils.env_utils import (
    load_environment,
    settings_from_environment,
    validate_loopback_host,
)
from speech_to_sub.web.jobs import (
    JobBusyError,
    JobNotFoundError,
    JobRegistry,
    JobStateError,
)
from speech_to_sub.web.job_store import SQLiteJobStore
from speech_to_sub.web.picker import (
    PickSelection,
    PickerError,
    collect_media_paths,
    pick_paths,
)
from speech_to_sub.web.schemas import (
    PickRequest,
    ProcessingSettingsPayload,
    RefreshRequest,
    TranscribeRequest,
)

ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
PROJECT_ROOT = ROOT_DIR.parents[1]
DEFAULT_JOB_DB = PROJECT_ROOT / "resources" / "state" / "jobs.sqlite3"
DEFAULT_WEB_PORT = 7862
SERVICE_MODULE = "speech_to_sub.service"
JOB_NOT_FOUND_DETAIL = "Задача не найдена."

_BAD_REQUEST_RESPONSE = {"description": "Некорректные входные данные."}
_JOB_NOT_FOUND_RESPONSE = {"description": JOB_NOT_FOUND_DETAIL}
_JOB_CONFLICT_RESPONSE = {"description": "Операция конфликтует с состоянием задачи."}
_INTERNAL_ERROR_RESPONSE = {"description": "Внутренняя ошибка локального сервиса."}

load_environment()
logger = logging.getLogger(__name__)


def _job_database_path() -> Path:
    configured = os.environ.get("WEB_JOB_DB", "").strip()
    path = Path(configured).expanduser() if configured else DEFAULT_JOB_DB
    return path if path.is_absolute() else PROJECT_ROOT / path


class ServiceAdapter:
    """Единая ленивая точка интеграции web со слоем batch service."""

    @staticmethod
    def get_preflight_status(settings: dict[str, Any]) -> dict[str, Any]:
        module = importlib.import_module(SERVICE_MODULE)
        result = module.get_preflight_status(settings)
        if not isinstance(result, Mapping):
            raise TypeError("Сервис должен вернуть результат предварительной проверки.")
        return dict(result)

    @staticmethod
    def build_items(paths: list[str], settings: dict[str, Any]) -> list[dict[str, Any]]:
        module = importlib.import_module(SERVICE_MODULE)
        result = module.build_items(paths, settings)
        if not isinstance(result, list) or any(not isinstance(item, Mapping) for item in result):
            raise TypeError("Сервис должен вернуть список карточек файлов.")
        return [dict(item) for item in result]

    @staticmethod
    def process_paths(
        paths: list[str],
        settings: dict[str, Any],
        emit_event: Any,
        log: logging.Logger,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        module = importlib.import_module(SERVICE_MODULE)
        result = module.process_paths(
            paths,
            settings,
            emit_event,
            log.info,
            cancel_check=cancel_check,
        )
        if not isinstance(result, list) or any(not isinstance(item, Mapping) for item in result):
            raise TypeError("Сервис должен вернуть список результатов файлов.")
        return [dict(item) for item in result]

    @staticmethod
    def unload() -> None:
        module = importlib.import_module("speech_to_sub.asr.registry")
        unload_backends = getattr(module, "unload_backends")
        unload_backends()
        aligner_module = importlib.import_module("speech_to_sub.alignment.registry")
        getattr(aligner_module, "unload_aligners")()

    @staticmethod
    def cleanup_workspaces() -> list[Path]:
        module = importlib.import_module(SERVICE_MODULE)
        return list(module.cleanup_stale_workspaces())


service_api = ServiceAdapter()
try:
    service_api.cleanup_workspaces()
except Exception as exc:
    logger.warning("Не удалось очистить устаревшие рабочие каталоги: %s", exc)
job_registry = JobRegistry(store=SQLiteJobStore(_job_database_path()))
_picker_refresh_lock = threading.Lock()

app = FastAPI(title="Speech to Sub Recognition", version=__version__)
app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")


@app.middleware("http")
async def protect_local_http_access(request: Request, call_next: Any) -> Any:
    """Блокирует удалённый доступ и cross-origin изменения локального API."""
    client_host = request.client.host if request.client is not None else ""
    if not _is_trusted_local_request(request, client_host=client_host):
        return JSONResponse(
            status_code=403,
            content={"detail": "Локальный API доступен только через loopback."},
        )
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        if not _is_trusted_state_change(request, client_host=client_host):
            return JSONResponse(
                status_code=403,
                content={"detail": "Изменение состояния разрешено только локальному origin."},
            )
    return await call_next(request)


@app.get("/")
def index() -> FileResponse:
    """Возвращает локальную страницу пакетного распознавания."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Проверяет команды и путь модели без загрузки весов."""
    settings, settings_error = _environment_settings()
    try:
        preflight = service_api.get_preflight_status(settings.model_dump(mode="json"))
    except Exception as exc:
        preflight = {
            "status": "degraded",
            "backend": {
                "id": settings.backend,
                "available": False,
                "error": str(exc) or exc.__class__.__name__,
            },
        }
    payload: dict[str, Any] = {
        "service": "speech-to-sub-recognition",
        "version": __version__,
        **preflight,
    }
    if settings_error is not None:
        payload["status"] = "degraded"
        payload["settings_error"] = settings_error
    return payload


@app.get("/api/ui-config")
def ui_config() -> dict[str, Any]:
    """Возвращает несекретные значения и варианты интерфейса."""
    settings, settings_error = _environment_settings()
    backend_labels = {
        "transformers": "Transformers / PyTorch",
        "faster-whisper": "faster-whisper / CTranslate2",
        "parakeet-tdt-v3": "Parakeet TDT v3",
        "qwen3-asr": "Qwen3-ASR 0.6B",
    }
    backend_model_paths = {
        "transformers": str(DEFAULT_TRANSFORMERS_MODEL_PATH),
        "faster-whisper": str(DEFAULT_FASTER_WHISPER_MODEL_PATH),
        "parakeet-tdt-v3": str(DEFAULT_PARAKEET_MODEL_PATH),
        "qwen3-asr": str(DEFAULT_QWEN_MODEL_PATH),
    }
    payload: dict[str, Any] = {
        "defaults": settings.model_dump(mode="json"),
        "backends": [
            {
                "value": name,
                "label": backend_labels.get(name, name),
                "model_path": backend_model_paths.get(name, ""),
            }
            for name in backend_names()
        ],
        "aligners": [
            {
                "value": name,
                "label": (
                    "Без дополнительного выравнивания"
                    if name == "none"
                    else "Qwen3 ForcedAligner 0.6B"
                ),
                "model_path": (
                    str(DEFAULT_QWEN_ALIGNER_MODEL_PATH)
                    if name == "qwen3-forced-aligner"
                    else ""
                ),
                "compatible_backends": (
                    list(backend_names())
                ),
            }
            for name in aligner_names()
        ],
        "devices": [
            {"value": "auto", "label": "Автоматически"},
            {"value": "cuda", "label": "CUDA"},
            {"value": "cpu", "label": "CPU"},
        ],
        "languages": [
            {"value": "en", "label": "Английский"},
            {"value": "ru", "label": "Русский"},
            {"value": "auto", "label": "Автоопределение"},
        ],
        "supported_extensions": sorted(SUPPORTED_MEDIA_EXTENSIONS),
    }
    if settings_error is not None:
        payload["settings_warning"] = settings_error
    return payload


@app.get("/api/active-job")
def active_job() -> dict[str, Any]:
    """Возвращает текущую или последнюю завершённую задачу."""
    return job_registry.current_snapshot()


@app.get("/api/jobs")
def list_jobs(
    limit: Annotated[int, Query(ge=1, le=100)] = 16,
) -> dict[str, Any]:
    """Возвращает bounded-историю сохранённых пакетных задач."""
    return {"jobs": job_registry.list_snapshots(limit=limit)}


@app.get("/api/jobs/{job_id}", responses={404: _JOB_NOT_FOUND_RESPONSE})
def get_job(job_id: str) -> dict[str, Any]:
    """Возвращает persisted snapshot и историю событий конкретной задачи."""
    try:
        return job_registry.snapshot(job_id, include_events=True)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=JOB_NOT_FOUND_DETAIL) from exc


@app.post(
    "/api/jobs/{job_id}/cancel",
    responses={404: _JOB_NOT_FOUND_RESPONSE, 409: _JOB_CONFLICT_RESPONSE},
)
def cancel_job(job_id: str) -> dict[str, Any]:
    """Запрашивает безопасную кооперативную отмену активной задачи."""
    try:
        snapshot = job_registry.cancel(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=JOB_NOT_FOUND_DETAIL) from exc
    except JobStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "job_id": job_id,
        "status": "cancelling",
        "latest_event_id": snapshot["latest_event_id"],
    }


@app.post(
    "/api/jobs/{job_id}/retry",
    responses={404: _JOB_NOT_FOUND_RESPONSE, 409: _JOB_CONFLICT_RESPONSE},
)
def retry_job(job_id: str) -> dict[str, Any]:
    """Запускает заново только failed/cancelled/interrupted элементы."""
    try:
        reservation = job_registry.reserve_start()
    except JobBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        job = job_registry.retry(
            job_id,
            service_api.process_paths,
            item_builder=service_api.build_items,
            reservation_token=reservation,
        )
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=JOB_NOT_FOUND_DETAIL) from exc
    except (JobBusyError, JobStateError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        job_registry.release_reservation(reservation)
    return {
        "job_id": job.job_id,
        "retry_of": job_id,
        "items": job.snapshot(active=True)["items"],
    }


@app.post(
    "/api/pick",
    responses={400: _BAD_REQUEST_RESPONSE, 500: _INTERNAL_ERROR_RESPONSE},
)
def pick(payload: PickRequest) -> dict[str, Any]:
    """Открывает локальный picker и строит карточки выбранных файлов."""
    with _picker_refresh_lock:
        try:
            selection = pick_paths(payload.kind, recursive=payload.settings.recursive)
        except PickerError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        settings = payload.settings.model_dump(mode="json")
        paths = [str(path) for path in selection.paths]
        items = _build_items(paths, settings) if paths else []
    return _selection_payload(selection, items)


@app.post("/api/refresh", responses={400: _BAD_REQUEST_RESPONSE})
def refresh(payload: RefreshRequest) -> dict[str, Any]:
    """Повторно строит карточки без запуска тяжёлого ASR."""
    with _picker_refresh_lock:
        paths = _normalize_source_paths(payload.paths)
        settings = payload.settings.model_dump(mode="json")
        return {"items": _build_items(paths, settings)}


@app.post(
    "/api/transcribe",
    responses={400: _BAD_REQUEST_RESPONSE, 409: _JOB_CONFLICT_RESPONSE},
)
def transcribe(payload: TranscribeRequest) -> dict[str, Any]:
    """Запускает единственную фоновую пакетную задачу."""
    source_paths = _normalize_source_paths(payload.paths)
    settings = payload.settings.model_dump(mode="json")
    try:
        reservation = job_registry.reserve_start()
    except JobBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        items = _build_items(source_paths, settings)
        paths = _media_paths_from_items(items)
        job = job_registry.start(
            paths=paths,
            settings=settings,
            items=items,
            processor=service_api.process_paths,
            source_paths=source_paths,
            reservation_token=reservation,
        )
    except JobBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        job_registry.release_reservation(reservation)
    return {"job_id": job.job_id, "items": items}


@app.get("/api/stream/{job_id}", responses={404: _JOB_NOT_FOUND_RESPONSE})
def stream(
    job_id: str,
    cursor: Annotated[int, Query(ge=0)] = 0,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    """Передаёт SSE-историю и гарантированно завершает terminal-подключение."""
    try:
        after_event_id = max(cursor, _parse_event_id(last_event_id))
        events = job_registry.iter_sse(job_id, after_event_id=after_event_id)
        job_registry.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=JOB_NOT_FOUND_DETAIL) from exc
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post(
    "/api/unload",
    responses={409: _JOB_CONFLICT_RESPONSE, 500: _INTERNAL_ERROR_RESPONSE},
)
def unload_models() -> dict[str, str]:
    """Выгружает локальную модель, если пакетный worker свободен."""
    try:
        reservation = job_registry.reserve_unload()
    except JobBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        try:
            service_api.unload()
        except Exception as exc:
            logger.exception("Не удалось выгрузить локальную модель: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Не удалось выгрузить локальную модель: {exc}",
            ) from exc
    finally:
        job_registry.release_reservation(reservation)
    return {"status": "ok", "message": "Локальная модель выгружена."}


def _build_items(paths: list[str], settings: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        expanded_paths = _expand_media_sources(
            paths,
            recursive=bool(settings.get("recursive")),
        )
        items = service_api.build_items(expanded_paths, settings)
        if len(items) > MAX_BATCH_PATHS:
            raise ValueError(f"За один запуск допускается не более {MAX_BATCH_PATHS} файлов.")
        return items
    except Exception as exc:
        logger.warning("Не удалось подготовить карточки медиафайлов: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось подготовить файлы: {exc}",
        ) from exc


def _expand_media_sources(paths: list[str], *, recursive: bool) -> list[str]:
    """Раскрывает каталоги с лимитом до запуска пофайлового ffprobe."""
    expanded: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        candidates = collect_media_paths(path, recursive=recursive) if path.is_dir() else (path,)
        for candidate in candidates:
            value = str(candidate)
            expanded.setdefault(value.casefold(), value)
            if len(expanded) > MAX_BATCH_PATHS:
                raise ValueError(
                    f"За один запуск допускается не более {MAX_BATCH_PATHS} файлов."
                )
    return list(expanded.values())


def _normalize_media_paths(raw_paths: list[str]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser().resolve(strict=False)
        if path.suffix.casefold() not in SUPPORTED_MEDIA_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый формат медиафайла: {path}",
            )
        key = str(path).casefold()
        if key not in seen:
            seen.add(key)
            paths.append(str(path))
    return paths


def _normalize_source_paths(raw_paths: list[str]) -> list[str]:
    """Нормализует входные медиафайлы и каталоги локального batch UI."""
    paths: list[str] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser().resolve(strict=False)
        if not path.is_dir() and path.suffix.casefold() not in SUPPORTED_MEDIA_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый входной путь: {path}",
            )
        key = str(path).casefold()
        if key not in seen:
            seen.add(key)
            paths.append(str(path))
    return paths


def _media_paths_from_items(items: list[dict[str, Any]]) -> list[str]:
    """Извлекает фактические файлы после раскрытия выбранных каталогов."""
    item_paths = [str(item.get("path") or "") for item in items]
    if not item_paths:
        raise HTTPException(status_code=400, detail="Не найдено поддерживаемых медиафайлов.")
    return _normalize_media_paths(item_paths)


def _selection_payload(
    selection: PickSelection,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "mode": selection.mode,
        "path": str(selection.folder) if selection.folder else "",
        "paths": [str(path) for path in selection.paths],
        "items": items,
        "cancelled": selection.folder is None and not selection.paths,
    }


def _environment_settings() -> tuple[ProcessingSettingsPayload, str | None]:
    try:
        settings = settings_from_environment()
        return ProcessingSettingsPayload.model_validate(settings.to_dict()), None
    except Exception:
        fallback = ProcessingSettings()
        warning = "Локальные настройки окружения некорректны."
        return ProcessingSettingsPayload.model_validate(fallback.to_dict()), warning


def _parse_event_id(value: str | None) -> int:
    if not value:
        return 0
    try:
        return max(0, int(value))
    except ValueError:
        return 0


def _is_trusted_state_change(request: Request, *, client_host: str) -> bool:
    """Проверяет loopback Host и, если он передан, точное совпадение Origin."""
    if not _is_trusted_local_request(request, client_host=client_host):
        return False
    allow_testserver = client_host.casefold() == "testclient"
    host = _local_authority(
        request.headers.get("host", ""),
        scheme=request.url.scheme,
        allow_testserver=allow_testserver,
    )
    if host is None:
        return False
    origin = request.headers.get("origin")
    if origin is None:
        return True
    return _origin_authority(origin, allow_testserver=allow_testserver) == host


def _is_trusted_local_request(request: Request, *, client_host: str) -> bool:
    """Разрешает HTTP-доступ только фактическому loopback-клиенту и Host."""
    allow_testserver = client_host.casefold() == "testclient"
    if not allow_testserver and _loopback_hostname(
        client_host,
        allow_testserver=False,
    ) is None:
        return False
    return _local_authority(
        request.headers.get("host", ""),
        scheme=request.url.scheme,
        allow_testserver=allow_testserver,
    ) is not None


def _local_authority(
    value: str,
    *,
    scheme: str,
    allow_testserver: bool,
) -> tuple[str, str, int] | None:
    """Нормализует доверенный loopback authority с учётом стандартного порта."""
    if scheme not in {"http", "https"} or not value or "@" in value:
        return None
    try:
        parsed = urlsplit(f"//{value}")
        hostname = parsed.hostname or ""
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError:
        return None
    if parsed.path or parsed.query or parsed.fragment:
        return None
    normalized_host = _loopback_hostname(hostname, allow_testserver=allow_testserver)
    return (scheme, normalized_host, port) if normalized_host is not None else None


def _origin_authority(
    value: str,
    *,
    allow_testserver: bool,
) -> tuple[str, str, int] | None:
    """Разбирает Origin без доверия к forwarded-заголовкам."""
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            return None
        authority = parsed.netloc
    except ValueError:
        return None
    return _local_authority(
        authority,
        scheme=parsed.scheme,
        allow_testserver=allow_testserver,
    )


def _loopback_hostname(value: str, *, allow_testserver: bool) -> str | None:
    hostname = value.strip().casefold()
    if allow_testserver and hostname == "testserver":
        return hostname
    if hostname == "localhost":
        return hostname
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return None
    return address.compressed if address.is_loopback else None


def _parse_web_port(value: str | int) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("WEB_PORT должен быть целым числом от 1 до 65535.") from exc
    if not 1 <= port <= 65_535:
        raise ValueError("WEB_PORT должен быть в диапазоне от 1 до 65535.")
    return port


def main() -> None:
    """Запускает локальный Python entrypoint с проверенным портом."""
    import uvicorn

    host = validate_loopback_host(os.environ.get("WEB_HOST", "127.0.0.1"))
    port = _parse_web_port(os.environ.get("WEB_PORT", DEFAULT_WEB_PORT))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
