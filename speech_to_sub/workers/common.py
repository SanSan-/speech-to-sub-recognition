"""Общие части протокола изолированных локальных worker-процессов."""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, Protocol

from speech_to_sub.asr.external_worker import decode_frame, encode_frame

WorkerProgressCallback = Callable[[int], None]
WorkerAction = Callable[
    [Mapping[str, Any], WorkerProgressCallback | None],
    dict[str, Any],
]


class WorkerRuntime(Protocol):
    """Минимальный контракт runtime для общего командного протокола."""

    def preflight(self, payload: Mapping[str, Any]) -> dict[str, Any]: ...

    def unload(self) -> dict[str, Any]: ...


class WorkerRequestHandler(Protocol):
    """Контракт обработчика одного NDJSON-запроса."""

    def __call__(
        self,
        runtime: Any,
        request: Mapping[str, Any],
        progress_callback: WorkerProgressCallback | None = None,
    ) -> tuple[dict[str, Any], bool]: ...


def handle_worker_request(
    runtime: WorkerRuntime,
    request: Mapping[str, Any],
    progress_callback: WorkerProgressCallback | None,
    *,
    action_name: str,
    worker_name: str,
) -> tuple[dict[str, Any], bool]:
    """Выполняет стандартные команды и одну специализированную команду worker-а."""
    request_id = str(request.get("id") or "")
    command = str(request.get("command") or "").strip().casefold()
    raw_payload = request.get("payload")
    payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
    try:
        result, should_stop = _execute_worker_command(
            runtime,
            command,
            payload,
            progress_callback,
            action_name=action_name,
            worker_name=worker_name,
        )
        return {"id": request_id, "ok": True, "result": result}, should_stop
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return {
            "id": request_id,
            "ok": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }, False


def make_worker_request_handler(
    *,
    action_name: str,
    worker_name: str,
) -> WorkerRequestHandler:
    """Связывает общее выполнение команд с типом конкретного worker-а."""
    return partial(
        handle_worker_request,
        action_name=action_name,
        worker_name=worker_name,
    )


def run_worker_loop(runtime: Any, handler: WorkerRequestHandler) -> int:
    """Читает framed NDJSON из stdin до команды shutdown или EOF."""
    _configure_streams()
    for line in sys.stdin:
        request = decode_frame(line)
        if request is None:
            continue
        request_id = str(request.get("id") or "")
        progress_callback = partial(_emit_progress, request_id)
        response, should_stop = handler(runtime, request, progress_callback)
        sys.stdout.write(encode_frame(response))
        sys.stdout.flush()
        if should_stop:
            break
    runtime.unload()
    return 0


def resolve_device(
    torch: Any,
    requested_device: str,
    *,
    device_name: str,
) -> tuple[str, str, Any]:
    """Выбирает CUDA/CPU и безопасный dtype для Qwen runtime."""
    requested = requested_device.strip().casefold()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Устройство {device_name} должно быть auto, cuda или cpu.")
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("Запрошена CUDA, но PyTorch worker-а не обнаружил CUDA-устройство.")
    device = "cuda" if cuda_available and requested != "cpu" else "cpu"
    if device == "cpu":
        return device, "float32", torch.float32
    supports_bfloat16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if supports_bfloat16:
        return device, "bfloat16", torch.bfloat16
    return device, "float16", torch.float16


def _execute_worker_command(
    runtime: WorkerRuntime,
    command: str,
    payload: Mapping[str, Any],
    progress_callback: WorkerProgressCallback | None,
    *,
    action_name: str,
    worker_name: str,
) -> tuple[dict[str, Any], bool]:
    if command == "ping":
        return {"protocol": 1}, False
    if command == "preflight":
        return runtime.preflight(payload), False
    if command == action_name:
        action: WorkerAction = getattr(runtime, action_name)
        return action(payload, progress_callback), False
    if command == "unload":
        return runtime.unload(), False
    if command == "shutdown":
        result = runtime.unload()
        result["shutdown"] = True
        return result, True
    raise ValueError(f"Неизвестная команда {worker_name} worker-а: {command or '?'}")


def _emit_progress(request_id: str, value: int) -> None:
    sys.stdout.write(
        encode_frame({"id": request_id, "event": "progress", "progress": value})
    )
    sys.stdout.flush()


def _configure_streams() -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="strict")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="strict", write_through=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", write_through=True)


__all__ = [
    "WorkerProgressCallback",
    "handle_worker_request",
    "make_worker_request_handler",
    "resolve_device",
    "run_worker_loop",
]
