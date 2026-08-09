from __future__ import annotations

import ipaddress
import math
import os
from pathlib import Path

from speech_to_sub.asr.registry import backend_names
from speech_to_sub.alignment.registry import aligner_names
from speech_to_sub.constants import (
    CLOUD_ASR_BACKENDS,
    DEFAULT_ASR_BACKEND,
    DEFAULT_ALIGNER,
    DEFAULT_AUDIO_LANGUAGE,
    DEFAULT_BACKEND_MODEL_PATHS,
    DEFAULT_BEAM_SIZE,
    DEFAULT_LANGUAGE,
    DEFAULT_LINE_LENGTH_GAP,
    DEFAULT_LONG_FORM_OVERLAP_SECONDS,
    DEFAULT_LONG_FORM_WINDOW_SECONDS,
    DEFAULT_MAX_CHARS_PER_LINE,
    DEFAULT_MAX_CPS,
    DEFAULT_MODEL_PATH,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_QWEN_ALIGNER_MODEL_PATH,
    DEFAULT_VAD_MIN_SILENCE_MS,
    MAX_LINE_LENGTH_GAP,
    SUPPORTED_OPENAI_MODELS,
)
from speech_to_sub.exceptions import ValidationError
from speech_to_sub.models import ProcessingSettings


def load_environment(env_path: Path | None = None) -> bool:
    """Загружает `.env`, не заменяя уже заданные переменные процесса."""
    path = env_path or Path.cwd() / ".env"
    if not path.exists():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return bool(load_dotenv(path, override=False))


def settings_from_environment() -> ProcessingSettings:
    """Возвращает настройки запуска из переменных окружения."""
    output_raw = os.getenv("ASR_OUTPUT_DIR", "").strip()
    backend = _read_choice(
        "ASR_BACKEND",
        DEFAULT_ASR_BACKEND,
        frozenset(backend_names()),
    )
    default_model_path = _default_model_path(backend)
    aligner = _read_choice(
        "ASR_ALIGNER",
        DEFAULT_ALIGNER,
        frozenset(aligner_names()),
    )
    aligner_path_raw = os.getenv("ASR_ALIGNER_MODEL_PATH", "").strip()
    worker_python_raw = os.getenv("ASR_WORKER_PYTHON", "").strip()
    aligner_worker_python_raw = os.getenv("ASR_ALIGNER_WORKER_PYTHON", "").strip()
    if aligner_path_raw:
        aligner_model_path = Path(aligner_path_raw)
    elif aligner == "qwen3-forced-aligner":
        aligner_model_path = DEFAULT_QWEN_ALIGNER_MODEL_PATH
    else:
        aligner_model_path = None
    return ProcessingSettings(
        backend=backend,
        model_path=Path(os.getenv("ASR_MODEL_PATH", str(default_model_path))),
        auto_download_model=_read_bool("ASR_AUTO_DOWNLOAD_MODEL", True),
        allow_cloud_processing=_read_bool("ASR_ALLOW_CLOUD_PROCESSING", False),
        openai_model=_read_choice(
            "OPENAI_TRANSCRIPTION_MODEL",
            DEFAULT_OPENAI_MODEL,
            SUPPORTED_OPENAI_MODELS,
        ),
        aligner=aligner,
        aligner_model_path=aligner_model_path,
        worker_python_path=Path(worker_python_raw) if worker_python_raw else None,
        aligner_worker_python_path=(
            Path(aligner_worker_python_raw) if aligner_worker_python_raw else None
        ),
        language=_read_choice(
            "ASR_LANGUAGE",
            DEFAULT_LANGUAGE,
            frozenset({"en", "ru", "auto"}),
        ),
        audio_language=(
            os.getenv("ASR_AUDIO_LANGUAGE", DEFAULT_AUDIO_LANGUAGE).strip()
            or DEFAULT_AUDIO_LANGUAGE
        ),
        audio_stream_index=_read_optional_non_negative_int("ASR_AUDIO_STREAM_INDEX"),
        device=_read_choice(
            "ASR_DEVICE",
            "auto",
            frozenset({"auto", "cuda", "cpu"}),
        ),
        quantization_enabled=_read_bool("ASR_QUANTIZATION", True),
        allow_cpu_fallback=_read_bool("ASR_ALLOW_CPU_FALLBACK", False),
        keep_audio=_read_bool("ASR_KEEP_AUDIO", False),
        max_chars_per_line=_read_positive_int(
            "ASR_MAX_CHARS_PER_LINE",
            DEFAULT_MAX_CHARS_PER_LINE,
        ),
        line_length_gap=_read_non_negative_int(
            "ASR_LINE_LENGTH_GAP",
            DEFAULT_LINE_LENGTH_GAP,
            maximum=MAX_LINE_LENGTH_GAP,
        ),
        max_cps=_read_positive_float("ASR_MAX_CPS", DEFAULT_MAX_CPS),
        long_form_window_seconds=_read_positive_int(
            "ASR_LONG_FORM_WINDOW_SECONDS",
            DEFAULT_LONG_FORM_WINDOW_SECONDS,
        ),
        long_form_overlap_seconds=_read_non_negative_int(
            "ASR_LONG_FORM_OVERLAP_SECONDS",
            DEFAULT_LONG_FORM_OVERLAP_SECONDS,
        ),
        vad_filter=_read_bool("ASR_VAD_FILTER", True),
        vad_min_silence_ms=_read_positive_int(
            "ASR_VAD_MIN_SILENCE_MS",
            DEFAULT_VAD_MIN_SILENCE_MS,
        ),
        beam_size=_read_positive_int("ASR_BEAM_SIZE", DEFAULT_BEAM_SIZE),
        condition_on_previous_text=_read_bool(
            "ASR_CONDITION_ON_PREVIOUS_TEXT",
            True,
        ),
        output_dir=Path(output_raw) if output_raw else None,
    )


def _default_model_path(backend: str) -> Path:
    """Возвращает локальный checkpoint выбранного backend-а."""
    if backend in CLOUD_ASR_BACKENDS:
        return DEFAULT_MODEL_PATH
    return DEFAULT_BACKEND_MODEL_PATHS[backend]


def get_command_path(variable: str, default: str) -> str:
    """Возвращает путь внешней команды из окружения."""
    return os.getenv(variable, default).strip() or default


def validate_loopback_host(raw_host: str) -> str:
    """Разрешает локальному web API слушать только loopback-интерфейс."""
    host = raw_host.strip() or "127.0.0.1"
    if host.casefold() == "localhost":
        return "localhost"
    candidate = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError as exc:
        raise ValidationError("WEB_HOST должен быть loopback-адресом.") from exc
    if not address.is_loopback:
        raise ValidationError("WEB_HOST должен быть loopback-адресом.")
    return candidate


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on", "да"}:
        return True
    if normalized in {"0", "false", "no", "off", "нет"}:
        return False
    raise ValidationError(
        f"Переменная {name} должна содержать логическое значение: 1/0, true/false, yes/no или on/off."
    )


def _read_choice(name: str, default: str, allowed: frozenset[str]) -> str:
    """Читает строковый параметр окружения из ограниченного набора значений."""
    value = os.getenv(name, default).strip().casefold() or default
    if value not in allowed:
        variants = ", ".join(sorted(allowed))
        raise ValidationError(f"Переменная {name} должна иметь одно из значений: {variants}.")
    return value


def _read_optional_non_negative_int(name: str) -> int | None:
    """Читает необязательное неотрицательное целое без необработанного ValueError."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValidationError(
            f"Переменная {name} должна содержать целое неотрицательное число."
        ) from exc
    if value < 0:
        raise ValidationError(
            f"Переменная {name} должна содержать целое неотрицательное число."
        )
    return value


def _read_non_negative_int(
    name: str,
    default: int,
    *,
    maximum: int | None = None,
) -> int:
    """Читает обязательное неотрицательное целое с безопасной ошибкой конфигурации."""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValidationError(
            f"Переменная {name} должна содержать целое неотрицательное число."
        ) from exc
    if value < 0 or (maximum is not None and value > maximum):
        expected = (
            f"целое число от 0 до {maximum}"
            if maximum is not None
            else "целое неотрицательное число"
        )
        raise ValidationError(
            f"Переменная {name} должна содержать {expected}."
        )
    return value


def _read_positive_int(name: str, default: int) -> int:
    """Читает обязательное положительное целое с безопасной ошибкой конфигурации."""
    value = _read_non_negative_int(name, default)
    if value == 0:
        raise ValidationError(f"Переменная {name} должна содержать положительное число.")
    return value


def _read_positive_float(name: str, default: float) -> float:
    """Читает конечное положительное число с безопасной ошибкой конфигурации."""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValidationError(
            f"Переменная {name} должна содержать положительное число."
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValidationError(
            f"Переменная {name} должна содержать положительное число."
        )
    return value
