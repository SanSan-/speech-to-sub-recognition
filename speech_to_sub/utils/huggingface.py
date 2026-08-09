"""Безопасная докачка локальных моделей из Hugging Face Hub."""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any, Final

from filelock import FileLock, Timeout

from speech_to_sub.exceptions import SpeechToSubError
from speech_to_sub.utils.io_utils import atomic_write_json

MODEL_READY_MARKER: Final = ".speech-to-sub-model.json"
DEFAULT_UNKNOWN_MODEL_SIZE_BYTES: Final = 10 * 1024**3
MIN_FREE_SPACE_RESERVE_BYTES: Final = 512 * 1024**2
FREE_SPACE_RESERVE_RATIO: Final = 0.1
DEFAULT_DOWNLOAD_LOCK_TIMEOUT_SECONDS: Final = 600.0

_JSON_PATTERN: Final = "*.json"
_CONFIG_FILENAME: Final = "config.json"
_MERGES_FILENAME: Final = "merges.txt"
_MODEL_SAFETENSORS_FILENAME: Final = "model.safetensors"
_MODEL_SAFETENSORS_PATTERN: Final = "model-*.safetensors"
_PREPROCESSOR_CONFIG_FILENAME: Final = "preprocessor_config.json"
_TOKENIZER_FILENAME: Final = "tokenizer.json"
_TOKENIZER_CONFIG_FILENAME: Final = "tokenizer_config.json"
_VOCAB_FILENAME: Final = "vocab.json"
_WEIGHT_INDEX_NAMES: Final = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
_WEIGHT_PATTERNS: Final = ("*.safetensors", "pytorch_model*.bin")
_BACKEND_ALIASES: Final = {
    "transformers": "transformers",
    "faster-whisper": "faster-whisper",
    "parakeet-tdt-v3": "parakeet-tdt-v3",
    "qwen3-asr": "qwen3-asr",
    "qwen3-forced-aligner": "qwen3-forced-aligner",
    "qwen3-aligner": "qwen3-forced-aligner",
    "qwen-forced-aligner": "qwen3-forced-aligner",
}
_DOWNLOAD_ALLOW_PATTERNS: Final = {
    "transformers": (
        _JSON_PATTERN,
        "added_tokens.json",
        _MERGES_FILENAME,
        _MODEL_SAFETENSORS_FILENAME,
        _MODEL_SAFETENSORS_PATTERN,
        "normalizer.json",
        _VOCAB_FILENAME,
    ),
    "faster-whisper": (
        _CONFIG_FILENAME,
        "model.bin",
        _PREPROCESSOR_CONFIG_FILENAME,
        _TOKENIZER_FILENAME,
        "vocabulary.json",
    ),
    "parakeet-tdt-v3": (
        _CONFIG_FILENAME,
        "generation_config.json",
        _MODEL_SAFETENSORS_FILENAME,
        "processor_config.json",
        _TOKENIZER_FILENAME,
        _TOKENIZER_CONFIG_FILENAME,
    ),
    "qwen3-asr": (
        _JSON_PATTERN,
        _MERGES_FILENAME,
        _MODEL_SAFETENSORS_FILENAME,
        _MODEL_SAFETENSORS_PATTERN,
        _VOCAB_FILENAME,
    ),
    "qwen3-forced-aligner": (
        _JSON_PATTERN,
        _MERGES_FILENAME,
        _MODEL_SAFETENSORS_FILENAME,
        _MODEL_SAFETENSORS_PATTERN,
        _VOCAB_FILENAME,
    ),
}
SUPPORTED_MODEL_BACKENDS: Final = tuple(
    backend for backend, canonical in _BACKEND_ALIASES.items() if backend == canonical
)


class ModelDownloadError(SpeechToSubError):
    """Базовая ошибка подготовки модели из Hugging Face Hub."""


class ModelDownloadDisabledError(ModelDownloadError):
    """Каталог неполон, а явное разрешение сетевой загрузки не дано."""


class ModelRepositoryUnavailableError(ModelDownloadError):
    """Репозиторий или его ревизия недоступны."""


class ModelNetworkError(ModelDownloadError):
    """Hugging Face Hub недоступен по сети."""


class ModelInsufficientSpaceError(ModelDownloadError):
    """На целевом диске недостаточно свободного места."""


class ModelIncompleteError(ModelDownloadError):
    """После загрузки модель не удовлетворяет контракту движка."""


class ModelDownloadLockError(ModelDownloadError):
    """Не удалось получить блокировку каталога модели."""


@dataclass(frozen=True)
class ModelReadiness:
    """Результат локальной проверки без сетевых запросов."""

    ready: bool
    missing: tuple[str, ...]


@dataclass(frozen=True)
class ModelDownloadProgress:
    """Состояние подготовки модели для интерфейса или журнала."""

    stage: str
    message: str
    completed_bytes: int | None = None
    total_bytes: int | None = None


ModelProgressCallback = Callable[[ModelDownloadProgress], None]


@dataclass(frozen=True)
class _RemoteSizeEstimate:
    download_bytes: int
    required_free_bytes: int
    metadata_complete: bool


def inspect_local_model(target_dir: Path | str, backend: str) -> ModelReadiness:
    """Проверяет обязательные файлы модели выбранного движка без сети."""
    path = Path(target_dir).expanduser().resolve()
    canonical_backend = _canonical_backend(backend)
    if not path.is_dir():
        return ModelReadiness(False, ("каталог модели",))
    if canonical_backend == "faster-whisper":
        missing = _missing_regular_files(
            path,
            (
                "model.bin",
                _CONFIG_FILENAME,
                _TOKENIZER_FILENAME,
                _PREPROCESSOR_CONFIG_FILENAME,
            ),
        )
    elif canonical_backend == "transformers":
        missing = _missing_regular_files(
            path,
            (
                _CONFIG_FILENAME,
                _PREPROCESSOR_CONFIG_FILENAME,
                _TOKENIZER_CONFIG_FILENAME,
            ),
        )
        missing.extend(_missing_huggingface_weights(path, allow_shards=True))
    elif canonical_backend == "parakeet-tdt-v3":
        missing = _missing_regular_files(
            path,
            (_CONFIG_FILENAME, "processor_config.json", _TOKENIZER_FILENAME),
        )
        if not any(
            _is_nonempty_file(path / name)
            for name in (_MODEL_SAFETENSORS_FILENAME, "pytorch_model.bin")
        ):
            missing.append("model.safetensors или pytorch_model.bin")
    else:
        missing = _missing_regular_files(
            path,
            (
                _CONFIG_FILENAME,
                _MERGES_FILENAME,
                _PREPROCESSOR_CONFIG_FILENAME,
                _TOKENIZER_CONFIG_FILENAME,
                _VOCAB_FILENAME,
            ),
        )
        missing.extend(_missing_huggingface_weights(path, allow_shards=True))
    return ModelReadiness(not missing, tuple(missing))


def ensure_huggingface_model(
    repo_id: str,
    target_dir: Path | str,
    backend: str,
    *,
    allow_download: bool = False,
    revision: str | None = None,
    token: str | None = None,
    progress_callback: ModelProgressCallback | None = None,
    lock_timeout_seconds: float = DEFAULT_DOWNLOAD_LOCK_TIMEOUT_SECONDS,
    unknown_size_bytes: int = DEFAULT_UNKNOWN_MODEL_SIZE_BYTES,
) -> Path:
    """Возвращает готовый ``model_path`` и при явном разрешении докачивает его.

    Полный локальный каталог никогда не вызывает Hugging Face Hub. Неполный каталог
    загружается непосредственно в ``target_dir``; ``snapshot_download`` повторно
    использует уже скачанные файлы и свои незавершённые части.
    """
    normalized_repo_id = _normalize_repo_id(repo_id)
    canonical_backend = _canonical_backend(backend)
    target = _resolve_target(target_dir)
    _validate_download_options(lock_timeout_seconds, unknown_size_bytes)
    _emit(progress_callback, "local-check", "Проверка локальной модели.")
    readiness = inspect_local_model(target, canonical_backend)
    if readiness.ready:
        _emit(progress_callback, "ready", "Локальная модель готова.")
        return target
    if not allow_download:
        _raise_download_disabled(readiness)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.download.lock"
    try:
        with FileLock(str(lock_path), timeout=lock_timeout_seconds):
            return _ensure_under_lock(
                normalized_repo_id,
                target,
                canonical_backend,
                revision=revision,
                token=token,
                progress_callback=progress_callback,
                unknown_size_bytes=unknown_size_bytes,
            )
    except Timeout as exc:
        raise ModelDownloadLockError(
            f"Каталог модели занят другим процессом: {target}"
        ) from exc


def _ensure_under_lock(
    repo_id: str,
    target: Path,
    backend: str,
    *,
    revision: str | None,
    token: str | None,
    progress_callback: ModelProgressCallback | None,
    unknown_size_bytes: int,
) -> Path:
    readiness = inspect_local_model(target, backend)
    if readiness.ready:
        _emit(progress_callback, "ready", "Локальная модель готова.")
        return target
    _emit(progress_callback, "metadata", "Получение сведений о файлах модели.")
    model_info = _load_remote_model_info(repo_id, revision=revision, token=token)
    resolved_revision = _resolved_model_revision(model_info)
    allow_patterns = _DOWNLOAD_ALLOW_PATTERNS[backend]
    estimate = _estimate_remote_size(
        model_info,
        target,
        unknown_size_bytes,
        allow_patterns=allow_patterns,
    )
    _require_free_space(target, estimate)
    _emit(
        progress_callback,
        "download",
        "Загрузка файлов модели.",
        completed_bytes=0,
        total_bytes=estimate.download_bytes,
    )
    target.mkdir(parents=True, exist_ok=True)
    _download_snapshot(
        repo_id,
        target,
        revision=resolved_revision,
        token=token,
        allow_patterns=allow_patterns,
    )
    _verify_download(target, backend)
    _write_ready_marker(target, repo_id, backend, resolved_revision)
    _emit(
        progress_callback,
        "ready",
        "Модель загружена и проверена.",
        completed_bytes=estimate.download_bytes,
        total_bytes=estimate.download_bytes,
    )
    return target


def _load_remote_model_info(
    repo_id: str, *, revision: str | None, token: str | None
) -> Any:
    try:
        api = _create_hf_api()
        return api.model_info(
            repo_id=repo_id,
            revision=revision,
            files_metadata=True,
            token=token,
        )
    except ImportError as exc:
        raise ModelDownloadError(
            "Для загрузки модели требуется huggingface_hub."
        ) from exc
    except Exception as exc:
        _raise_hub_access_error(exc, repo_id)


def _download_snapshot(
    repo_id: str,
    target: Path,
    *,
    revision: str | None,
    token: str | None,
    allow_patterns: Sequence[str],
) -> None:
    try:
        _run_snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(target),
            token=token,
            force_download=False,
            allow_patterns=list(allow_patterns),
        )
    except ImportError as exc:
        raise ModelDownloadError(
            "Для загрузки модели требуется huggingface_hub."
        ) from exc
    except Exception as exc:
        if _is_repository_error(exc):
            raise ModelRepositoryUnavailableError(
                f"Репозиторий модели недоступен: {repo_id}"
            ) from exc
        raise ModelNetworkError(
            f"Не удалось докачать модель из Hugging Face Hub: {repo_id}"
        ) from exc


def _estimate_remote_size(
    model_info: Any,
    target: Path,
    unknown_size_bytes: int,
    *,
    allow_patterns: Sequence[str],
) -> _RemoteSizeEstimate:
    siblings = _model_siblings(model_info)
    known_missing = 0
    selected_files = 0
    metadata_complete = bool(siblings)
    for sibling in siblings:
        filename = _sibling_value(sibling, "rfilename")
        if not isinstance(filename, str) or not filename.strip():
            metadata_complete = False
            continue
        local_file = _safe_local_sibling(target, filename)
        if not any(fnmatchcase(filename, pattern) for pattern in allow_patterns):
            continue
        selected_files += 1
        size = _sibling_value(sibling, "size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            metadata_complete = False
            continue
        if not local_file.is_file() or local_file.stat().st_size != size:
            known_missing += size
    metadata_complete = metadata_complete and selected_files > 0
    download_bytes = (
        known_missing if metadata_complete else max(known_missing, unknown_size_bytes)
    )
    reserve = max(
        MIN_FREE_SPACE_RESERVE_BYTES,
        math.ceil(download_bytes * FREE_SPACE_RESERVE_RATIO),
    )
    return _RemoteSizeEstimate(
        download_bytes, download_bytes + reserve, metadata_complete
    )


def _require_free_space(target: Path, estimate: _RemoteSizeEstimate) -> None:
    try:
        free_bytes = shutil.disk_usage(target.parent).free
    except OSError as exc:
        raise ModelDownloadError(
            f"Не удалось определить свободное место для каталога модели: {target}"
        ) from exc
    if free_bytes >= estimate.required_free_bytes:
        return
    raise ModelInsufficientSpaceError(
        "Недостаточно места для загрузки модели в "
        f"{target}: требуется не менее {_format_bytes(estimate.required_free_bytes)}, "
        f"доступно {_format_bytes(free_bytes)}."
    )


def _verify_download(target: Path, backend: str) -> None:
    readiness = inspect_local_model(target, backend)
    if readiness.ready:
        return
    missing = ", ".join(readiness.missing)
    raise ModelIncompleteError(
        f"Загрузка завершилась, но модель для {backend} неполна. Отсутствует: {missing}."
    )


def _raise_download_disabled(readiness: ModelReadiness) -> None:
    missing = ", ".join(readiness.missing)
    raise ModelDownloadDisabledError(
        "Локальная модель неполна, а загрузка из сети не разрешена. "
        f"Отсутствует: {missing}."
    )


def _write_ready_marker(
    target: Path,
    repo_id: str,
    backend: str,
    revision: str | None,
) -> None:
    atomic_write_json(
        target / MODEL_READY_MARKER,
        {
            "schema_version": 1,
            "status": "ready",
            "repo_id": repo_id,
            "revision": revision,
            "backend": backend,
        },
    )


def _missing_huggingface_weights(path: Path, *, allow_shards: bool) -> list[str]:
    indexes = [path / name for name in _WEIGHT_INDEX_NAMES if (path / name).is_file()]
    if allow_shards and indexes:
        missing = _missing_indexed_shards(path, indexes)
        return missing or []
    if any(
        _is_nonempty_file(candidate)
        for pattern in _WEIGHT_PATTERNS
        for candidate in path.glob(pattern)
    ):
        return []
    return ["веса *.safetensors или pytorch_model*.bin"]


def _missing_indexed_shards(path: Path, indexes: Sequence[Path]) -> list[str]:
    errors: list[str] = []
    for index_path in indexes:
        errors.extend(_indexed_shard_errors(path, index_path))
    return errors


def _indexed_shard_errors(path: Path, index_path: Path) -> list[str]:
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return [f"корректный индекс {index_path.name}"]
    weight_map = raw.get("weight_map") if isinstance(raw, Mapping) else None
    if not isinstance(weight_map, Mapping) or not weight_map:
        return [f"непустой weight_map в {index_path.name}"]
    shard_names = {str(value) for value in weight_map.values() if str(value)}
    if not shard_names:
        return [f"ссылки на шарды в {index_path.name}"]
    return _missing_shard_files(path, index_path, shard_names)


def _missing_shard_files(
    path: Path,
    index_path: Path,
    shard_names: set[str],
) -> list[str]:
    errors: list[str] = []
    for raw_name in shard_names:
        try:
            shard = _safe_local_sibling(path, raw_name)
        except ModelRepositoryUnavailableError:
            errors.append(f"безопасный путь шарда из {index_path.name}")
            continue
        if not _is_nonempty_file(shard):
            errors.append(raw_name)
    return errors


def _missing_regular_files(path: Path, names: Sequence[str]) -> list[str]:
    return [name for name in names if not _is_nonempty_file(path / name)]


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _model_siblings(model_info: Any) -> Sequence[Any]:
    value = (
        model_info.get("siblings")
        if isinstance(model_info, Mapping)
        else getattr(model_info, "siblings", ())
    )
    return (
        value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else ()
    )


def _resolved_model_revision(model_info: Any) -> str:
    """Возвращает неизменяемый SHA снимка, для которого получены метаданные."""
    value = (
        model_info.get("sha")
        if isinstance(model_info, Mapping)
        else getattr(model_info, "sha", None)
    )
    revision = str(value or "").strip()
    if not revision:
        raise ModelRepositoryUnavailableError(
            "Hugging Face Hub не вернул SHA снимка модели."
        )
    return revision


def _sibling_value(sibling: Any, name: str) -> Any:
    return (
        sibling.get(name)
        if isinstance(sibling, Mapping)
        else getattr(sibling, name, None)
    )


def _safe_local_sibling(target: Path, filename: str) -> Path:
    relative = PurePosixPath(filename.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ModelRepositoryUnavailableError(
            "Сведения Hugging Face Hub содержат небезопасный путь файла модели."
        )
    candidate = target.joinpath(*relative.parts).resolve()
    if candidate != target and target not in candidate.parents:
        raise ModelRepositoryUnavailableError(
            "Сведения Hugging Face Hub содержат путь вне каталога модели."
        )
    return candidate


def _resolve_target(target_dir: Path | str) -> Path:
    if not str(target_dir).strip():
        raise ModelDownloadError("Целевой каталог модели не задан.")
    target = Path(target_dir).expanduser().resolve()
    if target.exists() and not target.is_dir():
        raise ModelDownloadError(f"Путь модели не является каталогом: {target}")
    if not target.name:
        raise ModelDownloadError("Целевой каталог модели задан некорректно.")
    return target


def _normalize_repo_id(repo_id: str) -> str:
    normalized = str(repo_id).strip()
    if not normalized:
        raise ModelDownloadError("Идентификатор репозитория модели не задан.")
    return normalized


def _canonical_backend(backend: str) -> str:
    normalized = str(backend).strip().casefold()
    try:
        return _BACKEND_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_MODEL_BACKENDS)
        raise ModelDownloadError(
            f"Неизвестный движок модели '{backend}'. Поддерживаются: {supported}."
        ) from exc


def _validate_download_options(
    lock_timeout_seconds: float, unknown_size_bytes: int
) -> None:
    if not math.isfinite(lock_timeout_seconds) or lock_timeout_seconds < 0:
        raise ModelDownloadError(
            "Время ожидания блокировки должно быть неотрицательным."
        )
    if isinstance(unknown_size_bytes, bool) or unknown_size_bytes <= 0:
        raise ModelDownloadError(
            "Резерв для неизвестного размера модели должен быть положительным."
        )


def _raise_hub_access_error(exc: Exception, repo_id: str) -> None:
    if _is_repository_error(exc):
        raise ModelRepositoryUnavailableError(
            f"Репозиторий модели или его ревизия недоступны: {repo_id}"
        ) from exc
    raise ModelNetworkError(
        f"Не удалось получить сведения о модели из Hugging Face Hub: {repo_id}"
    ) from exc


def _is_repository_error(exc: Exception) -> bool:
    repository_errors = {
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "EntryNotFoundError",
        "GatedRepoError",
    }
    return any(base.__name__ in repository_errors for base in type(exc).__mro__)


def _create_hf_api() -> Any:
    from huggingface_hub import HfApi

    return HfApi()


def _run_snapshot_download(**kwargs: Any) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def _emit(
    callback: ModelProgressCallback | None,
    stage: str,
    message: str,
    *,
    completed_bytes: int | None = None,
    total_bytes: int | None = None,
) -> None:
    if callback is not None:
        callback(ModelDownloadProgress(stage, message, completed_bytes, total_bytes))


def _format_bytes(value: int) -> str:
    units = ("Б", "КиБ", "МиБ", "ГиБ", "ТиБ")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            break
        amount /= 1024
    return f"{amount:.1f} {unit}"


__all__ = [
    "DEFAULT_DOWNLOAD_LOCK_TIMEOUT_SECONDS",
    "DEFAULT_UNKNOWN_MODEL_SIZE_BYTES",
    "MODEL_READY_MARKER",
    "SUPPORTED_MODEL_BACKENDS",
    "ModelDownloadDisabledError",
    "ModelDownloadError",
    "ModelDownloadLockError",
    "ModelDownloadProgress",
    "ModelIncompleteError",
    "ModelInsufficientSpaceError",
    "ModelNetworkError",
    "ModelProgressCallback",
    "ModelReadiness",
    "ModelRepositoryUnavailableError",
    "ensure_huggingface_model",
    "inspect_local_model",
]
