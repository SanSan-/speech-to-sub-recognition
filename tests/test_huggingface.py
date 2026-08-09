from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from speech_to_sub.utils import huggingface
from speech_to_sub.utils.huggingface import (
    MODEL_READY_MARKER,
    ModelDownloadDisabledError,
    ModelIncompleteError,
    ModelInsufficientSpaceError,
    ModelNetworkError,
    ModelRepositoryUnavailableError,
    ensure_huggingface_model,
    inspect_local_model,
)


def _write_model(path: Path, backend: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    required = {
        "faster-whisper": (
            "model.bin",
            "config.json",
            "tokenizer.json",
            "preprocessor_config.json",
        ),
        "transformers": (
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "model.safetensors",
        ),
        "parakeet-tdt-v3": (
            "config.json",
            "processor_config.json",
            "tokenizer.json",
            "model.safetensors",
        ),
        "qwen3-asr": (
            "config.json",
            "merges.txt",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "model.safetensors",
        ),
        "qwen3-forced-aligner": (
            "config.json",
            "merges.txt",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "model.safetensors",
        ),
    }[backend]
    for name in required:
        (path / name).write_bytes(b"model-data")


def _info(
    *files: tuple[str, int | None],
    sha: str = "resolved-model-sha",
) -> SimpleNamespace:
    return SimpleNamespace(
        sha=sha,
        siblings=[SimpleNamespace(rfilename=name, size=size) for name, size in files]
    )


def _large_disk_usage(_path: Path) -> SimpleNamespace:
    return SimpleNamespace(total=100 * 1024**3, used=0, free=100 * 1024**3)


@pytest.mark.parametrize(
    "backend",
    (
        "faster-whisper",
        "transformers",
        "parakeet-tdt-v3",
        "qwen3-asr",
        "qwen3-forced-aligner",
    ),
)
def test_inspect_local_model_uses_backend_contract(
    tmp_path: Path, backend: str
) -> None:
    model_path = tmp_path / backend
    _write_model(model_path, backend)

    assert inspect_local_model(model_path, backend).ready is True


def test_inspect_local_model_checks_all_indexed_shards(tmp_path: Path) -> None:
    model_path = tmp_path / "qwen"
    model_path.mkdir()
    (model_path / "config.json").write_bytes(b"{}")
    (model_path / "model.safetensors.index.json").write_bytes(
        b'{"weight_map":{"a":"part-1.safetensors","b":"part-2.safetensors"}}'
    )
    (model_path / "part-1.safetensors").write_bytes(b"weights")

    readiness = inspect_local_model(model_path, "qwen3-asr")

    assert readiness.ready is False
    assert "part-2.safetensors" in readiness.missing


def test_empty_weight_index_is_not_a_complete_model(tmp_path: Path) -> None:
    model_path = tmp_path / "qwen"
    model_path.mkdir()
    (model_path / "config.json").write_bytes(b"{}")
    (model_path / "model.safetensors.index.json").write_bytes(
        b'{"weight_map":{"layer":""}}'
    )

    readiness = inspect_local_model(model_path, "qwen3-asr")

    assert readiness.ready is False
    assert any("ссылки на шарды" in item for item in readiness.missing)


def test_complete_model_never_touches_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    _write_model(model_path, "faster-whisper")
    progress: list[str] = []
    monkeypatch.setattr(
        huggingface,
        "_create_hf_api",
        lambda: pytest.fail("Сеть не должна использоваться для полной модели."),
    )
    monkeypatch.setattr(
        huggingface,
        "_run_snapshot_download",
        lambda **_kwargs: pytest.fail("Загрузка не должна запускаться."),
    )

    result = ensure_huggingface_model(
        "openai/whisper-large-v3",
        model_path,
        "faster-whisper",
        allow_download=True,
        progress_callback=lambda value: progress.append(value.stage),
    )

    assert result == model_path.resolve()
    assert progress == ["local-check", "ready"]
    assert not (tmp_path / ".model.download.lock").exists()


def test_incomplete_model_requires_explicit_download_permission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        huggingface,
        "_create_hf_api",
        lambda: pytest.fail("Без разрешения сеть не должна использоваться."),
    )

    model_path = tmp_path / "nested" / "model"
    with pytest.raises(
        ModelDownloadDisabledError, match="загрузка из сети не разрешена"
    ):
        ensure_huggingface_model("org/model", model_path, "faster-whisper")

    assert not model_path.parent.exists()


def test_download_resumes_into_same_model_path_and_writes_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    partial = model_path / "model.bin"
    partial.write_bytes(b"partial")
    token = "secret-token"
    calls: list[dict[str, Any]] = []
    progress: list[str] = []

    class Api:
        def model_info(self, **kwargs: Any) -> Any:
            assert kwargs["token"] == token
            return _info(
                ("model.bin", 10),
                ("config.json", 10),
                ("tokenizer.json", 10),
                ("preprocessor_config.json", 10),
            )

    def download(**kwargs: Any) -> str:
        calls.append(kwargs)
        assert partial.read_bytes() == b"partial"
        _write_model(model_path, "faster-whisper")
        return str(model_path)

    monkeypatch.setattr(huggingface, "_create_hf_api", Api)
    monkeypatch.setattr(huggingface, "_run_snapshot_download", download)
    monkeypatch.setattr(huggingface.shutil, "disk_usage", _large_disk_usage)

    result = ensure_huggingface_model(
        "org/model",
        model_path,
        "faster-whisper",
        allow_download=True,
        revision="main",
        token=token,
        progress_callback=lambda value: progress.append(value.stage),
    )

    assert result == model_path.resolve()
    assert calls == [
        {
            "repo_id": "org/model",
            "revision": "resolved-model-sha",
            "local_dir": str(model_path.resolve()),
            "token": token,
            "force_download": False,
            "allow_patterns": [
                "config.json",
                "model.bin",
                "preprocessor_config.json",
                "tokenizer.json",
                "vocabulary.json",
            ],
        }
    ]
    marker = model_path / MODEL_READY_MARKER
    marker_bytes = marker.read_bytes()
    assert not marker_bytes.startswith(b"\xef\xbb\xbf")
    marker_data = json.loads(marker_bytes.decode("utf-8"))
    assert marker_data == {
        "schema_version": 1,
        "status": "ready",
        "repo_id": "org/model",
        "revision": "resolved-model-sha",
        "backend": "faster-whisper",
    }
    assert token not in marker_bytes.decode("utf-8")
    assert progress == ["local-check", "metadata", "download", "ready"]


def test_insufficient_space_stops_before_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Api:
        def model_info(self, **_kwargs: Any) -> Any:
            return _info(("model.bin", 1024**3))

    monkeypatch.setattr(huggingface, "_create_hf_api", Api)
    monkeypatch.setattr(
        huggingface.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=1024, used=0, free=1024),
    )
    monkeypatch.setattr(
        huggingface,
        "_run_snapshot_download",
        lambda **_kwargs: pytest.fail(
            "При нехватке места загрузка не должна начинаться."
        ),
    )

    with pytest.raises(ModelInsufficientSpaceError, match="Недостаточно места"):
        ensure_huggingface_model(
            "org/model",
            tmp_path / "model",
            "faster-whisper",
            allow_download=True,
        )


def test_unknown_remote_sizes_use_conservative_estimate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Api:
        def model_info(self, **_kwargs: Any) -> Any:
            return _info(("model.bin", None))

    monkeypatch.setattr(huggingface, "_create_hf_api", Api)
    monkeypatch.setattr(
        huggingface.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=2048, used=0, free=2048),
    )

    with pytest.raises(ModelInsufficientSpaceError):
        ensure_huggingface_model(
            "org/model",
            tmp_path / "model",
            "faster-whisper",
            allow_download=True,
            unknown_size_bytes=4096,
        )


def test_download_size_ignores_unneeded_weight_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"

    class Api:
        def model_info(self, **_kwargs: Any) -> Any:
            return _info(
                ("config.json", 10),
                ("model.safetensors", 100),
                ("pytorch_model.bin", 10_000),
                ("flax_model.msgpack", 20_000),
            )

    observed_paths: list[Path] = []
    observed_patterns: list[str] = []

    def disk_usage(path: Path) -> SimpleNamespace:
        observed_paths.append(path)
        return SimpleNamespace(total=1024**3, used=0, free=600 * 1024**2)

    def download(**kwargs: Any) -> str:
        observed_patterns.extend(kwargs["allow_patterns"])
        _write_model(model_path, "transformers")
        return str(model_path)

    monkeypatch.setattr(huggingface, "_create_hf_api", Api)
    monkeypatch.setattr(huggingface, "_run_snapshot_download", download)
    monkeypatch.setattr(huggingface.shutil, "disk_usage", disk_usage)

    ensure_huggingface_model(
        "openai/whisper-large-v3",
        model_path,
        "transformers",
        allow_download=True,
    )

    assert observed_paths == [model_path.parent]
    assert "model.safetensors" in observed_patterns
    assert "pytorch_model.bin" not in observed_patterns
    assert "flax_model.msgpack" not in observed_patterns


def test_repository_error_is_mapped_without_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RepositoryNotFoundError(Exception):
        pass

    class Api:
        def model_info(self, **_kwargs: Any) -> Any:
            raise RepositoryNotFoundError("secret-token")

    monkeypatch.setattr(huggingface, "_create_hf_api", Api)

    with pytest.raises(ModelRepositoryUnavailableError) as caught:
        ensure_huggingface_model(
            "org/missing",
            tmp_path / "model",
            "qwen3-asr",
            allow_download=True,
            token="secret-token",
        )

    assert "secret-token" not in str(caught.value)


def test_download_requires_immutable_revision_from_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Api:
        def model_info(self, **_kwargs: Any) -> Any:
            return _info(("model.bin", 10), sha="")

    monkeypatch.setattr(huggingface, "_create_hf_api", Api)

    with pytest.raises(ModelRepositoryUnavailableError, match="SHA снимка"):
        ensure_huggingface_model(
            "org/model",
            tmp_path / "model",
            "faster-whisper",
            allow_download=True,
        )


def test_network_error_is_mapped_without_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Api:
        def model_info(self, **_kwargs: Any) -> Any:
            raise TimeoutError("secret-token")

    monkeypatch.setattr(huggingface, "_create_hf_api", Api)

    with pytest.raises(ModelNetworkError) as caught:
        ensure_huggingface_model(
            "org/model",
            tmp_path / "model",
            "qwen3-asr",
            allow_download=True,
            token="secret-token",
        )

    assert "secret-token" not in str(caught.value)


def test_incomplete_snapshot_is_not_marked_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"

    class Api:
        def model_info(self, **_kwargs: Any) -> Any:
            return _info(("config.json", 2), ("model.safetensors", 7))

    def incomplete_download(**_kwargs: Any) -> str:
        model_path.mkdir(exist_ok=True)
        (model_path / "config.json").write_bytes(b"{}")
        return str(model_path)

    monkeypatch.setattr(huggingface, "_create_hf_api", Api)
    monkeypatch.setattr(huggingface, "_run_snapshot_download", incomplete_download)
    monkeypatch.setattr(huggingface.shutil, "disk_usage", _large_disk_usage)

    with pytest.raises(ModelIncompleteError, match="модель для qwen3-asr неполна"):
        ensure_huggingface_model(
            "org/model",
            model_path,
            "qwen3-asr",
            allow_download=True,
        )

    assert not (model_path / MODEL_READY_MARKER).exists()


def test_concurrent_calls_download_snapshot_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    download_started = threading.Event()
    release_download = threading.Event()
    calls = 0

    class Api:
        def model_info(self, **_kwargs: Any) -> Any:
            return _info(
                ("model.bin", 10),
                ("config.json", 10),
                ("tokenizer.json", 10),
                ("preprocessor_config.json", 10),
            )

    def download(**_kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        download_started.set()
        assert release_download.wait(timeout=5)
        _write_model(model_path, "faster-whisper")
        return str(model_path)

    monkeypatch.setattr(huggingface, "_create_hf_api", Api)
    monkeypatch.setattr(huggingface, "_run_snapshot_download", download)
    monkeypatch.setattr(huggingface.shutil, "disk_usage", _large_disk_usage)

    def ensure() -> Path:
        return ensure_huggingface_model(
            "org/model",
            model_path,
            "faster-whisper",
            allow_download=True,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(ensure)
        assert download_started.wait(timeout=5)
        second = executor.submit(ensure)
        release_download.set()
        assert first.result(timeout=5) == model_path.resolve()
        assert second.result(timeout=5) == model_path.resolve()

    assert calls == 1


def test_unsafe_remote_filename_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Api:
        def model_info(self, **_kwargs: Any) -> Any:
            return _info(("../outside.bin", 10))

    monkeypatch.setattr(huggingface, "_create_hf_api", Api)

    with pytest.raises(ModelRepositoryUnavailableError, match="небезопасный путь"):
        ensure_huggingface_model(
            "org/model",
            tmp_path / "model",
            "faster-whisper",
            allow_download=True,
        )
