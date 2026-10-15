from __future__ import annotations

from pathlib import Path

import pytest

from speech_to_sub.exceptions import ValidationError
from speech_to_sub.models import ProcessingSettings
from speech_to_sub.utils import worker_runtime

_PROFILES = (
    ("qwen3-asr", "worker_python_path"),
    ("parakeet-tdt-v3", "worker_python_path"),
    ("qwen3-forced-aligner", "aligner_worker_python_path"),
)
_ENVIRONMENT_NAMES = (
    "QWEN_ASR_PYTHON",
    "PARAKEET_ASR_PYTHON",
    "ASR_ALIGNER_WORKER_PYTHON",
)


@pytest.mark.parametrize("profile_id, path_setting", _PROFILES)
def test_explicit_worker_path_overrides_environment_and_resolves_relative_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile_id: str,
    path_setting: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in _ENVIRONMENT_NAMES:
        monkeypatch.setenv(name, str(tmp_path / name / "python.exe"))
    settings = ProcessingSettings(**{path_setting: Path("explicit-python.exe")})

    resolved = worker_runtime.resolve_worker_python(settings, profile_id)

    assert resolved == (tmp_path / "explicit-python.exe").resolve()
    assert not resolved.exists()


@pytest.mark.parametrize(
    "profile_id, empty_aligner_environment, expected_environment",
    [
        ("qwen3-asr", False, "QWEN_ASR_PYTHON"),
        ("parakeet-tdt-v3", False, "PARAKEET_ASR_PYTHON"),
        ("qwen3-forced-aligner", False, "ASR_ALIGNER_WORKER_PYTHON"),
        ("qwen3-forced-aligner", True, "QWEN_ASR_PYTHON"),
    ],
)
def test_worker_path_uses_profile_environment_priority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile_id: str,
    empty_aligner_environment: bool,
    expected_environment: str,
) -> None:
    for name in _ENVIRONMENT_NAMES:
        monkeypatch.setenv(name, str(tmp_path / name / "python.exe"))
    if empty_aligner_environment:
        monkeypatch.setenv("ASR_ALIGNER_WORKER_PYTHON", "")
    settings = ProcessingSettings()

    resolved = worker_runtime.resolve_worker_python(settings, profile_id)

    assert resolved == (tmp_path / expected_environment / "python.exe").resolve()


@pytest.mark.parametrize(
    "profile_id, runtime_directory",
    [
        ("qwen3-asr", "qwen"),
        ("parakeet-tdt-v3", "parakeet"),
        ("qwen3-forced-aligner", "qwen"),
    ],
)
def test_worker_path_defaults_to_profile_under_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile_id: str,
    runtime_directory: str,
) -> None:
    for name in _ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(worker_runtime, "BASE_DIR", tmp_path)
    settings = ProcessingSettings()

    resolved = worker_runtime.resolve_worker_python(settings, profile_id)

    assert resolved == (
        tmp_path / "resources/runtimes" / runtime_directory / "Scripts/python.exe"
    ).resolve()
    assert not resolved.exists()


def test_aligner_does_not_reuse_asr_worker_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(worker_runtime, "BASE_DIR", tmp_path)
    settings = ProcessingSettings(worker_python_path=tmp_path / "parakeet-python.exe")

    resolved = worker_runtime.resolve_worker_python(settings, "qwen3-forced-aligner")

    assert resolved == (tmp_path / "resources/runtimes/qwen/Scripts/python.exe").resolve()


@pytest.mark.parametrize(
    "profile_id, path_setting, runtime_name, setting_hint",
    [
        ("qwen3-asr", "worker_python_path", "Qwen", "QWEN_ASR_PYTHON или worker_python_path"),
        (
            "parakeet-tdt-v3",
            "worker_python_path",
            "Parakeet",
            "PARAKEET_ASR_PYTHON или worker_python_path",
        ),
        (
            "qwen3-forced-aligner",
            "aligner_worker_python_path",
            "Qwen",
            "ASR_ALIGNER_WORKER_PYTHON или aligner_worker_python_path",
        ),
    ],
)
def test_missing_worker_preserves_profile_diagnostic(
    tmp_path: Path,
    profile_id: str,
    path_setting: str,
    runtime_name: str,
    setting_hint: str,
) -> None:
    missing_python = tmp_path / "missing-python.exe"
    settings = ProcessingSettings(**{path_setting: missing_python})

    with pytest.raises(ValidationError) as failure:
        worker_runtime.require_worker_python(settings, profile_id)

    assert str(failure.value) == (
        f"Python изолированного {runtime_name} runtime не найден: "
        f"{missing_python.resolve()}. Укажите {setting_hint}."
    )


@pytest.mark.parametrize("profile_id, path_setting", _PROFILES)
def test_worker_file_is_accepted_without_running_it(
    tmp_path: Path,
    profile_id: str,
    path_setting: str,
) -> None:
    python_path = tmp_path / "python.exe"
    python_path.write_bytes(b"")
    settings = ProcessingSettings(**{path_setting: python_path})

    assert worker_runtime.require_worker_python(settings, profile_id) == python_path.resolve()


@pytest.mark.parametrize("profile_id, path_setting", _PROFILES)
def test_selected_profile_check_accepts_uppercase_identifiers(
    tmp_path: Path,
    profile_id: str,
    path_setting: str,
) -> None:
    selected_field = "aligner" if profile_id == "qwen3-forced-aligner" else "backend"
    settings = ProcessingSettings(
        **{
            selected_field: profile_id.upper(),
            path_setting: tmp_path / "missing-python.exe",
        }
    )

    with pytest.raises(ValidationError, match="Python изолированного"):
        worker_runtime.validate_selected_worker_pythons(settings)


@pytest.mark.parametrize("backend", ["transformers", "faster-whisper", "openai-api"])
def test_unselected_worker_paths_do_not_block_other_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    for name in _ENVIRONMENT_NAMES:
        monkeypatch.setenv(name, str(tmp_path / "missing-environment-python.exe"))
    settings = ProcessingSettings(
        backend=backend,
        aligner="none",
        worker_python_path=tmp_path / "missing-asr-python.exe",
        aligner_worker_python_path=tmp_path / "missing-aligner-python.exe",
    )

    worker_runtime.validate_selected_worker_pythons(settings)
