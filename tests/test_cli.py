from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from speech_to_sub import __version__
from speech_to_sub import cli
from speech_to_sub.constants import (
    DEFAULT_QWEN_ALIGNER_MODEL_PATH,
    DEFAULT_QWEN_MODEL_PATH,
)
from speech_to_sub.exceptions import ValidationError
from speech_to_sub.models import ProcessingSettings


def test_cli_reports_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"speech-to-sub {__version__}"


def _install_cli_dependencies(monkeypatch: pytest.MonkeyPatch) -> logging.Logger:
    logger = logging.getLogger("speech-to-sub-cli-tests")
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(cli, "settings_from_environment", ProcessingSettings)
    monkeypatch.setattr(cli, "setup_logging", lambda _verbose: logger)
    return logger


@pytest.mark.parametrize(
    "results, expected_code",
    [
        ([{"state": "done"}, {"state": "cached"}, {"state": "skipped"}], 0),
        ([{"state": "done"}, {"state": "error"}], 2),
    ],
)
def test_main_returns_batch_exit_codes(
    results: list[dict[str, str]],
    expected_code: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_cli_dependencies(monkeypatch)
    received: dict[str, Any] = {}

    def fake_process(
        paths: list[Path],
        settings: dict[str, Any],
        emit_event: Callable[[dict[str, Any]], None],
        log: Callable[[str], None],
    ) -> list[dict[str, str]]:
        del emit_event, log
        received["paths"] = paths
        received["settings"] = settings
        return results

    monkeypatch.setattr(cli, "process_paths", fake_process)

    code = cli.main(
        [
            "--input",
            "sample.mp4",
            "--backend",
            "faster-whisper",
            "--worker-python-path",
            "backend-python.exe",
            "--aligner-worker-python-path",
            "aligner-python.exe",
            "--language",
            "ru",
            "--output-format",
            "ass",
            "--max-chars-per-line",
            "40",
            "--line-length-gap",
            "6",
            "--max-cps",
            "16.5",
            "--long-form-window-seconds",
            "240",
            "--long-form-overlap-seconds",
            "3",
            "--no-vad",
            "--beam-size",
            "3",
            "--no-condition-on-previous-text",
            "--no-auto-download-model",
            "--force",
        ]
    )

    assert code == expected_code
    assert received["paths"] == [Path("sample.mp4")]
    assert received["settings"]["backend"] == "faster-whisper"
    assert received["settings"]["worker_python_path"] == "backend-python.exe"
    assert received["settings"]["aligner_worker_python_path"] == "aligner-python.exe"
    assert received["settings"]["language"] == "ru"
    assert received["settings"]["output_format"] == "ass"
    assert received["settings"]["max_chars_per_line"] == 40
    assert received["settings"]["line_length_gap"] == 6
    assert received["settings"]["max_cps"] == 16.5
    assert received["settings"]["force"] is True
    assert received["settings"]["long_form_window_seconds"] == 240
    assert received["settings"]["long_form_overlap_seconds"] == 3
    assert received["settings"]["vad_filter"] is False
    assert received["settings"]["beam_size"] == 3
    assert received["settings"]["condition_on_previous_text"] is False
    assert received["settings"]["auto_download_model"] is False


@pytest.mark.parametrize(
    "error",
    [
        ValidationError("неверные параметры"),
        OSError("ошибка файловой системы"),
        ValueError("некорректное значение"),
    ],
)
def test_main_returns_one_when_batch_cannot_start(
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_cli_dependencies(monkeypatch)

    def fail_process(*_args: Any, **_kwargs: Any) -> list[dict[str, str]]:
        raise error

    monkeypatch.setattr(cli, "process_paths", fail_process)

    assert cli.main(["--input", "sample.mp4"]) == 1


def test_cli_backend_and_aligner_select_matching_local_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_cli_dependencies(monkeypatch)
    monkeypatch.delenv("ASR_MODEL_PATH", raising=False)
    monkeypatch.delenv("ASR_ALIGNER_MODEL_PATH", raising=False)
    received: dict[str, Any] = {}

    def fake_process(
        _paths: list[Path],
        settings: dict[str, Any],
        emit_event: Callable[[dict[str, Any]], None],
        log: Callable[[str], None],
    ) -> list[dict[str, str]]:
        del emit_event, log
        received.update(settings)
        return [{"state": "done"}]

    monkeypatch.setattr(cli, "process_paths", fake_process)

    assert cli.main(
        [
            "--input",
            "sample.mp4",
            "--backend",
            "qwen3-asr",
            "--aligner",
            "qwen3-forced-aligner",
        ]
    ) == 0
    assert Path(received["model_path"]) == DEFAULT_QWEN_MODEL_PATH
    assert Path(received["aligner_model_path"]) == DEFAULT_QWEN_ALIGNER_MODEL_PATH


def test_cli_selects_openai_with_explicit_cloud_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_cli_dependencies(monkeypatch)
    monkeypatch.delenv("ASR_MODEL_PATH", raising=False)
    received: dict[str, Any] = {}

    def fake_process(
        _paths: list[Path],
        settings: dict[str, Any],
        emit_event: Callable[[dict[str, Any]], None],
        log: Callable[[str], None],
    ) -> list[dict[str, str]]:
        del emit_event, log
        received.update(settings)
        return [{"state": "done"}]

    monkeypatch.setattr(cli, "process_paths", fake_process)

    assert cli.main(
        [
            "--input",
            "sample.mp4",
            "--backend",
            "openai-api",
            "--allow-cloud-processing",
            "--openai-model",
            "whisper-1",
        ]
    ) == 0
    assert received["backend"] == "openai-api"
    assert received["allow_cloud_processing"] is True
    assert received["openai_model"] == "whisper-1"


def test_parser_returns_configuration_code_for_invalid_invocation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_cli_dependencies(monkeypatch)

    assert cli.main([]) == 1
    captured = capsys.readouterr()
    assert "Пакет не запущен" in captured.err
    assert "--input" in captured.err


def test_parser_rejects_unknown_backend_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_cli_dependencies(monkeypatch)

    assert cli.main(["--input", "sample.mp4", "--backend", "unknown"]) == 1
    captured = capsys.readouterr()
    assert "--backend" in captured.err
    assert "Traceback" not in captured.err


def test_parser_rejects_unknown_output_format_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_cli_dependencies(monkeypatch)

    assert cli.main(["--input", "sample.mp4", "--output-format", "txt"]) == 1
    captured = capsys.readouterr()
    assert "--output-format" in captured.err
    assert "Traceback" not in captured.err


def test_environment_error_returns_one_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_cli_dependencies(monkeypatch)
    monkeypatch.setattr(
        cli,
        "settings_from_environment",
        lambda: (_ for _ in ()).throw(ValidationError("некорректный ASR_DEVICE")),
    )

    assert cli.main(["--input", "sample.mp4"]) == 1
    captured = capsys.readouterr()
    assert "некорректный ASR_DEVICE" in captured.err
    assert "Traceback" not in captured.err
