from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from speech_to_sub.asr.registry import unload_backends
from speech_to_sub.service import process_paths
from speech_to_sub.subtitles.validator import validate_srt_text
from speech_to_sub.utils.env_utils import load_environment, settings_from_environment
from speech_to_sub.utils.io_utils import read_text_utf8


@pytest.mark.integration
def test_real_batch_continues_after_error_and_reuses_cache(tmp_path: Path) -> None:
    """Проверяет реальную пачку done/done/error и повтор cached/cached/error."""
    load_environment()
    if not _enabled("RUN_BATCH_ASR_TEST"):
        pytest.skip("Установите RUN_BATCH_ASR_TEST=1 для реальной batch-проверки.")

    media_value = (
        os.getenv("ASR_BATCH_TEST_MEDIA", "").strip()
        or os.getenv("ASR_TEST_MEDIA", "").strip()
    )
    if not media_value:
        pytest.skip("Для batch-проверки задайте ASR_BATCH_TEST_MEDIA.")
    source = Path(media_value).expanduser().resolve()
    if not source.is_file():
        pytest.skip(f"Тестовый медиафайл не найден: {source}")

    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    suffix = source.suffix.casefold()
    shutil.copyfile(source, inputs / f"01-valid{suffix}")
    shutil.copyfile(source, inputs / f"02-valid{suffix}")
    (inputs / "03-broken.mp4").write_bytes(b"intentionally broken media")

    base_settings = settings_from_environment()
    language = os.getenv("ASR_TEST_LANGUAGE", base_settings.language).strip()
    settings = replace(
        base_settings,
        language=language,
        output_dir=outputs,
        force=False,
        keep_audio=False,
        recursive=False,
    )
    first_events: list[dict[str, object]] = []

    try:
        first = process_paths(
            [inputs],
            settings.to_dict(),
            emit_event=first_events.append,
            log=lambda _message: None,
        )
        second = process_paths(
            [inputs],
            settings.to_dict(),
            emit_event=lambda _event: None,
            log=lambda _message: None,
        )
    finally:
        unload_backends()

    assert [item["state"] for item in first] == ["done", "done", "error"]
    assert [item["state"] for item in second] == ["cached", "cached", "error"]
    assert any(event.get("state") == "error" for event in first_events)
    for item in first[:2]:
        srt_path = Path(str(item["srt_output"]))
        sidecar_path = Path(str(item["sidecar_output"]))
        validate_srt_text(read_text_utf8(srt_path))
        sidecar = json.loads(read_text_utf8(sidecar_path))
        assert sidecar["status"] == "done"
        assert sidecar["started_at"] <= sidecar["finished_at"]


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().casefold() in {"1", "true", "yes", "on", "да"}
