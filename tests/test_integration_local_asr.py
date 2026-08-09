from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from speech_to_sub.asr.registry import unload_backends
from speech_to_sub.service import process_paths
from speech_to_sub.subtitles.validator import validate_srt_text
from speech_to_sub.utils.env_utils import load_environment, settings_from_environment
from speech_to_sub.utils.io_utils import read_text_utf8


@pytest.mark.integration
def test_real_local_whisper_creates_valid_srt(tmp_path: Path) -> None:
    """Запускает выбранный локальный backend только по явному флагу."""
    load_environment()
    if not _enabled("RUN_LOCAL_ASR_TEST"):
        pytest.skip("Установите RUN_LOCAL_ASR_TEST=1 для тяжёлой проверки.")

    media_value = os.getenv("ASR_TEST_MEDIA", "").strip()
    if not media_value:
        pytest.skip("Для тяжёлой проверки задайте ASR_TEST_MEDIA.")
    media_path = Path(media_value).expanduser().resolve()
    if not media_path.is_file():
        pytest.skip(f"Тестовый медиафайл не найден: {media_path}")

    base_settings = settings_from_environment()
    language = os.getenv("ASR_TEST_LANGUAGE", base_settings.language).strip()
    settings = replace(
        base_settings,
        language=language,
        output_dir=tmp_path,
        force=True,
        keep_audio=False,
        recursive=False,
    )

    try:
        results = process_paths(
            [media_path],
            settings.to_dict(),
            emit_event=lambda _event: None,
            log=lambda _message: None,
        )
    finally:
        unload_backends()

    assert len(results) == 1
    assert results[0]["state"] == "done", results[0].get("error")
    srt_path = Path(str(results[0]["srt_output"]))
    sidecar_path = Path(str(results[0]["sidecar_output"]))
    assert srt_path.is_file()
    assert sidecar_path.is_file()
    assert not srt_path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert not sidecar_path.read_bytes().startswith(b"\xef\xbb\xbf")
    validate_srt_text(read_text_utf8(srt_path))
    sidecar = json.loads(read_text_utf8(sidecar_path))
    assert sidecar["recognition_settings"]["runtime"]["backend"] == settings.backend


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().casefold() in {"1", "true", "yes", "on", "да"}
