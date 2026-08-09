from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from speech_to_sub.asr.registry import unload_backends
from speech_to_sub.media.ffmpeg import probe_media
from speech_to_sub.service import process_paths
from speech_to_sub.subtitles.validator import validate_srt_text
from speech_to_sub.utils.env_utils import load_environment, settings_from_environment
from speech_to_sub.utils.io_utils import read_text_utf8


@pytest.mark.integration
def test_real_multitrack_media_uses_explicit_audio_stream(tmp_path: Path) -> None:
    """Проверяет две реальные дорожки одним локальным checkpoint без повторной загрузки."""
    load_environment()
    if not _enabled("RUN_MULTITRACK_ASR_TEST"):
        pytest.skip("Установите RUN_MULTITRACK_ASR_TEST=1 для multi-track проверки.")

    media_value = os.getenv("ASR_MULTITRACK_MEDIA", "").strip()
    if not media_value:
        pytest.skip("Для multi-track проверки задайте ASR_MULTITRACK_MEDIA.")
    media_path = Path(media_value).expanduser().resolve()
    if not media_path.is_file():
        pytest.skip(f"Multi-track медиафайл не найден: {media_path}")

    probe = probe_media(media_path)
    assert len(probe.streams) >= 2, "Тест требует минимум две аудиодорожки."
    cases = (
        (_stream_index("ASR_MULTITRACK_RU_STREAM_INDEX", 0), "ru"),
        (_stream_index("ASR_MULTITRACK_EN_STREAM_INDEX", 1), "en"),
    )
    base_settings = settings_from_environment()

    try:
        for stream_index, language in cases:
            output_dir = tmp_path / language
            settings = replace(
                base_settings,
                language=language,
                audio_stream_index=stream_index,
                output_dir=output_dir,
                force=True,
                keep_audio=False,
                recursive=False,
            )
            results = process_paths(
                [media_path],
                settings.to_dict(),
                emit_event=lambda _event: None,
                log=lambda _message: None,
            )

            assert len(results) == 1
            assert results[0]["state"] == "done", results[0].get("error")
            srt_path = Path(str(results[0]["srt_output"]))
            sidecar_path = Path(str(results[0]["sidecar_output"]))
            sidecar = json.loads(read_text_utf8(sidecar_path))
            assert sidecar["selected_stream"]["ordinal"] == stream_index
            assert sidecar["recognition_settings"]["language"] == language
            assert sidecar["transcript"]["language"] == language
            assert sidecar["transcript"]["text"].strip()
            validate_srt_text(read_text_utf8(srt_path))
    finally:
        unload_backends()


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().casefold() in {"1", "true", "yes", "on", "да"}


def _stream_index(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise AssertionError(f"{name} должен быть целым числом.") from exc
    if value < 0:
        raise AssertionError(f"{name} не может быть отрицательным.")
    return value
