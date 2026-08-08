from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from speech_to_sub.exceptions import MediaError
from speech_to_sub.media.ffmpeg import (
    build_ffmpeg_normalize_args,
    decode_audio_float32,
    get_media_duration,
    normalize_audio,
    parse_ffprobe_payload,
    probe_media,
    select_audio_stream,
)
from speech_to_sub.models import AudioStreamInfo


def _stream(
    ordinal: int,
    index: int,
    *,
    language: str | None = None,
    title: str | None = None,
) -> AudioStreamInfo:
    return AudioStreamInfo(
        ordinal=ordinal,
        index=index,
        codec_name="aac",
        sample_rate=48_000,
        channels=2,
        language=language,
        title=title,
        duration=12.5,
    )


def test_probe_media_uses_argument_list_and_parses_audio_streams() -> None:
    payload = {
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264"},
            {
                "index": 2,
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
                "duration": "12.500",
                "tags": {"language": "eng", "title": "English"},
            },
        ],
        "format": {"duration": "12.750", "format_name": "mov,mp4"},
    }
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

    probe = probe_media(Path("лекция с пробелом.mp4"), "custom-ffprobe", runner=runner)

    assert probe.duration == 12.75
    assert probe.format_name == "mov,mp4"
    assert probe.streams == (_stream(0, 2, language="eng", title="English"),)
    args, kwargs = calls[0]
    assert args[0] == "custom-ffprobe"
    assert args[-1] == "лекция с пробелом.mp4"
    assert kwargs["shell"] is False
    assert kwargs["encoding"] == "utf-8"


def test_parse_ffprobe_uses_stream_duration_when_format_duration_is_missing() -> None:
    probe = parse_ffprobe_payload(
        "audio.mka",
        {
            "streams": [
                {
                    "index": 3,
                    "codec_type": "audio",
                    "codec_name": "flac",
                    "tags": {"DURATION": "00:01:02.500000000"},
                }
            ]
        },
    )

    assert probe.duration == 62.5
    assert get_media_duration(probe) == 62.5


def test_probe_media_rejects_tool_error_and_invalid_json() -> None:
    def failed(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, "", "нет аудиопотока")

    def invalid(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, "not-json", "")

    with pytest.raises(MediaError, match="нет аудиопотока"):
        probe_media("broken.mp4", runner=failed)
    with pytest.raises(MediaError, match="некорректный JSON"):
        probe_media("broken.mp4", runner=invalid)


def test_select_audio_stream_follows_priority_and_matches_title() -> None:
    streams = (
        _stream(0, 1, language="jpn", title="Japanese"),
        _stream(1, 4, language=None, title="English commentary"),
    )

    assert select_audio_stream(streams, 0, "eng") == (streams[0], None)
    assert select_audio_stream(streams, preferred_language="eng") == (streams[1], None)


def test_select_audio_stream_warns_on_ambiguous_match_and_fallback() -> None:
    streams = (
        _stream(0, 1, language="eng"),
        _stream(1, 4, language="en"),
    )
    warnings: list[str] = []

    selected, warning = select_audio_stream(
        streams,
        preferred_language="eng",
        warning_callback=warnings.append,
    )
    assert selected is streams[0]
    assert warning is not None and "Несколько" in warning
    assert warning == warnings[-1]

    warnings.clear()
    selected, warning = select_audio_stream(streams, warning_callback=warnings.append)
    assert selected is streams[0]
    assert warning is not None and "первый" in warning
    assert warning == warnings[-1]


def test_select_audio_stream_rejects_missing_explicit_ordinal() -> None:
    with pytest.raises(MediaError, match="доступны: 0"):
        select_audio_stream((_stream(0, 7),), requested_ordinal=7)


def test_normalize_audio_builds_safe_command_and_requires_created_file(tmp_path: Path) -> None:
    source = tmp_path / "исходник с пробелом.mp4"
    source.write_bytes(b"media")
    destination = tmp_path / "work" / "normalized.flac"
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        Path(args[-1]).write_bytes(b"fLaC")
        return subprocess.CompletedProcess(args, 0, "", "")

    result = normalize_audio(
        source,
        destination,
        _stream(0, 3),
        "custom-ffmpeg",
        runner=runner,
    )

    assert result == destination
    args, kwargs = calls[0]
    assert args == build_ffmpeg_normalize_args(
        source,
        destination,
        _stream(0, 3),
        "custom-ffmpeg",
    )
    assert "-nostdin" in args
    assert args[args.index("-map") + 1] == "0:3"
    assert "-vn" in args
    assert args[args.index("-ac") + 1] == "1"
    assert args[args.index("-ar") + 1] == "16000"
    assert "-n" in args
    assert kwargs["shell"] is False


def test_normalize_audio_does_not_hide_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    destination = tmp_path / "normalized.flac"
    source.write_bytes(b"media")
    destination.write_bytes(b"ready")

    with pytest.raises(MediaError, match="перезапись не разрешена"):
        normalize_audio(source, destination, _stream(0, 1))

    args = build_ffmpeg_normalize_args(source, destination, _stream(0, 1), overwrite=True)
    assert "-y" in args
    assert "-n" not in args


def test_decode_audio_float32_uses_configured_ffmpeg(tmp_path: Path) -> None:
    source = tmp_path / "normalized.flac"
    source.write_bytes(b"fLaC")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args,
            0,
            b"\x00\x00\x00\x00\x00\x00\x80\x3f",
            b"",
        )

    samples = decode_audio_float32(source, "D:\\Tools\\ffmpeg.exe", runner=runner)

    assert samples.tolist() == [0.0, 1.0]
    args, kwargs = calls[0]
    assert args[0] == "D:\\Tools\\ffmpeg.exe"
    assert args[-1] == "pipe:1"
    assert "-nostdin" in args
    assert kwargs["shell"] is False
