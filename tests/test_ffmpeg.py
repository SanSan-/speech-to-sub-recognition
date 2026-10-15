from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from speech_to_sub.exceptions import MediaError
from speech_to_sub.media import ffmpeg, windows_process
from speech_to_sub.media.ffmpeg import (
    build_ffmpeg_normalize_args,
    decode_audio_float32,
    get_media_duration,
    normalize_audio,
    parse_ffprobe_payload,
    probe_media,
    select_audio_stream,
)
from speech_to_sub.media.windows_process import CREATE_SUSPENDED, WindowsProcessJob
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
    assert kwargs["timeout"] == 30.0


def test_probe_media_timeout_is_a_bounded_russian_domain_error() -> None:
    def timed_out(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    with pytest.raises(MediaError, match=r"ffprobe не завершился за 0\.25 с"):
        probe_media("зависший файл.mp4", runner=timed_out, timeout_seconds=0.25)


def test_probe_media_real_subprocess_is_terminated_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ffmpeg,
        "build_ffprobe_args",
        lambda *_args, **_kwargs: [
            sys.executable,
            "-c",
            (
                "import subprocess, sys, time; "
                "subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(8)']); "
                "time.sleep(8)"
            ),
        ],
    )
    started = time.monotonic()

    with pytest.raises(MediaError, match="ffprobe не завершился"):
        probe_media("зависший файл.mp4", timeout_seconds=0.1)

    assert time.monotonic() - started < 2.0


@pytest.mark.skipif(os.name != "nt", reason="Проверка завершения дерева Windows")
@pytest.mark.parametrize("parent_exits", [False, True])
def test_terminate_process_tree_without_external_kill_utility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parent_exits: bool
) -> None:
    child_pid_path = tmp_path / "child.pid"
    script = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(8)']); "
        "pid_path = pathlib.Path(sys.argv[1]); "
        "pending_path = pid_path.with_suffix('.tmp'); "
        "pending_path.write_text(str(child.pid), encoding='utf-8'); "
        "pending_path.replace(pid_path); "
        f"time.sleep({0 if parent_exits else 8})"
    )
    job = WindowsProcessJob()
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(child_pid_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED,
    )
    child = None

    def unavailable_taskkill(args: list[str], **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", unavailable_taskkill)
    try:
        job.assign_and_resume(process.pid)
        ready_deadline = time.monotonic() + 3.0
        while not child_pid_path.is_file() and time.monotonic() < ready_deadline:
            time.sleep(0.01)
        assert child_pid_path.is_file()
        child = psutil.Process(int(child_pid_path.read_text(encoding="utf-8")))
        assert child.is_running()
        if parent_exits:
            process.wait(timeout=2.0)
        with pytest.raises(subprocess.TimeoutExpired):
            process.communicate(timeout=0.01)
        started = time.monotonic()

        ffmpeg._terminate_process_tree(process, windows_job=job)

        assert time.monotonic() - started < 2.0
        assert process.poll() is not None
        assert not child.is_running()
    finally:
        job.close()
        if child is not None and child.is_running():
            child.kill()
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=2.0)


@pytest.mark.skipif(os.name != "nt", reason="Проверка владения деревом Windows")
def test_probe_media_does_not_start_when_job_assignment_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = tmp_path / "started.txt"
    processes: list[subprocess.Popen[str]] = []
    popen = subprocess.Popen

    def capture_process(*args: object, **kwargs: object) -> subprocess.Popen[str]:
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    kernel32 = windows_process._load_kernel32()
    monkeypatch.setattr(kernel32, "AssignProcessToJobObject", lambda *_args: False)
    monkeypatch.setattr(windows_process, "_load_kernel32", lambda: kernel32)

    monkeypatch.setattr(subprocess, "Popen", capture_process)
    monkeypatch.setattr(
        ffmpeg,
        "build_ffprobe_args",
        lambda *_args, **_kwargs: [
            sys.executable,
            "-c",
            "import pathlib, sys; pathlib.Path(sys.argv[1]).write_text('started', encoding='utf-8')",
            str(marker),
        ],
    )

    with pytest.raises(OSError):
        probe_media("sample.mp4", timeout_seconds=0.1)

    assert not marker.exists()
    assert len(processes) == 1
    assert processes[0].poll() is not None


@pytest.mark.skipif(os.name != "nt", reason="Проверка дескрипторов Windows")
@pytest.mark.parametrize("operation", ["CreateJobObjectW", "SetInformationJobObject"])
def test_windows_process_job_releases_handle_on_setup_failure(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    kernel32 = windows_process._load_kernel32()
    close_handle = kernel32.CloseHandle
    closed_handles: list[int] = []

    def record_close(handle: int) -> int:
        closed_handles.append(handle)
        return close_handle(handle)

    monkeypatch.setattr(kernel32, operation, lambda *_args: 0)
    monkeypatch.setattr(kernel32, "CloseHandle", record_close)
    monkeypatch.setattr(windows_process, "_load_kernel32", lambda: kernel32)

    with pytest.raises(OSError):
        WindowsProcessJob()

    assert len(closed_handles) == (1 if operation == "SetInformationJobObject" else 0)


@pytest.mark.skipif(os.name != "nt", reason="Проверка дескрипторов Windows")
def test_windows_process_job_rejects_invalid_process_without_leaking_job() -> None:
    job = WindowsProcessJob()
    try:
        with pytest.raises(OSError):
            job.assign_and_resume(0)
    finally:
        job.close()
    job.close()


def test_probe_media_real_subprocess_returns_valid_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ffmpeg,
        "build_ffprobe_args",
        lambda *_args, **_kwargs: [
            sys.executable,
            "-c",
            "print('{\"streams\": [], \"format\": {\"duration\": \"12.5\"}}')",
        ],
    )

    probe = probe_media("sample.mp4")

    assert probe.duration == 12.5
    assert probe.streams == ()


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_probe_media_rejects_invalid_timeout(timeout: float) -> None:
    with pytest.raises(MediaError, match="Тайм-аут ffprobe"):
        probe_media("sample.mp4", timeout_seconds=timeout)


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
    streams = (_stream(0, 7),)
    with pytest.raises(MediaError, match="доступны: 0"):
        select_audio_stream(streams, requested_ordinal=7)


def test_normalize_audio_builds_safe_command_and_requires_created_file(
    tmp_path: Path,
) -> None:
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
    stream = _stream(0, 1)

    with pytest.raises(MediaError, match="перезапись не разрешена"):
        normalize_audio(source, destination, stream)

    args = build_ffmpeg_normalize_args(
        source, destination, _stream(0, 1), overwrite=True
    )
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
