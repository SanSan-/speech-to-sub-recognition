from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from filelock import FileLock

from speech_to_sub import service
from speech_to_sub.exceptions import MediaError, ValidationError
from speech_to_sub.models import (
    AudioStreamInfo,
    MediaProbe,
    ProcessingSettings,
    RuntimeSignature,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)
from speech_to_sub.subtitles.validator import parse_srt


TEST_ENGINE_VERSION = "test-engine-1"


class FakeBackend:
    """Лёгкий backend без импорта checkpoint и ML-зависимостей."""

    backend_id = "transformers"

    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.durations: list[float] = []

    def transcribe(
        self,
        audio_path: Path,
        settings: ProcessingSettings,
        duration: float,
        progress_callback: Any = None,
        cancel_check: Any = None,
    ) -> Transcript:
        if cancel_check and cancel_check():
            raise AssertionError("FakeBackend не должен запускаться после отмены")
        assert audio_path.read_bytes() == b"normalized-audio"
        self.calls.append(audio_path)
        self.durations.append(duration)
        if progress_callback:
            progress_callback(50)
        return Transcript(
            text="Тестовая расшифровка.",
            language=settings.language,
            duration=duration,
            segments=(TranscriptSegment(0.0, 1.2, "Тестовая расшифровка."),),
            model=str(settings.model_path),
            device="cpu",
            quantized=False,
            metadata={
                "runtime": self.backend_id,
                "engine_version": TEST_ENGINE_VERSION,
                "compute_type": "float32",
            },
        )

    def preflight(self, settings: ProcessingSettings) -> Path:
        return settings.model_path

    def expected_runtime_signature(
        self,
        settings: ProcessingSettings,
    ) -> RuntimeSignature:
        del settings
        return {
            "backend": self.backend_id,
            "engine_version": TEST_ENGINE_VERSION,
            "device": "cpu",
            "compute_type": "float32",
            "quantized": False,
        }

    def runtime_signature(
        self,
        settings: ProcessingSettings,
    ) -> RuntimeSignature | None:
        del settings
        return None

    def unload(self) -> None:
        return None


class RuntimeAwareFakeBackend(FakeBackend):
    """Имитирует сохранённый CPU backend после разрешённого fallback."""

    def expected_runtime_signature(
        self,
        settings: ProcessingSettings,
    ) -> RuntimeSignature:
        del settings
        return {
            "backend": self.backend_id,
            "engine_version": TEST_ENGINE_VERSION,
            "device": "cuda",
            "compute_type": "int8",
            "quantized": True,
        }

    def runtime_signature(
        self,
        settings: ProcessingSettings,
    ) -> RuntimeSignature | None:
        if settings.device != "auto" or not settings.allow_cpu_fallback:
            return None
        return {
            "backend": self.backend_id,
            "engine_version": TEST_ENGINE_VERSION,
            "device": "cpu",
            "compute_type": "float32",
            "quantized": False,
        }


def _probe(path: str | Path) -> MediaProbe:
    return MediaProbe(
        path=Path(path),
        duration=2.0,
        streams=(
            AudioStreamInfo(
                ordinal=0,
                index=1,
                codec_name="aac",
                sample_rate=48_000,
                channels=2,
                language="rus",
            ),
        ),
        format_name="mov,mp4",
    )


def _install_fast_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> list[Path]:
    normalized_sources: list[Path] = []
    monkeypatch.setattr(service, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(service, "_command_exists", lambda _command: True)
    monkeypatch.setattr(
        service,
        "probe_media",
        lambda path, ffprobe_path="ffprobe": _probe(path),
    )

    def fake_normalize(
        source: str | Path,
        destination: str | Path,
        stream: AudioStreamInfo,
        ffmpeg_path: str | Path = "ffmpeg",
        *,
        overwrite: bool = False,
    ) -> Path:
        del stream, ffmpeg_path
        assert overwrite is True
        source_path = Path(source)
        destination_path = Path(destination)
        normalized_sources.append(source_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"normalized-audio")
        return destination_path

    monkeypatch.setattr(service, "normalize_audio", fake_normalize)
    monkeypatch.setattr(service, "get_media_duration", lambda *_args, **_kwargs: 1.5)
    lookup_backend = FakeBackend()
    monkeypatch.setattr(service, "get_backend", lambda _name: lookup_backend)
    monkeypatch.setattr(service, "activate_backend", lambda _name: lookup_backend)
    return normalized_sources


def test_build_items_reports_idle_skipped_and_probe_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idle = tmp_path / "01-idle.mp4"
    skipped = tmp_path / "02-skipped.mp4"
    broken = tmp_path / "03-broken.mp4"
    for path in (idle, skipped, broken):
        path.write_bytes(b"media")
    skipped.with_name("02-skipped.en.srt").write_text(
        "existing",
        encoding="utf-8",
        newline="\n",
    )

    def fake_probe(path: str | Path, ffprobe_path: str = "ffprobe") -> MediaProbe:
        del ffprobe_path
        if Path(path) == broken.resolve():
            raise MediaError("повреждённый контейнер")
        return _probe(path)

    monkeypatch.setattr(service, "probe_media", fake_probe)
    items = service.build_items([tmp_path])
    by_name = {item["name"]: item for item in items}

    assert by_name[idle.name]["state"] == "idle"
    assert by_name[idle.name]["selected_stream"]["index"] == 1
    assert by_name[idle.name]["source_fingerprint"]["sha256"]
    assert by_name[idle.name]["source_fingerprint"]["path"] == str(idle.resolve())
    assert by_name[skipped.name]["state"] == "skipped"
    assert by_name[skipped.name]["skipped"] is True
    assert by_name[broken.name]["state"] == "error"
    assert by_name[broken.name]["source_fingerprint"]["sha256"]
    assert "повреждённый контейнер" in by_name[broken.name]["error"]


def test_cleanup_stale_workspaces_removes_only_safe_unlocked_job_directories(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    stale = work_dir / "job-stale"
    active = work_dir / "job-active"
    fresh = work_dir / "job-fresh"
    unmanaged = work_dir / "job-unmanaged"
    wrong_prefix = work_dir / "session-stale"
    outside = tmp_path / "job-outside"
    for path in (stale, active, fresh, unmanaged, wrong_prefix, outside):
        path.mkdir()
        (path / "data.flac").write_bytes(b"audio")
    (stale / service.WORKSPACE_LOCK_NAME).write_text("", encoding="utf-8")
    (fresh / service.WORKSPACE_LOCK_NAME).write_text("", encoding="utf-8")
    active_lock = FileLock(str(active / service.WORKSPACE_LOCK_NAME), timeout=0)
    active_lock.acquire()
    try:
        for path in (
            stale,
            stale / service.WORKSPACE_LOCK_NAME,
            active,
            active / service.WORKSPACE_LOCK_NAME,
            unmanaged,
            wrong_prefix,
            outside,
        ):
            os.utime(path, (100.0, 100.0))
        for path in (fresh, fresh / service.WORKSPACE_LOCK_NAME):
            os.utime(path, (950.0, 950.0))

        removed = service.cleanup_stale_workspaces(
            work_dir,
            ttl_seconds=100,
            now=1_000.0,
        )
    finally:
        active_lock.release()

    assert removed == [stale.resolve()]
    assert not stale.exists()
    assert active.is_dir()
    assert fresh.is_dir()
    assert unmanaged.is_dir()
    assert wrong_prefix.is_dir()
    assert outside.is_dir()


def test_preflight_uses_selected_registry_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend()
    backend.backend_id = "faster-whisper"
    selected: list[str] = []
    monkeypatch.setattr(service, "_command_exists", lambda _command: True)

    def get_selected(name: str) -> FakeBackend:
        selected.append(name)
        return backend

    monkeypatch.setattr(service, "get_backend", get_selected)

    status = service.get_preflight_status(
        {
            "backend": "faster-whisper",
            "model_path": str(tmp_path / "ctranslate2-model"),
        }
    )

    assert status["status"] == "ok"
    assert status["backend"] == {
        "id": "faster-whisper",
        "available": True,
        "error": None,
    }
    assert selected == ["faster-whisper"]


def test_process_paths_writes_utf8_srt_sidecar_and_reuses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "лекция.mp4"
    original_media = b"original-media"
    media.write_bytes(original_media)
    normalized_sources = _install_fast_runtime(monkeypatch, tmp_path)
    backend = FakeBackend()
    events: list[dict[str, Any]] = []
    logs: list[str] = []

    first = service.process_paths(
        [media],
        {"language": "ru"},
        events.append,
        logs.append,
        backend=backend,
    )

    srt_path = tmp_path / "лекция.ru.srt"
    sidecar_path = tmp_path / "лекция.ru.asr.json"
    assert first[0]["state"] == "done"
    assert first[0]["srt_output"] == str(srt_path)
    assert srt_path.read_bytes()[:3] != b"\xef\xbb\xbf"
    assert "Тестовая расшифровка." in srt_path.read_text(encoding="utf-8")
    assert sidecar_path.read_bytes()[:3] != b"\xef\xbb\xbf"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["status"] == "done"
    assert sidecar["started_at"] <= sidecar["finished_at"]
    assert sidecar["created_at"] == sidecar["finished_at"]
    assert sidecar["sidecar_schema_version"] == 2
    assert sidecar["source"]["sha256"]
    assert "settings" not in sidecar
    assert sidecar["recognition_settings"]["language"] == "ru"
    assert sidecar["recognition_settings"]["runtime"] == {
        "backend": "transformers",
        "engine_version": TEST_ENGINE_VERSION,
        "device": "cpu",
        "compute_type": "float32",
        "quantized": False,
    }
    assert sidecar["layout_settings"] == {
        "srt_builder_version": "3",
        "max_chars_per_line": 42,
        "line_length_gap": 8,
        "max_cps": 17.0,
    }
    assert sidecar["normalized_audio_duration"] == 1.5
    assert sidecar["transcript"]["text"] == "Тестовая расшифровка."
    assert sidecar["transcript"]["duration"] == 1.5
    assert sidecar["transcript"]["segments"][0]["words"] == []
    assert sidecar["subtitle_layout"] == {
        "reconciled_segments": 0,
        "alignment_text_segments": 0,
        "synthetic_timing_segments": 1,
        "retimed_leading_islands": 0,
        "adjusted_boundaries": 0,
        "max_boundary_drift_ms": 0,
        "timing_anomaly_adjustments": 0,
    }
    assert any(
        "синтезированы временные метки сегментов: 1" in message
        for message in logs
    )
    assert not any(
        "фрагменты с аномальными начальными метками" in message
        for message in logs
    )
    assert media.read_bytes() == original_media
    assert len(backend.calls) == 1
    assert normalized_sources == [media.resolve()]
    assert backend.durations == [1.5]
    assert any(event.get("state") == "done" for event in events)

    cards = service.build_items([media], {"language": "ru"})
    assert cards[0]["state"] == "cached"
    assert cards[0]["cached"] is True

    cached = service.process_paths(
        [media],
        {"language": "ru"},
        events.append,
        logs.append,
        backend=backend,
    )
    assert cached[0]["state"] == "cached"
    assert cached[0]["cached"] is True
    assert len(backend.calls) == 1
    assert normalized_sources == [media.resolve()]


def test_layout_change_requires_force_and_reuses_recognition_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "layout.mp4"
    media.write_bytes(b"source")
    normalized_sources = _install_fast_runtime(monkeypatch, tmp_path)
    backend = FakeBackend()
    callbacks = (lambda _event: None, lambda _message: None)

    first = service.process_paths([media], {"language": "ru"}, *callbacks, backend=backend)
    srt_path = Path(first[0]["srt_output"])
    original_srt = srt_path.read_bytes()
    sidecar_path = Path(first[0]["sidecar_output"])
    first_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    skipped = service.process_paths(
        [media],
        {"language": "ru", "line_length_gap": 0},
        *callbacks,
        backend=backend,
    )

    assert skipped[0]["state"] == "skipped"
    assert srt_path.read_bytes() == original_srt
    assert json.loads(sidecar_path.read_text(encoding="utf-8")) == first_sidecar

    rebuilt = service.process_paths(
        [media],
        {"language": "ru", "line_length_gap": 0, "force": True},
        *callbacks,
        backend=backend,
    )
    rebuilt_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert rebuilt[0]["state"] == "done"
    assert srt_path.read_bytes() == original_srt
    assert len(backend.calls) == 1
    assert normalized_sources == [media.resolve()]
    assert rebuilt_sidecar["recognition_settings"] == first_sidecar["recognition_settings"]
    assert rebuilt_sidecar["layout_settings"]["line_length_gap"] == 0


def test_missing_srt_is_rebuilt_from_recognition_cache_without_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "missing-srt.mp4"
    media.write_bytes(b"source")
    normalized_sources = _install_fast_runtime(monkeypatch, tmp_path)
    backend = FakeBackend()
    callbacks = (lambda _event: None, lambda _message: None)
    first = service.process_paths([media], {"language": "ru"}, *callbacks, backend=backend)
    srt_path = Path(first[0]["srt_output"])
    srt_path.unlink()
    cards = service.build_items([media], {"language": "ru"})

    rebuilt = service.process_paths(
        [media],
        {"language": "ru"},
        *callbacks,
        backend=backend,
    )

    assert rebuilt[0]["state"] == "done"
    assert cards[0]["state"] == "idle"
    assert cards[0]["stage"] == "Пересборка SRT из кеша распознавания"
    assert srt_path.is_file()
    assert len(backend.calls) == 1
    assert normalized_sources == [media.resolve()]


def test_recognition_setting_change_runs_backend_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "recognition-change.mp4"
    media.write_bytes(b"source")
    normalized_sources = _install_fast_runtime(monkeypatch, tmp_path)
    backend = FakeBackend()
    callbacks = (lambda _event: None, lambda _message: None)
    service.process_paths([media], {"language": "ru"}, *callbacks, backend=backend)

    result = service.process_paths(
        [media],
        {"language": "ru", "long_form_window_seconds": 240, "force": True},
        *callbacks,
        backend=backend,
    )

    assert result[0]["state"] == "done"
    assert len(backend.calls) == 2
    assert normalized_sources == [media.resolve(), media.resolve()]


def test_malformed_cached_transcript_falls_back_to_recognition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "malformed-cache.mp4"
    media.write_bytes(b"source")
    normalized_sources = _install_fast_runtime(monkeypatch, tmp_path)
    backend = FakeBackend()
    callbacks = (lambda _event: None, lambda _message: None)
    first = service.process_paths([media], {"language": "ru"}, *callbacks, backend=backend)
    Path(first[0]["srt_output"]).unlink()
    sidecar_path = Path(first[0]["sidecar_output"])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["transcript"]["duration"] = "1.5"
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    result = service.process_paths(
        [media],
        {"language": "ru"},
        *callbacks,
        backend=backend,
    )

    assert result[0]["state"] == "done"
    assert len(backend.calls) == 2
    assert normalized_sources == [media.resolve(), media.resolve()]


def test_source_mismatch_does_not_reuse_recognition_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "source-change.mp4"
    media.write_bytes(b"source-v1")
    normalized_sources = _install_fast_runtime(monkeypatch, tmp_path)
    backend = FakeBackend()
    callbacks = (lambda _event: None, lambda _message: None)
    first = service.process_paths([media], {"language": "ru"}, *callbacks, backend=backend)
    Path(first[0]["srt_output"]).unlink()
    media.write_bytes(b"source-v2")

    result = service.process_paths(
        [media],
        {"language": "ru"},
        *callbacks,
        backend=backend,
    )

    assert result[0]["state"] == "done"
    assert len(backend.calls) == 2
    assert normalized_sources == [media.resolve(), media.resolve()]


def test_legacy_pipeline7_builder2_sidecar_rebuilds_without_recognition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "legacy-cache.mp4"
    media.write_bytes(b"source")
    normalized_sources = _install_fast_runtime(monkeypatch, tmp_path)
    backend = FakeBackend()
    callbacks = (lambda _event: None, lambda _message: None)
    first = service.process_paths([media], {"language": "ru"}, *callbacks, backend=backend)
    Path(first[0]["srt_output"]).unlink()
    sidecar_path = Path(first[0]["sidecar_output"])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    recognition = sidecar.pop("recognition_settings")
    sidecar.pop("sidecar_schema_version")
    sidecar.pop("layout_settings")
    sidecar["settings"] = {
        **recognition,
        "srt_builder_version": "2",
        "max_chars_per_line": 42,
    }
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    result = service.process_paths(
        [media],
        {"language": "ru"},
        *callbacks,
        backend=backend,
    )
    migrated = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert result[0]["state"] == "done"
    assert len(backend.calls) == 1
    assert normalized_sources == [media.resolve()]
    assert migrated["sidecar_schema_version"] == 2
    assert "settings" not in migrated


def test_process_paths_extends_final_word_cue_within_audio_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "финал.mp4"
    media.write_bytes(b"media")
    _install_fast_runtime(monkeypatch, tmp_path)

    class FinalWordBackend(FakeBackend):
        def transcribe(
            self,
            audio_path: Path,
            settings: ProcessingSettings,
            duration: float,
            progress_callback: Any = None,
            cancel_check: Any = None,
        ) -> Transcript:
            del audio_path, progress_callback, cancel_check
            word = TranscriptWord(start=1.3, end=1.5, text="Финал.")
            return Transcript(
                text=word.text,
                language=settings.language,
                duration=duration,
                segments=(
                    TranscriptSegment(
                        start=word.start,
                        end=word.end,
                        text=word.text,
                        words=(word,),
                    ),
                ),
                model=str(settings.model_path),
                device="cpu",
                quantized=False,
                metadata={
                    "runtime": self.backend_id,
                    "engine_version": TEST_ENGINE_VERSION,
                    "compute_type": "float32",
                },
            )

    result = service.process_paths(
        [media],
        {"language": "ru"},
        lambda _event: None,
        lambda _message: None,
        backend=FinalWordBackend(),
    )

    assert result[0]["state"] == "done"
    srt_text = Path(result[0]["srt_output"]).read_text(encoding="utf-8")
    cue, = parse_srt(srt_text)
    assert cue.end == pytest.approx(1.5)
    assert cue.end - cue.start >= 0.8


def test_layout_diagnostics_log_contains_only_nonzero_events() -> None:
    messages: list[str] = []

    service._log_subtitle_layout_diagnostics(
        Path("лекция.mp4"),
        messages.append,
        {
            "reconciled_segments": 2,
            "alignment_text_segments": 0,
            "synthetic_timing_segments": 0,
            "retimed_leading_islands": 1,
            "adjusted_boundaries": 3,
            "max_boundary_drift_ms": 120,
        },
    )

    assert messages == [
        "лекция.mp4: разметка SRT — восстановлены текст и пунктуация сегментов: 2; "
        "перенесены фрагменты с аномальными начальными метками: 1; "
        "скорректированы временные границы реплик: 3; максимальный сдвиг границы, мс: 120."
    ]


def test_service_runs_exclusive_alignment_as_independent_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "alignment.mp4"
    media.write_bytes(b"media")
    _install_fast_runtime(monkeypatch, tmp_path)

    class ExclusiveBackend(FakeBackend):
        backend_id = "faster-whisper"

        def __init__(self) -> None:
            super().__init__()
            self.unload_calls = 0

        def unload(self) -> None:
            self.unload_calls += 1

    class FakeAligner:
        aligner_id = "qwen3-forced-aligner"
        requires_exclusive_runtime = True

        def __init__(self) -> None:
            self.unload_calls = 0
            self.calls: list[tuple[Path, float]] = []

        def preflight(self, settings: ProcessingSettings) -> Path | None:
            return settings.aligner_model_path

        def expected_runtime_signature(
            self,
            settings: ProcessingSettings,
        ) -> RuntimeSignature:
            del settings
            return {
                "backend": self.aligner_id,
                "engine_version": TEST_ENGINE_VERSION,
                "device": "cpu",
                "compute_type": "float32",
                "quantized": False,
            }

        def runtime_signature(
            self,
            settings: ProcessingSettings,
            transcript: Transcript | None = None,
        ) -> RuntimeSignature:
            del settings, transcript
            return self.expected_runtime_signature(ProcessingSettings())

        def align(
            self,
            audio_path: Path,
            transcript: Transcript,
            settings: ProcessingSettings,
            duration: float,
            progress_callback: Any = None,
            cancel_check: Any = None,
        ) -> Transcript:
            del settings
            assert cancel_check is not None and cancel_check() is False
            assert audio_path.read_bytes() == b"normalized-audio"
            self.calls.append((audio_path, duration))
            if progress_callback:
                progress_callback(50)
            return Transcript(
                text=transcript.text,
                language=transcript.language,
                duration=transcript.duration,
                segments=transcript.segments,
                model=transcript.model,
                device=transcript.device,
                quantized=transcript.quantized,
                metadata={
                    **transcript.metadata,
                    "alignment_status": "fallback",
                    "alignment_fallback": {"segment_index": 449},
                },
            )

        def unload(self) -> None:
            self.unload_calls += 1

    backend = ExclusiveBackend()
    aligner = FakeAligner()
    monkeypatch.setattr(service, "get_aligner", lambda _name: aligner)

    messages: list[str] = []
    result = service.process_paths(
        [media],
        {
            "backend": "faster-whisper",
            "aligner": "qwen3-forced-aligner",
            "aligner_model_path": str(tmp_path / "aligner"),
            "language": "ru",
        },
        lambda _event: None,
        messages.append,
        backend=backend,
        cancel_check=lambda: False,
    )

    assert result[0]["state"] == "done"
    assert len(aligner.calls) == 1
    assert aligner.calls[0][1] == pytest.approx(1.5)
    assert aligner.unload_calls == 1
    assert backend.unload_calls == 1
    assert any(
        "ForcedAligner не вернул слова для сегмента 449" in message
        and "faster-whisper" in message
        for message in messages
    )


@pytest.mark.parametrize(
    "max_cps",
    [4.9, 60.1, float("nan"), float("inf"), "17", True],
)
def test_service_rejects_invalid_max_cps(max_cps: Any) -> None:
    settings = ProcessingSettings(max_cps=max_cps)
    with pytest.raises(ValidationError, match="CPS"):
        service._validate_output_settings(settings)


@pytest.mark.parametrize("line_length_gap", [-1, 21, "8", True])
def test_service_rejects_invalid_line_length_gap(line_length_gap: Any) -> None:
    settings = ProcessingSettings(line_length_gap=line_length_gap)
    with pytest.raises(ValidationError, match="Допуск длины строки"):
        service._validate_output_settings(settings)


def test_cache_uses_loaded_backend_runtime_after_cpu_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "fallback.mp4"
    media.write_bytes(b"source")
    normalized_sources = _install_fast_runtime(monkeypatch, tmp_path)
    backend = RuntimeAwareFakeBackend()
    monkeypatch.setattr(service, "get_backend", lambda _name: backend)
    monkeypatch.setattr(service, "activate_backend", lambda _name: backend)

    first = service.process_paths(
        [media],
        {
            "language": "ru",
            "device": "auto",
            "quantization_enabled": True,
            "allow_cpu_fallback": True,
        },
        lambda _event: None,
        lambda _message: None,
        backend=backend,
    )
    cards = service.build_items(
        [media],
        {
            "language": "ru",
            "device": "auto",
            "quantization_enabled": True,
            "allow_cpu_fallback": True,
        },
    )
    second = service.process_paths(
        [media],
        {
            "language": "ru",
            "device": "auto",
            "quantization_enabled": True,
            "allow_cpu_fallback": True,
        },
        lambda _event: None,
        lambda _message: None,
        backend=backend,
    )

    assert first[0]["state"] == "done"
    assert cards[0]["state"] == "cached"
    assert second[0]["state"] == "cached"
    assert len(backend.calls) == 1
    assert normalized_sources == [media.resolve()]


def test_process_paths_activates_selected_registry_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "selected.mp4"
    media.write_bytes(b"source")
    _install_fast_runtime(monkeypatch, tmp_path)
    backend = FakeBackend()
    backend.backend_id = "faster-whisper"
    selected: list[str] = []

    monkeypatch.setattr(service, "get_backend", lambda _name: backend)

    def activate(name: str) -> FakeBackend:
        selected.append(name)
        return backend

    monkeypatch.setattr(service, "activate_backend", activate)

    results = service.process_paths(
        [media],
        {"backend": "faster-whisper", "language": "ru"},
        lambda _event: None,
        lambda _message: None,
    )

    assert results[0]["state"] == "done"
    assert selected == ["faster-whisper"]
    assert len(backend.calls) == 1


def test_existing_srt_is_skipped_unless_force_is_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"source")
    srt_path = tmp_path / "sample.en.srt"
    srt_path.write_text("existing", encoding="utf-8", newline="\n")
    normalized_sources = _install_fast_runtime(monkeypatch, tmp_path)
    backend = FakeBackend()

    skipped = service.process_paths(
        [media],
        {"language": "en"},
        lambda _event: None,
        lambda _message: None,
        backend=backend,
    )

    assert skipped[0]["state"] == "skipped"
    assert skipped[0]["skipped"] is True
    assert srt_path.read_text(encoding="utf-8") == "existing"
    assert normalized_sources == []
    assert backend.calls == []

    forced = service.process_paths(
        [media],
        {"language": "en", "force": True},
        lambda _event: None,
        lambda _message: None,
        backend=backend,
    )

    assert forced[0]["state"] == "done"
    assert "Тестовая расшифровка." in srt_path.read_text(encoding="utf-8")
    assert normalized_sources == [media.resolve()]
    assert len(backend.calls) == 1


def test_cache_adds_requested_audio_without_reloading_asr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"source")
    normalized_sources = _install_fast_runtime(monkeypatch, tmp_path)
    backend = FakeBackend()

    first = service.process_paths(
        [media],
        {"language": "ru", "device": "cpu"},
        lambda _event: None,
        lambda _message: None,
        backend=backend,
    )
    cards = service.build_items(
        [media],
        {"language": "ru", "device": "cpu", "keep_audio": True},
    )
    second = service.process_paths(
        [media],
        {"language": "ru", "device": "cpu", "keep_audio": True},
        lambda _event: None,
        lambda _message: None,
        backend=backend,
    )

    audio_path = tmp_path / "sample.ru.asr.flac"
    assert first[0]["state"] == "done"
    assert first[0]["audio_output"] is None
    assert cards[0]["state"] == "idle"
    assert cards[0]["stage"] == "Требуется сохранить аудио"
    assert second[0]["state"] == "cached"
    assert second[0]["audio_output"] == str(audio_path)
    assert audio_path.read_bytes() == b"normalized-audio"
    assert len(backend.calls) == 1
    assert normalized_sources == [media.resolve(), media.resolve()]


def test_process_paths_continues_after_one_file_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = tmp_path / "01-broken.mp4"
    valid = tmp_path / "02-valid.mp4"
    broken.write_bytes(b"broken")
    valid.write_bytes(b"valid")
    normalized_sources = _install_fast_runtime(monkeypatch, tmp_path)
    successful_normalize = service.normalize_audio

    def partial_normalize(
        source: str | Path,
        destination: str | Path,
        stream: AudioStreamInfo,
        ffmpeg_path: str | Path = "ffmpeg",
        *,
        overwrite: bool = False,
    ) -> Path:
        if Path(source) == broken.resolve():
            raise MediaError("тестовая ошибка FFmpeg")
        return successful_normalize(
            source,
            destination,
            stream,
            ffmpeg_path,
            overwrite=overwrite,
        )

    monkeypatch.setattr(service, "normalize_audio", partial_normalize)
    backend = FakeBackend()
    events: list[dict[str, Any]] = []

    results = service.process_paths(
        [broken, valid],
        {"language": "ru"},
        events.append,
        lambda _message: None,
        backend=backend,
    )

    assert [item["state"] for item in results] == ["error", "done"]
    assert "тестовая ошибка FFmpeg" in results[0]["error"]
    assert results[1]["srt_output"] == str(tmp_path / "02-valid.ru.srt")
    assert normalized_sources == [valid.resolve()]
    assert len(backend.calls) == 1
    assert any(
        event.get("path") == str(broken.resolve()) and event.get("state") == "error"
        for event in events
    )
    assert any(
        event.get("path") == str(valid.resolve()) and event.get("state") == "done"
        for event in events
    )


def test_process_paths_cancels_remaining_files_after_safe_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "01-first.mp4"
    second = tmp_path / "02-second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    normalized_sources = _install_fast_runtime(monkeypatch, tmp_path)
    events: list[dict[str, Any]] = []
    cancelled = False

    def emit(event: dict[str, Any]) -> None:
        nonlocal cancelled
        events.append(event)
        if event.get("path") == str(first.resolve()) and event.get("state") == "done":
            cancelled = True

    results = service.process_paths(
        [first, second],
        {"language": "ru"},
        emit,
        lambda _message: None,
        backend=FakeBackend(),
        cancel_check=lambda: cancelled,
    )

    assert [item["state"] for item in results] == ["done", "cancelled"]
    assert normalized_sources == [first.resolve()]
    assert not (tmp_path / "02-second.ru.srt").exists()
    assert any(
        event.get("path") == str(second.resolve())
        and event.get("state") == "cancelled"
        for event in events
    )


def test_same_basename_in_different_containers_gets_distinct_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mp4 = tmp_path / "sample.mp4"
    mkv = tmp_path / "sample.mkv"
    mp4.write_bytes(b"mp4")
    mkv.write_bytes(b"mkv")
    _install_fast_runtime(monkeypatch, tmp_path)

    results = service.process_paths(
        [mp4, mkv],
        {"language": "en", "output_dir": str(tmp_path / "out")},
        lambda _event: None,
        lambda _message: None,
        backend=FakeBackend(),
    )

    outputs = {Path(item["srt_output"]).name for item in results}
    assert outputs == {"sample.mp4.en.srt", "sample.mkv.en.srt"}
    assert all(Path(item["srt_output"]).is_file() for item in results)


def test_same_filename_without_common_root_gets_path_hash_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first" / "sample.mp4"
    second = tmp_path / "second" / "sample.mp4"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    _install_fast_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(service, "_common_input_root", lambda _paths: None)

    results = service.process_paths(
        [first, second],
        {"language": "en", "output_dir": str(tmp_path / "out")},
        lambda _event: None,
        lambda _message: None,
        backend=FakeBackend(),
    )

    output_names = [Path(item["srt_output"]).name for item in results]
    assert len(set(output_names)) == 2
    assert all(name.startswith("sample.mp4.") and name.endswith(".en.srt") for name in output_names)
    assert all(Path(item["srt_output"]).is_file() for item in results)


def test_output_lock_rejects_parallel_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"source")
    _install_fast_runtime(monkeypatch, tmp_path)
    settings = ProcessingSettings(language="en")
    outputs = service.build_output_paths(media.resolve(), settings)
    lock = FileLock(str(service._output_lock_path(outputs)), timeout=0)

    with lock:
        results = service.process_paths(
            [media],
            settings.to_dict(),
            lambda _event: None,
            lambda _message: None,
            backend=FakeBackend(),
        )

    assert results[0]["state"] == "error"
    assert "другим процессом" in results[0]["error"]
    assert not outputs.srt_path.exists()


def test_artifact_commit_restores_previous_files_on_partial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    srt = tmp_path / "sample.en.srt"
    sidecar = tmp_path / "sample.en.asr.json"
    staged_srt = tmp_path / ".sample.srt.stage"
    staged_sidecar = tmp_path / ".sample.json.stage"
    srt.write_text("old srt", encoding="utf-8", newline="")
    sidecar.write_text("old sidecar", encoding="utf-8", newline="")
    staged_srt.write_text("new srt", encoding="utf-8", newline="")
    staged_sidecar.write_text("new sidecar", encoding="utf-8", newline="")
    real_replace = service.os.replace

    def fail_sidecar_publish(source: Path, destination: Path) -> None:
        if Path(source) == staged_sidecar and Path(destination) == sidecar:
            raise OSError("тестовый сбой commit")
        real_replace(source, destination)

    monkeypatch.setattr(service.os, "replace", fail_sidecar_publish)

    with pytest.raises(OSError, match="тестовый сбой"):
        service._commit_staged_artifacts(
            {srt: staged_srt, sidecar: staged_sidecar},
            "transaction",
        )

    assert srt.read_text(encoding="utf-8") == "old srt"
    assert sidecar.read_text(encoding="utf-8") == "old sidecar"
