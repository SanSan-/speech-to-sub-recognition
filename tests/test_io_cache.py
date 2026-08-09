from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from speech_to_sub.exceptions import ValidationError
from speech_to_sub.constants import SIDECAR_SCHEMA_VERSION
from speech_to_sub.models import ProcessingSettings
from speech_to_sub.utils import io_utils
from speech_to_sub.utils.cache import (
    build_layout_fingerprint,
    build_recognition_fingerprint,
    build_source_fingerprint,
    load_sidecar,
    sidecar_fingerprints,
    sidecar_matches,
    sidecar_recognition_matches,
    write_sidecar,
)
from speech_to_sub.utils.io_utils import (
    atomic_write_text_utf8,
    discover_media,
    has_utf8_bom,
    read_text_utf8,
    validate_model_path,
)


def test_utf8_reader_is_strict_and_atomic_writer_does_not_add_bom(
    tmp_path: Path,
) -> None:
    target = tmp_path / "результат.txt"
    atomic_write_text_utf8(target, "Первая строка\nSecond line\n")

    expected = "Первая строка\nSecond line\n".encode("utf-8")
    assert target.read_bytes() == expected
    assert read_text_utf8(target) == expected.decode("utf-8")
    assert has_utf8_bom(target) is False

    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"valid\xffinvalid")
    with pytest.raises(UnicodeDecodeError):
        read_text_utf8(invalid)


def test_atomic_writer_keeps_old_file_and_cleans_temporary_on_publish_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "result.txt"
    target.write_text("старое значение", encoding="utf-8", newline="")

    def fail_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("ошибка публикации")

    monkeypatch.setattr(io_utils.os, "replace", fail_replace)

    with pytest.raises(OSError, match="ошибка публикации"):
        atomic_write_text_utf8(target, "новое значение")

    assert target.read_text(encoding="utf-8") == "старое значение"
    assert list(tmp_path.iterdir()) == [target]


def test_discover_media_is_recursive_sorted_and_deduplicated(tmp_path: Path) -> None:
    first = tmp_path / "A.MP4"
    second = tmp_path / "b.wav"
    nested = tmp_path / "nested" / "c.mkv"
    unsupported = tmp_path / "notes.txt"
    for path in (first, second, nested, unsupported):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")

    shallow = discover_media((tmp_path, first), recursive=False)
    recursive = discover_media((tmp_path, first), recursive=True)

    assert shallow == [first.resolve(), second.resolve()]
    assert recursive == [first.resolve(), second.resolve(), nested.resolve()]


def test_discover_media_ignores_generated_asr_audio_in_folders(tmp_path: Path) -> None:
    source = tmp_path / "lecture.mp4"
    generated = tmp_path / "lecture.en.asr.flac"
    source.write_bytes(b"source")
    generated.write_bytes(b"generated")

    assert discover_media((tmp_path,), recursive=False) == [source.resolve()]
    assert discover_media((generated,), recursive=False) == [generated.resolve()]


def test_validate_model_path_requires_local_configs_and_weights(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
        (model_path / name).write_text("{}\n", encoding="utf-8", newline="\n")

    with pytest.raises(ValidationError, match="веса"):
        validate_model_path(model_path)

    (model_path / "model-00001-of-00002.safetensors").write_bytes(b"weights")
    assert validate_model_path(model_path) == model_path.resolve()

    (model_path / "tokenizer_config.json").unlink()
    with pytest.raises(ValidationError, match="tokenizer_config.json"):
        validate_model_path(model_path)


def test_cache_fingerprints_separate_recognition_and_layout_settings(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"media-v1")

    source = build_source_fingerprint(source_path)
    assert source["path"] == str(source_path.resolve())
    assert source["size"] == len(b"media-v1")
    assert source["sha256"] == hashlib.sha256(b"media-v1").hexdigest()

    settings = ProcessingSettings(
        backend="transformers",
        model_path=tmp_path / "model",
        language="ru",
    )
    runtime = {
        "backend": "transformers",
        "engine_version": "5.6.2",
        "device": "cpu",
        "compute_type": "float32",
        "quantized": False,
    }
    recognition = build_recognition_fingerprint(
        settings,
        stream_ordinal=2,
        runtime=runtime,
    )
    layout = build_layout_fingerprint(settings)
    operational_only = replace(
        settings,
        force=True,
        keep_audio=True,
        output_dir=tmp_path / "outputs",
        verbose=True,
    )
    assert build_recognition_fingerprint(operational_only, 2, runtime=runtime) == recognition
    assert build_layout_fingerprint(operational_only) == layout
    assert build_recognition_fingerprint(
        replace(settings, language="en"),
        2,
        runtime=runtime,
    ) != recognition
    for changed_layout in (
        replace(settings, max_chars_per_line=40),
        replace(settings, line_length_gap=0),
        replace(settings, max_cps=16.5),
    ):
        assert build_recognition_fingerprint(
            changed_layout,
            2,
            runtime=runtime,
        ) == recognition
        assert build_layout_fingerprint(changed_layout) != layout
    assert recognition["pipeline_version"] == "7"
    assert "srt_builder_version" not in recognition
    assert layout == {
        "srt_builder_version": "3",
        "max_chars_per_line": 42,
        "line_length_gap": 8,
        "max_cps": 17.0,
    }
    assert build_recognition_fingerprint(settings, 1, runtime=runtime) != recognition
    assert build_recognition_fingerprint(
        settings,
        2,
        runtime={
            "backend": "transformers",
            "engine_version": "5.6.2",
            "device": "cuda",
            "compute_type": "int8",
            "quantized": True,
        },
    ) != recognition

    faster_settings = replace(settings, backend="faster-whisper")
    faster_runtime = {**runtime, "backend": "faster-whisper"}
    faster_fingerprint = build_recognition_fingerprint(
        faster_settings,
        2,
        runtime=faster_runtime,
    )
    assert build_recognition_fingerprint(
        replace(faster_settings, long_form_window_seconds=240),
        2,
        runtime=faster_runtime,
    ) != faster_fingerprint
    assert build_recognition_fingerprint(
        replace(settings, long_form_window_seconds=240),
        2,
        runtime=runtime,
    ) == recognition
    assert build_recognition_fingerprint(
        settings,
        2,
        runtime={
            **runtime,
            "backend": "faster-whisper",
        },
    ) != recognition
    assert build_recognition_fingerprint(
        settings,
        2,
        runtime={
            **runtime,
            "engine_version": "5.7.0",
        },
    ) != recognition

    aligner_settings = replace(
        settings,
        backend="qwen3-asr",
        aligner="qwen3-forced-aligner",
        aligner_model_path=tmp_path / "qwen-aligner",
    )
    qwen_runtime = {
        **runtime,
        "backend": "qwen3-asr",
        "engine_version": "0.0.6",
        "device": "cuda",
        "compute_type": "bfloat16",
    }
    aligner_runtime = {
        **qwen_runtime,
        "backend": "qwen3-forced-aligner",
    }
    aligned_fingerprint = build_recognition_fingerprint(
        aligner_settings,
        2,
        runtime=qwen_runtime,
        aligner_runtime=aligner_runtime,
    )
    assert aligned_fingerprint["aligner"]["id"] == "qwen3-forced-aligner"
    assert aligned_fingerprint["aligner"]["parameters"] == {
        "max_segment_seconds": 180,
    }
    assert build_recognition_fingerprint(
        replace(aligner_settings, aligner_model_path=tmp_path / "other-aligner"),
        2,
        runtime=qwen_runtime,
        aligner_runtime=aligner_runtime,
    ) != aligned_fingerprint
    assert build_recognition_fingerprint(
        aligner_settings,
        2,
        runtime=qwen_runtime,
        aligner_runtime={**aligner_runtime, "engine_version": "0.0.7"},
    ) != aligned_fingerprint

    source_path.write_bytes(b"media-v2")
    assert build_source_fingerprint(source_path)["sha256"] != source["sha256"]


def test_cloud_fingerprint_excludes_local_runtime_options_and_permission(
    tmp_path: Path,
) -> None:
    settings = ProcessingSettings(
        backend="openai-api",
        model_path=tmp_path / "ignored-local-model",
        allow_cloud_processing=True,
        openai_model="whisper-1",
        device="cuda",
        quantization_enabled=True,
    )
    runtime = {
        "backend": "openai-api",
        "engine_version": "2.14.0",
        "device": "cloud",
        "compute_type": "remote:whisper-1",
        "quantized": False,
    }

    fingerprint = build_recognition_fingerprint(settings, 0, runtime=runtime)

    assert fingerprint["openai_model"] == "whisper-1"
    assert "model_path" not in fingerprint
    assert "requested_device" not in fingerprint
    assert "requested_quantization_enabled" not in fingerprint
    assert build_recognition_fingerprint(
        replace(
            settings,
            model_path=tmp_path / "other",
            allow_cloud_processing=False,
            auto_download_model=False,
            device="cpu",
            quantization_enabled=False,
        ),
        0,
        runtime=runtime,
    ) == fingerprint


def test_sidecar_round_trip_and_match_validation(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "sample.asr.json"
    source = {"sha256": "source-hash"}
    recognition = {"pipeline_version": "7", "language": "ru"}
    layout = {
        "srt_builder_version": "3",
        "max_chars_per_line": 42,
        "line_length_gap": 8,
        "max_cps": 17.0,
    }
    payload = {
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
        "status": "done",
        "source": source,
        "recognition_settings": recognition,
        "layout_settings": layout,
        "message": "Готово",
    }

    write_sidecar(sidecar_path, payload)

    assert sidecar_path.read_bytes()[:3] != b"\xef\xbb\xbf"
    assert load_sidecar(sidecar_path) == payload
    assert sidecar_matches(load_sidecar(sidecar_path), source, recognition, layout) is True
    assert sidecar_recognition_matches(payload, source, recognition) is True
    assert sidecar_matches(
        {**payload, "status": "error"},
        source,
        recognition,
        layout,
    ) is False
    assert sidecar_matches(
        payload,
        {"sha256": "other"},
        recognition,
        layout,
    ) is False
    assert sidecar_matches(
        payload,
        source,
        recognition,
        {**layout, "max_cps": 15.0},
    ) is False
    assert sidecar_recognition_matches(
        payload,
        source,
        {**recognition, "language": "en"},
    ) is False

    sidecar_path.write_bytes(b"not-json")
    assert load_sidecar(sidecar_path) is None
    sidecar_path.write_text("[]\n", encoding="utf-8", newline="\n")
    assert load_sidecar(sidecar_path) is None


def test_legacy_pipeline7_builder2_sidecar_reuses_only_recognition() -> None:
    recognition = {"pipeline_version": "7", "language": "ru"}
    legacy = {
        "status": "done",
        "source": {"sha256": "source-hash"},
        "settings": {
            **recognition,
            "srt_builder_version": "2",
            "max_chars_per_line": 42,
        },
    }

    fingerprints = sidecar_fingerprints(legacy)

    assert fingerprints == (
        recognition,
        {
            "srt_builder_version": "2",
            "max_chars_per_line": 42,
        },
    )
    assert sidecar_recognition_matches(
        legacy,
        legacy["source"],
        recognition,
    ) is True
    extended_layout = {
        **legacy,
        "settings": {
            **legacy["settings"],
            "line_length_gap": 8,
            "max_cps": 17.0,
        },
    }
    assert sidecar_fingerprints(extended_layout) == (
        recognition,
        {
            "srt_builder_version": "2",
            "max_chars_per_line": 42,
            "line_length_gap": 8,
            "max_cps": 17.0,
        },
    )
    assert sidecar_fingerprints(
        {
            **legacy,
            "settings": {**legacy["settings"], "pipeline_version": "6"},
        }
    ) is None
    assert sidecar_fingerprints(
        {
            **legacy,
            "settings": {**legacy["settings"], "srt_builder_version": "3"},
        }
    ) is None
    assert sidecar_fingerprints({**legacy, "sidecar_schema_version": 99}) is None
    assert sidecar_fingerprints({**legacy, "sidecar_schema_version": 2.0}) is None
