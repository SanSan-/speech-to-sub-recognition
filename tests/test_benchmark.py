from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from speech_to_sub import benchmark
from speech_to_sub.benchmark import ResourceSampler, analyze_srt
from tools.benchmark_asr import _build_parser


def test_benchmark_parser_accepts_separate_aligner_worker() -> None:
    args = _build_parser().parse_args(
        [
            "--input",
            "sample.mkv",
            "--backend",
            "parakeet-tdt-v3",
            "--model-path",
            "parakeet-model",
            "--aligner",
            "qwen3-forced-aligner",
            "--aligner-model-path",
            "qwen-aligner-model",
            "--worker-python-path",
            "parakeet-python.exe",
            "--aligner-worker-python-path",
            "qwen-python.exe",
            "--output-dir",
            "outputs",
            "--result",
            "result.json",
        ]
    )

    assert args.worker_python_path == Path("parakeet-python.exe")
    assert args.aligner_worker_python_path == Path("qwen-python.exe")


def test_analyze_srt_reports_quality_and_boundary_metrics() -> None:
    sidecar = {
        "transcript": {
            "duration": 12.0,
            "metadata": {"window_seconds": 7.0, "overlap_seconds": 2.0},
            "segments": [
                {
                    "words": [
                        {"start": 4.0, "end": 4.2, "text": "one"},
                        {"start": 4.3, "end": 4.5, "text": "two"},
                        {"start": 4.6, "end": 4.8, "text": "three"},
                        {"start": 5.1, "end": 5.3, "text": "one"},
                        {"start": 5.4, "end": 5.6, "text": "two"},
                        {"start": 5.7, "end": 5.9, "text": "three"},
                    ]
                }
            ],
        }
    }
    content = (
        "1\n00:00:00,000 --> 00:00:01,000\none two three\n\n"
        "2\n00:00:02,000 --> 00:00:10,000\n"
        "one two three one two three one two three one two three\n"
    )

    metrics = analyze_srt(content, audio_duration=12.0, sidecar=sidecar)

    assert metrics["cue_count"] == 2
    assert metrics["overlap_count"] == 0
    assert metrics["short_cue_count"] == 0
    assert metrics["long_cue_count"] == 1
    assert metrics["long_line_count"] == 1
    assert metrics["repeated_5gram_occurrences"] > 0
    assert metrics["boundary_repeated_3gram_occurrences"] == 1
    assert metrics["word_timestamp_count"] == 6
    assert metrics["non_monotonic_word_timestamp_count"] == 0
    assert metrics["word_timestamps_after_300_seconds"] == 0
    assert metrics["last_word_end_seconds"] == 5.9
    assert metrics["tail_gap_seconds"] == 2.0


def test_analyze_srt_counts_fast_and_short_cue() -> None:
    content = (
        "1\n00:00:00,000 --> 00:00:00,500\n"
        "This subtitle is intentionally much too fast for half a second.\n"
    )

    metrics = analyze_srt(content, audio_duration=1.0)

    assert metrics["short_cue_count"] == 1
    assert metrics["reading_speed_violations"] == 1


def test_resource_sampler_counts_worker_process_rss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = SimpleNamespace(pid=2, memory_info=lambda: SimpleNamespace(rss=200))
    parent = SimpleNamespace(
        pid=1,
        memory_info=lambda: SimpleNamespace(rss=100),
        children=lambda recursive: [child] if recursive else [],
    )
    monkeypatch.setattr(benchmark.psutil, "Process", lambda _pid: parent)
    monkeypatch.setattr(benchmark, "_gpu_used_vram_mib", lambda: None)

    sampler = ResourceSampler()
    sampler._sample_once(include_vram=False)

    assert sampler.start_rss_bytes == 300
    assert sampler.peak_rss_bytes == 300
    assert sampler.to_dict()["rss_scope"] == "process_tree"
