"""Запускает один изолированный ASR benchmark и атомарно сохраняет JSON-результат."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speech_to_sub.alignment.registry import aligner_names, unload_aligners  # noqa: E402
from speech_to_sub.asr.registry import backend_names, unload_backends  # noqa: E402
from speech_to_sub.benchmark import ResourceSampler, analyze_srt  # noqa: E402
from speech_to_sub.media.ffmpeg import probe_media  # noqa: E402
from speech_to_sub.models import ProcessingSettings  # noqa: E402
from speech_to_sub.service import process_paths  # noqa: E402
from speech_to_sub.utils.io_utils import (  # noqa: E402
    atomic_write_json,
    read_text_utf8,
)


def main(argv: list[str] | None = None) -> int:
    _configure_console()
    parser = _build_parser()
    args = parser.parse_args(argv)
    started_at = datetime.now(timezone.utc)
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    result_path = args.result.expanduser().resolve()
    settings_values: dict[str, Any] = {
        "backend": args.backend,
        "model_path": args.model_path,
        "aligner": args.aligner,
        "aligner_model_path": args.aligner_model_path,
        "worker_python_path": args.worker_python_path,
        "aligner_worker_python_path": args.aligner_worker_python_path,
        "language": args.language,
        "device": args.device,
        "quantization_enabled": args.quantization,
        "allow_cpu_fallback": args.allow_cpu_fallback,
        "force": True,
        "keep_audio": False,
        "recursive": False,
        "output_dir": output_dir,
        "long_form_window_seconds": args.window_seconds,
        "long_form_overlap_seconds": args.overlap_seconds,
        "vad_filter": args.vad_filter,
        "vad_min_silence_ms": args.vad_min_silence_ms,
        "beam_size": args.beam_size,
        "condition_on_previous_text": args.condition_on_previous_text,
    }
    settings = ProcessingSettings.from_mapping(settings_values)
    media_duration = probe_media(input_path).duration
    emitted: list[dict[str, Any]] = []
    wall_started = time.perf_counter()
    error: str | None = None
    results: list[dict[str, Any]] = []
    with ResourceSampler() as sampler:
        try:
            results = process_paths(
                [input_path],
                settings.to_dict(),
                emit_event=emitted.append,
                log=lambda message: print(message, flush=True),
            )
        except Exception as exc:  # benchmark обязан сохранить диагностический JSON
            error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                unload_backends()
            except Exception as unload_error:
                if error is None:
                    error = f"Ошибка выгрузки backend: {unload_error}"
            try:
                unload_aligners()
            except Exception as unload_error:
                if error is None:
                    error = f"Ошибка выгрузки aligner: {unload_error}"
    wall_seconds = time.perf_counter() - wall_started
    finished_at = datetime.now(timezone.utc)
    item = results[0] if results else {}
    if error is None and item.get("state") != "done":
        error = str(item.get("error") or f"Неожиданное состояние: {item.get('state')}")
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "input": str(input_path),
        "backend": settings.backend,
        "model_path": str(settings.model_path),
        "aligner": settings.aligner,
        "aligner_model_path": (
            str(settings.aligner_model_path) if settings.aligner_model_path else None
        ),
        "worker_python_path": (
            str(settings.worker_python_path) if settings.worker_python_path else None
        ),
        "aligner_worker_python_path": (
            str(settings.aligner_worker_python_path)
            if settings.aligner_worker_python_path
            else None
        ),
        "language": settings.language,
        "device": settings.device,
        "quantization_enabled": settings.quantization_enabled,
        "media_duration_seconds": media_duration,
        "wall_seconds": round(wall_seconds, 3),
        "rtf": round(wall_seconds / media_duration, 6) if media_duration > 0 else None,
        "resources": sampler.to_dict(),
        "state": item.get("state", "error"),
        "error": error,
        "event_count": len(emitted),
    }
    if error is None:
        srt_path = Path(str(item["srt_output"]))
        sidecar_path = Path(str(item["sidecar_output"]))
        sidecar = json.loads(read_text_utf8(sidecar_path))
        report["srt_path"] = str(srt_path)
        report["sidecar_path"] = str(sidecar_path)
        recognition_settings = sidecar.get("recognition_settings")
        if (
            not isinstance(recognition_settings, dict)
            and "sidecar_schema_version" not in sidecar
        ):
            recognition_settings = sidecar.get("settings", {})
        if not isinstance(recognition_settings, dict):
            recognition_settings = {}
        report["runtime"] = recognition_settings.get("runtime")
        report["metrics"] = analyze_srt(
            read_text_utf8(srt_path),
            audio_duration=float(sidecar.get("audio", {}).get("duration", media_duration)),
            sidecar=sidecar,
            max_chars_per_line=settings.max_chars_per_line,
            line_length_gap=settings.line_length_gap,
        )
    atomic_write_json(result_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if error is None else 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Локальный ASR long-form benchmark")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--backend", choices=backend_names(), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--aligner", choices=aligner_names(), default="none")
    parser.add_argument("--aligner-model-path", type=Path)
    parser.add_argument("--worker-python-path", type=Path)
    parser.add_argument("--aligner-worker-python-path", type=Path)
    parser.add_argument("--language", choices=("en", "ru", "auto"), default="auto")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=300.0)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    parser.add_argument("--vad-min-silence-ms", type=int, default=600)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--no-quantization", dest="quantization", action="store_false")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--no-vad", dest="vad_filter", action="store_false")
    parser.add_argument(
        "--no-condition-on-previous-text",
        dest="condition_on_previous_text",
        action="store_false",
    )
    parser.set_defaults(quantization=True, vad_filter=True, condition_on_previous_text=True)
    return parser


def _configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
