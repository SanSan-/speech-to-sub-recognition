from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

from speech_to_sub.asr.registry import backend_names
from speech_to_sub.alignment.registry import aligner_names
from speech_to_sub.constants import (
    DEFAULT_BACKEND_MODEL_PATHS,
    DEFAULT_QWEN_ALIGNER_MODEL_PATH,
    SUPPORTED_OPENAI_MODELS,
)
from speech_to_sub.exceptions import SpeechToSubError, ValidationError
from speech_to_sub.models import ProcessingSettings
from speech_to_sub.service import process_paths
from speech_to_sub.utils.env_utils import load_environment, settings_from_environment
from speech_to_sub.utils.logging_utils import setup_logging


class _CliArgumentParser(argparse.ArgumentParser):
    """Преобразует ошибки аргументов в документированный код конфигурации."""

    def error(self, message: str) -> None:
        raise ValidationError(f"Некорректные аргументы командной строки: {message}")


def build_parser() -> argparse.ArgumentParser:
    """Создаёт CLI parser."""
    parser = _CliArgumentParser(
        prog="speech-to-sub",
        description="Пакетное распознавание речи и создание SRT.",
    )
    parser.add_argument("--input", nargs="+", required=True, type=Path, help="Файл или папка.")
    parser.add_argument("--backend", choices=backend_names(), help="Движок распознавания.")
    parser.add_argument(
        "--aligner",
        choices=aligner_names(),
        help="Необязательное выравнивание слов.",
    )
    parser.add_argument("--aligner-model-path", type=Path)
    parser.add_argument("--worker-python-path", type=Path)
    parser.add_argument("--aligner-worker-python-path", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Корневой каталог результатов.")
    parser.add_argument("--recursive", action="store_true", default=None)
    parser.add_argument("--language", choices=("en", "ru", "auto"))
    parser.add_argument("--audio-language")
    parser.add_argument("--audio-stream-index", type=int)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--no-auto-download-model",
        dest="auto_download_model",
        action="store_false",
        default=None,
        help="Запретить докачивание неполной локальной модели.",
    )
    parser.add_argument(
        "--allow-cloud-processing",
        action="store_true",
        default=None,
        help="Разрешить передачу аудио выбранному облачному сервису.",
    )
    parser.add_argument(
        "--openai-model",
        choices=tuple(sorted(SUPPORTED_OPENAI_MODELS)),
        help="Модель распознавания OpenAI API.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--no-quantization",
        dest="quantization_enabled",
        action="store_false",
        default=None,
        help="Отключить 8-битную загрузку, не отключая CUDA.",
    )
    parser.add_argument("--allow-cpu-fallback", action="store_true", default=None)
    parser.add_argument("--max-chars-per-line", type=int)
    parser.add_argument("--line-length-gap", type=int)
    parser.add_argument("--max-cps", type=float)
    parser.add_argument("--long-form-window-seconds", type=int)
    parser.add_argument("--long-form-overlap-seconds", type=int)
    parser.add_argument("--no-vad", dest="vad_filter", action="store_false", default=None)
    parser.add_argument("--vad-min-silence-ms", type=int)
    parser.add_argument("--beam-size", type=int)
    parser.add_argument(
        "--no-condition-on-previous-text",
        dest="condition_on_previous_text",
        action="store_false",
        default=None,
    )
    parser.add_argument("--keep-audio", action="store_true", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        default=None,
        help=(
            "Обойти кеш готового SRT и распознавания, заново выполнить распознавание и "
            "выбранное выравнивание, затем атомарно заменить целевые результаты."
        ),
    )
    parser.add_argument("--verbose", action="store_true", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Запускает CLI и возвращает документированный код завершения."""
    _configure_console_utf8()
    logger: logging.Logger | None = None
    try:
        load_environment()
        args = build_parser().parse_args(argv)
        settings = _merge_settings(settings_from_environment(), args)
        logger = setup_logging(settings.verbose)
        results = process_paths(
            args.input,
            settings.to_dict(),
            emit_event=lambda event: _log_event(logger, event),
            log=lambda _message: None,
        )
    except (SpeechToSubError, OSError, ValueError) as exc:
        if logger is not None:
            logger.error("Пакет не запущен: %s", exc)
        else:
            print(f"Пакет не запущен: {exc}", file=sys.stderr)
        return 1
    assert logger is not None
    failed = [item for item in results if item.get("state") == "error"]
    ready = [item for item in results if item.get("state") in {"done", "cached"}]
    skipped = [item for item in results if item.get("state") == "skipped"]
    logger.info(
        "Итог: готово=%d, пропущено=%d, ошибок=%d.",
        len(ready),
        len(skipped),
        len(failed),
    )
    return 2 if failed else 0


def _configure_console_utf8() -> None:
    """Явно включает UTF-8 для русских сообщений CLI в Windows-терминале."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                continue


def _merge_settings(base: ProcessingSettings, args: argparse.Namespace) -> ProcessingSettings:
    changes: dict[str, object] = {}
    for name in (
        "backend",
        "aligner",
        "aligner_model_path",
        "worker_python_path",
        "aligner_worker_python_path",
        "output_dir",
        "recursive",
        "language",
        "audio_language",
        "audio_stream_index",
        "model_path",
        "auto_download_model",
        "allow_cloud_processing",
        "openai_model",
        "device",
        "quantization_enabled",
        "allow_cpu_fallback",
        "max_chars_per_line",
        "line_length_gap",
        "max_cps",
        "long_form_window_seconds",
        "long_form_overlap_seconds",
        "vad_filter",
        "vad_min_silence_ms",
        "beam_size",
        "condition_on_previous_text",
        "keep_audio",
        "force",
        "verbose",
    ):
        value = getattr(args, name, None)
        if value is not None:
            changes[name] = value
    if (
        args.backend in DEFAULT_BACKEND_MODEL_PATHS
        and args.model_path is None
        and not os.getenv("ASR_MODEL_PATH")
    ):
        changes["model_path"] = DEFAULT_BACKEND_MODEL_PATHS[args.backend]
    if args.aligner is not None and args.aligner_model_path is None:
        if args.aligner == "qwen3-forced-aligner" and not os.getenv(
            "ASR_ALIGNER_MODEL_PATH"
        ):
            changes["aligner_model_path"] = DEFAULT_QWEN_ALIGNER_MODEL_PATH
        elif args.aligner == "none":
            changes["aligner_model_path"] = None
    return ProcessingSettings.from_mapping({**base.to_dict(), **changes})


def _log_event(logger: logging.Logger, event: dict[str, object]) -> None:
    if event.get("type") != "file":
        return
    if event.get("state") in {"done", "cached", "skipped", "error"}:
        logger.debug(
            "Состояние %s: %s.",
            event.get("state"),
            event.get("path"),
        )


__all__ = ["build_parser", "main"]
