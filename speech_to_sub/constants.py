from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = BASE_DIR / "resources"
CACHE_DIR = RESOURCES_DIR / "cache"
WORK_DIR = RESOURCES_DIR / "work"
LOGS_DIR = BASE_DIR / "logs"

DEFAULT_TRANSFORMERS_MODEL_PATH = Path(
    r"D:\Projects\-ai\+automatic-speech-recognition\whisper-large-v3"
)
DEFAULT_FASTER_WHISPER_MODEL_PATH = Path(
    r"D:\Projects\-ai\+automatic-speech-recognition\whisper-large-v3-ct2"
)
DEFAULT_PARAKEET_MODEL_PATH = Path(
    r"D:\Projects\-ai\+automatic-speech-recognition\parakeet-tdt-0.6b-v3"
)
DEFAULT_QWEN_MODEL_PATH = Path(
    r"D:\Projects\-ai\+automatic-speech-recognition\Qwen3-ASR-0.6B"
)
DEFAULT_QWEN_ALIGNER_MODEL_PATH = Path(
    r"D:\Projects\-ai\+automatic-speech-recognition\Qwen3-ForcedAligner-0.6B"
)
DEFAULT_BACKEND_MODEL_PATHS = {
    "transformers": DEFAULT_TRANSFORMERS_MODEL_PATH,
    "faster-whisper": DEFAULT_FASTER_WHISPER_MODEL_PATH,
    "parakeet-tdt-v3": DEFAULT_PARAKEET_MODEL_PATH,
    "qwen3-asr": DEFAULT_QWEN_MODEL_PATH,
}
DEFAULT_ASR_BACKEND = "faster-whisper"
DEFAULT_ALIGNER = "none"
DEFAULT_MODEL_PATH = DEFAULT_FASTER_WHISPER_MODEL_PATH
DEFAULT_LANGUAGE = "en"
DEFAULT_AUDIO_LANGUAGE = "eng"
DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 7862
DEFAULT_MAX_CHARS_PER_LINE = 42
DEFAULT_CHUNK_LENGTH_SECONDS = 30
DEFAULT_STRIDE_LENGTH_SECONDS = 5
DEFAULT_LONG_FORM_WINDOW_SECONDS = 300
DEFAULT_LONG_FORM_OVERLAP_SECONDS = 2
QWEN_ALIGNMENT_MAX_SEGMENT_SECONDS = 180
DEFAULT_VAD_MIN_SILENCE_MS = 600
DEFAULT_BEAM_SIZE = 5
MAX_BATCH_PATHS = 10_000
PIPELINE_VERSION = "6"
SRT_BUILDER_VERSION = "1"

SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".m4v", ".mov", ".mkv", ".webm"})
SUPPORTED_AUDIO_EXTENSIONS = frozenset(
    {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".mka"}
)
SUPPORTED_MEDIA_EXTENSIONS = SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS
