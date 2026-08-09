from __future__ import annotations

import json
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from speech_to_sub.asr.openai_api import (
    MAX_UPLOAD_BYTES,
    OpenAiApiBackend,
    OpenAiAuthenticationError,
    OpenAiAudioPreparationError,
    OpenAiRateLimitError,
    OpenAiResponseError,
)
from speech_to_sub.exceptions import ProcessingCancelled, ValidationError
from speech_to_sub.models import ProcessingSettings


class _FakeTranscriptions:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.audio = SimpleNamespace(transcriptions=_FakeTranscriptions(responses))
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _response(text: str, words: list[Any], duration: float) -> Any:
    return SimpleNamespace(
        text=text,
        language="english",
        duration=duration,
        words=words,
        segments=[SimpleNamespace(start=0.0, end=duration, text=text)],
    )


def _word(start: float, end: float, word: str) -> Any:
    return SimpleNamespace(start=start, end=end, word=word)


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "normalized.flac"
    path.write_bytes(b"audio")
    return path


def _encoder(calls: list[list[str]], *, size: int = 5) -> Any:
    def run(args: list[str], **_kwargs: Any) -> Any:
        calls.append(args)
        Path(args[-1]).write_bytes(b"x" * size)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def _settings(
    *,
    language: str = "en",
    openai_model: str = "whisper-1",
) -> ProcessingSettings:
    return ProcessingSettings(
        backend="openai-api",
        language=language,
        allow_cloud_processing=True,
        openai_model=openai_model,
    )


def test_preflight_requires_explicit_backend_and_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAiApiBackend()
    default_settings = ProcessingSettings()
    cloud_without_consent = ProcessingSettings(backend="openai-api")
    cloud_settings = _settings()

    with pytest.raises(ValidationError, match="явного выбора"):
        backend.preflight(default_settings)
    with pytest.raises(ValidationError, match="явного согласия"):
        backend.preflight(cloud_without_consent)
    with pytest.raises(OpenAiAuthenticationError, match="OPENAI_API_KEY"):
        backend.preflight(cloud_settings)


def test_expected_runtime_needs_neither_key_nor_consent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    runtime = OpenAiApiBackend().expected_runtime_signature(
        ProcessingSettings(backend="openai-api")
    )

    assert runtime["device"] == "cloud"
    assert runtime["compute_type"] == "remote:whisper-1"


def test_whisper_chunks_audio_and_normalizes_absolute_word_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-token")
    monkeypatch.setenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")
    responses = [
        _response(
            "Hello boundary",
            [_word(10.0, 10.5, "Hello"), _word(898.5, 899.5, "boundary")],
            900.0,
        ),
        _response(
            "boundary world.",
            [_word(0.5, 1.5, "boundary"), _word(10.0, 10.5, "world.")],
            102.0,
        ),
    ]
    client = _FakeClient(responses)
    factory_calls: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> _FakeClient:
        factory_calls.append(kwargs)
        return client

    encode_calls: list[list[str]] = []
    backend = OpenAiApiBackend(factory, _encoder(encode_calls))
    progress: list[int] = []

    transcript = backend.transcribe(
        _audio(tmp_path),
        _settings(),
        1000.0,
        progress_callback=progress.append,
    )

    words = [word for segment in transcript.segments for word in segment.words]
    assert [(word.start, word.end, word.text) for word in words] == [
        (10.0, 10.5, "Hello"),
        (898.5, 899.5, "boundary"),
        (908.0, 908.5, "world."),
    ]
    assert transcript.text == "Hello boundary world."
    assert transcript.device == "cloud"
    assert transcript.metadata["timestamp_fallback"] == "none"
    assert progress == [0, 90, 99, 100]
    assert len(encode_calls) == 2
    assert factory_calls[0]["max_retries"] == 2
    assert factory_calls[0]["api_key"] == "unit-test-token"
    first_call, second_call = client.audio.transcriptions.calls
    assert first_call["response_format"] == "verbose_json"
    assert first_call["timestamp_granularities"] == ["segment", "word"]
    assert second_call["prompt"] == "Hello boundary"
    assert "unit-test-token" not in json.dumps(transcript.to_dict(), ensure_ascii=False)


def test_other_openai_models_are_outside_v15_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    backend = OpenAiApiBackend()
    settings = _settings(language="auto", openai_model="gpt-transcribe")

    with pytest.raises(ValidationError, match="не поддерживается"):
        backend.preflight(settings)


def test_explicit_language_uses_whisper_language_parameter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")
    client = _FakeClient([_response("Текст.", [_word(0.0, 0.8, "Текст.")], 1.0)])
    backend = OpenAiApiBackend(lambda **_kwargs: client, _encoder([]))

    backend.transcribe(_audio(tmp_path), _settings(language="ru"), 1.0)

    request = client.audio.transcriptions.calls[0]
    assert request["language"] == "ru"


def test_rate_limit_is_mapped_without_exposing_provider_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "private-marker")
    request = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")
    response = httpx.Response(429, request=request)
    provider_error = openai.RateLimitError(
        "provider detail private-marker",
        response=response,
        body=None,
    )
    client = _FakeClient([provider_error])
    backend = OpenAiApiBackend(lambda **_kwargs: client, _encoder([]))
    audio_path = _audio(tmp_path)
    settings = _settings()

    with pytest.raises(OpenAiRateLimitError) as caught:
        backend.transcribe(audio_path, settings, 1.0)

    assert "private-marker" not in str(caught.value)
    rendered = "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb)
    )
    assert "private-marker" not in rendered


def test_empty_response_is_rejected_as_typed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    client = _FakeClient([SimpleNamespace(text="", words=[], segments=[])])
    backend = OpenAiApiBackend(lambda **_kwargs: client, _encoder([]))
    audio_path = _audio(tmp_path)
    settings = _settings()

    with pytest.raises(OpenAiResponseError, match="пустое распознавание"):
        backend.transcribe(audio_path, settings, 1.0)


def test_cancellation_happens_before_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    encode_calls: list[list[str]] = []
    backend = OpenAiApiBackend(
        lambda **_kwargs: _FakeClient([]), _encoder(encode_calls)
    )
    audio_path = _audio(tmp_path)
    settings = _settings()

    def cancel_check() -> bool:
        return True

    with pytest.raises(ProcessingCancelled, match="отменено"):
        backend.transcribe(
            audio_path,
            settings,
            1.0,
            cancel_check=cancel_check,
        )

    assert encode_calls == []


def test_oversized_encoded_chunk_is_rejected_before_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    client = _FakeClient([])
    backend = OpenAiApiBackend(
        lambda **_kwargs: client,
        _encoder([], size=MAX_UPLOAD_BYTES + 1),
    )
    audio_path = _audio(tmp_path)
    settings = _settings()

    with pytest.raises(OpenAiAudioPreparationError, match="предел загрузки"):
        backend.transcribe(audio_path, settings, 1.0)

    assert client.audio.transcriptions.calls == []


def test_unload_closes_client_and_clears_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    client = _FakeClient([_response("Text.", [_word(0.0, 0.5, "Text.")], 1.0)])
    backend = OpenAiApiBackend(lambda **_kwargs: client, _encoder([]))

    backend.transcribe(_audio(tmp_path), _settings(), 1.0)
    assert backend.runtime_signature(_settings()) is not None

    backend.unload()

    assert client.closed is True
    assert backend.runtime_signature(_settings()) is None
