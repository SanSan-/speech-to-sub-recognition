from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from speech_to_sub.asr import registry
from speech_to_sub.exceptions import ValidationError
from speech_to_sub.models import ProcessingSettings, RuntimeSignature, Transcript


class _FakeBackend:
    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id
        self.unload_calls = 0

    def preflight(self, settings: ProcessingSettings) -> Path:
        return settings.model_path

    def expected_runtime_signature(
        self,
        settings: ProcessingSettings,
    ) -> RuntimeSignature:
        del settings
        return {
            "backend": self.backend_id,
            "engine_version": "test-1",
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

    def transcribe(
        self,
        audio_path: Path,
        settings: ProcessingSettings,
        duration: float,
        progress_callback: Any = None,
    ) -> Transcript:
        del audio_path, settings, duration, progress_callback
        raise AssertionError("Инференс не должен запускаться в тесте registry.")

    def unload(self) -> None:
        self.unload_calls += 1


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    factories: dict[str, Any] = {}
    monkeypatch.setattr(registry, "_FACTORIES", factories)
    monkeypatch.setattr(registry, "_INSTANCES", {})
    monkeypatch.setattr(registry, "_ACTIVE_BACKEND_ID", None)
    return factories


def test_get_backend_is_lazy_singleton_and_names_are_stable(
    isolated_registry: dict[str, Any],
) -> None:
    created: list[_FakeBackend] = []

    def factory() -> _FakeBackend:
        backend = _FakeBackend("alpha")
        created.append(backend)
        return backend

    isolated_registry.update(
        {
            "alpha": factory,
            "beta": lambda: _FakeBackend("beta"),
        }
    )

    assert registry.backend_names() == ("alpha", "beta")
    assert created == []
    first = registry.get_backend(" ALPHA ")
    second = registry.get_backend("alpha")

    assert first is second
    assert created == [first]


def test_activate_backend_unloads_previous_and_unload_all_reuses_instances(
    isolated_registry: dict[str, Any],
) -> None:
    alpha = _FakeBackend("alpha")
    beta = _FakeBackend("beta")
    isolated_registry.update({"alpha": lambda: alpha, "beta": lambda: beta})

    assert registry.activate_backend("alpha") is alpha
    assert registry.activate_backend("alpha") is alpha
    assert alpha.unload_calls == 0

    assert registry.activate_backend("beta") is beta
    assert alpha.unload_calls == 1

    registry.unload_backends()
    assert alpha.unload_calls == 2
    assert beta.unload_calls == 1
    assert registry.get_backend("alpha") is alpha


def test_unknown_backend_is_rejected_before_factory_resolution(
    isolated_registry: dict[str, Any],
) -> None:
    isolated_registry["known"] = lambda: _FakeBackend("known")

    with pytest.raises(ValidationError, match="Неизвестный ASR backend"):
        registry.get_backend("unknown")


def test_default_faster_whisper_factory_is_a_lazy_import_reference() -> None:
    reference = registry._FACTORIES.get("faster-whisper")

    assert reference == "speech_to_sub.asr.faster_whisper:FasterWhisperBackend"


def test_default_openai_factory_is_a_lazy_import_reference() -> None:
    reference = registry._FACTORIES.get("openai-api")

    assert reference == "speech_to_sub.asr.openai_api:OpenAiApiBackend"
