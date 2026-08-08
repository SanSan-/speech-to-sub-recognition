from __future__ import annotations

import pytest

from speech_to_sub import __version__
from speech_to_sub.exceptions import ValidationError
from speech_to_sub.models import ProcessingSettings


def test_unknown_backend_without_explicit_model_is_domain_error() -> None:
    with pytest.raises(ValidationError, match="Неизвестный ASR backend 'unknown'"):
        ProcessingSettings.from_mapping({"backend": "unknown"})


def test_package_version_matches_v14_milestone() -> None:
    assert __version__ == "1.4.0"
