from __future__ import annotations

import numpy as np
import pytest

from speech_to_sub.exceptions import ValidationError
from speech_to_sub.media.audio_windows import iter_pcm_s16_windows


def _pcm(values: list[int]) -> bytes:
    return np.asarray(values, dtype="<i2").tobytes()


def test_pcm_windows_keep_overlap_and_mark_only_last_as_final() -> None:
    windows = list(
        iter_pcm_s16_windows(
            (_pcm(list(range(5))), _pcm(list(range(5, 18)))),
            window_seconds=1.0,
            overlap_seconds=0.2,
            sample_rate=10,
        )
    )

    assert [(item.offset, item.duration, item.is_final) for item in windows] == [
        (0.0, 1.0, False),
        (0.8, 1.0, True),
    ]
    np.testing.assert_allclose(windows[0].samples * 32_768, np.arange(10))
    np.testing.assert_allclose(windows[1].samples * 32_768, np.arange(8, 18))


def test_pcm_windows_emit_tail_only_when_it_contains_new_samples() -> None:
    windows = list(
        iter_pcm_s16_windows(
            (_pcm(list(range(19))),),
            window_seconds=1.0,
            overlap_seconds=0.2,
            sample_rate=10,
        )
    )

    assert [(item.offset, item.duration, item.is_final) for item in windows] == [
        (0.0, 1.0, False),
        (0.8, 1.0, False),
        (1.6, 0.3, True),
    ]
    np.testing.assert_allclose(windows[-1].samples * 32_768, np.arange(16, 19))


def test_pcm_windows_reject_invalid_settings_and_partial_sample() -> None:
    with pytest.raises(ValidationError, match="короче"):
        list(
            iter_pcm_s16_windows(
                (_pcm([1, 2]),),
                window_seconds=1.0,
                overlap_seconds=1.0,
                sample_rate=10,
            )
        )
    with pytest.raises(ValidationError, match="неполным"):
        list(
            iter_pcm_s16_windows(
                (b"\x01",),
                window_seconds=1.0,
                overlap_seconds=0.0,
                sample_rate=10,
            )
        )
