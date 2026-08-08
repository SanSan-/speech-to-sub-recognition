from pathlib import Path

import pytest

from speech_to_sub.web.picker import PickerError, filter_media_paths


def test_filter_media_paths_stops_at_explicit_limit(tmp_path: Path) -> None:
    paths = [tmp_path / f"sample-{index}.mp4" for index in range(3)]
    for path in paths:
        path.write_bytes(b"media")

    with pytest.raises(PickerError, match="не более 2"):
        filter_media_paths(paths, max_paths=2)
