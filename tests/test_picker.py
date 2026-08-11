from pathlib import Path

import pytest

from speech_to_sub import service
from speech_to_sub.constants import MAX_BATCH_PATHS
from speech_to_sub.web import picker
from speech_to_sub.web.picker import (
    PickerError,
    collect_media_paths,
    filter_media_paths,
)


def test_filter_media_paths_stops_at_explicit_limit(tmp_path: Path) -> None:
    paths = [tmp_path / f"sample-{index}.mp4" for index in range(3)]
    for path in paths:
        path.write_bytes(b"media")

    with pytest.raises(PickerError, match="не более 2"):
        filter_media_paths(paths, max_paths=2)


def test_filter_media_paths_keeps_supported_missing_path_for_service_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "исчезнувший.MP4"
    generated = tmp_path / "служебный.ru.asr.flac"
    unsupported = tmp_path / "заметки.txt"
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("stat запрещён")),
    )

    assert filter_media_paths([missing, generated, unsupported, missing]) == (missing,)


def test_filter_media_paths_accepts_exact_default_limit_and_rejects_10_001(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / f"sample-{index:05}.mp4" for index in range(MAX_BATCH_PATHS)]

    assert len(filter_media_paths(paths)) == 10_000

    with pytest.raises(PickerError, match="не более 10000"):
        filter_media_paths([*paths, tmp_path / "overflow.mp4"])


def test_collect_media_paths_recurses_and_ignores_only_generated_flac(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "уровень 1" / "уровень 2"
    nested.mkdir(parents=True)
    video = tmp_path / "верхний.MKV"
    audio = nested / "обычная дорожка.FLAC"
    generated = nested / "обычная дорожка.ru.ASR.FLAC"
    ignored = nested / "заметки.txt"
    for path in (video, audio, generated, ignored):
        path.write_bytes(b"data")

    collected = collect_media_paths(tmp_path, recursive=True)

    assert collected == tuple(
        sorted((video, audio), key=lambda value: str(value).casefold())
    )


def test_collect_media_paths_reports_progress_and_tolerates_callback_failure(
    tmp_path: Path,
) -> None:
    for index in range(260):
        (tmp_path / f"sample-{index:03}.mp4").write_bytes(b"media")
    calls = 0

    def broken_callback(_event: dict[str, object]) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("интерфейс уже закрыт")

    collected = collect_media_paths(
        tmp_path,
        recursive=True,
        progress_callback=broken_callback,
    )

    assert len(collected) == 260
    assert calls >= 3


def test_collect_media_paths_surfaces_unreadable_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        picker.os,
        "scandir",
        lambda _path: (_ for _ in ()).throw(PermissionError("нет доступа")),
    )

    with pytest.raises(PickerError, match="Не удалось прочитать каталог"):
        collect_media_paths(tmp_path, recursive=True)


def test_collect_media_paths_skips_unreadable_nested_directory_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = tmp_path / "недоступный"
    available = tmp_path / "доступный"
    blocked.mkdir()
    available.mkdir()
    media = available / "лекция.mp4"
    media.write_bytes(b"media")
    events: list[dict[str, object]] = []
    real_scandir = picker.os.scandir

    def selective_scandir(path: str | Path):
        if Path(path) == blocked:
            raise PermissionError("нет доступа")
        return real_scandir(path)

    monkeypatch.setattr(picker.os, "scandir", selective_scandir)

    collected = collect_media_paths(
        tmp_path,
        recursive=True,
        progress_callback=events.append,
    )

    assert collected == (media,)
    assert any("Пропущен недоступный" in str(event.get("message")) for event in events)


def test_blocked_supported_file_reaches_resilient_service_as_error_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = tmp_path / "valid.mp4"
    blocked = tmp_path / "blocked.mp4"
    valid.write_bytes(b"media")

    class Entry:
        def __init__(self, path: Path, *, blocked_file: bool) -> None:
            self.path = str(path)
            self.name = path.name
            self._blocked_file = blocked_file

        @staticmethod
        def is_symlink() -> bool:
            return False

        @staticmethod
        def is_dir(*, follow_symlinks: bool) -> bool:
            assert follow_symlinks is False
            return False

        def is_file(self, *, follow_symlinks: bool) -> bool:
            assert follow_symlinks is False
            if self._blocked_file:
                raise PermissionError("контролируемый запрет stat")
            return True

    class Entries:
        def __enter__(self):
            return [Entry(blocked, blocked_file=True), Entry(valid, blocked_file=False)]

        @staticmethod
        def __exit__(*args: object) -> None:
            del args

    monkeypatch.setattr(picker.os, "scandir", lambda _path: Entries())

    collected = collect_media_paths(tmp_path)
    items = service.build_pending_items(collected)
    by_name = {item["name"]: item for item in items}

    assert collected == (blocked, valid)
    assert by_name[blocked.name]["state"] == "error"
    assert "не найден" in by_name[blocked.name]["error"]
    assert by_name[valid.name]["state"] == "queued"


def test_collect_media_paths_accepts_exact_limit_and_rejects_next_file(
    tmp_path: Path,
) -> None:
    first = tmp_path / "01.mp4"
    second = tmp_path / "02.wav"
    third = tmp_path / "03.mkv"
    for path in (first, second, third):
        path.write_bytes(b"media")

    third.rename(tmp_path / "03.txt")
    assert collect_media_paths(tmp_path, max_paths=2) == (first, second)

    (tmp_path / "03.txt").rename(third)
    with pytest.raises(PickerError, match="не более 2"):
        collect_media_paths(tmp_path, max_paths=2)


def test_collect_media_paths_does_not_follow_directory_symlink(tmp_path: Path) -> None:
    root = tmp_path / "выбрано"
    outside = tmp_path / "снаружи"
    root.mkdir()
    outside.mkdir()
    local_media = root / "локальный.mp4"
    outside_media = outside / "внешний.mp4"
    local_media.write_bytes(b"local")
    outside_media.write_bytes(b"outside")
    link = root / "ссылка"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Символические ссылки недоступны: {exc}")

    assert collect_media_paths(root, recursive=True) == (local_media,)


def test_collect_directory_entry_does_not_descend_into_windows_junction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class JunctionEntry:
        name = "junction"
        path = r"D:\выбрано\junction"

        @staticmethod
        def is_symlink() -> bool:
            return False

        @staticmethod
        def is_dir(*, follow_symlinks: bool) -> bool:
            raise AssertionError(
                f"junction не должен проверяться как каталог: {follow_symlinks}"
            )

        @staticmethod
        def is_file(*, follow_symlinks: bool) -> bool:
            raise AssertionError(
                f"junction не должен проверяться как файл: {follow_symlinks}"
            )

    monkeypatch.setattr(
        picker,
        "is_link_or_junction",
        lambda path: str(path) == JunctionEntry.path,
    )

    assert (
        picker._collect_directory_entry(
            JunctionEntry(),
            recursive=True,
            unique={},
            max_paths=10,
            progress_callback=None,
        )
        is None
    )
