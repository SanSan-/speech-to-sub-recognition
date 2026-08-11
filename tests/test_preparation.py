from __future__ import annotations

from speech_to_sub.web.preparation import PREPARATION_LOG_LIMIT, PreparationTracker


def test_preparation_tracker_exposes_correlated_progress_and_terminal_error() -> None:
    tracker = PreparationTracker()
    first = tracker.begin("pick", phase="dialog", message="Открыт диалог.")
    tracker.update(
        first,
        phase="probing",
        discovered=80,
        processed=31,
        total=80,
        message="Проверены аудиопотоки: 31 из 80.",
    )

    running = tracker.snapshot()

    assert running["generation"] == 1
    assert running["operation_id"] == first
    assert running["operation"] == "pick"
    assert running["status"] == "running"
    assert running["phase"] == "probing"
    assert running["active"] is True
    assert running["discovered"] == 80
    assert running["processed"] == 31
    assert running["total"] == 80
    assert running["error"] is None
    assert [entry["message"] for entry in running["logs"]] == [
        "Открыт диалог.",
        "Проверены аудиопотоки: 31 из 80.",
    ]

    tracker.fail(first, message="ffprobe не завершился за 30 с")
    failed = tracker.snapshot()

    assert failed["status"] == "error"
    assert failed["phase"] == "error"
    assert failed["active"] is False
    assert failed["error"] == "ffprobe не завершился за 30 с"
    assert failed["finished_at"] is not None


def test_preparation_tracker_ignores_stale_updates_and_bounds_history() -> None:
    tracker = PreparationTracker()
    stale = tracker.begin("pick", phase="dialog", message="Старый выбор.")
    current = tracker.begin("refresh", phase="collecting", message="Новое обновление.")

    assert tracker.update(stale, phase="error", message="Запоздалая ошибка.") is False
    for index in range(PREPARATION_LOG_LIMIT + 20):
        assert tracker.update(
            current,
            phase="probing",
            processed=index,
            message=f"Шаг {index}.",
        )

    snapshot = tracker.snapshot()

    assert snapshot["generation"] == 2
    assert snapshot["operation"] == "refresh"
    assert snapshot["status"] == "running"
    assert len(snapshot["logs"]) == PREPARATION_LOG_LIMIT
    assert all("Запоздалая" not in entry["message"] for entry in snapshot["logs"])


def test_preparation_snapshot_is_independent_from_internal_state() -> None:
    tracker = PreparationTracker()
    operation_id = tracker.begin("transcribe", phase="queueing", message="Очередь.")
    snapshot = tracker.snapshot()
    snapshot["logs"].clear()
    snapshot["message"] = "изменено снаружи"

    tracker.finish(operation_id, message="Задача создана.")
    actual = tracker.snapshot()

    assert actual["message"] == "Задача создана."
    assert actual["status"] == "done"
    assert len(actual["logs"]) == 2


def test_preparation_counters_never_decrease_within_one_generation() -> None:
    tracker = PreparationTracker()
    operation_id = tracker.begin(
        "refresh", phase="collecting", message="Начат большой сбор."
    )
    tracker.update(
        operation_id,
        discovered=1_000,
        processed=900,
        total=1_000,
    )

    tracker.update(
        operation_id,
        discovered=1,
        processed=1,
        total=1,
    )

    snapshot = tracker.snapshot()
    assert snapshot["discovered"] == 1_000
    assert snapshot["processed"] == 900
    assert snapshot["total"] == 1_000
