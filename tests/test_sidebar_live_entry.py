"""Tests for the sidebar's live-task discrimination.

Regression: while an analysis is running, browsing a history record set
``viewing_history`` and there was no non-destructive way back to the live
progress view — the running task's own entry in 未完成任务 was disabled like
every other incomplete entry. ``_is_live_entry`` identifies that one entry so
it can stay clickable and switch the view back instead of restarting the run.
"""

from __future__ import annotations

from web.components.sidebar import _is_live_entry
from web.progress import ProgressTracker


def _entry(ticker: str, trade_date: str) -> dict:
    return {"ticker": ticker, "trade_date": trade_date, "status": "running"}


def test_matches_running_tracker():
    tracker = ProgressTracker(ticker="300750", trade_date="2026-07-01")
    tracker.is_running = True
    assert _is_live_entry(_entry("300750", "2026-07-01"), tracker) is True


def test_matches_paused_tracker():
    # Paused runs keep is_running=True, so the entry must still count as live.
    tracker = ProgressTracker(ticker="300750", trade_date="2026-07-01")
    tracker.is_running = True
    tracker.is_paused = True
    assert _is_live_entry(_entry("300750", "2026-07-01"), tracker) is True


def test_different_ticker_or_date_is_not_live():
    tracker = ProgressTracker(ticker="300750", trade_date="2026-07-01")
    tracker.is_running = True
    assert _is_live_entry(_entry("600519", "2026-07-01"), tracker) is False
    assert _is_live_entry(_entry("300750", "2026-06-30"), tracker) is False


def test_no_tracker_or_not_running_is_not_live():
    assert _is_live_entry(_entry("300750", "2026-07-01"), None) is False

    stopped = ProgressTracker(ticker="300750", trade_date="2026-07-01")
    stopped.is_running = False  # completed / errored / stopped
    assert _is_live_entry(_entry("300750", "2026-07-01"), stopped) is False
