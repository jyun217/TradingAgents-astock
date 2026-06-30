"""Tests for Web history helpers."""

from __future__ import annotations

import json
import threading

from web import history


def test_results_dir_honors_configured_results_dir(tmp_path, monkeypatch):
    """_results_dir reads DEFAULT_CONFIG['results_dir'] (e.g. analysis_data/logs)."""
    monkeypatch.setitem(history.DEFAULT_CONFIG, "results_dir", str(tmp_path / "analysis_data" / "logs"))
    assert history._results_dir() == tmp_path / "analysis_data" / "logs"


def test_get_history_scans_configured_dir(tmp_path, monkeypatch):
    """get_history finds logs under the configured results_dir, not the home default."""
    logs = tmp_path / "analysis_data" / "logs"
    log_dir = logs / "688234" / "TradingAgentsStrategy_logs"
    log_dir.mkdir(parents=True)
    (log_dir / "full_states_log_2026-06-30.json").write_text(
        json.dumps({"final_trade_decision": "BUY"}), encoding="utf-8"
    )
    monkeypatch.setitem(history.DEFAULT_CONFIG, "results_dir", str(logs))

    entries = history.get_history()

    assert entries == [
        {"ticker": "688234", "date": "2026-06-30", "path": str(log_dir / "full_states_log_2026-06-30.json")}
    ]


def test_extract_rating_reads_5_tier_from_final_decision():
    assert history.extract_rating({"final_trade_decision": "**Rating**: Underweight\n减持"}) == "Underweight"
    assert history.extract_rating({"final_trade_decision": "**Rating**: Sell"}) == "Sell"
    assert history.extract_rating({"final_trade_decision": "**Rating**: Overweight"}) == "Overweight"


def test_extract_rating_falls_back_to_signal_when_no_decision():
    # No final_trade_decision text → fall back to 3-tier extract_signal
    assert history.extract_rating({"investment_plan": "we should BUY"}) == "Buy"
    assert history.extract_rating({}) == "N/A"


def test_signal_style_maps_full_5_tier():
    from web.components.report_viewer import _signal_style

    assert _signal_style("Buy") == ("#22c55e", "买入")
    assert _signal_style("Overweight") == ("#4ade80", "增持")
    assert _signal_style("Hold") == ("#fbbf24", "持有")
    assert _signal_style("Underweight") == ("#fb923c", "减持")
    assert _signal_style("Sell") == ("#ef4444", "卖出")


def test_incomplete_task_round_trip(tmp_path, monkeypatch):
    index = tmp_path / "incomplete_tasks.json"
    logs = tmp_path / "logs"
    monkeypatch.setattr(history, "_INCOMPLETE_TASKS_FILE", index)
    monkeypatch.setattr(history, "_results_dir", lambda: logs)
    monkeypatch.setattr(history, "_checkpoint_step", lambda ticker, trade_date: 3)

    history.record_incomplete_task(
        "600370",
        "2026-06-02",
        status="error",
        error="quota exceeded",
        completed_stages=["market", "news"],
    )

    entries = history.get_incomplete_history()

    assert entries == [
        {
            "ticker": "600370",
            "trade_date": "2026-06-02",
            "status": "error",
            "error": "quota exceeded",
            "completed_stages": ["market", "news"],
            "updated_at": entries[0]["updated_at"],
            "checkpoint_step": 3,
        }
    ]


def test_completed_history_hides_incomplete_task(tmp_path, monkeypatch):
    index = tmp_path / "incomplete_tasks.json"
    logs = tmp_path / "logs"
    log_dir = logs / "600370" / "TradingAgentsStrategy_logs"
    log_dir.mkdir(parents=True)
    (log_dir / "full_states_log_2026-06-02.json").write_text(
        json.dumps({"final_trade_decision": "HOLD"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(history, "_INCOMPLETE_TASKS_FILE", index)
    monkeypatch.setattr(history, "_results_dir", lambda: logs)
    monkeypatch.setattr(history, "_checkpoint_step", lambda ticker, trade_date: 3)

    history.record_incomplete_task("600370", "2026-06-02", status="running")

    assert history.get_incomplete_history() == []


def test_incomplete_task_writes_are_thread_safe(tmp_path, monkeypatch):
    index = tmp_path / "incomplete_tasks.json"
    logs = tmp_path / "logs"
    monkeypatch.setattr(history, "_INCOMPLETE_TASKS_FILE", index)
    monkeypatch.setattr(history, "_results_dir", lambda: logs)
    monkeypatch.setattr(history, "_checkpoint_step", lambda ticker, trade_date: 1)

    def write_task(i: int) -> None:
        history.record_incomplete_task(
            f"60037{i % 10}",
            "2026-06-02",
            status="running",
            completed_stages=["market"],
        )

    threads = [threading.Thread(target=write_task, args=(i,)) for i in range(30)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    entries = history.get_incomplete_history()

    assert len(entries) == 10
    assert {entry["status"] for entry in entries} == {"running"}
    assert not list(tmp_path.glob("*.tmp"))
