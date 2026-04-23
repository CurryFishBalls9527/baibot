"""Tests for the scheduler heartbeat (PR3)."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.automation import heartbeat as hb


@pytest.fixture
def tmp_results(tmp_path: Path, monkeypatch):
    """Point the heartbeat module at a fresh filesystem root."""
    daily_root = tmp_path / "daily_reviews"
    weekly_root = tmp_path / "weekly_reviews"
    log_path = tmp_path / "scheduler.log"
    daily_root.mkdir()
    weekly_root.mkdir()
    monkeypatch.setattr(hb, "_DAILY_REVIEW_ROOT", daily_root)
    monkeypatch.setattr(hb, "_WEEKLY_REVIEW_ROOT", weekly_root)
    monkeypatch.setattr(hb, "_SCHEDULER_LOG_PATH", log_path)
    return {"daily": daily_root, "weekly": weekly_root, "log": log_path}


# ── daily ───────────────────────────────────────────────────────

class TestDailyHeartbeat:
    def test_disabled_returns_status_disabled(self, tmp_results):
        r = hb.check_daily_review_ran(
            {"daily_review_heartbeat_enabled": False},
            review_date=date(2026, 4, 24),  # Friday
        )
        assert r["status"] == "disabled"

    def test_weekend_skip(self, tmp_results):
        # Sat
        r = hb.check_daily_review_ran({}, review_date=date(2026, 4, 25))
        assert r["status"] == "weekend_skip"

    def test_ok_when_files_exist(self, tmp_results):
        review_date = date(2026, 4, 24)
        day_dir = tmp_results["daily"] / review_date.isoformat()
        day_dir.mkdir()
        (day_dir / "mechanical_AAPL.md").write_text("# test")
        r = hb.check_daily_review_ran({}, review_date=review_date)
        assert r["status"] == "ok"
        assert r["files"] == 1

    def test_missing_alerts_notifier(self, tmp_results):
        review_date = date(2026, 4, 24)
        fake_notifier = MagicMock()
        with patch.object(hb, "_notifier_for", return_value=fake_notifier):
            r = hb.check_daily_review_ran({}, review_date=review_date)
        assert r["status"] == "missing"
        fake_notifier.send.assert_called_once()
        call_kwargs = fake_notifier.send.call_args.kwargs
        assert call_kwargs["priority"] == "high"
        assert "heartbeat:daily_review:2026-04-24" == call_kwargs["dedupe_key"]

    def test_log_line_counts_as_proof(self, tmp_results):
        review_date = date(2026, 4, 24)
        # Write a log entry within the 90-min window.
        now = datetime.now()
        recent_ts = (now - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
        tmp_results["log"].write_text(
            f"{recent_ts} [INFO] scheduler: JOB DONE: Daily Trade Review — ok\n"
        )
        fake_notifier = MagicMock()
        with patch.object(hb, "_notifier_for", return_value=fake_notifier):
            r = hb.check_daily_review_ran({}, review_date=review_date)
        assert r["status"] == "ok"
        assert r["log_seen"] is True
        fake_notifier.send.assert_not_called()

    def test_stale_log_line_does_not_count(self, tmp_results):
        review_date = date(2026, 4, 24)
        # 4h old — past the 90-min window.
        stale_ts = (datetime.now() - timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S")
        tmp_results["log"].write_text(
            f"{stale_ts} [INFO] scheduler: JOB DONE: Daily Trade Review — ok\n"
        )
        fake_notifier = MagicMock()
        with patch.object(hb, "_notifier_for", return_value=fake_notifier):
            r = hb.check_daily_review_ran({}, review_date=review_date)
        assert r["status"] == "missing"
        fake_notifier.send.assert_called_once()


# ── weekly ──────────────────────────────────────────────────────

class TestWeeklyHeartbeat:
    def test_ok_when_files_exist(self, tmp_results):
        review_date = date(2026, 4, 25)  # Saturday
        iso = hb._iso_week(review_date)
        week_dir = tmp_results["weekly"] / iso
        week_dir.mkdir()
        (week_dir / "chan_v2.md").write_text("# Weekly test")
        r = hb.check_weekly_review_ran({}, review_date=review_date)
        assert r["status"] == "ok"

    def test_missing_alerts(self, tmp_results):
        review_date = date(2026, 4, 25)
        fake_notifier = MagicMock()
        with patch.object(hb, "_notifier_for", return_value=fake_notifier):
            r = hb.check_weekly_review_ran({}, review_date=review_date)
        assert r["status"] == "missing"
        fake_notifier.send.assert_called_once()


# ── startup banner ──────────────────────────────────────────────

class TestStartupBanner:
    def test_warns_when_both_disabled(self, caplog):
        with caplog.at_level("WARNING"):
            hb.log_notifier_banner({"ntfy_enabled": False, "telegram_enabled": False})
        assert any(
            "No notifier backends are enabled" in record.message
            for record in caplog.records
        )

    def test_info_when_ntfy_enabled(self, caplog):
        with caplog.at_level("INFO"):
            hb.log_notifier_banner({
                "ntfy_enabled": True, "ntfy_topic": "test_topic",
                "telegram_enabled": False,
            })
        # Expect an INFO line mentioning ntfy.
        assert any(
            "Notifier backends enabled" in record.message and "ntfy" in record.message
            for record in caplog.records
            if record.levelname == "INFO"
        )
