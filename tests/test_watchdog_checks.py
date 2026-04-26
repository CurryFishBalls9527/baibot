"""Tests for watchdog check_* functions.

Each check is exercised with synthetic input (events.jsonl, scheduler log,
sqlite DB, or mocked Alpaca) and inspected for the expected alerts. No
network or live state.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import pytest

from tradingagents.automation import events
from tradingagents.watchdog import checks
from tradingagents.watchdog.state import WatchdogState, TailedFileState

ET = ZoneInfo("US/Eastern")


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("BAIBOT_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("BAIBOT_REPO_ROOT", str(tmp_path))
    yield


def _make_state(tmp_path: Path) -> WatchdogState:
    return WatchdogState(
        events_jsonl=TailedFileState(path=str(tmp_path / "service_logs" / "events.jsonl")),
        service_log=TailedFileState(path=str(tmp_path / "service_logs" / "automation_service.out.log")),
    )


# ── Check 2: event tail ───────────────────────────────────────────


def test_event_tail_fires_on_position_stranded(tmp_path):
    events.emit_event(
        events.Categories.POSITION_STRANDED,
        level="error",
        variant="intraday_mechanical",
        symbol="KDP",
        message="flatten failed",
    )
    state = _make_state(tmp_path)
    alerts = checks.check_event_tail(state)
    assert len(alerts) == 1
    a = alerts[0]
    assert a.category == events.Categories.POSITION_STRANDED
    assert "KDP" in a.title
    assert a.priority == "urgent"
    assert "intraday_mechanical" in a.dedupe_key
    assert "KDP" in a.dedupe_key


def test_event_tail_skips_info_level(tmp_path):
    events.emit_event(
        events.Categories.RECONCILER_TICK,
        level="info",
        variant="mechanical",
    )
    state = _make_state(tmp_path)
    alerts = checks.check_event_tail(state)
    assert alerts == []


def test_event_tail_batches_order_rejects(tmp_path):
    for _ in range(5):
        events.emit_event(
            events.Categories.ORDER_REJECT,
            level="error",
            variant="mechanical",
            symbol="X",
            code="42210000",
            message="reject",
        )
    state = _make_state(tmp_path)
    alerts = checks.check_event_tail(state)
    # Single batched alert, not 5 individual ones.
    reject_alerts = [a for a in alerts if a.category == events.Categories.ORDER_REJECT]
    assert len(reject_alerts) == 1
    assert "5" in reject_alerts[0].title


def test_event_tail_advances_offset(tmp_path):
    events.emit_event(events.Categories.JOB_FAILED, message="first")
    state = _make_state(tmp_path)
    alerts1 = checks.check_event_tail(state)
    assert len(alerts1) == 1

    # Second call sees no new lines.
    alerts2 = checks.check_event_tail(state)
    assert alerts2 == []

    events.emit_event(events.Categories.JOB_FAILED, message="second")
    alerts3 = checks.check_event_tail(state)
    assert len(alerts3) == 1


def test_event_tail_handles_truncation(tmp_path):
    events.emit_event(events.Categories.JOB_FAILED, message="pre-truncate")
    state = _make_state(tmp_path)
    checks.check_event_tail(state)

    # Simulate rotation: rewrite file from scratch (different inode).
    events_path = Path(state.events_jsonl.path)
    events_path.unlink()
    events.emit_event(events.Categories.NAKED_POSITION, variant="x", symbol="Y", message="post")
    alerts = checks.check_event_tail(state)
    # The new event must be picked up despite the offset > new file size.
    assert any(a.category == events.Categories.NAKED_POSITION for a in alerts)


def test_event_tail_dedupes_via_state(tmp_path):
    """A second tail on the same data shouldn't re-add anything new.

    Note: dedupe of alert *sending* happens in the monitor, not in the check
    itself; the check just reads the file delta. So the second call here
    sees no new events because the offset advanced.
    """
    events.emit_event(events.Categories.NAKED_POSITION, variant="m", symbol="A", message="x")
    state = _make_state(tmp_path)
    a1 = checks.check_event_tail(state)
    a2 = checks.check_event_tail(state)
    assert len(a1) == 1
    assert a2 == []


# ── Check 8: log error sweep ──────────────────────────────────────


def test_log_error_sweep_detects_error_lines(tmp_path):
    log = tmp_path / "service_logs" / "automation_service.out.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "2026-04-24 10:00:00 INFO scheduler: tick\n"
        "2026-04-24 10:01:00 ERROR orchestrator: AAOI bracket reject\n"
        "2026-04-24 10:02:00 INFO scheduler: tick\n"
    )
    state = _make_state(tmp_path)
    alerts = checks.check_log_error_sweep(state)
    assert len(alerts) >= 1
    assert any("error" in a.title.lower() for a in alerts)


def test_log_error_sweep_skips_watchdog_self_logs(tmp_path):
    log = tmp_path / "service_logs" / "automation_service.out.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "2026-04-24 10:00:00 ERROR watchdog: self-noise\n"
    )
    state = _make_state(tmp_path)
    alerts = checks.check_log_error_sweep(state)
    assert alerts == []


def test_log_error_sweep_advances_offset(tmp_path):
    log = tmp_path / "service_logs" / "automation_service.out.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "2026-04-24 10:01:00 ERROR orchestrator: first\n"
    )
    state = _make_state(tmp_path)
    a1 = checks.check_log_error_sweep(state)
    assert len(a1) == 1
    a2 = checks.check_log_error_sweep(state)
    # Same fingerprint, dedupe by alert_dedupe — but at this layer we
    # haven't called mark_alerted, so a fingerprint-based gate isn't in
    # play. The offset advanced though, so no new lines.
    assert a2 == []


# ── Check 9: intraday regime gate stuck ────────────────────────────


def _seed_regime_blocked_log(tmp_path: Path, dates: list) -> None:
    """Write a service log with one 'regime gate blocked' line per date."""
    log = tmp_path / "service_logs" / "automation_service.out.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for d in dates:
        lines.append(
            f"{d} 10:00:01 [INFO] tradingagents.automation.intraday_orchestrator: "
            "Minervini regime gate: skipping scan (regime=market_correction, "
            "filter=confirmed_uptrend_only)"
        )
    log.write_text("\n".join(lines) + "\n")


def test_regime_gate_stuck_alerts_on_3_of_5_blocked(tmp_path, monkeypatch):
    """Alert fires when 3+ of last 5 trading days had a regime-blocked scan."""
    from datetime import date as _date
    today = _date(2026, 4, 24)  # Friday
    fixed_now = datetime(2026, 4, 24, 17, 0, tzinfo=checks.ET)
    monkeypatch.setattr(checks, "now_et", lambda: fixed_now)

    # 5 most recent trading days ending Friday 4/24: Mon-Fri 4/20-4/24
    blocked = ["2026-04-22", "2026-04-23", "2026-04-24"]  # Wed/Thu/Fri
    _seed_regime_blocked_log(tmp_path, blocked)
    state = _make_state(tmp_path)
    alerts = checks.check_intraday_regime_gate_stuck(state)
    assert len(alerts) == 1
    assert "3/5" in alerts[0].title
    assert "2026-04-22" in alerts[0].body


def test_regime_gate_stuck_does_not_fire_on_2_of_5(tmp_path, monkeypatch):
    """Threshold is 3; 2 blocked days is not enough to alert."""
    fixed_now = datetime(2026, 4, 24, 17, 0, tzinfo=checks.ET)
    monkeypatch.setattr(checks, "now_et", lambda: fixed_now)
    _seed_regime_blocked_log(tmp_path, ["2026-04-23", "2026-04-24"])
    state = _make_state(tmp_path)
    assert checks.check_intraday_regime_gate_stuck(state) == []


def test_regime_gate_stuck_ignores_old_blocked_days(tmp_path, monkeypatch):
    """Blocked days outside the last-5-trading-day window don't count."""
    fixed_now = datetime(2026, 4, 24, 17, 0, tzinfo=checks.ET)
    monkeypatch.setattr(checks, "now_et", lambda: fixed_now)
    # All blocked dates are >5 trading days ago
    _seed_regime_blocked_log(tmp_path, ["2026-04-01", "2026-04-02", "2026-04-03"])
    state = _make_state(tmp_path)
    assert checks.check_intraday_regime_gate_stuck(state) == []


def test_regime_gate_stuck_no_log_returns_empty(tmp_path, monkeypatch):
    fixed_now = datetime(2026, 4, 24, 17, 0, tzinfo=checks.ET)
    monkeypatch.setattr(checks, "now_et", lambda: fixed_now)
    state = _make_state(tmp_path)
    # No log file written
    assert checks.check_intraday_regime_gate_stuck(state) == []


def test_log_error_sweep_skips_yfinance_noise(tmp_path):
    log = tmp_path / "service_logs" / "automation_service.out.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "2026-04-24 10:00:00 [ERROR] yfinance: AXTI: No data found, symbol may be delisted\n"
        "2026-04-24 10:00:01 [ERROR] yfinance: SNDK: rate limited\n"
    )
    state = _make_state(tmp_path)
    alerts = checks.check_log_error_sweep(state)
    assert alerts == []


def test_log_error_sweep_skips_notifier_http_noise(tmp_path):
    log = tmp_path / "service_logs" / "automation_service.out.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "2026-04-24 10:00:00 [ERROR] tradingagents.automation.notifier: "
        "Failed to send ntfy notification: HTTP Error 400: Bad Request\n"
        "2026-04-24 10:00:01 [ERROR] tradingagents.automation.telegram_notifier: "
        "Failed to send Telegram notification: HTTP Error 429: Too Many Requests\n"
    )
    state = _make_state(tmp_path)
    alerts = checks.check_log_error_sweep(state)
    assert alerts == []


def test_log_error_sweep_still_alerts_on_real_error_alongside_noise(tmp_path):
    log = tmp_path / "service_logs" / "automation_service.out.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "2026-04-24 10:00:00 [ERROR] yfinance: AXTI: delisted\n"
        "2026-04-24 10:01:00 [ERROR] tradingagents.automation.orchestrator: "
        "AAPL bracket reject — 42210000\n"
    )
    state = _make_state(tmp_path)
    alerts = checks.check_log_error_sweep(state)
    assert len(alerts) == 1
    assert "orchestrator" in alerts[0].body or "AAPL" in alerts[0].body


# ── Check 1: scheduler liveness ────────────────────────────────────


def test_scheduler_liveness_alerts_on_missing_log(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    alerts = checks.check_scheduler_liveness(state)
    # Missing log → urgent alert
    assert any("missing" in a.title.lower() for a in alerts)


def test_scheduler_liveness_silent_when_fresh(tmp_path):
    log = tmp_path / "service_logs" / "automation_service.out.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    log.write_text(f"{now_str} INFO scheduler: JOB DONE: Exit Check Pass — ok\n")
    state = _make_state(tmp_path)
    alerts = checks.check_scheduler_liveness(state)
    assert alerts == []


def test_scheduler_liveness_silent_on_weekend(tmp_path):
    """Saturday/Sunday: log naturally goes idle. Don't alert unless dead >14h."""
    log = tmp_path / "service_logs" / "automation_service.out.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    # Last JOB DONE 6h ago — well under the 14h "really dead" threshold.
    fake_now = datetime(2026, 4, 25, 17, 0, tzinfo=ET)  # Saturday
    last_done = (fake_now - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
    log.write_text(f"{last_done} INFO scheduler: JOB DONE: Weekly Heartbeat — ok\n")
    # Backdate the file mtime so the freshness check doesn't trivially pass.
    import os as _os
    old_ts = (fake_now - timedelta(hours=6)).timestamp()
    _os.utime(log, (old_ts, old_ts))

    state = _make_state(tmp_path)
    with mock.patch("tradingagents.watchdog.checks.now_et", return_value=fake_now):
        alerts = checks.check_scheduler_liveness(state)
    assert alerts == []


def test_scheduler_liveness_fires_when_dead_overnight(tmp_path):
    """Saturday after 14h+ of no JOB DONE: scheduler is genuinely dead — fire."""
    log = tmp_path / "service_logs" / "automation_service.out.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fake_now = datetime(2026, 4, 25, 17, 0, tzinfo=ET)  # Saturday
    last_done = (fake_now - timedelta(hours=20)).strftime("%Y-%m-%d %H:%M:%S")
    log.write_text(f"{last_done} INFO scheduler: JOB DONE: Stale — ok\n")
    import os as _os
    old_ts = (fake_now - timedelta(hours=20)).timestamp()
    _os.utime(log, (old_ts, old_ts))

    state = _make_state(tmp_path)
    with mock.patch("tradingagents.watchdog.checks.now_et", return_value=fake_now):
        alerts = checks.check_scheduler_liveness(state)
    assert len(alerts) == 1
    assert "stale" in alerts[0].title.lower()


def test_scheduler_liveness_off_hours_silent_with_recent_job(tmp_path):
    """Weekday 22:00 ET: scheduler is idle but ran something a few hours ago."""
    log = tmp_path / "service_logs" / "automation_service.out.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fake_now = datetime(2026, 4, 24, 22, 0, tzinfo=ET)  # Friday 22:00 ET
    last_done = (fake_now - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S")
    log.write_text(f"{last_done} INFO scheduler: JOB DONE: Daily Review — ok\n")
    import os as _os
    old_ts = (fake_now - timedelta(hours=5)).timestamp()
    _os.utime(log, (old_ts, old_ts))

    state = _make_state(tmp_path)
    with mock.patch("tradingagents.watchdog.checks.now_et", return_value=fake_now):
        alerts = checks.check_scheduler_liveness(state)
    assert alerts == []


# ── Check 7: daily activity sanity ────────────────────────────────


def test_daily_activity_sanity_flags_empty_db(tmp_path, monkeypatch):
    # Build a stub mechanical DB with daily_snapshots table but no row for today.
    dbfile = tmp_path / "trading_mechanical.db"
    con = sqlite3.connect(str(dbfile))
    con.execute(
        "CREATE TABLE daily_snapshots ("
        " date TEXT, equity REAL, cash REAL, buying_power REAL, positions_json TEXT"
        ")"
    )
    con.commit()
    con.close()

    monkeypatch.setenv("ALPACA_MECHANICAL_API_KEY", "stub")
    monkeypatch.setenv("ALPACA_MECHANICAL_SECRET_KEY", "stub")

    state = _make_state(tmp_path)
    # Force date to a weekday.
    fake_today = datetime(2026, 4, 24, tzinfo=ET)
    with mock.patch("tradingagents.watchdog.checks.date") as mock_date, \
         mock.patch("tradingagents.watchdog.checks.now_et") as mock_now:
        mock_date.today.return_value = fake_today.date()
        mock_now.return_value = fake_today
        alerts = checks.check_daily_activity_sanity(state)

    # We may get alerts for any variant whose env var is set; mechanical is.
    mech_alerts = [a for a in alerts if "mechanical" in a.title and "v2" not in a.title]
    assert len(mech_alerts) == 1
    assert mech_alerts[0].category == events.Categories.ACTIVITY_GAP


def test_daily_activity_sanity_silent_when_row_present(tmp_path, monkeypatch):
    dbfile = tmp_path / "trading_mechanical.db"
    con = sqlite3.connect(str(dbfile))
    con.execute(
        "CREATE TABLE daily_snapshots ("
        " date TEXT, equity REAL, cash REAL, buying_power REAL, positions_json TEXT"
        ")"
    )
    fake_today = datetime(2026, 4, 24, tzinfo=ET)
    con.execute(
        "INSERT INTO daily_snapshots(date, equity) VALUES (?, ?)",
        (fake_today.date().isoformat(), 100000.0),
    )
    con.commit()
    con.close()

    monkeypatch.setenv("ALPACA_MECHANICAL_API_KEY", "stub")
    monkeypatch.setenv("ALPACA_MECHANICAL_SECRET_KEY", "stub")

    state = _make_state(tmp_path)
    with mock.patch("tradingagents.watchdog.checks.date") as mock_date:
        mock_date.today.return_value = fake_today.date()
        alerts = checks.check_daily_activity_sanity(state)

    mech_alerts = [a for a in alerts if "mechanical" in a.title and "v2" not in a.title]
    assert mech_alerts == []


# ── Check 6: job execution sanity ────────────────────────────────


def test_job_execution_sanity_flags_shortfall(tmp_path):
    log = tmp_path / "service_logs" / "automation_service.out.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now(ET).date().isoformat()
    log.write_text(
        f"{today} 09:30:00 INFO scheduler: JOB DONE: Daily Swing Analysis — ok\n"
        f"{today} 09:35:00 INFO scheduler: JOB DONE: Exit Check Pass — ok\n"
    )
    state = _make_state(tmp_path)
    alerts = checks.check_job_execution_sanity(state)
    # Exit Check expects ≥80, we have 1 — should fire.
    titles = [a.title for a in alerts]
    assert any("Exit Check Pass" in t for t in titles)


# ── Naked position check (DB query path; Alpaca mocked) ───────────


def test_naked_position_detection(tmp_path, monkeypatch):
    dbfile = tmp_path / "trading_mechanical.db"
    con = sqlite3.connect(str(dbfile))
    con.execute(
        "CREATE TABLE position_states ("
        " symbol TEXT, entry_price REAL, current_stop REAL, stop_order_id TEXT"
        ")"
    )
    con.execute(
        "INSERT INTO position_states(symbol, entry_price, current_stop, stop_order_id) "
        "VALUES (?, ?, ?, ?)",
        ("NAKED", 100.0, 92.0, None),
    )
    con.commit()
    con.close()

    monkeypatch.setenv("ALPACA_MECHANICAL_API_KEY", "stub")
    monkeypatch.setenv("ALPACA_MECHANICAL_SECRET_KEY", "stub")

    fake_position = mock.Mock(symbol="NAKED")
    fake_client = mock.Mock(get_all_positions=mock.Mock(return_value=[fake_position]))
    with mock.patch("alpaca.trading.client.TradingClient", return_value=fake_client):
        state = _make_state(tmp_path)
        alerts = checks.check_naked_positions(state)

    naked = [a for a in alerts if a.category == events.Categories.NAKED_POSITION]
    assert len(naked) == 1
    assert "NAKED" in naked[0].title
    assert "mechanical" in naked[0].title
