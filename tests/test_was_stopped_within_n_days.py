"""Regression tests for `TradingDatabase.was_stopped_within_n_days`.

Cross-day extension of `was_stopped_today`. Motivated by W17 AAOI:
chan_v2 was stopped on 4/22 and re-entered 4/23 because the same-day
guard didn't see across the midnight boundary.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from tradingagents.storage.database import TradingDatabase


def _seed_stop(db: TradingDatabase, symbol: str, ts_iso: str, *, status: str = "filled"):
    """Insert a filled bracket_stop_loss SELL row at the given timestamp.

    Bypasses `log_trade` because that helper sets timestamp via DB default
    and we need to control the date for the cross-day test cases.
    """
    db.conn.execute(
        """INSERT INTO trades
              (symbol, side, qty, status, filled_qty, reasoning, timestamp)
           VALUES (?, 'sell', 100, ?, 100, 'bracket_stop_loss', ?)""",
        (symbol, status, ts_iso),
    )
    db.conn.commit()


def _mk_db(tmp_path: Path) -> TradingDatabase:
    return TradingDatabase(str(tmp_path / "t.db"))


class TestWasStoppedWithinNDays:
    def test_n_equals_1_matches_was_stopped_today(self, tmp_path):
        db = _mk_db(tmp_path)
        today = date.today().isoformat()
        _seed_stop(db, "AAOI", f"{today} 14:30:00")
        assert db.was_stopped_today("AAOI", as_of=today) is True
        assert db.was_stopped_within_n_days("AAOI", 1, as_of=today) is True

    def test_n_equals_3_catches_yesterday(self, tmp_path):
        """W17 AAOI scenario: stop on day D, scan on day D+1."""
        db = _mk_db(tmp_path)
        today = date.today().isoformat()
        yday = (date.today() - timedelta(days=1)).isoformat()
        _seed_stop(db, "AAOI", f"{yday} 14:30:00")
        # Same-day-only misses it.
        assert db.was_stopped_today("AAOI", as_of=today) is False
        # 3-day window catches it.
        assert db.was_stopped_within_n_days("AAOI", 3, as_of=today) is True

    def test_n_day_window_excludes_old_stops(self, tmp_path):
        """A stop 5 days ago is outside a 3-day window."""
        db = _mk_db(tmp_path)
        today = date.today().isoformat()
        old = (date.today() - timedelta(days=5)).isoformat()
        _seed_stop(db, "AAOI", f"{old} 14:30:00")
        assert db.was_stopped_within_n_days("AAOI", 3, as_of=today) is False
        # Wider window catches it.
        assert db.was_stopped_within_n_days("AAOI", 7, as_of=today) is True

    def test_only_filled_stops_count(self, tmp_path):
        """Cancelled or rejected stop orders don't trigger the guard."""
        db = _mk_db(tmp_path)
        today = date.today().isoformat()
        _seed_stop(db, "AAOI", f"{today} 14:30:00", status="canceled")
        assert db.was_stopped_within_n_days("AAOI", 3, as_of=today) is False

    def test_symbol_isolation(self, tmp_path):
        """A stop on one symbol doesn't block re-entry on another."""
        db = _mk_db(tmp_path)
        yday = (date.today() - timedelta(days=1)).isoformat()
        today = date.today().isoformat()
        _seed_stop(db, "AAOI", f"{yday} 14:30:00")
        assert db.was_stopped_within_n_days("AAOI", 3, as_of=today) is True
        assert db.was_stopped_within_n_days("VALE", 3, as_of=today) is False

    def test_n_zero_or_negative_falls_through_to_today_only(self, tmp_path):
        """Defensive: n_days <= 1 should be equivalent to was_stopped_today."""
        db = _mk_db(tmp_path)
        yday = (date.today() - timedelta(days=1)).isoformat()
        today = date.today().isoformat()
        _seed_stop(db, "AAOI", f"{yday} 14:30:00")
        assert db.was_stopped_within_n_days("AAOI", 0, as_of=today) is False
        assert db.was_stopped_within_n_days("AAOI", 1, as_of=today) is False
