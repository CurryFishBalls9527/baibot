"""Tests for signal_metadata migration + intraday serialization safety.

Covers PR1c:
  1. Idempotent ALTER — column appears on fresh + existing DBs.
  2. log_signal accepts signal_metadata as optional JSON string.
  3. log_signal NULL-tolerant when metadata not provided.
  4. Intraday serialization strips NaN / inf without raising.
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pytest

from tradingagents.storage.database import TradingDatabase


def test_signal_metadata_column_exists_on_fresh_db(tmp_path: Path):
    db = TradingDatabase(str(tmp_path / "fresh.db"))
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(signals)").fetchall()}
    assert "signal_metadata" in cols


def test_migration_is_idempotent(tmp_path: Path):
    """Opening the DB twice shouldn't raise `duplicate column name`."""
    path = tmp_path / "twice.db"
    db1 = TradingDatabase(str(path))
    db1.close()
    db2 = TradingDatabase(str(path))  # triggers _ensure_columns again
    cols = {r[1] for r in db2.conn.execute("PRAGMA table_info(signals)").fetchall()}
    assert "signal_metadata" in cols
    db2.close()


def test_migration_on_pre_existing_signals_table(tmp_path: Path):
    """Create a signals table *without* signal_metadata, then migrate."""
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            confidence REAL DEFAULT 0,
            reasoning TEXT,
            stop_loss REAL,
            take_profit REAL,
            timeframe TEXT,
            full_analysis TEXT,
            executed INTEGER DEFAULT 0,
            rejected_reason TEXT
        );
        INSERT INTO signals (symbol, action) VALUES ('NVDA', 'BUY');
        """
    )
    conn.commit()
    conn.close()

    # Now open through TradingDatabase — should migrate without data loss.
    db = TradingDatabase(str(path))
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(signals)").fetchall()}
    assert "signal_metadata" in cols
    row = db.conn.execute("SELECT symbol, signal_metadata FROM signals").fetchone()
    assert row["symbol"] == "NVDA"
    assert row["signal_metadata"] is None


def test_log_signal_accepts_metadata_json(tmp_path: Path):
    db = TradingDatabase(str(tmp_path / "s.db"))
    sid = db.log_signal(
        symbol="NVDA", action="BUY", reasoning="test",
        signal_metadata=json.dumps({"setup_family": "orb_breakout", "orb_high": 150.0}),
    )
    row = db.conn.execute(
        "SELECT signal_metadata FROM signals WHERE id = ?", (sid,)
    ).fetchone()
    assert row["signal_metadata"] is not None
    parsed = json.loads(row["signal_metadata"])
    assert parsed["setup_family"] == "orb_breakout"
    assert parsed["orb_high"] == 150.0


def test_log_signal_without_metadata_is_null(tmp_path: Path):
    db = TradingDatabase(str(tmp_path / "s.db"))
    sid = db.log_signal(symbol="NVDA", action="BUY", reasoning="no meta")
    row = db.conn.execute(
        "SELECT signal_metadata FROM signals WHERE id = ?", (sid,)
    ).fetchone()
    assert row["signal_metadata"] is None


class TestIntradaySerializationSafety:
    """The intraday orchestrator builds `signal["metadata"]` and serializes to
    JSON before calling log_signal. Real-world feature frames can contain NaN
    / inf from ratio math; ensure JSON never raises."""

    def _clean(self, v):
        # Mirror the inline helper in intraday_orchestrator._execute_entry.
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    def test_nan_values_become_null(self):
        raw = {"vwap": float("nan"), "orb_high": 150.0}
        cleaned = {k: self._clean(v) for k, v in raw.items()}
        assert cleaned["vwap"] is None
        assert cleaned["orb_high"] == 150.0
        # JSON round-trips cleanly.
        parsed = json.loads(json.dumps(cleaned))
        assert parsed["vwap"] is None

    def test_inf_values_become_null(self):
        raw = {"volume_ratio": float("inf")}
        cleaned = {k: self._clean(v) for k, v in raw.items()}
        assert cleaned["volume_ratio"] is None
        json.dumps(cleaned)  # does not raise

    def test_negative_inf_values_become_null(self):
        raw = {"distance_pct": float("-inf")}
        cleaned = {k: self._clean(v) for k, v in raw.items()}
        assert cleaned["distance_pct"] is None

    def test_ordinary_floats_preserved(self):
        raw = {"a": 0.0, "b": 1.5, "c": -0.33}
        cleaned = {k: self._clean(v) for k, v in raw.items()}
        assert cleaned == raw
