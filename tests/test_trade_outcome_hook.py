"""Tests for the shared `log_closed_trade` helper + position_states migration.

Covers:
  1. Shared helper writes expected fields given a pos_state dict.
  2. Chan-shaped pos_state (T-type in base_pattern) round-trips.
  3. Intraday-shaped pos_state (setup_family in base_pattern) round-trips.
  4. Minervini-shaped pos_state (base_label + rs + stage) round-trips.
  5. MFE/MAE kill switch respected.
  6. position_states migration is idempotent + NULL-tolerant.
  7. Helper never raises on broker/compute failures.
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from tradingagents.automation.trade_outcome import log_closed_trade
from tradingagents.storage.database import TradingDatabase


def _mk_db(tmp_path: Path) -> TradingDatabase:
    return TradingDatabase(str(tmp_path / "t.db"))


def _base_pos_state(**overrides) -> dict:
    entry_date = (date.today() - timedelta(days=3)).isoformat()
    state = {
        "entry_price": 100.0,
        "entry_date": entry_date,
        "highest_close": 110.0,
        "current_stop": 95.0,
        "partial_taken": False,
    }
    state.update(overrides)
    return state


def _stub_broker_no_bars(symbol: str = "ANY"):
    """Broker whose data_client returns empty bars — MFE/MAE will be NULL."""
    broker = MagicMock()
    broker.data_client.get_stock_bars.return_value = SimpleNamespace(data={symbol: []})
    return broker


class TestSharedHelper:
    def test_writes_expected_fields(self, tmp_path):
        db = _mk_db(tmp_path)
        state = _base_pos_state(
            base_pattern="vcp", regime_at_entry="confirmed_uptrend",
            rs_at_entry=85.0, stage_at_entry=2.0,
        )
        oid = log_closed_trade(
            db=db, symbol="AAPL", pos_state=state,
            exit_price=110.0, exit_reason="trailing_stop",
            broker=None, excursion_enabled=False,
        )
        assert isinstance(oid, int) and oid > 0
        row = db.conn.execute(
            "SELECT * FROM trade_outcomes WHERE id = ?", (oid,)
        ).fetchone()
        assert row["symbol"] == "AAPL"
        assert row["entry_price"] == 100.0
        assert row["exit_price"] == 110.0
        assert row["return_pct"] == 0.10
        assert row["base_pattern"] == "vcp"
        assert row["regime_at_entry"] == "confirmed_uptrend"
        assert row["rs_at_entry"] == 85.0
        assert row["stage_at_entry"] == 2.0
        assert row["exit_reason"] == "trailing_stop"
        # MFE/MAE NULL because compute disabled
        assert row["max_favorable_excursion"] is None
        assert row["max_adverse_excursion"] is None

    def test_chan_t_type_roundtrip(self, tmp_path):
        db = _mk_db(tmp_path)
        state = _base_pos_state(base_pattern="T1+T2S", regime_at_entry=None)
        oid = log_closed_trade(
            db=db, symbol="NVDA", pos_state=state,
            exit_price=108.0, exit_reason="chan_sell_bsp",
            broker=None, excursion_enabled=False,
        )
        row = db.conn.execute(
            "SELECT * FROM trade_outcomes WHERE id = ?", (oid,)
        ).fetchone()
        assert row["base_pattern"] == "T1+T2S"
        assert row["regime_at_entry"] is None
        assert row["rs_at_entry"] is None
        assert row["stage_at_entry"] is None

    def test_intraday_setup_family_roundtrip(self, tmp_path):
        db = _mk_db(tmp_path)
        state = _base_pos_state(base_pattern="orb_breakout")
        oid = log_closed_trade(
            db=db, symbol="MRVL", pos_state=state,
            exit_price=95.0, exit_reason="eod_flatten",
            broker=None, excursion_enabled=False,
        )
        row = db.conn.execute(
            "SELECT * FROM trade_outcomes WHERE id = ?", (oid,)
        ).fetchone()
        assert row["base_pattern"] == "orb_breakout"
        # Loss: (95-100)/100 = -0.05
        assert row["return_pct"] == -0.05

    def test_kill_switch_prevents_excursion_call(self, tmp_path):
        db = _mk_db(tmp_path)
        state = _base_pos_state()
        broker = MagicMock()
        log_closed_trade(
            db=db, symbol="AAPL", pos_state=state,
            exit_price=110.0, exit_reason="test",
            broker=broker, excursion_enabled=False,
        )
        broker.data_client.get_stock_bars.assert_not_called()

    def test_missing_entry_price_returns_none(self, tmp_path):
        db = _mk_db(tmp_path)
        state = _base_pos_state(entry_price=0)
        oid = log_closed_trade(
            db=db, symbol="AAPL", pos_state=state,
            exit_price=100.0, exit_reason="test",
        )
        assert oid is None
        # No row inserted
        n = db.conn.execute("SELECT COUNT(*) FROM trade_outcomes").fetchone()[0]
        assert n == 0

    def test_broker_failure_does_not_raise(self, tmp_path):
        db = _mk_db(tmp_path)
        state = _base_pos_state()
        broker = MagicMock()
        broker.data_client.get_stock_bars.side_effect = RuntimeError("down")
        # Should not raise.
        oid = log_closed_trade(
            db=db, symbol="AAPL", pos_state=state,
            exit_price=105.0, exit_reason="test",
            broker=broker, excursion_enabled=True,
        )
        assert oid is not None  # outcome row still written
        row = db.conn.execute(
            "SELECT * FROM trade_outcomes WHERE id = ?", (oid,)
        ).fetchone()
        assert row["max_favorable_excursion"] is None


class TestPositionStateMigration:
    def test_new_columns_exist_on_fresh_db(self, tmp_path):
        db = _mk_db(tmp_path)
        cols = {r[1] for r in db.conn.execute("PRAGMA table_info(position_states)").fetchall()}
        for c in ("regime_at_entry", "base_pattern", "rs_at_entry", "stage_at_entry"):
            assert c in cols, f"column {c} missing"

    def test_migration_is_idempotent(self, tmp_path):
        path = tmp_path / "x.db"
        db1 = TradingDatabase(str(path))
        db1.close()
        db2 = TradingDatabase(str(path))  # re-migrate
        cols = {r[1] for r in db2.conn.execute("PRAGMA table_info(position_states)").fetchall()}
        assert "base_pattern" in cols
        db2.close()

    def test_migration_on_preexisting_table_preserves_data(self, tmp_path):
        """Create a legacy position_states without new columns, migrate, assert data kept."""
        path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(path))
        conn.executescript(
            """
            CREATE TABLE position_states (
                symbol TEXT PRIMARY KEY,
                entry_price REAL,
                entry_date TEXT,
                highest_close REAL,
                current_stop REAL,
                partial_taken INTEGER DEFAULT 0,
                stop_type TEXT,
                updated_at TEXT
            );
            INSERT INTO position_states
              (symbol, entry_price, entry_date, highest_close, current_stop, partial_taken, stop_type, updated_at)
            VALUES ('NVDA', 100.0, '2026-04-01', 110.0, 95.0, 0, 'trailing', '2026-04-01 10:00:00');
            """
        )
        conn.commit()
        conn.close()
        # Open through TradingDatabase → migration runs.
        db = TradingDatabase(str(path))
        row = db.conn.execute(
            "SELECT symbol, entry_price, base_pattern, regime_at_entry FROM position_states"
        ).fetchone()
        assert row["symbol"] == "NVDA"
        assert row["entry_price"] == 100.0
        # New columns default to NULL on legacy rows.
        assert row["base_pattern"] is None
        assert row["regime_at_entry"] is None
        db.close()

    def test_upsert_persists_new_fields(self, tmp_path):
        db = _mk_db(tmp_path)
        db.upsert_position_state("AAPL", {
            "entry_price": 100.0, "entry_date": "2026-04-01",
            "highest_close": 110.0, "current_stop": 95.0,
            "partial_taken": False,
            "base_pattern": "T1", "regime_at_entry": "confirmed_uptrend",
            "rs_at_entry": 88.0, "stage_at_entry": 2.0,
        })
        state = db.get_position_state("AAPL")
        assert state["base_pattern"] == "T1"
        assert state["regime_at_entry"] == "confirmed_uptrend"
        assert state["rs_at_entry"] == 88.0
        assert state["stage_at_entry"] == 2.0

    def test_upsert_merge_preserves_base_pattern(self, tmp_path):
        """An upsert that doesn't include base_pattern shouldn't erase it."""
        db = _mk_db(tmp_path)
        db.upsert_position_state("AAPL", {
            "entry_price": 100.0, "entry_date": "2026-04-01",
            "highest_close": 110.0, "current_stop": 95.0,
            "partial_taken": False,
            "base_pattern": "vcp",
        })
        # Second upsert with a state-update that doesn't re-specify base_pattern
        # (e.g. from the exit manager ratcheting current_stop).
        db.upsert_position_state("AAPL", {
            "entry_price": 100.0, "entry_date": "2026-04-01",
            "highest_close": 112.0, "current_stop": 98.0,
            "partial_taken": False,
        })
        state = db.get_position_state("AAPL")
        assert state["base_pattern"] == "vcp"  # preserved
        assert state["highest_close"] == 112.0  # updated
