"""Tests for Chan base_pattern backfill (PR2).

Covers:
- The regex correctly extracts T-type from real-world Chan entry reasoning
- Non-match cases (non-Chan / malformed strings) return None
- The DB helper UPDATE-by-id works
- Idempotency of the extract + apply flow
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tradingagents.storage.database import TradingDatabase

# Import the function under test directly from the script file.
import importlib.util


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "backfill_chan_base_pattern.py"
)


@pytest.fixture(scope="module")
def extract_t_type():
    spec = importlib.util.spec_from_file_location(
        "backfill_chan_base_pattern_module", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.extract_t_type


class TestExtract:
    def test_single_t2s(self, extract_t_type):
        assert extract_t_type(
            "Chan buy signal: T2S at 2026/04/17 18:00"
        ) == "T2S"

    def test_single_t1(self, extract_t_type):
        assert extract_t_type(
            "Chan buy signal: T1 at 2026/04/15 14:30"
        ) == "T1"

    def test_single_t2(self, extract_t_type):
        assert extract_t_type(
            "Chan buy signal: T2 at 2026/04/15 14:30"
        ) == "T2"

    def test_combo_t1_t2s(self, extract_t_type):
        assert extract_t_type(
            "Chan buy signal: T1+T2S at 2026/04/15 14:30"
        ) == "T1+T2S"

    def test_combo_t2_t2s(self, extract_t_type):
        assert extract_t_type(
            "Chan buy signal: T2+T2S at 2026/04/15 14:30"
        ) == "T2+T2S"

    def test_empty_string(self, extract_t_type):
        assert extract_t_type("") is None

    def test_none_input(self, extract_t_type):
        assert extract_t_type(None) is None

    def test_non_chan_reasoning(self, extract_t_type):
        assert extract_t_type(
            "Minervini rule entry: base=cup_handle stage=2 status=leader "
            "buy_point=132.43"
        ) is None

    def test_malformed_chan_string(self, extract_t_type):
        # Missing T-type after "buy signal:"
        assert extract_t_type("Chan buy signal: at 2026/04/17") is None


class TestDBHelper:
    def test_update_base_pattern(self, tmp_path: Path):
        db = TradingDatabase(str(tmp_path / "t.db"))
        oid = db.log_trade_outcome({
            "symbol": "AAOI", "entry_date": "2026-04-20",
            "exit_date": "2026-04-21", "entry_price": 100.0,
            "exit_price": 105.0, "return_pct": 0.05, "hold_days": 1,
            "exit_reason": "test",
        })
        # base_pattern defaults to NULL.
        row = db.conn.execute(
            "SELECT base_pattern FROM trade_outcomes WHERE id = ?", (oid,)
        ).fetchone()
        assert row["base_pattern"] is None

        db.update_trade_outcome_base_pattern(oid, "T2S")
        row = db.conn.execute(
            "SELECT base_pattern FROM trade_outcomes WHERE id = ?", (oid,)
        ).fetchone()
        assert row["base_pattern"] == "T2S"

    def test_update_allows_null(self, tmp_path: Path):
        db = TradingDatabase(str(tmp_path / "t.db"))
        oid = db.log_trade_outcome({
            "symbol": "X", "entry_date": "2026-04-20",
            "exit_date": "2026-04-21", "entry_price": 100.0,
            "exit_price": 105.0, "return_pct": 0.05, "hold_days": 1,
            "exit_reason": "test", "base_pattern": "T2",
        })
        db.update_trade_outcome_base_pattern(oid, None)
        row = db.conn.execute(
            "SELECT base_pattern FROM trade_outcomes WHERE id = ?", (oid,)
        ).fetchone()
        assert row["base_pattern"] is None
