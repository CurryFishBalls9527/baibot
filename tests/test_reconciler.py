"""Tests for OrderReconciler (Track P-SYNC)."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tradingagents.automation.reconciler import OrderReconciler, _is_open
from tradingagents.broker.models import OrderResult
from tradingagents.storage.database import TradingDatabase


def test_is_open_classifies_statuses():
    assert _is_open("OrderStatus.PENDING_NEW")
    assert _is_open("OrderStatus.NEW")
    assert _is_open("OrderStatus.ACCEPTED")
    assert _is_open("OrderStatus.PARTIALLY_FILLED")
    assert _is_open("pending_new")
    # terminal
    assert not _is_open("OrderStatus.FILLED")
    assert not _is_open("OrderStatus.CANCELED")
    assert not _is_open("OrderStatus.REJECTED")
    assert not _is_open("OrderStatus.EXPIRED")
    assert not _is_open(None)
    assert not _is_open("")


def _mkdb(tmp_path: Path, variant: str = "mechanical"):
    return TradingDatabase(str(tmp_path / "test.db"), variant=variant)


def test_reconcile_updates_drifted_row(tmp_path):
    db = _mkdb(tmp_path)
    db.log_trade(
        symbol="NVDA",
        side="buy",
        qty=10,
        status="OrderStatus.PENDING_NEW",
        filled_qty=0,
        filled_price=None,
        order_id="abc-123",
    )
    broker = MagicMock()
    broker.get_order.return_value = OrderResult(
        order_id="abc-123",
        symbol="NVDA",
        side="buy",
        qty=10,
        notional=None,
        order_type="market",
        status="OrderStatus.FILLED",
        filled_qty=10,
        filled_avg_price=198.55,
    )

    rec = OrderReconciler(broker=broker, db=db, variant="mechanical")
    result = rec.reconcile_once()

    assert result["checked"] == 1
    assert result["drifted"] == 1
    assert result["updated"] == 1

    row = db.conn.execute(
        "SELECT status, filled_qty, filled_price FROM trades WHERE order_id=?",
        ("abc-123",),
    ).fetchone()
    assert row["status"] == "OrderStatus.FILLED"
    assert row["filled_qty"] == 10
    assert row["filled_price"] == 198.55


def test_reconcile_no_drift_no_update(tmp_path):
    db = _mkdb(tmp_path)
    db.log_trade(
        symbol="AAPL",
        side="buy",
        qty=5,
        status="OrderStatus.FILLED",
        filled_qty=5,
        filled_price=180.0,
        order_id="ok-999",
    )
    broker = MagicMock()
    # Mock order has MagicMock stop/tp_order_id; sweep must skip non-str legs.
    rec = OrderReconciler(broker=broker, db=db, variant="mechanical")
    result = rec.reconcile_once()
    # Filled already → not in drift-pass open set.
    assert result["checked"] == 0
    # Bracket-leg sweep still probes the filled parent, but MagicMock leg_ids
    # are rejected defensively and no sell rows are inserted.
    assert result["leg_inserted"] == 0


def test_reconcile_skips_rows_without_order_id(tmp_path):
    db = _mkdb(tmp_path)
    db.log_trade(
        symbol="TSLA",
        side="buy",
        qty=2,
        status="OrderStatus.PENDING_NEW",
        filled_qty=0,
        order_id=None,
    )
    broker = MagicMock()
    rec = OrderReconciler(broker=broker, db=db, variant="mechanical")
    result = rec.reconcile_once()
    assert result["checked"] == 0
    broker.get_order.assert_not_called()


def test_reconcile_handles_broker_error(tmp_path):
    db = _mkdb(tmp_path)
    db.log_trade(
        symbol="AMD",
        side="buy",
        qty=3,
        status="OrderStatus.NEW",
        order_id="err-1",
    )
    broker = MagicMock()
    broker.get_order.side_effect = RuntimeError("broker unavailable")
    rec = OrderReconciler(broker=broker, db=db, variant="mechanical")
    result = rec.reconcile_once()
    assert result["checked"] == 1
    # Broker error hits both drift pass and sweep pass (same symbol) → 2 errs.
    assert result["errors"] == 2
    assert result["updated"] == 0
    # Local row untouched.
    row = db.conn.execute(
        "SELECT status FROM trades WHERE order_id=?", ("err-1",)
    ).fetchone()
    assert row["status"] == "OrderStatus.NEW"


def _mk_filled_leg(order_id, status="OrderStatus.FILLED", filled_qty=10,
                   filled_price=95.0, qty=10):
    leg = MagicMock()
    leg.order_id = order_id
    leg.status = status
    leg.filled_qty = filled_qty
    leg.filled_avg_price = filled_price
    leg.qty = qty
    leg.filled_at = None
    return leg


def test_bracket_fill_logs_trade_outcome(tmp_path):
    """Regression: bracket leg fill must write a trade_outcomes row.

    Closes Gap A — _insert_leg_fill used to write trades but not outcomes,
    so the daily review silently missed every bracket-fired exit.
    """
    db = _mkdb(tmp_path, variant="llm")
    # Simulate an open position: buy row + position_state.
    db.log_trade(
        symbol="AAOI", side="buy", qty=67,
        status="OrderStatus.FILLED", filled_qty=67, filled_price=151.80,
        order_id="parent-oid",
    )
    db.upsert_position_state("AAOI", {
        "entry_price": 151.80,
        "entry_date": "2026-04-13",
        "highest_close": 165.0,
        "current_stop": 151.80,
        "partial_taken": False,
        "variant": "llm",
        "base_pattern": "leader_continuation",
        "regime_at_entry": "confirmed_uptrend",
    })

    parent = OrderResult(
        order_id="parent-oid", symbol="AAOI", side="buy", qty=67,
        notional=None, order_type="market", status="OrderStatus.FILLED",
        filled_qty=67, filled_avg_price=151.80,
        stop_order_id="sl-oid", tp_order_id="tp-oid",
    )
    broker = MagicMock()
    broker.data_client = None  # skip excursion network call
    broker.get_order.side_effect = lambda oid: {
        "parent-oid": parent,
        "sl-oid": _mk_filled_leg("sl-oid", filled_price=151.71, qty=67),
        "tp-oid": _mk_filled_leg(
            "tp-oid", status="OrderStatus.CANCELED", filled_qty=0,
            filled_price=None, qty=67,
        ),
    }[oid]

    rec = OrderReconciler(broker=broker, db=db, variant="llm")
    result = rec.reconcile_once()

    # Trade row inserted for the stop fill
    assert result["leg_inserted"] == 1
    # trade_outcomes row materialized
    outcomes = db.conn.execute(
        "SELECT symbol, exit_reason, return_pct, entry_price, exit_price "
        "FROM trade_outcomes WHERE symbol='AAOI'"
    ).fetchall()
    assert len(outcomes) == 1
    row = outcomes[0]
    assert row["exit_reason"] == "bracket_stop_loss"
    assert row["entry_price"] == 151.80
    assert row["exit_price"] == 151.71
    # Return: (151.71 - 151.80) / 151.80 ≈ -0.000593
    assert row["return_pct"] < 0
    assert row["return_pct"] > -0.01


def test_bracket_fill_outcome_idempotent(tmp_path):
    """Re-running reconcile on the same fill must not duplicate the outcome."""
    db = _mkdb(tmp_path, variant="llm")
    db.log_trade(
        symbol="AAOI", side="buy", qty=67,
        status="OrderStatus.FILLED", filled_qty=67, filled_price=151.80,
        order_id="parent-oid",
    )
    db.upsert_position_state("AAOI", {
        "entry_price": 151.80, "entry_date": "2026-04-13",
        "highest_close": 151.80, "current_stop": 151.80,
        "partial_taken": False, "variant": "llm",
    })
    # Pre-existing outcome — simulates the backfill script already ran.
    db.log_trade_outcome({
        "symbol": "AAOI", "entry_date": "2026-04-13", "exit_date": "2026-04-23",
        "entry_price": 151.80, "exit_price": 151.71,
        "return_pct": -0.0006, "hold_days": 10,
        "exit_reason": "bracket_stop_loss",
    })
    parent = OrderResult(
        order_id="parent-oid", symbol="AAOI", side="buy", qty=67,
        notional=None, order_type="market", status="OrderStatus.FILLED",
        filled_qty=67, filled_avg_price=151.80,
        stop_order_id="sl-oid", tp_order_id="tp-oid",
    )
    broker = MagicMock()
    broker.data_client = None
    broker.get_order.side_effect = lambda oid: {
        "parent-oid": parent,
        "sl-oid": _mk_filled_leg("sl-oid", filled_price=151.71, qty=67),
        "tp-oid": _mk_filled_leg(
            "tp-oid", status="OrderStatus.CANCELED", filled_qty=0,
            filled_price=None, qty=67,
        ),
    }[oid]

    rec = OrderReconciler(broker=broker, db=db, variant="llm")
    rec.reconcile_once()

    outcomes = db.conn.execute(
        "SELECT COUNT(*) c FROM trade_outcomes WHERE symbol='AAOI'"
    ).fetchone()
    assert outcomes["c"] == 1  # still just the one, no duplicate


def test_bracket_fill_no_outcome_when_state_missing(tmp_path):
    """If position_state is already gone, skip silently (no crash)."""
    db = _mkdb(tmp_path, variant="llm")
    db.log_trade(
        symbol="AAOI", side="buy", qty=67,
        status="OrderStatus.FILLED", filled_qty=67, filled_price=151.80,
        order_id="parent-oid",
    )
    # NOTE: no position_state row.
    parent = OrderResult(
        order_id="parent-oid", symbol="AAOI", side="buy", qty=67,
        notional=None, order_type="market", status="OrderStatus.FILLED",
        filled_qty=67, filled_avg_price=151.80,
        stop_order_id="sl-oid", tp_order_id="tp-oid",
    )
    broker = MagicMock()
    broker.data_client = None
    broker.get_order.side_effect = lambda oid: {
        "parent-oid": parent,
        "sl-oid": _mk_filled_leg("sl-oid", filled_price=151.71, qty=67),
        "tp-oid": _mk_filled_leg(
            "tp-oid", status="OrderStatus.CANCELED", filled_qty=0,
            filled_price=None, qty=67,
        ),
    }[oid]
    rec = OrderReconciler(broker=broker, db=db, variant="llm")
    rec.reconcile_once()  # must not raise

    outcomes = db.conn.execute(
        "SELECT COUNT(*) c FROM trade_outcomes"
    ).fetchone()
    assert outcomes["c"] == 0


def test_reconcile_respects_lookback(tmp_path):
    db = _mkdb(tmp_path)
    db.log_trade(
        symbol="META",
        side="buy",
        qty=1,
        status="OrderStatus.NEW",
        order_id="old-1",
    )
    # Backdate the row past the lookback window.
    old_ts = (datetime.utcnow() - timedelta(days=30)).isoformat()
    db.conn.execute(
        "UPDATE trades SET timestamp=? WHERE order_id=?", (old_ts, "old-1")
    )
    db.conn.commit()

    broker = MagicMock()
    rec = OrderReconciler(broker=broker, db=db, variant="mechanical", lookback_days=7)
    result = rec.reconcile_once()
    assert result["checked"] == 0
    broker.get_order.assert_not_called()
