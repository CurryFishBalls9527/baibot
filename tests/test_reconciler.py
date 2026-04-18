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
    rec = OrderReconciler(broker=broker, db=db, variant="mechanical")
    result = rec.reconcile_once()
    # Filled already → not in open set; broker should never be called.
    assert result["checked"] == 0
    broker.get_order.assert_not_called()


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
    assert result["errors"] == 1
    assert result["updated"] == 0
    # Local row untouched.
    row = db.conn.execute(
        "SELECT status FROM trades WHERE order_id=?", ("err-1",)
    ).fetchone()
    assert row["status"] == "OrderStatus.NEW"


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
