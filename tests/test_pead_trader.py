"""Tests for PEAD live paper trader.

Coverage:
  * State persistence (round-trip + corruption tolerance)
  * Signal selection (surprise threshold, universe gate, capacity cap)
  * Exit timing (only closes when exit_target_date reached)
  * Direction-aware slippage signing
  * Dry-run no-op behavior
  * Idempotent re-runs (skip already-held symbols)
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


# Pre-warm to dodge pre-existing broker.__init__ circular import. Same pattern
# as test_intraday_xsection_paper_trader.
def _prewarm_imports():
    if "tradingagents.automation.events" in sys.modules:
        return
    stub = types.ModuleType("tradingagents.automation.events")

    class _Categories:
        ORDER_REJECT = "ORDER_REJECT"
        WASH_TRADE_REJECT = "WASH_TRADE_REJECT"

    stub.Categories = _Categories
    stub.emit_event = lambda *a, **k: None
    sys.modules["tradingagents.automation.events"] = stub


_prewarm_imports()


def _import_models():
    from tradingagents.broker.models import OrderResult
    return OrderResult


def _import_pead():
    from tradingagents.research.pead_trader import (
        PEADConfig, PEADPositionState, PEADTrader,
    )
    return PEADConfig, PEADPositionState, PEADTrader


# ---------------------------------------------------------------- broker mock


def _make_fake_broker(quotes=None, equity: float = 100_000.0,
                     calendar_dates=None):
    """Calendar dates: list of date strings ('YYYY-MM-DD') that are trading days.
    Default: every weekday in 2026 May."""
    OrderResult = _import_models()

    @dataclass
    class _CalEntry:
        date: object  # datetime.date

    if calendar_dates is None:
        # Every weekday May 2026
        calendar_dates = []
        d = pd.Timestamp("2026-05-01")
        while d <= pd.Timestamp("2026-06-30"):
            if d.weekday() < 5:
                calendar_dates.append(d.date())
            d += pd.Timedelta(days=1)
    else:
        calendar_dates = [
            (pd.Timestamp(d).date() if isinstance(d, str) else d)
            for d in calendar_dates
        ]

    class _TradingClient:
        def __init__(self):
            self._calendar_dates = calendar_dates

        def get_calendar(self, req):
            start = req.start
            end = req.end
            return [_CalEntry(date=d) for d in self._calendar_dates
                    if start <= d <= end]

    class FakeBroker:
        def __init__(self):
            self.quotes = dict(quotes or {})
            self.equity = equity
            self.submitted_orders: list = []
            self.closed_symbols: list[str] = []
            self.trading_client = _TradingClient()
            self._order_book: dict = {}

        def get_account(self):
            @dataclass
            class _A:
                equity: float = self.equity

            return _A()

        def get_latest_prices(self, symbols):
            return {sym: self.quotes.get(sym, 0.0) for sym in symbols}

        def submit_order(self, order):
            self.submitted_orders.append(order)
            intent = self.quotes.get(order.symbol, 100.0)
            order_id = f"oid_{len(self.submitted_orders)}"
            result = OrderResult(
                order_id=order_id, symbol=order.symbol, side=order.side,
                qty=order.qty, notional=None, order_type=order.order_type,
                status="filled", filled_qty=float(order.qty),
                filled_avg_price=intent,
                submitted_at=datetime.now(timezone.utc),
                filled_at=datetime.now(timezone.utc),
            )
            self._order_book[order_id] = result
            return result

        def close_position(self, symbol):
            self.closed_symbols.append(symbol)
            order_id = f"close_{symbol}"
            result = OrderResult(
                order_id=order_id, symbol=symbol, side="sell",
                qty=1, notional=None, order_type="market",
                status="filled", filled_qty=1.0,
                filled_avg_price=self.quotes.get(symbol, 100.0),
                submitted_at=datetime.now(timezone.utc),
                filled_at=datetime.now(timezone.utc),
            )
            self._order_book[order_id] = result
            return result

        def get_order(self, order_id):
            return self._order_book.get(
                order_id,
                OrderResult(
                    order_id=order_id, symbol="?", side="?", qty=None,
                    notional=None, order_type="market", status="filled",
                    filled_qty=0.0, filled_avg_price=None,
                ),
            )

    return FakeBroker()


def _config(tmp_path: Path, **kwargs):
    PEADConfig, _, _ = _import_pead()
    base = dict(
        min_positive_surprise_pct=5.0,
        max_positive_surprise_pct=50.0,
        hold_days=20,
        position_pct=0.05,
        max_concurrent_positions=10,
        max_gross_exposure=0.5,
        log_dir=tmp_path,
        dry_run=True,
    )
    base.update(kwargs)
    return PEADConfig(**base)


# ---------------------------------------------------------------- tests


def test_state_round_trip(tmp_path):
    PEADConfig, PEADPositionState, PEADTrader = _import_pead()
    cfg = _config(tmp_path)
    trader = PEADTrader(cfg, _make_fake_broker())

    positions = [
        PEADPositionState(
            symbol="NVDA", surprise_pct=15.5,
            event_date="2026-04-29", entry_date="2026-04-30",
            exit_target_date="2026-05-28",
            entry_order_id="oid_1", entry_price=850.0,
            intent_price=849.5, shares=5, dry_run=False,
        ),
        PEADPositionState(
            symbol="META", surprise_pct=8.2,
            event_date="2026-04-30", entry_date="2026-05-01",
            exit_target_date="2026-05-29",
            entry_order_id="oid_2", entry_price=600.0,
            intent_price=599.8, shares=8, dry_run=False,
        ),
    ]
    trader.save_state(positions)
    loaded = trader.load_state()
    assert len(loaded) == 2
    assert {p.symbol for p in loaded} == {"NVDA", "META"}
    nvda = next(p for p in loaded if p.symbol == "NVDA")
    assert nvda.shares == 5
    assert nvda.surprise_pct == 15.5
    assert nvda.exit_target_date == "2026-05-28"


def test_state_corrupt_file_returns_empty(tmp_path):
    PEADConfig, _, PEADTrader = _import_pead()
    cfg = _config(tmp_path)
    (tmp_path / "positions.json").write_text("not valid json{{{")
    trader = PEADTrader(cfg, _make_fake_broker())
    assert trader.load_state() == []


def test_run_once_skips_non_trading_day(tmp_path):
    PEADConfig, _, PEADTrader = _import_pead()
    cfg = _config(tmp_path)
    # Empty calendar = no trading days
    trader = PEADTrader(cfg, _make_fake_broker(calendar_dates=[]))
    summary = trader.run_once(today=datetime(2026, 5, 1))
    assert summary["action"] == "skip"
    assert summary["reason"] == "not_trading_day"


def test_close_due_positions_only_closes_on_exit_date(tmp_path, monkeypatch):
    PEADConfig, PEADPositionState, PEADTrader = _import_pead()
    cfg = _config(tmp_path, dry_run=False)
    broker = _make_fake_broker(quotes={"NVDA": 900.0, "META": 620.0})
    trader = PEADTrader(cfg, broker)

    positions = [
        # Exit due TODAY (2026-05-15)
        PEADPositionState(
            symbol="NVDA", surprise_pct=15.5,
            event_date="2026-04-15", entry_date="2026-04-16",
            exit_target_date="2026-05-15",
            entry_order_id="oid_1", entry_price=850.0,
            intent_price=850.0, shares=5, dry_run=False,
        ),
        # Exit due in the future
        PEADPositionState(
            symbol="META", surprise_pct=8.2,
            event_date="2026-04-30", entry_date="2026-05-01",
            exit_target_date="2026-05-29",
            entry_order_id="oid_2", entry_price=600.0,
            intent_price=600.0, shares=8, dry_run=False,
        ),
    ]
    still_open, just_closed = trader.close_due_positions(
        positions, today=datetime(2026, 5, 15),
    )
    assert {p.symbol for p in still_open} == {"META"}
    assert {p.symbol for p in just_closed} == {"NVDA"}
    assert "NVDA" in broker.closed_symbols
    assert "META" not in broker.closed_symbols


def test_open_new_positions_skips_already_held(tmp_path):
    PEADConfig, _, PEADTrader = _import_pead()
    cfg = _config(tmp_path, dry_run=False)
    broker = _make_fake_broker(
        quotes={"NVDA": 850.0, "META": 600.0, "AAPL": 200.0},
        equity=100_000.0,
    )
    trader = PEADTrader(cfg, broker)
    signals = [
        {"symbol": "NVDA", "surprise_pct": 12.0, "event_date": "2026-04-29",
         "first_eligible_session": "2026-04-30", "time_hint": "amc"},
        {"symbol": "META", "surprise_pct": 8.0, "event_date": "2026-04-29",
         "first_eligible_session": "2026-04-30", "time_hint": "amc"},
        {"symbol": "AAPL", "surprise_pct": 6.5, "event_date": "2026-04-29",
         "first_eligible_session": "2026-04-30", "time_hint": "amc"},
    ]
    # Already holding NVDA
    new_states = trader.open_new_positions(
        signals, already_open_symbols={"NVDA"}, target_dollars=5000.0,
    )
    # Only META and AAPL should get opened
    assert {s.symbol for s in new_states} == {"META", "AAPL"}
    assert len(broker.submitted_orders) == 2
    assert {o.symbol for o in broker.submitted_orders} == {"META", "AAPL"}


def test_max_concurrent_caps_new_signals(tmp_path):
    PEADConfig, _, PEADTrader = _import_pead()
    cfg = _config(tmp_path, dry_run=False, max_concurrent_positions=3)
    broker = _make_fake_broker(
        quotes={"A": 100.0, "B": 100.0, "C": 100.0, "D": 100.0, "E": 100.0},
        equity=100_000.0,
    )
    trader = PEADTrader(cfg, broker)
    # 5 signals with various surprise pct
    signals = [
        {"symbol": s, "surprise_pct": pct, "event_date": "2026-04-29",
         "first_eligible_session": "2026-04-30", "time_hint": "amc"}
        for s, pct in [("A", 6.0), ("B", 12.0), ("C", 8.0), ("D", 15.0), ("E", 20.0)]
    ]
    # Already holding 1 → room for 2 more
    new_states = trader.open_new_positions(
        signals, already_open_symbols={"X"}, target_dollars=5000.0,
    )
    assert len(new_states) == 2
    # Should pick the 2 with highest surprise (E=20, D=15)
    assert {s.symbol for s in new_states} == {"D", "E"}


def test_dry_run_does_not_submit(tmp_path):
    PEADConfig, _, PEADTrader = _import_pead()
    cfg = _config(tmp_path, dry_run=True)
    broker = _make_fake_broker(quotes={"NVDA": 850.0})
    trader = PEADTrader(cfg, broker)
    signals = [
        {"symbol": "NVDA", "surprise_pct": 12.0, "event_date": "2026-04-29",
         "first_eligible_session": "2026-04-30", "time_hint": "amc"},
    ]
    new_states = trader.open_new_positions(
        signals, already_open_symbols=set(), target_dollars=5000.0,
    )
    # In dry-run we still record the position
    assert len(new_states) == 1
    # But broker was NOT called
    assert broker.submitted_orders == []


def test_compute_target_dollars_caps_via_max_gross(tmp_path):
    PEADConfig, _, PEADTrader = _import_pead()
    cfg = _config(tmp_path, position_pct=0.10, max_gross_exposure=0.05)
    broker = _make_fake_broker(equity=100_000.0)
    trader = PEADTrader(cfg, broker)
    target = trader.compute_target_dollars()
    # min(0.10, 0.05) * 100k = 5000
    assert target == 5000.0


def test_slippage_bps_signed():
    _, _, PEADTrader = _import_pead()
    # Buy filled higher than intent → unfavorable → positive bps
    assert PEADTrader._slippage_bps(100.0, 100.10, "buy") == 10.0
    # Sell filled lower than intent → unfavorable → positive bps
    assert PEADTrader._slippage_bps(100.0, 99.90, "sell") == 10.0
    # Buy filled cheap → favorable → negative bps
    assert PEADTrader._slippage_bps(100.0, 99.95, "buy") == -5.0
    # Missing data
    assert PEADTrader._slippage_bps(None, 100.0, "buy") is None
    assert PEADTrader._slippage_bps(100.0, None, "buy") is None


def test_idempotent_rerun_doesnt_double_submit(tmp_path, monkeypatch):
    """Re-running on the same day with the same state should not submit twice."""
    PEADConfig, PEADPositionState, PEADTrader = _import_pead()
    cfg = _config(tmp_path, dry_run=False)
    broker = _make_fake_broker(
        quotes={"NVDA": 850.0},
        equity=100_000.0,
        calendar_dates=["2026-05-01"],
    )
    trader = PEADTrader(cfg, broker)
    # Pre-load NVDA position
    trader.save_state([
        PEADPositionState(
            symbol="NVDA", surprise_pct=12.0,
            event_date="2026-04-29", entry_date="2026-04-30",
            exit_target_date="2026-05-28",
            entry_order_id="oid_1", entry_price=850.0,
            intent_price=850.0, shares=5, dry_run=False,
        ),
    ])
    # Mock find_new_signals to return NVDA again (which we already hold)
    monkeypatch.setattr(
        trader, "find_new_signals",
        lambda today: [
            {"symbol": "NVDA", "surprise_pct": 12.0, "event_date": "2026-04-29",
             "first_eligible_session": "2026-05-01", "time_hint": "amc"},
        ],
    )
    summary = trader.run_once(today=datetime(2026, 5, 1))
    # NVDA was held, signal was for NVDA → no new orders
    assert summary["new_positions_opened"] == 0
    assert broker.submitted_orders == []
    # State preserved
    assert len(trader.load_state()) == 1
