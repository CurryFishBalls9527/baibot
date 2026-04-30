"""Tests for the XSection paper trader.

Two threads of coverage:
  1. Parity: select_basket() returns the same longs/shorts as the backtester
     for an identical synthetic universe. This is the guard against drift —
     the backtester and the paper trader implement the selection twice and
     they must agree.
  2. Mechanics: dry-run logs without submitting; intent capture; flatten;
     slippage signing; safety knobs.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


# Pre-existing repo-level import cycle: tradingagents.broker.__init__ eagerly
# imports AlpacaBroker, which imports tradingagents.automation.events, which
# triggers automation/__init__.py → orchestrator.py → AlpacaBroker (mid-load)
# → ImportError. Most tests dodge this by happening to load other modules
# first; running this file in isolation hits the cycle. Pre-stubbing
# automation.events in sys.modules short-circuits the cycle without needing
# real automation modules. See test_alpaca_broker_bracket_legs.py for an
# alternative (in-method imports).
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
    from tradingagents.broker.models import OrderResult, Position
    return OrderResult, Position


def _import_backtester():
    from tradingagents.research.intraday_xsection_backtester import (
        XSectionReversionBacktester,
        XSectionReversionConfig,
    )
    return XSectionReversionBacktester, XSectionReversionConfig


def _import_paper_trader():
    from tradingagents.research.intraday_xsection_paper_trader import (
        XSectionPaperTrader,
        XSectionPaperTraderConfig,
    )
    return XSectionPaperTrader, XSectionPaperTraderConfig


# ---------------------------------------------------------------- mocks


def _make_fake_broker(
    positions=None,
    quotes=None,
    equity: float = 100_000.0,
):
    OrderResult, _Position = _import_models()

    class FakeBroker:
        def __init__(self):
            self.positions = list(positions or [])
            self.quotes = dict(quotes or {})
            self.equity = equity
            self.submitted_orders: list = []
            self.flatten_calls: int = 0
            self.data_client = None
            self._order_book: dict[str, "OrderResult"] = {}

        def get_account(self):
            @dataclass
            class _A:
                equity: float = self.equity

            return _A()

        def get_positions(self):
            return list(self.positions)

        def get_latest_prices(self, symbols):
            return {sym: self.quotes.get(sym, 0.0) for sym in symbols}

        def _make_filled(self, order_id, symbol, side, qty, fill_price):
            return OrderResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                qty=qty,
                notional=None,
                order_type="market",
                status="filled",
                filled_qty=float(qty),
                filled_avg_price=fill_price,
                submitted_at=datetime.now(timezone.utc),
                filled_at=datetime.now(timezone.utc),
            )

        def submit_order(self, order):
            self.submitted_orders.append(order)
            intent = self.quotes.get(order.symbol, 100.0)
            order_id = f"oid_{len(self.submitted_orders)}"
            result = self._make_filled(
                order_id, order.symbol, order.side, order.qty, intent
            )
            self._order_book[order_id] = result
            return result

        def get_order(self, order_id):
            return self._order_book.get(
                order_id,
                OrderResult(
                    order_id=order_id, symbol="?", side="?", qty=None,
                    notional=None, order_type="market", status="unknown",
                    filled_qty=0.0, filled_avg_price=None,
                ),
            )

        def close_all_positions(self):
            self.flatten_calls += 1
            out = []
            for pos in self.positions:
                order_id = f"flat_{pos.symbol}"
                fill_px = self.quotes.get(pos.symbol, pos.current_price)
                result = self._make_filled(
                    order_id, pos.symbol,
                    "sell" if pos.side == "long" else "buy",
                    pos.qty, fill_px,
                )
                self._order_book[order_id] = result
                out.append(result)
            self.positions = []
            return out

    return FakeBroker()


def _config(**kwargs):
    _, XSectionReversionConfig = _import_backtester()
    base = dict(
        initial_cash=100_000.0,
        min_dollar_volume_avg=0.0,
        interval_minutes=30,
        formation_minutes=30,
        hold_minutes=60,
        n_long=1,
        n_short=1,
        dollar_neutral=True,
        target_gross_exposure=1.0,
        earliest_rebalance_time=dtime(9, 0),
        latest_rebalance_time=dtime(9, 0),
        flatten_at_close_time=dtime(11, 0),
        half_spread_bps=0.0,
        slippage_bps=0.0,
        short_borrow_bps_per_day=0.0,
    )
    base.update(kwargs)
    return XSectionReversionConfig(**base)


def _live(tmp_path: Path, **kwargs):
    _, XSectionPaperTraderConfig = _import_paper_trader()
    base = dict(
        log_dir=tmp_path,
        dry_run=True,
        max_gross_exposure=0.5,
        bar_grace_seconds=0.0,
        bar_lookback_days=1,
    )
    base.update(kwargs)
    return XSectionPaperTraderConfig(**base)


def _make_synthetic_frames():
    """Mirror the backtester-test fixture shape: 4 bars 30m apart, with a
    bar past the rebalance time so the backtester's EOD-wash guard doesn't
    skip the schedule."""
    index = pd.to_datetime([
        "2026-04-01 08:30:00",
        "2026-04-01 09:00:00",
        "2026-04-01 10:00:00",
        "2026-04-01 11:00:00",
    ])
    # 30m formation: rank by bar 0 close → bar 1 close.
    # AAA -10% (loser → long), BBB +10% (winner → short),
    # CCC +5%, DDD -5%, EEE +0% (middle).
    payload = {
        "AAA": [100, 90, 95, 101],
        "BBB": [100, 110, 105, 99],
        "CCC": [100, 105, 102, 104],
        "DDD": [100, 95, 99, 97],
        "EEE": [100, 100, 100, 100],
    }
    frames = {}
    for sym, closes in payload.items():
        frames[sym] = pd.DataFrame({
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1_000_000] * len(closes),
        }, index=index)
    return frames


# ---------------------------------------------------------------- parity


def test_paper_trader_select_basket_matches_backtester(tmp_path, monkeypatch):
    frames = _make_synthetic_frames()
    cfg = _config(n_long=1, n_short=1)
    XSectionReversionBacktester, _ = _import_backtester()
    XSectionPaperTrader, _ = _import_paper_trader()

    # Run the backtester to extract its selection.
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_symbol_frames",
        lambda *a, **kw: frames,
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_sector_map",
        lambda *a, **kw: {},
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._compute_dollar_volume",
        lambda frame, lookback_bars: pd.Series(10_000_000.0, index=frame.index),
    )
    bt_result = XSectionReversionBacktester(cfg).backtest(
        symbols=list(frames.keys()),
        begin="2026-04-01",
        end="2026-04-01",
        intraday_db_path="dummy.duckdb",
    )
    bt_trades = {t.symbol: t.side for t in bt_result.trades}

    # Now drive the paper trader's selection directly.
    broker = _make_fake_broker()
    trader = XSectionPaperTrader(cfg, _live(tmp_path), broker, list(frames.keys()))
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_paper_trader._compute_dollar_volume",
        lambda frame, lookback_bars: pd.Series(10_000_000.0, index=frame.index),
    )
    decision = trader.select_basket(frames, pd.Timestamp("2026-04-01 09:00:00"))

    assert decision["action"] == "rebalance"
    assert set(decision["longs"]) == {sym for sym, side in bt_trades.items() if side == "long"}
    assert set(decision["shorts"]) == {sym for sym, side in bt_trades.items() if side == "short"}


# ---------------------------------------------------------------- dry-run


def test_dry_run_logs_intent_without_submitting(tmp_path, monkeypatch):
    frames = _make_synthetic_frames()
    cfg = _config(n_long=1, n_short=1)
    XSectionPaperTrader, _ = _import_paper_trader()
    quotes = {"AAA": 90.0, "DDD": 95.0, "BBB": 110.0, "CCC": 105.0, "EEE": 100.0}
    broker = _make_fake_broker(quotes=quotes, equity=100_000.0)
    trader = XSectionPaperTrader(cfg, _live(tmp_path), broker, list(frames.keys()))
    monkeypatch.setattr(trader, "fetch_bars", lambda: frames)
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_paper_trader._compute_dollar_volume",
        lambda frame, lookback_bars: pd.Series(10_000_000.0, index=frame.index),
    )

    summary = trader.run_one_rebalance(pd.Timestamp("2026-04-01 09:00:00"))
    assert summary["action"] == "rebalance"
    assert summary["open_count"] == 2  # 1 long + 1 short
    assert broker.submitted_orders == []
    assert broker.flatten_calls == 0

    fills = (tmp_path / "fills.jsonl").read_text().strip().splitlines()
    records = [json.loads(line) for line in fills]
    assert {r["symbol"] for r in records} == {"AAA", "BBB"}
    assert all(r["fill_status"] == "dry_run" for r in records)
    assert all(r["dry_run"] is True for r in records)


# ---------------------------------------------------------------- live submit


def test_live_submit_routes_orders_with_intent_and_fill(tmp_path, monkeypatch):
    frames = _make_synthetic_frames()
    cfg = _config(n_long=1, n_short=1)
    XSectionPaperTrader, _ = _import_paper_trader()
    quotes = {"AAA": 90.0, "DDD": 95.0, "BBB": 110.0, "CCC": 105.0, "EEE": 100.0}
    broker = _make_fake_broker(quotes=quotes, equity=10_000.0)
    trader = XSectionPaperTrader(
        cfg, _live(tmp_path, dry_run=False), broker, list(frames.keys())
    )
    monkeypatch.setattr(trader, "fetch_bars", lambda: frames)
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_paper_trader._compute_dollar_volume",
        lambda frame, lookback_bars: pd.Series(10_000_000.0, index=frame.index),
    )

    trader.run_one_rebalance(pd.Timestamp("2026-04-01 09:00:00"))
    assert len(broker.submitted_orders) == 2
    sides = {o.side for o in broker.submitted_orders}
    assert sides == {"buy", "sell"}

    fills = [
        json.loads(line)
        for line in (tmp_path / "fills.jsonl").read_text().strip().splitlines()
    ]
    # All four fills should have an intent_price and fill_price; mock fills at
    # intent so slippage is exactly 0.
    for r in fills:
        assert r["dry_run"] is False
        assert r["intent_price"] > 0
        assert r["fill_price"] == r["intent_price"]
        assert r["slippage_bps"] == 0.0


# ---------------------------------------------------------------- flatten


def test_flatten_all_in_dry_run_emits_records_without_calling_broker(tmp_path):
    cfg = _config()
    XSectionPaperTrader, _ = _import_paper_trader()
    _OrderResult, Position = _import_models()
    positions = [
        Position(
            symbol="AAA", qty=100, side="long",
            avg_entry_price=90.0, current_price=92.0,
            market_value=9200.0, unrealized_pl=200.0, unrealized_plpc=0.022,
        ),
        Position(
            symbol="BBB", qty=50, side="short",
            avg_entry_price=110.0, current_price=108.0,
            market_value=5400.0, unrealized_pl=100.0, unrealized_plpc=0.018,
        ),
    ]
    broker = _make_fake_broker(
        positions=positions,
        quotes={"AAA": 91.5, "BBB": 109.0},
    )
    trader = XSectionPaperTrader(cfg, _live(tmp_path), broker, ["AAA", "BBB"])

    records = trader.flatten_all(pd.Timestamp("2026-04-01 14:55:00"))
    assert {r.symbol for r in records} == {"AAA", "BBB"}
    aaa = next(r for r in records if r.symbol == "AAA")
    bbb = next(r for r in records if r.symbol == "BBB")
    assert aaa.side == "sell" and aaa.action == "flatten"
    assert bbb.side == "buy" and bbb.action == "flatten"
    assert all(r.dry_run for r in records)
    # broker.close_all_positions() must NOT have been called in dry-run
    assert broker.flatten_calls == 0
    # but positions are still present (we didn't mutate them)
    assert len(broker.positions) == 2


def test_flatten_all_live_invokes_close_all_and_logs_fills(tmp_path):
    cfg = _config()
    XSectionPaperTrader, _ = _import_paper_trader()
    _OrderResult, Position = _import_models()
    positions = [
        Position(
            symbol="AAA", qty=100, side="long",
            avg_entry_price=90.0, current_price=92.0,
            market_value=9200.0, unrealized_pl=200.0, unrealized_plpc=0.022,
        ),
    ]
    broker = _make_fake_broker(positions=positions, quotes={"AAA": 91.0})
    trader = XSectionPaperTrader(
        cfg, _live(tmp_path, dry_run=False), broker, ["AAA"]
    )
    records = trader.flatten_all(pd.Timestamp("2026-04-01 14:55:00"))
    assert broker.flatten_calls == 1
    assert len(records) == 1
    assert records[0].fill_status == "filled"
    # Broker filled at quote 91 vs intent 91 → zero slippage
    assert records[0].slippage_bps == 0.0


# ---------------------------------------------------------------- slippage signing


def test_slippage_bps_signed_correctly():
    XSectionPaperTrader, _ = _import_paper_trader()
    # Buy at 100.10 vs intent 100.00 → fill higher than intent → +10 bps (bad for us)
    assert XSectionPaperTrader._slippage_bps(100.0, 100.10, "buy") == 10.0
    # Sell at 99.90 vs intent 100.00 → fill lower than intent → +10 bps (bad for us)
    assert XSectionPaperTrader._slippage_bps(100.0, 99.90, "sell") == 10.0
    # Buy filled cheap → favorable → negative bps
    assert XSectionPaperTrader._slippage_bps(100.0, 99.95, "buy") == -5.0
    # Missing data → None
    assert XSectionPaperTrader._slippage_bps(None, 100.0, "buy") is None
    assert XSectionPaperTrader._slippage_bps(100.0, None, "buy") is None


# ---------------------------------------------------------------- safety cap


def test_max_gross_exposure_caps_per_name_dollars(tmp_path):
    cfg = _config(target_gross_exposure=1.0, n_long=10, n_short=10)
    XSectionPaperTrader, _ = _import_paper_trader()
    # Strategy says 100% gross; safety cap clamps to 50%.
    broker = _make_fake_broker(equity=100_000.0)
    trader = XSectionPaperTrader(cfg, _live(tmp_path, max_gross_exposure=0.5), broker, [])
    long_per, short_per = trader._compute_target_dollars()
    # 100k * 0.5 / 10 names = 5000 per name (cap binds, NOT 10000)
    assert long_per == 5000.0
    assert short_per == 5000.0


# ---------------------------------------------------------------- regime gate


def test_regime_gate_blocks_when_disallowed(tmp_path, monkeypatch):
    frames = _make_synthetic_frames()
    cfg = _config(allowed_market_regimes=("confirmed_uptrend",))
    XSectionPaperTrader, _ = _import_paper_trader()
    broker = _make_fake_broker(equity=100_000.0)
    trader = XSectionPaperTrader(cfg, _live(tmp_path), broker, list(frames.keys()))
    monkeypatch.setattr(
        trader, "_regime_lookup",
        lambda ts: ("market_correction", 1),
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_paper_trader._compute_dollar_volume",
        lambda frame, lookback_bars: pd.Series(10_000_000.0, index=frame.index),
    )
    decision = trader.select_basket(frames, pd.Timestamp("2026-04-01 09:00:00"))
    assert decision["action"] == "regime_blocked"
