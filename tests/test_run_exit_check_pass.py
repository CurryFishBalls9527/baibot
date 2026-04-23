"""Regression tests for Orchestrator.run_exit_check_pass.

Covers:
  - daily scan parity (exit manager decisions fire the same way as before)
  - idempotency: concurrent callers don't double-submit SELL
  - `ai_review_enabled=False` skips LLM review path (for 5-min cron usage)
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tradingagents.automation.orchestrator import Orchestrator
from tradingagents.broker.models import Account, Position


def _make_account() -> Account:
    return Account(
        account_id="paper",
        equity=100_000.0, cash=50_000.0,
        buying_power=100_000.0, portfolio_value=100_000.0,
        status="ACTIVE",
    )


def _make_position(symbol="AAPL", qty=100, entry=100.0, current=85.0) -> Position:
    return Position(
        symbol=symbol, qty=qty, side="long",
        avg_entry_price=entry, current_price=current,
        market_value=qty * current,
        unrealized_pl=(current - entry) * qty,
        unrealized_plpc=(current / entry) - 1,
    )


def _make_orch(broker, db, config=None):
    """Build a minimal Orchestrator skipping __init__."""
    o = Orchestrator.__new__(Orchestrator)
    o.broker = broker
    o.db = db
    o.config = config or {
        "exit_manager_enabled": True,
        "exit_manager_version": "v1",
        "mechanical_only_mode": True,
        "use_50dma_exit": True,
        "trail_stop_pct": 0.10,
    }
    o.risk_engine = MagicMock()
    o.risk_engine.check_order.return_value = SimpleNamespace(passed=True)
    o.notifier = SimpleNamespace(enabled=False, send=lambda *a, **kw: None)
    # Stub private methods we don't want to exercise here.
    o._get_latest_features = lambda sym: {}
    o._log_trade_outcome = lambda *args, **kwargs: None
    o._analyze_and_trade = MagicMock(
        return_value={"symbol": "AAPL", "action": "HOLD", "traded": False}
    )
    # Stub _execute_structured_signal — the real method requires broker
    # methods (get_latest_price, etc.) that aren't relevant to exit-pass
    # idempotency. We care that it's called (or not), not what it does.
    o._execute_structured_signal = MagicMock(
        return_value={"symbol": "AAPL", "action": "SELL", "traded": True}
    )
    # These are real methods on the class; no need to stub.
    return o


class FakeBroker:
    """Tracks submit_order, supports sequencing of get_live_orders returns."""

    def __init__(self, positions=None, open_order_sequence=None):
        self._positions = positions or []
        # List of lists: each call to get_live_orders returns the next entry.
        self._open_order_sequence = open_order_sequence or []
        self._get_open_orders_calls = 0
        self.submitted_orders = []

    def get_account(self):
        return _make_account()

    def get_positions(self):
        return list(self._positions)

    def get_live_orders(self, symbol=None):
        idx = min(self._get_open_orders_calls, len(self._open_order_sequence) - 1)
        self._get_open_orders_calls += 1
        orders = self._open_order_sequence[idx] if self._open_order_sequence else []
        if symbol:
            return [o for o in orders if getattr(o, "symbol", None) == symbol]
        return orders

    def submit_order(self, order):
        self.submitted_orders.append(order)
        return SimpleNamespace(
            order_id=f"new-{len(self.submitted_orders)}",
            status="accepted",
            filled_qty=0.0,
            filled_avg_price=None,
        )


class FakeDB:
    """In-memory stand-in for TradingDatabase."""

    def __init__(self, states=None):
        self._states = states or {}
        self.deletes = []
        self.logged_trades = []
        self.logged_signals = []

    def get_position_state(self, symbol):
        return self._states.get(symbol)

    def upsert_position_state(self, symbol, state):
        existing = self._states.get(symbol, {}) or {}
        self._states[symbol] = {**existing, **state}

    def delete_position_state(self, symbol):
        self.deletes.append(symbol)
        self._states.pop(symbol, None)

    def log_signal(self, **kwargs):
        self.logged_signals.append(kwargs)
        return len(self.logged_signals)

    def mark_signal_executed(self, sid):
        pass

    def mark_signal_rejected(self, sid, reason):
        pass

    def log_trade(self, **kwargs):
        self.logged_trades.append(kwargs)


# ── SELL idempotency ─────────────────────────────────────────────────────


class TestSellIdempotency:
    def _setup_stop_below_trail(self):
        """Position where trailing stop should fire SELL."""
        position = _make_position("AAPL", qty=100, entry=100.0, current=85.0)
        state = {
            "entry_price": 100.0,
            "entry_date": "2026-04-01",
            "highest_close": 120.0,
            # Trail_stop = 120 * 0.9 = 108. Current 85 < 108 → SELL.
            "current_stop": 108.0,
            "partial_taken": False,
        }
        return position, state

    def test_first_call_submits_sell(self):
        position, state = self._setup_stop_below_trail()
        broker = FakeBroker(
            positions=[position],
            open_order_sequence=[[]],  # no existing open sells
        )
        db = FakeDB(states={"AAPL": state})
        orch = _make_orch(broker, db)

        results = orch.run_exit_check_pass(ai_review_enabled=False)

        assert "AAPL" in results
        # SELL is dispatched via _execute_structured_signal (stubbed).
        assert orch._execute_structured_signal.call_count == 1
        call_kwargs = orch._execute_structured_signal.call_args.kwargs
        assert call_kwargs["structured"]["action"] == "SELL"
        assert call_kwargs["symbol"] == "AAPL"
        assert db.deletes == ["AAPL"]

    def test_second_call_with_in_flight_sell_is_noop(self):
        """When an open sell exists at broker, skip — don't duplicate."""
        position, state = self._setup_stop_below_trail()
        existing_sell = SimpleNamespace(
            order_id="in-flight-1",
            symbol="AAPL",
            side="sell",
            status="accepted",
        )
        broker = FakeBroker(
            positions=[position],
            open_order_sequence=[[existing_sell]],
        )
        db = FakeDB(states={"AAPL": state})
        orch = _make_orch(broker, db)

        results = orch.run_exit_check_pass(ai_review_enabled=False)

        assert len(broker.submitted_orders) == 0
        assert db.deletes == []  # don't delete state if we didn't submit
        assert results["AAPL"]["action"] == "SKIP"
        assert "idempotency" in results["AAPL"]["screen_rejected"].lower()

    def test_back_to_back_calls_submit_only_once(self):
        """Simulate 5-min cron + daily scan racing. First call submits,
        second call sees in-flight sell, skips. No duplicate submission."""
        position, state = self._setup_stop_below_trail()
        # Call 1: no open orders → submit fires. Call 2: the submitted
        # order is now open → guard fires.
        submitted_sell = SimpleNamespace(
            order_id="new-1", symbol="AAPL", side="sell", status="accepted",
        )
        broker = FakeBroker(
            positions=[position],
            open_order_sequence=[[], [submitted_sell]],
        )
        # Simulate mid-flight race: state not yet deleted when second call
        # queries. Override delete to be a no-op so the second call still
        # sees the position_state row.
        db = FakeDB(states={"AAPL": state})
        db.delete_position_state = lambda sym: db.deletes.append(sym)
        orch = _make_orch(broker, db)

        # Call 1 — submits via _execute_structured_signal stub
        orch.run_exit_check_pass(ai_review_enabled=False)
        assert orch._execute_structured_signal.call_count == 1
        # Call 2 — guard sees in-flight sell, skips submit
        r2 = orch.run_exit_check_pass(ai_review_enabled=False)
        assert orch._execute_structured_signal.call_count == 1, (
            "Second call must not re-submit SELL (idempotency guard)"
        )
        assert r2["AAPL"]["action"] == "SKIP"


# ── HOLD path idempotency ──────────────────────────────────────────────


class TestHoldIsPureNoop:
    def test_hold_decision_doesnt_submit_anything(self):
        """HOLD must never trigger an order, no matter how often it's called."""
        position = _make_position("AAPL", qty=100, entry=100.0, current=105.0)
        state = {
            "entry_price": 100.0,
            "entry_date": "2026-04-01",
            "highest_close": 105.0,
            # Trail_stop = 105 * 0.9 = 94.5 < current 105 → no stop fired.
            "current_stop": 92.0,
            "partial_taken": False,
        }
        broker = FakeBroker(positions=[position], open_order_sequence=[[], [], []])
        db = FakeDB(states={"AAPL": state})
        orch = _make_orch(broker, db)

        for _ in range(3):
            orch.run_exit_check_pass(ai_review_enabled=False)

        assert len(broker.submitted_orders) == 0
        assert db.deletes == []
        # State should still exist (updated by each HOLD tick).
        assert db.get_position_state("AAPL") is not None


# ── AI review toggle ───────────────────────────────────────────────────


class TestAIReviewToggle:
    def test_ai_review_disabled_skips_llm_path(self):
        """When ai_review_enabled=False, warnings don't escalate to AI.

        Sets up a non-mechanical variant (no mechanical_only_mode) so the
        AI path WOULD normally fire. With the flag off, it must not.
        """
        position = _make_position("AAPL", qty=100, entry=100.0, current=98.0)
        state = {
            "entry_price": 100.0,
            "entry_date": "2026-04-01",
            "highest_close": 105.0,
            "current_stop": 94.0,  # stays below 98 → no stop
            "partial_taken": False,
        }
        broker = FakeBroker(positions=[position])
        db = FakeDB(states={"AAPL": state})
        config = {
            "exit_manager_enabled": True,
            "exit_manager_version": "v1",
            "mechanical_only_mode": False,  # would normally run AI
            "use_50dma_exit": True,
            "trail_stop_pct": 0.10,
        }
        orch = _make_orch(broker, db, config=config)
        # Feed warning-triggering feature so warnings will accumulate.
        orch._get_latest_features = lambda sym: {
            "rs_percentile": 30,  # < 50 triggers warning
            "sma_50": 100.0,  # price 98 < 102 triggers warning
            "adx_14": 10,  # < 15 triggers warning
        }

        orch.run_exit_check_pass(ai_review_enabled=False)

        # AI path must not have been called even though warnings present.
        orch._analyze_and_trade.assert_not_called()

    def test_ai_review_enabled_runs_llm_path_on_warnings(self):
        position = _make_position("AAPL", qty=100, entry=100.0, current=98.0)
        state = {
            "entry_price": 100.0,
            "entry_date": "2026-04-01",
            "highest_close": 105.0,
            "current_stop": 94.0,
            "partial_taken": False,
        }
        broker = FakeBroker(positions=[position])
        db = FakeDB(states={"AAPL": state})
        config = {
            "exit_manager_enabled": True,
            "exit_manager_version": "v1",
            "mechanical_only_mode": False,
            # Disable 50DMA exit so we get HOLD (not SELL) and fall through
            # to the warnings-and-AI-review path.
            "use_50dma_exit": False,
            "trail_stop_pct": 0.10,
        }
        orch = _make_orch(broker, db, config=config)
        orch._get_latest_features = lambda sym: {
            "rs_percentile": 30,  # < 50 → warning
            "sma_50": 100.0,      # price 98 < 102 → warning
            "adx_14": 10,         # < 15 → warning
        }

        orch.run_exit_check_pass(ai_review_enabled=True)

        orch._analyze_and_trade.assert_called_once()


# ── no-positions / edge cases ─────────────────────────────────────────


class TestEdgeCases:
    def test_no_held_positions_returns_empty(self):
        broker = FakeBroker(positions=[])
        db = FakeDB()
        orch = _make_orch(broker, db)
        assert orch.run_exit_check_pass() == {}

    def test_exit_manager_disabled_mechanical_mode_is_noop(self):
        position = _make_position("AAPL")
        broker = FakeBroker(positions=[position])
        db = FakeDB()
        config = {
            "exit_manager_enabled": False,
            "mechanical_only_mode": True,
        }
        orch = _make_orch(broker, db, config=config)

        # Daily scan: exit_manager disabled + mechanical_only_mode → HOLD only.
        results = orch.run_exit_check_pass(ai_review_enabled=True)
        assert results["AAPL"]["action"] == "HOLD"
        assert len(broker.submitted_orders) == 0

        # 5-min cron: ai_review_enabled=False → short-circuits entirely.
        results_cron = orch.run_exit_check_pass(ai_review_enabled=False)
        assert results_cron == {}
