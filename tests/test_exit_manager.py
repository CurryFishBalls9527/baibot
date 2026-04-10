"""Tests for the ExitManager (Phase 2)."""

import pytest
from dataclasses import dataclass
from datetime import date, timedelta

from tradingagents.portfolio.exit_manager import ExitManager


@dataclass
class MockPosition:
    symbol: str
    current_price: float
    qty: float
    avg_entry_price: float
    market_value: float = 0.0


class TestExitManager:
    def _default_config(self):
        return {
            "trail_stop_pct": 0.10,
            "breakeven_trigger_pct": 0.05,
            "partial_profit_trigger_pct": 0.12,
            "partial_profit_fraction": 0.33,
            "max_hold_days": 60,
            "use_50dma_exit": True,
        }

    def _default_state(self, entry_price=100.0, days_ago=10):
        entry_date = (date.today() - timedelta(days=days_ago)).isoformat()
        return {
            "entry_price": entry_price,
            "entry_date": entry_date,
            "highest_close": entry_price * 1.05,
            "current_stop": entry_price * 0.92,
            "partial_taken": False,
        }

    def test_hold_when_healthy(self):
        """Position in good shape should HOLD."""
        em = ExitManager(self._default_config())
        pos = MockPosition("AAPL", 108.0, 100, 100.0)
        state = self._default_state(100.0, days_ago=5)
        features = {"sma_50": 95.0}

        decision = em.check_position(pos, features, state)
        assert decision.action == "HOLD"
        assert decision.updated_state is not None

    def test_trailing_stop_triggers_sell(self):
        """Price dropped below trailing stop should SELL."""
        em = ExitManager(self._default_config())
        # highest_close = 120, trail_stop = 120 * 0.9 = 108
        state = self._default_state(100.0)
        state["highest_close"] = 120.0
        state["current_stop"] = 108.0

        pos = MockPosition("AAPL", 107.0, 100, 100.0)  # below 108
        features = {"sma_50": 95.0}

        decision = em.check_position(pos, features, state)
        assert decision.action == "SELL"
        assert decision.reason == "trailing_stop"
        assert decision.qty == 100

    def test_lost_50dma_triggers_sell(self):
        """Price below 50 DMA should SELL."""
        em = ExitManager(self._default_config())
        # Price 94, sma_50=95.  highest_close=100 -> trail=90, current_stop=92.
        # Price 94 > trail(90) and > stop(92) so trailing stop won't fire.
        # But price < sma_50(95) so 50dma exit fires.
        pos = MockPosition("AAPL", 94.0, 100, 100.0)
        state = self._default_state(100.0)
        state["highest_close"] = 100.0  # trail_stop = 100*0.9 = 90
        state["current_stop"] = 92.0
        features = {"sma_50": 95.0}

        decision = em.check_position(pos, features, state)
        assert decision.action == "SELL"
        assert decision.reason == "lost_50dma"

    def test_50dma_disabled(self):
        """When use_50dma_exit=False, don't sell on 50 DMA breach."""
        config = self._default_config()
        config["use_50dma_exit"] = False
        em = ExitManager(config)
        pos = MockPosition("AAPL", 94.0, 100, 100.0)
        state = self._default_state(100.0)
        state["highest_close"] = 100.0  # trail_stop = 90
        state["current_stop"] = 92.0
        features = {"sma_50": 95.0}

        decision = em.check_position(pos, features, state)
        assert decision.action == "HOLD"

    def test_max_hold_days_triggers_sell(self):
        """Position held too long should SELL."""
        em = ExitManager(self._default_config())
        pos = MockPosition("AAPL", 105.0, 100, 100.0)
        state = self._default_state(100.0, days_ago=65)  # > 60 max_hold_days
        state["current_stop"] = 85.0
        features = {"sma_50": 95.0}

        decision = em.check_position(pos, features, state)
        assert decision.action == "SELL"
        assert decision.reason == "max_hold_days"

    def test_partial_profit_triggers(self):
        """Price up 12%+ should trigger partial sell."""
        em = ExitManager(self._default_config())
        # entry=100, price=113 -> 13% gain, above 12% trigger
        pos = MockPosition("AAPL", 113.0, 100, 100.0)
        state = self._default_state(100.0, days_ago=10)
        state["current_stop"] = 92.0
        features = {"sma_50": 95.0}

        decision = em.check_position(pos, features, state)
        assert decision.action == "PARTIAL_SELL"
        assert decision.reason == "partial_profit"
        assert decision.qty == 33  # 100 * 0.33
        assert decision.updated_state["partial_taken"] is True

    def test_no_double_partial(self):
        """If partial already taken, don't trigger again."""
        em = ExitManager(self._default_config())
        pos = MockPosition("AAPL", 115.0, 67, 100.0)
        state = self._default_state(100.0, days_ago=10)
        state["partial_taken"] = True
        state["current_stop"] = 100.0  # breakeven
        features = {"sma_50": 95.0}

        decision = em.check_position(pos, features, state)
        assert decision.action == "HOLD"

    def test_breakeven_stop_ratchet(self):
        """Once price is up 5%, stop should ratchet to at least entry price."""
        em = ExitManager(self._default_config())
        pos = MockPosition("AAPL", 106.0, 100, 100.0)  # 6% gain
        state = self._default_state(100.0)
        state["current_stop"] = 92.0
        state["highest_close"] = 106.0
        features = {"sma_50": 90.0}

        decision = em.check_position(pos, features, state)
        assert decision.action == "HOLD"
        # Stop should be at least entry price (100) due to breakeven
        assert decision.updated_state["current_stop"] >= 100.0

    def test_empty_features(self):
        """Should work with empty features dict."""
        em = ExitManager(self._default_config())
        pos = MockPosition("AAPL", 105.0, 100, 100.0)
        state = self._default_state(100.0)
        features = {}

        decision = em.check_position(pos, features, state)
        assert decision.action in ("HOLD", "PARTIAL_SELL")
