"""Tests for volatility-aware position sizing (Phase 5)."""

import pytest
from tradingagents.portfolio.position_sizer import PositionSizer
from tradingagents.broker.models import Account


def make_account(equity=100000, cash=100000, buying_power=200000):
    return Account(
        account_id="test", equity=equity, cash=cash,
        buying_power=buying_power, portfolio_value=equity,
        status="ACTIVE",
    )


class TestPositionSizerATR:
    def test_atr_based_sizing(self):
        """With ATR but no stop-loss, should use 2x ATR for risk-per-share."""
        sizer = PositionSizer({
            "max_position_pct": 0.10,
            "max_total_exposure": 0.80,
            "risk_per_trade": 0.02,
        })
        signal = {"action": "BUY", "symbol": "NVDA", "confidence": 0.7}
        account = make_account()
        # ATR = $5, so risk_per_share = $10 (2x ATR)
        # max_risk = 100000 * 0.02 = $2000
        # qty_by_risk = 2000 / 10 = 200
        # qty_by_capital = min(10000, 80000, 200000) / 175 = ~57
        order = sizer.calculate(signal, account, current_price=175.0, atr=5.0)
        assert order is not None
        assert order.qty == 57  # capital-limited

    def test_stop_loss_takes_priority_over_atr(self):
        """When both stop_loss and ATR are available, stop_loss wins."""
        sizer = PositionSizer({
            "max_position_pct": 0.10,
            "max_total_exposure": 0.80,
            "risk_per_trade": 0.02,
        })
        signal = {"action": "BUY", "symbol": "NVDA", "confidence": 0.7,
                  "stop_loss": 165.0}  # $10 risk per share
        account = make_account()
        # risk_per_share from stop_loss = 175 - 165 = $10
        # max_risk = $2000, qty_by_risk = 200
        order = sizer.calculate(signal, account, current_price=175.0, atr=3.0)
        assert order is not None
        # Should use stop_loss, not ATR
        # qty_by_risk = 2000/10 = 200, qty_by_capital = 10000/175 = 57
        assert order.qty == 57

    def test_no_atr_no_stop_uses_confidence(self):
        """Without ATR or stop-loss, falls back to confidence-based scaling."""
        sizer = PositionSizer({
            "max_position_pct": 0.10,
            "max_total_exposure": 0.80,
            "risk_per_trade": 0.02,
        })
        signal = {"action": "BUY", "symbol": "NVDA", "confidence": 0.8}
        account = make_account()
        order = sizer.calculate(signal, account, current_price=175.0)
        assert order is not None
        # scale = 0.5 + 0.8*0.5 = 0.9
        # available = min(10000, 80000, 200000) = 10000
        # qty = floor(10000 * 0.9 / 175) = floor(51.4) = 51
        assert order.qty == 51

    def test_high_atr_reduces_position(self):
        """High ATR (volatile stock) should result in smaller position."""
        sizer = PositionSizer({
            "max_position_pct": 0.10,
            "max_total_exposure": 0.80,
            "risk_per_trade": 0.02,
        })
        signal = {"action": "BUY", "symbol": "NVDA", "confidence": 0.7}
        account = make_account()
        # ATR = $20, risk_per_share = $40 (2x ATR)
        # max_risk = $2000, qty_by_risk = 50
        # qty_by_capital = 10000/175 = 57
        order = sizer.calculate(signal, account, current_price=175.0, atr=20.0)
        assert order is not None
        assert order.qty == 50  # risk-limited

    def test_sell_ignores_atr(self):
        """SELL should work regardless of ATR."""
        from tradingagents.broker.models import Position
        sizer = PositionSizer({
            "max_position_pct": 0.10,
            "max_total_exposure": 0.80,
            "risk_per_trade": 0.02,
        })
        signal = {"action": "SELL", "symbol": "NVDA"}
        account = make_account()
        pos = Position(
            symbol="NVDA", qty=50, side="long", avg_entry_price=150.0,
            current_price=175.0, market_value=8750.0,
            unrealized_pl=1250.0, unrealized_plpc=0.1667,
        )
        order = sizer.calculate(signal, account, current_price=175.0,
                                current_position=pos, atr=5.0)
        assert order is not None
        assert order.side == "sell"
        assert order.qty == 50


class TestSignalProcessorWithScreener:
    def test_screener_overrides(self):
        """Screener data should override LLM-guessed values."""
        from tradingagents.graph.signal_processing import SignalProcessor

        class MockLLM:
            def invoke(self, messages):
                class Resp:
                    content = '{"action": "BUY", "confidence": 0.6, "reasoning": "test", "stop_loss_pct": 0.05, "take_profit_pct": 0.10, "timeframe": "swing"}'
                return Resp()

        proc = SignalProcessor(MockLLM())
        screener_data = {
            "current_price": 100.0,
            "initial_stop_price": 92.0,  # 8% stop
            "template_score": 8,
            "rs_percentile": 85,
        }
        result = proc.process_signal_with_screener("test report", "NVDA", screener_data)

        assert result["action"] == "BUY"
        assert result["stop_loss_pct"] == pytest.approx(0.08, abs=0.001)
        assert result["stop_loss"] == 92.0
        # TP should be at least 3x SL = 24%
        assert result["take_profit_pct"] >= 0.24
        # Confidence from screener: 0.55 + 8/25 + 85/250 = 0.55 + 0.32 + 0.34 = 1.21 -> capped at 0.95
        assert result["confidence"] == pytest.approx(0.95, abs=0.01)

    def test_screener_empty_data(self):
        """With empty screener data, should return base signal unchanged."""
        from tradingagents.graph.signal_processing import SignalProcessor

        class MockLLM:
            def invoke(self, messages):
                class Resp:
                    content = '{"action": "HOLD", "confidence": 0.5, "reasoning": "test", "stop_loss_pct": 0.05, "take_profit_pct": 0.15, "timeframe": "swing"}'
                return Resp()

        proc = SignalProcessor(MockLLM())
        result = proc.process_signal_with_screener("test report", "NVDA", {})

        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.5
