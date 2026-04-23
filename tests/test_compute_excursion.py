"""Tests for Orchestrator._compute_excursion.

The compute is kill-switched (default OFF) and wraps all Alpaca calls in
try/except. These tests verify:
  1. Kill switch OFF → returns (None, None) without touching the broker.
  2. Simple win case → MFE > 0, MAE close to 0.
  3. Simple loss case → MAE < 0, MFE small positive.
  4. Missing broker.data_client → (None, None), no crash.
  5. Empty bar series → (None, None).
  6. Broker raises → (None, None), warning logged, no bubble-up.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from tradingagents.automation.orchestrator import Orchestrator


def _make_orch(config, broker):
    o = Orchestrator.__new__(Orchestrator)
    o.config = config
    o.broker = broker
    return o


def _bar(high, low):
    return SimpleNamespace(high=high, low=low)


def _stub_broker_with_bars(symbol: str, bars: list):
    """Build a MagicMock broker whose data_client returns these bars."""
    broker = MagicMock()
    broker.data_client.get_stock_bars.return_value = SimpleNamespace(
        data={symbol: bars}
    )
    return broker


class TestKillSwitch:
    def test_disabled_returns_none(self):
        broker = MagicMock()
        orch = _make_orch({"trade_outcome_excursion_enabled": False}, broker)
        mfe, mae = orch._compute_excursion("AAPL", "2026-04-01", "2026-04-05", 100.0)
        assert mfe is None and mae is None
        # Crucially: broker was never touched.
        broker.data_client.get_stock_bars.assert_not_called()

    def test_disabled_is_default(self):
        """No flag in config → defaults OFF."""
        broker = MagicMock()
        orch = _make_orch({}, broker)
        mfe, mae = orch._compute_excursion("AAPL", "2026-04-01", "2026-04-05", 100.0)
        assert mfe is None and mae is None
        broker.data_client.get_stock_bars.assert_not_called()


class TestHappyPath:
    def test_winner_has_positive_mfe(self):
        bars = [_bar(105, 99), _bar(110, 101), _bar(115, 105), _bar(112, 107)]
        broker = _stub_broker_with_bars("NVDA", bars)
        orch = _make_orch({"trade_outcome_excursion_enabled": True}, broker)
        mfe, mae = orch._compute_excursion("NVDA", "2026-04-01", "2026-04-05", 100.0)
        assert mfe == 0.15  # (115 - 100) / 100
        assert mae == -0.01  # (99 - 100) / 100

    def test_loser_has_negative_mae(self):
        bars = [_bar(101, 95), _bar(100, 90), _bar(98, 88)]
        broker = _stub_broker_with_bars("XYZ", bars)
        orch = _make_orch({"trade_outcome_excursion_enabled": True}, broker)
        mfe, mae = orch._compute_excursion("XYZ", "2026-04-01", "2026-04-03", 100.0)
        assert mfe == 0.01
        assert mae == -0.12

    def test_rounding_to_four_decimals(self):
        bars = [_bar(100.12345, 99.87654)]
        broker = _stub_broker_with_bars("X", bars)
        orch = _make_orch({"trade_outcome_excursion_enabled": True}, broker)
        mfe, mae = orch._compute_excursion("X", "2026-04-01", "2026-04-02", 100.0)
        # (100.12345 - 100) / 100 == 0.0012345 → 0.0012
        assert mfe == 0.0012
        assert mae == -0.0012


class TestFailsSoft:
    def test_missing_data_client(self):
        broker = SimpleNamespace()  # no .data_client attribute
        orch = _make_orch({"trade_outcome_excursion_enabled": True}, broker)
        mfe, mae = orch._compute_excursion("AAPL", "2026-04-01", "2026-04-05", 100.0)
        assert mfe is None and mae is None

    def test_empty_series(self):
        broker = _stub_broker_with_bars("AAPL", [])
        orch = _make_orch({"trade_outcome_excursion_enabled": True}, broker)
        mfe, mae = orch._compute_excursion("AAPL", "2026-04-01", "2026-04-05", 100.0)
        assert mfe is None and mae is None

    def test_broker_raises(self):
        broker = MagicMock()
        broker.data_client.get_stock_bars.side_effect = RuntimeError("api down")
        orch = _make_orch({"trade_outcome_excursion_enabled": True}, broker)
        mfe, mae = orch._compute_excursion("AAPL", "2026-04-01", "2026-04-05", 100.0)
        assert mfe is None and mae is None

    def test_missing_entry_date(self):
        broker = _stub_broker_with_bars("AAPL", [_bar(110, 95)])
        orch = _make_orch({"trade_outcome_excursion_enabled": True}, broker)
        mfe, mae = orch._compute_excursion("AAPL", "", "2026-04-05", 100.0)
        assert mfe is None and mae is None

    def test_zero_entry_price(self):
        broker = _stub_broker_with_bars("AAPL", [_bar(110, 95)])
        orch = _make_orch({"trade_outcome_excursion_enabled": True}, broker)
        mfe, mae = orch._compute_excursion("AAPL", "2026-04-01", "2026-04-05", 0.0)
        assert mfe is None and mae is None
