"""Tests for the shared `compute_excursion` helper in automation/trade_outcome.

The compute is kill-switched (via `enabled` param) and wraps all Alpaca
calls in try/except. These tests verify:
  1. Kill switch OFF → returns (None, None) without touching the broker.
  2. Simple win case → MFE > 0, MAE close to 0.
  3. Simple loss case → MAE < 0, MFE small positive.
  4. Missing data_client → (None, None), no crash.
  5. Empty bar series → (None, None).
  6. Client raises → (None, None), warning logged, no bubble-up.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from tradingagents.automation.trade_outcome import compute_excursion


def _bar(high, low):
    return SimpleNamespace(high=high, low=low)


def _data_client_with_bars(symbol: str, bars: list):
    client = MagicMock()
    client.get_stock_bars.return_value = SimpleNamespace(data={symbol: bars})
    return client


class TestKillSwitch:
    def test_disabled_returns_none(self):
        client = MagicMock()
        mfe, mae = compute_excursion(
            data_client=client, symbol="AAPL",
            entry_date_str="2026-04-01", exit_date_str="2026-04-05",
            entry_price=100.0, enabled=False,
        )
        assert mfe is None and mae is None
        client.get_stock_bars.assert_not_called()


class TestHappyPath:
    def test_winner_has_positive_mfe(self):
        bars = [_bar(105, 99), _bar(110, 101), _bar(115, 105), _bar(112, 107)]
        client = _data_client_with_bars("NVDA", bars)
        mfe, mae = compute_excursion(
            data_client=client, symbol="NVDA",
            entry_date_str="2026-04-01", exit_date_str="2026-04-05",
            entry_price=100.0,
        )
        assert mfe == 0.15   # (115 - 100) / 100
        assert mae == -0.01  # (99 - 100) / 100

    def test_loser_has_negative_mae(self):
        bars = [_bar(101, 95), _bar(100, 90), _bar(98, 88)]
        client = _data_client_with_bars("XYZ", bars)
        mfe, mae = compute_excursion(
            data_client=client, symbol="XYZ",
            entry_date_str="2026-04-01", exit_date_str="2026-04-03",
            entry_price=100.0,
        )
        assert mfe == 0.01
        assert mae == -0.12

    def test_rounding_to_four_decimals(self):
        bars = [_bar(100.12345, 99.87654)]
        client = _data_client_with_bars("X", bars)
        mfe, mae = compute_excursion(
            data_client=client, symbol="X",
            entry_date_str="2026-04-01", exit_date_str="2026-04-02",
            entry_price=100.0,
        )
        assert mfe == 0.0012
        assert mae == -0.0012


class TestFailsSoft:
    def test_missing_data_client(self):
        mfe, mae = compute_excursion(
            data_client=None, symbol="AAPL",
            entry_date_str="2026-04-01", exit_date_str="2026-04-05",
            entry_price=100.0,
        )
        assert mfe is None and mae is None

    def test_empty_series(self):
        client = _data_client_with_bars("AAPL", [])
        mfe, mae = compute_excursion(
            data_client=client, symbol="AAPL",
            entry_date_str="2026-04-01", exit_date_str="2026-04-05",
            entry_price=100.0,
        )
        assert mfe is None and mae is None

    def test_client_raises(self):
        client = MagicMock()
        client.get_stock_bars.side_effect = RuntimeError("api down")
        mfe, mae = compute_excursion(
            data_client=client, symbol="AAPL",
            entry_date_str="2026-04-01", exit_date_str="2026-04-05",
            entry_price=100.0,
        )
        assert mfe is None and mae is None

    def test_missing_entry_date(self):
        client = _data_client_with_bars("AAPL", [_bar(110, 95)])
        mfe, mae = compute_excursion(
            data_client=client, symbol="AAPL",
            entry_date_str="", exit_date_str="2026-04-05",
            entry_price=100.0,
        )
        assert mfe is None and mae is None

    def test_zero_entry_price(self):
        client = _data_client_with_bars("AAPL", [_bar(110, 95)])
        mfe, mae = compute_excursion(
            data_client=client, symbol="AAPL",
            entry_date_str="2026-04-01", exit_date_str="2026-04-05",
            entry_price=0.0,
        )
        assert mfe is None and mae is None
