"""Tests for tradingagents.analytics.benchmark — compute_benchmark_comparison."""
from __future__ import annotations

from datetime import date

import pytest

from tradingagents.analytics.benchmark import (
    _pearson,
    compute_benchmark_comparison,
)


class TestPearson:
    def test_perfect_positive(self):
        assert _pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert _pearson([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)

    def test_zero_correlation(self):
        # Constant y → denominator 0 → None.
        assert _pearson([1, 2, 3], [5, 5, 5]) is None

    def test_length_mismatch_returns_none(self):
        assert _pearson([1, 2], [1, 2, 3]) is None

    def test_single_element_returns_none(self):
        assert _pearson([1.0], [1.0]) is None


class TestCompute:
    def test_injected_returns_no_network(self):
        result = compute_benchmark_comparison(
            ticker="SPY",
            start=date(2026, 4, 1),
            end=date(2026, 4, 7),
            strategy_daily_returns=[0.01, 0.02, -0.01, 0.005, 0.0],
            benchmark_returns=[0.005, 0.01, -0.005, 0.0, 0.002],
        )
        assert result["ticker"] == "SPY"
        assert result["num_days"] == 5
        # Compounded: strat ≈ 1.01 * 1.02 * 0.99 * 1.005 * 1.0 - 1
        assert result["strategy_return"] == pytest.approx(0.0251, abs=1e-3)
        assert result["benchmark_return"] == pytest.approx(0.01201, abs=1e-3)
        # Excess = strat - bench, allowing for per-field rounding drift (1e-5).
        assert result["excess_return"] == pytest.approx(
            result["strategy_return"] - result["benchmark_return"], abs=1e-5
        )
        assert result["correlation"] is not None
        assert 0.9 < result["correlation"] <= 1.0  # strongly correlated

    def test_empty_series(self):
        result = compute_benchmark_comparison(
            ticker="SPY",
            start=date(2026, 4, 1),
            end=date(2026, 4, 7),
            strategy_daily_returns=[],
            benchmark_returns=[],
        )
        assert result["num_days"] == 0
        assert result["strategy_return"] == 0.0
        assert result["benchmark_return"] == 0.0
        assert result["excess_return"] == 0.0
        assert result["correlation"] is None

    def test_trims_to_shorter(self):
        """Strategy has one more day than benchmark — trim to benchmark length."""
        result = compute_benchmark_comparison(
            ticker="SPY",
            start=date(2026, 4, 1),
            end=date(2026, 4, 10),
            strategy_daily_returns=[0.01, 0.01, 0.01, 0.01],  # 4 days
            benchmark_returns=[0.005, 0.005, 0.005],           # 3 days
        )
        assert result["num_days"] == 3
        # Only last 3 strat returns contribute.
        assert result["strategy_return"] == pytest.approx(
            (1.01 * 1.01 * 1.01) - 1, abs=1e-4
        )

    def test_negative_strategy_flat_benchmark(self):
        result = compute_benchmark_comparison(
            ticker="QQQ",
            start=date(2026, 4, 1),
            end=date(2026, 4, 7),
            strategy_daily_returns=[-0.01, -0.005, -0.02],
            benchmark_returns=[0.0, 0.0, 0.0],
        )
        assert result["strategy_return"] < 0
        assert result["benchmark_return"] == 0.0
        assert result["excess_return"] < 0

    def test_raises_without_credentials_or_returns(self):
        with pytest.raises(ValueError):
            compute_benchmark_comparison(
                ticker="SPY",
                start=date(2026, 4, 1),
                end=date(2026, 4, 7),
                strategy_daily_returns=[0.01, 0.01],
                # No benchmark_returns, no alpaca_credentials
            )
