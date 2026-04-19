import pandas as pd

from tradingagents.research.intraday_universe import (
    IntradayUniverseFilterConfig,
    filter_symbols_by_tradability,
)


class _FakeWarehouse:
    def __init__(self, db_path, read_only=True):
        self.db_path = db_path
        self.read_only = read_only

    def get_daily_bars_bulk(self, symbols, start_date=None, end_date=None):
        idx = pd.date_range("2026-01-01", periods=60, freq="D")
        return {
            "GOOD": pd.DataFrame(
                {
                    "close": [50.0] * len(idx),
                    "volume": [2_000_000] * len(idx),
                },
                index=idx,
            ),
            "CHEAP": pd.DataFrame(
                {
                    "close": [5.0] * len(idx),
                    "volume": [5_000_000] * len(idx),
                },
                index=idx,
            ),
            "THIN": pd.DataFrame(
                {
                    "close": [40.0] * len(idx),
                    "volume": [100_000] * len(idx),
                },
                index=idx,
            ),
            "SHORT": pd.DataFrame(
                {
                    "close": [60.0] * 20,
                    "volume": [3_000_000] * 20,
                },
                index=idx[:20],
            ),
        }

    def close(self):
        return None


def test_filter_symbols_by_tradability(monkeypatch):
    monkeypatch.setattr(
        "tradingagents.research.intraday_universe.MarketDataWarehouse",
        _FakeWarehouse,
    )

    filtered, diagnostics = filter_symbols_by_tradability(
        ["GOOD", "CHEAP", "THIN", "SHORT", "MISSING"],
        as_of_date="2026-03-31",
        config=IntradayUniverseFilterConfig(
            lookback_days=60,
            min_median_close=20.0,
            min_median_dollar_volume=50_000_000.0,
            min_trading_days=40,
        ),
    )

    assert filtered == ["GOOD"]
    reasons = diagnostics.set_index("symbol")["reason"].to_dict()
    assert reasons["GOOD"] == "eligible"
    assert reasons["CHEAP"] == "low_price"
    assert reasons["THIN"] == "low_dollar_volume"
    assert reasons["SHORT"] == "insufficient_history"
    assert reasons["MISSING"] == "missing_daily_bars"
