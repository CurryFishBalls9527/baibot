"""Research-only intraday universe filtering helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .warehouse import MarketDataWarehouse


@dataclass
class IntradayUniverseFilterConfig:
    daily_db_path: str = "research_data/market_data.duckdb"
    lookback_days: int = 60
    min_median_close: float = 20.0
    min_median_dollar_volume: float = 50_000_000.0
    min_trading_days: int = 40


def filter_symbols_by_tradability(
    symbols: list[str],
    as_of_date: str,
    config: IntradayUniverseFilterConfig | None = None,
) -> tuple[list[str], pd.DataFrame]:
    cfg = config or IntradayUniverseFilterConfig()
    warehouse = MarketDataWarehouse(cfg.daily_db_path, read_only=True)
    try:
        bars_map = warehouse.get_daily_bars_bulk(symbols, end_date=as_of_date)
    finally:
        warehouse.close()

    rows: list[dict] = []
    for symbol in symbols:
        df = bars_map.get(symbol)
        if df is None or df.empty:
            rows.append(
                {
                    "symbol": symbol,
                    "eligible": False,
                    "reason": "missing_daily_bars",
                    "trading_days": 0,
                    "median_close": None,
                    "median_dollar_volume": None,
                }
            )
            continue

        tail = df.tail(cfg.lookback_days).copy()
        trading_days = int(len(tail))
        median_close = float(tail["close"].median()) if "close" in tail and not tail.empty else None
        median_dollar_volume = (
            float((tail["close"] * tail["volume"]).median())
            if {"close", "volume"}.issubset(tail.columns) and not tail.empty
            else None
        )

        reason = "eligible"
        eligible = True
        if trading_days < cfg.min_trading_days:
            eligible = False
            reason = "insufficient_history"
        elif median_close is None or median_close < cfg.min_median_close:
            eligible = False
            reason = "low_price"
        elif median_dollar_volume is None or median_dollar_volume < cfg.min_median_dollar_volume:
            eligible = False
            reason = "low_dollar_volume"

        rows.append(
            {
                "symbol": symbol,
                "eligible": eligible,
                "reason": reason,
                "trading_days": trading_days,
                "median_close": round(median_close, 4) if median_close is not None else None,
                "median_dollar_volume": round(median_dollar_volume, 2)
                if median_dollar_volume is not None
                else None,
            }
        )

    diagnostics = pd.DataFrame(rows)
    filtered = diagnostics.loc[diagnostics["eligible"], "symbol"].tolist()
    return filtered, diagnostics
