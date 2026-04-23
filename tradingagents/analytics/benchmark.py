"""Strategy vs benchmark comparison utilities.

Pure-function module used by the weekly strategy review. Fetches benchmark
daily bars via Alpaca and computes excess return / correlation against a
caller-provided strategy daily-return series.

No broker writes. Testable without network via the `benchmark_returns`
injection point.
"""

from __future__ import annotations

import math
import statistics
from datetime import date, datetime, timedelta
from typing import Optional, Sequence


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Pearson correlation. Returns None for degenerate inputs."""
    if len(x) != len(y) or len(x) < 2:
        return None
    try:
        mean_x = statistics.fmean(x)
        mean_y = statistics.fmean(y)
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
        if denom_x == 0 or denom_y == 0:
            return None
        return num / (denom_x * denom_y)
    except Exception:
        return None


def _fetch_benchmark_daily_returns(
    ticker: str,
    start: date,
    end: date,
    alpaca_credentials: tuple[str, str],
) -> list[float]:
    """Fetch daily returns for `ticker` between start and end (inclusive).

    Thin wrapper around Alpaca's StockHistoricalDataClient. Returns close-to-
    close daily returns. Factored out so callers can inject mocks.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    api_key, secret_key = alpaca_credentials
    client = StockHistoricalDataClient(api_key, secret_key)
    req = StockBarsRequest(
        symbol_or_symbols=[ticker],
        timeframe=TimeFrame.Day,
        start=datetime.combine(start - timedelta(days=1), datetime.min.time()),
        end=datetime.combine(end + timedelta(days=1), datetime.min.time()),
        # Paper-account credentials aren't authorized for recent SIP data;
        # IEX-only is the documented workaround.
        feed="iex",
    )
    bars = client.get_stock_bars(req)
    series = getattr(bars, "data", {}).get(ticker) or []
    if len(series) < 2:
        return []
    closes = [float(b.close) for b in series]
    returns = [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]
    return returns


def compute_benchmark_comparison(
    ticker: str,
    start: date,
    end: date,
    strategy_daily_returns: Sequence[float],
    *,
    benchmark_returns: Optional[Sequence[float]] = None,
    alpaca_credentials: Optional[tuple[str, str]] = None,
) -> dict:
    """Compare a strategy's daily returns to a benchmark over [start, end].

    Parameters
    ----------
    ticker : str
        Benchmark symbol (e.g. "SPY", "QQQ").
    start, end : date
        Window bounds (inclusive).
    strategy_daily_returns : list of floats
        Strategy's per-trading-day returns in order (decimal, 0.01 == +1 %).
    benchmark_returns : optional list of floats
        Inject a pre-fetched benchmark series. If provided, no network call.
    alpaca_credentials : optional (api_key, secret_key)
        Required when benchmark_returns is None — fetches live from Alpaca.

    Returns
    -------
    dict with:
        benchmark_return   — compounded benchmark return over the window
        strategy_return    — compounded strategy return over the window
        excess_return      — strategy_return - benchmark_return
        correlation        — Pearson over the shared-length daily series
        num_days           — length of the aligned series (min of the two)
        ticker             — echoed back
    """
    if benchmark_returns is None:
        if alpaca_credentials is None:
            raise ValueError(
                "Either benchmark_returns or alpaca_credentials must be provided"
            )
        benchmark_returns = _fetch_benchmark_daily_returns(
            ticker, start, end, alpaca_credentials
        )

    # Trim to the shorter series — guards against weekend / holiday edges
    # where the strategy DB may have a differently-dated snapshot cadence
    # than the benchmark's trading-day calendar.
    n = min(len(strategy_daily_returns), len(benchmark_returns))
    strat = list(strategy_daily_returns[-n:]) if n else []
    bench = list(benchmark_returns[-n:]) if n else []

    def _compound(series: Sequence[float]) -> float:
        prod = 1.0
        for r in series:
            prod *= 1.0 + float(r)
        return prod - 1.0

    strategy_return = _compound(strat) if strat else 0.0
    benchmark_return = _compound(bench) if bench else 0.0

    return {
        "ticker": ticker,
        "num_days": n,
        "strategy_return": round(strategy_return, 6),
        "benchmark_return": round(benchmark_return, 6),
        "excess_return": round(strategy_return - benchmark_return, 6),
        "correlation": (
            round(_pearson(strat, bench), 4)
            if _pearson(strat, bench) is not None
            else None
        ),
    }
