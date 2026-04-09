#!/usr/bin/env python3
"""Compare all strategies: original vs hybrid (pre-screen + weekly-signal).

Runs both single-stock and portfolio-level backtests, then prints a comparison
showing returns AND estimated LLM API savings.

Usage:
    python scripts/run_backtest_comparison.py
    python scripts/run_backtest_comparison.py --symbols AAPL NVDA MSFT --period 2y
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.backtesting.engine import BacktestEngine
from tradingagents.backtesting.strategies import (
    SwingStrategy, DayTradingStrategy, CombinedStrategy,
)
from tradingagents.backtesting.hybrid_strategies import (
    PreScreenStrategy, WeeklySignalDailyExecStrategy,
)
from tradingagents.backtesting.portfolio_backtest import (
    PortfolioBacktestEngine,
    PortfolioSwingStrategy,
    PortfolioCombinedStrategy,
)
from tradingagents.backtesting.hybrid_strategies import (
    PortfolioPreScreenStrategy,
    PortfolioWeeklySignalStrategy,
)

from datetime import datetime, timedelta


def parse_args():
    p = argparse.ArgumentParser(description="Backtest comparison: original vs hybrid strategies")
    p.add_argument(
        "--symbols", nargs="+",
        default=["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA"],
        help="Symbols to test",
    )
    p.add_argument("--period", default="1y", help="Lookback period: 1y, 2y, 6m")
    p.add_argument("--cash", type=float, default=100_000, help="Initial cash")
    p.add_argument("--single-symbol", default="NVDA", help="Symbol for single-stock tests")
    return p.parse_args()


def period_to_dates(period: str):
    end = datetime.now()
    if period.endswith("y"):
        days = int(period[:-1]) * 365
    elif period.endswith("m"):
        days = int(period[:-1]) * 30
    else:
        days = 365
    start = end - timedelta(days=days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def run_single_stock_comparison(symbol: str, start: str, end: str, cash: float):
    """Compare all strategies on a single stock."""
    engine = BacktestEngine(initial_cash=cash)

    strategies = [
        ("Swing (original)", SwingStrategy, {}),
        ("DayTrading (original)", DayTradingStrategy, {}),
        ("Combined (original)", CombinedStrategy, {}),
        ("PreScreen (hybrid)", PreScreenStrategy, {}),
        ("WeeklySignal (hybrid)", WeeklySignalDailyExecStrategy, {}),
        # Also test weekly signal with different intervals
        ("WeeklySignal-3d", WeeklySignalDailyExecStrategy, {"signal_interval": 3}),
        ("WeeklySignal-10d", WeeklySignalDailyExecStrategy, {"signal_interval": 10}),
    ]

    print(f"\n{'#' * 80}")
    print(f"  SINGLE STOCK COMPARISON: {symbol} ({start} to {end})")
    print(f"{'#' * 80}")

    results = []
    for name, strat_class, params in strategies:
        try:
            r = engine.run_single(symbol, strat_class, start, end, params)
            r["label"] = name
            results.append(r)
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            continue

    # Print comparison table
    print(f"\n  {'Strategy':<28} {'Return':>10} {'B&H':>10} {'Beat?':>6} "
          f"{'WinRate':>8} {'MaxDD':>8} {'Trades':>7} {'Sharpe':>8} {'PF':>8}")
    print(f"  {'─' * 95}")

    for r in results:
        if "error" in r:
            continue
        beat = "YES" if r["beats_buy_hold"] else "no"
        print(
            f"  {r['label']:<28} {r['total_return_pct']:>10} {r['buy_hold_return_pct']:>10} "
            f"{beat:>6} {r['win_rate_pct']:>8} {r['max_drawdown_pct']:>8} "
            f"{r['total_trades']:>7} {str(r['sharpe_ratio']):>8} {str(r['profit_factor']):>8}"
        )

    # API savings analysis
    print(f"\n  --- API Usage Estimate ---")
    total_days = None
    for r in results:
        if "error" in r:
            continue
        # Estimate trading days from period
        if total_days is None:
            # rough: 252 trading days/year
            from datetime import datetime as dt
            d1 = dt.strptime(start, "%Y-%m-%d")
            d2 = dt.strptime(end, "%Y-%m-%d")
            total_days = int((d2 - d1).days * 252 / 365)

    if total_days:
        print(f"  Trading days in period: ~{total_days}")
        print(f"  Original approach: {total_days} LLM calls (1 per day per stock)")
        print(f"  PreScreen approach: ~{int(total_days * 0.25)} LLM calls (skip ~75% of days)")
        print(f"  WeeklySignal-5d: ~{total_days // 5} LLM calls ({100 - (total_days // 5) / total_days * 100:.0f}% saved)")
        print(f"  WeeklySignal-10d: ~{total_days // 10} LLM calls ({100 - (total_days // 10) / total_days * 100:.0f}% saved)")

    return results


def run_portfolio_comparison(symbols: list, start: str, end: str, cash: float):
    """Compare portfolio-level strategies."""
    engine = PortfolioBacktestEngine(initial_cash=cash)

    strategies = [
        ("PortfolioSwing (orig)", PortfolioSwingStrategy, {}),
        ("PortfolioCombined (orig)", PortfolioCombinedStrategy, {}),
        ("PortfolioPreScreen", PortfolioPreScreenStrategy, {}),
        ("PortfolioWeeklySignal", PortfolioWeeklySignalStrategy, {}),
        ("PortfolioWeeklySignal-10d", PortfolioWeeklySignalStrategy, {"signal_interval": 10}),
    ]

    print(f"\n\n{'#' * 80}")
    print(f"  PORTFOLIO COMPARISON: {', '.join(symbols)} ({start} to {end})")
    print(f"{'#' * 80}")

    results = []
    for name, strat_class, params in strategies:
        try:
            r = engine.run(symbols, strat_class, start, end, params)
            r["label"] = name
            results.append(r)
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            results.append({"label": name, "error": str(e)})
            continue

    # Print comparison table
    print(f"\n  {'Strategy':<30} {'Return':>10} {'SPY':>10} {'Beat?':>6} "
          f"{'WinRate':>8} {'MaxDD':>8} {'Trades':>7} {'Sharpe':>8}")
    print(f"  {'─' * 90}")

    for r in results:
        if "error" in r:
            print(f"  {r['label']:<30} ERROR: {r['error']}")
            continue
        beat = "YES" if r["beats_spy"] else "no"
        print(
            f"  {r['label']:<30} {r['total_return_pct']:>10} {r['spy_return_pct']:>10} "
            f"{beat:>6} {r['win_rate_pct']:>8} {r['max_drawdown_pct']:>8} "
            f"{r['total_trades']:>7} {str(r['sharpe_ratio']):>8}"
        )

    # API cost comparison
    n_stocks = len(symbols)
    from datetime import datetime as dt
    d1 = dt.strptime(start, "%Y-%m-%d")
    d2 = dt.strptime(end, "%Y-%m-%d")
    total_days = int((d2 - d1).days * 252 / 365)

    print(f"\n  --- Portfolio API Usage Estimate ({n_stocks} stocks) ---")
    daily_calls = total_days * n_stocks * 13  # ~13 LLM calls per stock
    print(f"  Original (daily LLM, all stocks):  ~{daily_calls:,} LLM calls")
    print(f"  PreScreen (skip ~75%):             ~{int(daily_calls * 0.25):,} LLM calls")
    weekly_calls = (total_days // 5) * n_stocks * 13
    print(f"  WeeklySignal-5d:                   ~{weekly_calls:,} LLM calls ({100 - weekly_calls / daily_calls * 100:.0f}% saved)")
    biweekly_calls = (total_days // 10) * n_stocks * 13
    print(f"  WeeklySignal-10d:                  ~{biweekly_calls:,} LLM calls ({100 - biweekly_calls / daily_calls * 100:.0f}% saved)")

    # Cost estimate (gpt-4o-mini pricing: ~$0.15/1K input + $0.60/1K output tokens)
    # Average ~2000 tokens per call
    cost_per_call = 0.002  # rough estimate for gpt-4o-mini
    print(f"\n  Estimated monthly cost (gpt-4o-mini, ~$0.002/call):")
    monthly_days = 21
    monthly_daily = monthly_days * n_stocks * 13 * cost_per_call
    monthly_prescreen = monthly_daily * 0.25
    monthly_weekly = (monthly_days // 5) * n_stocks * 13 * cost_per_call
    monthly_biweekly = (monthly_days // 10) * n_stocks * 13 * cost_per_call
    print(f"    Original:          ${monthly_daily:>8.2f}/month")
    print(f"    PreScreen:         ${monthly_prescreen:>8.2f}/month")
    print(f"    WeeklySignal-5d:   ${monthly_weekly:>8.2f}/month")
    print(f"    WeeklySignal-10d:  ${monthly_biweekly:>8.2f}/month")

    return results


def main():
    args = parse_args()
    start, end = period_to_dates(args.period)

    print(f"\nBacktest Configuration:")
    print(f"  Period: {start} to {end}")
    print(f"  Initial cash: ${args.cash:,.0f}")
    print(f"  Single stock: {args.single_symbol}")
    print(f"  Portfolio: {', '.join(args.symbols)}")

    # 1. Single stock comparison
    single_results = run_single_stock_comparison(
        args.single_symbol, start, end, args.cash
    )

    # 2. Portfolio comparison
    portfolio_results = run_portfolio_comparison(
        args.symbols, start, end, args.cash
    )

    # Final summary
    print(f"\n\n{'=' * 80}")
    print(f"  CONCLUSION")
    print(f"{'=' * 80}")
    print(f"""
  The hybrid strategies (PreScreen, WeeklySignal) aim to match or exceed
  the performance of daily-analysis strategies while using 75-90% fewer
  LLM API calls.

  Key trade-off:
  - PreScreen: Same frequency, but skips "no signal" days → ~75% fewer calls
  - WeeklySignal-5d: Analyzes weekly, executes daily on rules → ~80% fewer calls
  - WeeklySignal-10d: Analyzes bi-weekly → ~90% fewer calls, but may miss signals

  If hybrid returns are within 1-2% of original strategies, the API savings
  make them clearly superior for production use.
""")


if __name__ == "__main__":
    main()
