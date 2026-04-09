#!/usr/bin/env python3
"""Run backtests against historical data.

Usage:
    # Quick test: one stock, one strategy
    python run_backtest.py --symbols NVDA

    # Compare all strategies on multiple stocks
    python run_backtest.py --symbols AAPL,NVDA,MSFT,GOOGL

    # Screen large-cap stocks first, then backtest top 10
    python run_backtest.py --screen

    # Custom date range (default: last 1 year)
    python run_backtest.py --symbols NVDA --start 2024-01-01 --end 2025-12-31

    # Two years of data
    python run_backtest.py --symbols AAPL,NVDA --period 2y

    # Only run one strategy
    python run_backtest.py --symbols NVDA --strategy swing
"""

import argparse
import sys
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.WARNING)

from tradingagents.backtesting.engine import BacktestEngine
from tradingagents.backtesting.strategies import (
    SwingStrategy,
    DayTradingStrategy,
    CombinedStrategy,
)
from tradingagents.backtesting.screener import LargeCapScreener

STRATEGY_MAP = {
    "swing": ("Swing", SwingStrategy),
    "day": ("DayTrading", DayTradingStrategy),
    "combined": ("Combined", CombinedStrategy),
}


def main():
    parser = argparse.ArgumentParser(description="Backtest trading strategies")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--screen", action="store_true",
                        help="Screen large-cap stocks (>$30B) then backtest top 10")
    parser.add_argument("--screen-count", type=int, default=10,
                        help="Number of stocks to screen (default 10)")
    parser.add_argument("--min-cap", type=float, default=30,
                        help="Min market cap in billions (default 30)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--period", type=str, default="1y",
                        help="Lookback period: 6m, 1y, 2y, 3y (default 1y)")
    parser.add_argument("--cash", type=float, default=100_000,
                        help="Starting capital (default 100000)")
    parser.add_argument("--strategy", type=str, choices=["swing", "day", "combined"],
                        help="Run only one strategy")

    args = parser.parse_args()

    # Determine date range
    if args.end:
        end_date = args.end
    else:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if args.start:
        start_date = args.start
    else:
        period_map = {"6m": 180, "1y": 365, "2y": 730, "3y": 1095}
        days = period_map.get(args.period, 365)
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Determine symbols
    if args.screen:
        print(f"\nScreening stocks with market cap >= ${args.min_cap}B...")
        screener = LargeCapScreener(min_market_cap_b=args.min_cap)
        stocks = screener.screen()

        print(f"\nFound {len(stocks)} qualifying stocks:")
        print(f"{'Symbol':>8} {'Name':<30} {'Market Cap':>12} {'Sector':<20}")
        print("-" * 75)
        for s in stocks[:args.screen_count + 5]:
            print(f"{s['symbol']:>8} {s['name']:<30} ${s['market_cap_b']:>9.1f}B {s['sector']:<20}")

        symbols = [s["symbol"] for s in stocks[:args.screen_count]]
        print(f"\nBacktesting top {len(symbols)}: {symbols}\n")

    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN"]
        print(f"No symbols specified, using default: {symbols}")

    # Run backtests
    engine = BacktestEngine(initial_cash=args.cash)

    if args.strategy:
        # Single strategy mode
        name, strat_class = STRATEGY_MAP[args.strategy]
        print(f"\nRunning {name} strategy on {symbols}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Starting capital: ${args.cash:,.0f}")

        for symbol in symbols:
            try:
                result = engine.run_single(symbol, strat_class, start_date, end_date)
                engine.print_results(result)
            except Exception as e:
                print(f"\n  {symbol}: ERROR - {e}")
    else:
        # Comparison mode: all 3 strategies
        print(f"\nComparing ALL strategies on {symbols}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Starting capital: ${args.cash:,.0f}\n")

        comparison = engine.run_comparison(symbols, start_date, end_date)

        # Print each result
        for r in comparison["results"]:
            if "error" not in r:
                engine.print_results(r)

        # Print summary
        engine.print_comparison(comparison)

    print()


if __name__ == "__main__":
    main()
