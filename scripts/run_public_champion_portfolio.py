#!/usr/bin/env python3
"""Run a public Minervini-style portfolio backtest on a curated growth universe."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tradingagents.research import (
    BacktestConfig,
    MarketDataWarehouse,
    MinerviniConfig,
    MinerviniScreener,
    PortfolioMinerviniBacktester,
    resolve_universe,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Portfolio-level public-champion backtest")
    parser.add_argument("--db", default=str(ROOT / "research_data" / "portfolio_eval.duckdb"))
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "minervini"))
    parser.add_argument("--universe", default="combined", help="large | growth | combined")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--start", help="Warmup start date YYYY-MM-DD")
    parser.add_argument("--trade-start", help="Trade evaluation start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-03-23")
    parser.add_argument("--period", default="2y", help="Warmup fetch period if --start omitted")
    parser.add_argument("--refresh-data", action="store_true")
    return parser.parse_args()


def resolve_dates(args):
    end_date = args.end
    if args.start:
        start_date = args.start
    else:
        days = {"1y": 365, "2y": 730, "3y": 1095}
        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days.get(args.period, 730))
        ).strftime("%Y-%m-%d")
    if args.trade_start:
        trade_start = args.trade_start
    else:
        trade_start = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)
        ).strftime("%Y-%m-%d")
    return start_date, trade_start, end_date


def save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    args = parse_args()
    start_date, trade_start, end_date = resolve_dates(args)
    symbols = resolve_universe(args.universe)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = end_date

    warehouse = MarketDataWarehouse(args.db)
    try:
        required = sorted(set(symbols + [args.benchmark]))
        available = set(warehouse.available_symbols())
        if args.refresh_data or not set(required).issubset(available):
            fetch_counts = warehouse.fetch_and_store_daily_bars(required, start_date, end_date)
        else:
            fetch_counts = {symbol: -1 for symbol in required}

        data_by_symbol = {
            symbol: warehouse.get_daily_bars(symbol, start_date, end_date)
            for symbol in symbols
        }
        data_by_symbol = {symbol: df for symbol, df in data_by_symbol.items() if not df.empty}
        benchmark_df = warehouse.get_daily_bars(args.benchmark, start_date, end_date)

        screener = MinerviniScreener(
            MinerviniConfig(
                require_fundamentals=False,
                require_market_uptrend=False,
                max_stage_number=3,
                max_buy_zone_pct=0.07,
                pivot_buffer_pct=0.0,
            )
        )
        backtester = PortfolioMinerviniBacktester(
            screener=screener,
            config=BacktestConfig(
                max_position_pct=0.12,
                risk_per_trade=0.012,
                stop_loss_pct=0.08,
                trail_stop_pct=0.12,
                max_hold_days=90,
                min_template_score=7,
                require_volume_surge=False,
                require_market_regime=False,
                progressive_entries=True,
                initial_entry_fraction=0.50,
                add_on_trigger_pct_1=0.025,
                add_on_trigger_pct_2=0.05,
                add_on_fraction_1=0.30,
                add_on_fraction_2=0.20,
                breakeven_trigger_pct=0.05,
                partial_profit_trigger_pct=0.12,
                partial_profit_fraction=0.33,
                use_ema21_exit=True,
                use_close_range_filter=True,
                min_close_range_pct=0.55,
                scale_exposure_in_weak_market=True,
                weak_market_position_scale=0.60,
                target_exposure_confirmed_uptrend=1.00,
                target_exposure_uptrend_under_pressure=0.60,
                target_exposure_market_correction=0.00,
                allow_new_entries_in_correction=False,
                max_positions=6,
            ),
        )

        result = backtester.backtest_portfolio(
            data_by_symbol,
            benchmark_df=benchmark_df,
            trade_start_date=trade_start,
        )

        prefix = f"portfolio_public_champion_{args.universe}_{stamp}"
        save(pd.DataFrame([result.summary]), results_dir / f"{prefix}_metrics.csv")
        save(result.trades, results_dir / f"{prefix}_trades.csv")
        save(result.equity_curve, results_dir / f"{prefix}_equity_curve.csv")
        save(result.daily_state, results_dir / f"{prefix}_daily_state.csv")
        save(result.symbol_summary, results_dir / f"{prefix}_symbol_summary.csv")
        save(
            pd.DataFrame(
                [{"symbol": symbol, "rows_fetched": count} for symbol, count in fetch_counts.items()]
            ),
            results_dir / f"{prefix}_fetch_counts.csv",
        )
        if not result.symbol_summary.empty:
            save(
                result.symbol_summary.head(10),
                results_dir / f"{prefix}_top10_total_return.csv",
            )
            save(
                result.symbol_summary.tail(10),
                results_dir / f"{prefix}_bottom10_total_return.csv",
            )

        print(pd.DataFrame([result.summary]).to_string(index=False))
        if not result.symbol_summary.empty:
            print("\nTop symbols:")
            print(result.symbol_summary.head(10).to_string(index=False))
        print(f"\nSaved outputs with prefix {prefix}")
    finally:
        warehouse.close()


if __name__ == "__main__":
    main()
