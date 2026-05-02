#!/usr/bin/env python3
"""Build a local Minervini-style research dataset, screen setups, and backtest.

Examples:
    python run_minervini_research.py --refresh-data --screen
    python run_minervini_research.py --refresh-data --screen --backtest
    python run_minervini_research.py --symbols NVDA,MSFT,ANET --refresh-data --backtest
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from tradingagents.backtesting.screener import LARGE_CAP_UNIVERSE
from tradingagents.research import MarketDataWarehouse, MinerviniBacktester, MinerviniConfig, MinerviniScreener


def parse_args():
    parser = argparse.ArgumentParser(description="Minervini-style swing research runner")
    parser.add_argument("--db", type=str, default="research_data/market_data.duckdb")
    parser.add_argument("--results-dir", type=str, default="results/minervini")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--benchmark", type=str, default="SPY")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--trade-start",
        type=str,
        help="Optional trade evaluation start date (YYYY-MM-DD) while retaining earlier bars for indicator warmup",
    )
    parser.add_argument("--period", type=str, default="3y", help="Lookback period: 1y, 2y, 3y, 5y")
    parser.add_argument("--refresh-data", action="store_true", help="Download and store daily bars")
    parser.add_argument(
        "--refresh-fundamentals",
        action="store_true",
        help="Download and store the latest fundamentals snapshot",
    )
    parser.add_argument("--screen", action="store_true", help="Run the latest screen")
    parser.add_argument("--backtest", action="store_true", help="Run the Minervini-style backtest")
    parser.add_argument("--screen-count", type=int, default=20, help="Number of top screen rows to show")
    parser.add_argument("--min-rs", type=float, default=70.0, help="Minimum RS percentile")
    parser.add_argument("--min-revenue-growth", type=float, default=0.15)
    parser.add_argument("--min-eps-growth", type=float, default=0.15)
    parser.add_argument("--min-roe", type=float, default=0.15)
    parser.add_argument("--max-stage", type=int, default=2, help="Maximum preferred base stage number")
    parser.add_argument(
        "--require-acceleration",
        action="store_true",
        help="Require non-negative revenue/EPS acceleration in the screen",
    )
    parser.add_argument(
        "--allow-missing-fundamentals",
        action="store_true",
        help="Do not require revenue/EPS/ROE filters in the screen",
    )
    parser.add_argument(
        "--allow-market-correction",
        action="store_true",
        help="Do not require a confirmed benchmark uptrend",
    )
    return parser.parse_args()


def resolve_dates(args):
    if args.end:
        end_date = args.end
    else:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if args.start:
        start_date = args.start
    else:
        period_days = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825}
        start_date = (datetime.now() - timedelta(days=period_days.get(args.period, 1095))).strftime("%Y-%m-%d")

    return start_date, end_date


def resolve_symbols(args):
    if args.symbols:
        return [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    return LARGE_CAP_UNIVERSE


def save_dataframe(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    args = parse_args()
    run_screen = args.screen or (not args.screen and not args.backtest)
    run_backtest = args.backtest or (not args.screen and not args.backtest)
    start_date, end_date = resolve_dates(args)
    symbols = resolve_symbols(args)
    results_dir = Path(args.results_dir)

    warehouse = MarketDataWarehouse(args.db)
    try:
        available = set(warehouse.available_symbols())
        required = set(symbols + [args.benchmark])
        if args.refresh_data or not required.issubset(available):
            counts = warehouse.fetch_and_store_daily_bars(required, start_date, end_date)
            print(f"Stored daily bars for {sum(1 for _, count in counts.items() if count > 0)} symbols")
        if args.refresh_fundamentals or args.refresh_data:
            warehouse.fetch_and_store_fundamentals(symbols)
            warehouse.fetch_and_store_quarterly_fundamentals(symbols)
            warehouse.fetch_and_store_earnings_events(symbols)
            print(f"Stored fundamentals snapshots for {len(symbols)} symbols")

        data_by_symbol = {
            symbol: warehouse.get_daily_bars(symbol, start_date, end_date)
            for symbol in symbols
        }
        data_by_symbol = {symbol: df for symbol, df in data_by_symbol.items() if not df.empty}
        benchmark_df = warehouse.get_daily_bars(args.benchmark, start_date, end_date)
        fundamentals_df = warehouse.get_latest_fundamentals(symbols)
        if fundamentals_df.empty and not args.allow_missing_fundamentals:
            warehouse.fetch_and_store_fundamentals(symbols)
            warehouse.fetch_and_store_quarterly_fundamentals(symbols)
            warehouse.fetch_and_store_earnings_events(symbols)
            fundamentals_df = warehouse.get_latest_fundamentals(symbols)
            print(f"Stored fundamentals snapshots for {len(symbols)} symbols")
        quarterly_df = warehouse.get_quarterly_fundamentals(symbols)
        # earnings_events lives in the attached read-only earnings DB
        # (ed) since the 2026-05-01 split. See warehouse.py:EARNINGS_DB_PATH.
        earnings_events_df = warehouse.conn.execute(
            """
            SELECT symbol, event_datetime, eps_estimate, reported_eps,
                   surprise_pct, revenue_average, is_future
            FROM ed.earnings_events
            WHERE symbol IN ({placeholders})
            ORDER BY symbol, event_datetime DESC
            """.format(placeholders=", ".join(["?"] * len(symbols))),
            symbols,
        ).fetchdf()

        screener = MinerviniScreener(
            MinerviniConfig(
                min_rs_percentile=args.min_rs,
                min_revenue_growth=args.min_revenue_growth,
                min_eps_growth=args.min_eps_growth,
                min_return_on_equity=args.min_roe,
                require_fundamentals=not args.allow_missing_fundamentals,
                require_market_uptrend=not args.allow_market_correction,
                require_acceleration=args.require_acceleration,
                max_stage_number=args.max_stage,
            )
        )
        latest_screen = None
        regime = screener.analyze_market_regime(benchmark_df)
        regime_path = results_dir / f"market_regime_{end_date}.csv"
        save_dataframe(
            pd.DataFrame([regime]),
            regime_path,
        )
        print(
            f"Market regime: {regime['regime']} "
            f"(close={regime['benchmark_close']}, 50dma={regime['benchmark_sma_50']}, 200dma={regime['benchmark_sma_200']})"
        )
        print(f"Saved market regime snapshot to {regime_path}")
        if not fundamentals_df.empty:
            fundamentals_path = results_dir / f"fundamentals_{end_date}.csv"
            save_dataframe(fundamentals_df, fundamentals_path)
            print(f"Saved fundamentals snapshot to {fundamentals_path}")
        if not quarterly_df.empty:
            quarterly_path = results_dir / f"quarterly_fundamentals_{end_date}.csv"
            save_dataframe(quarterly_df, quarterly_path)
            print(f"Saved quarterly fundamentals to {quarterly_path}")
        if not earnings_events_df.empty:
            earnings_path = results_dir / f"earnings_events_{end_date}.csv"
            save_dataframe(earnings_events_df, earnings_path)
            print(f"Saved earnings events to {earnings_path}")

        if run_screen:
            latest_screen = screener.screen_universe(
                data_by_symbol,
                benchmark_df,
                fundamentals_df=fundamentals_df,
            )
            if latest_screen.empty:
                print("No candidates found.")
            else:
                screen_path = results_dir / f"screen_{end_date}.csv"
                save_dataframe(latest_screen, screen_path)
                print("\nTop screen results:")
                print(
                    latest_screen[
                        [
                            "symbol",
                            "close",
                            "rs_percentile",
                            "revenue_growth",
                            "eps_growth",
                            "eps_acceleration",
                            "template_score",
                            "passed_template",
                            "stage_number",
                            "base_label",
                            "candidate_status",
                            "breakout_ready",
                            "buy_point",
                            "buy_limit_price",
                            "initial_stop_price",
                            "rule_watch_candidate",
                            "rule_entry_candidate",
                        ]
                    ]
                    .head(args.screen_count)
                    .to_string(index=False)
                )
                print(f"\nSaved screen results to {screen_path}")

        if run_backtest:
            candidate_symbols = list(data_by_symbol.keys())
            if latest_screen is not None and not latest_screen.empty:
                filtered = latest_screen[latest_screen["passed_template"]]
                if not filtered.empty:
                    candidate_symbols = filtered["symbol"].head(args.screen_count).tolist()

            backtester = MinerviniBacktester(screener=screener)
            backtest_data = {symbol: data_by_symbol[symbol] for symbol in candidate_symbols}
            results = backtester.backtest_universe(
                backtest_data,
                benchmark_df=benchmark_df,
                trade_start_date=args.trade_start,
            )

            summary = results["summary"]
            trades = results["trades"]
            summary_path = results_dir / f"backtest_summary_{end_date}.csv"
            trades_path = results_dir / f"backtest_trades_{end_date}.csv"
            save_dataframe(summary, summary_path)
            if not trades.empty:
                save_dataframe(trades, trades_path)

            print("\nBacktest summary:")
            if summary.empty:
                print("No backtest results generated.")
            else:
                print(
                    summary[
                        [
                            "symbol",
                            "total_return",
                            "benchmark_return",
                            "max_drawdown",
                            "total_trades",
                            "win_rate",
                            "profit_factor",
                        ]
                    ].to_string(index=False)
                )
            print(f"\nSaved backtest summary to {summary_path}")
            if not trades.empty:
                print(f"Saved trade log to {trades_path}")

            print("\nPortfolio summary:")
            for key, value in results["portfolio_summary"].items():
                print(f"  {key}: {value}")
    finally:
        warehouse.close()


if __name__ == "__main__":
    main()
