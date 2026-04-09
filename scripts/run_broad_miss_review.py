#!/usr/bin/env python3
"""Review recent missed opportunities from the dynamic broad universe.

This is intentionally approximate: it uses the current broad-market slice,
then replays recent weeks on daily bars to see which non-growth names would
have produced trades under the portfolio backtest rules.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from tradingagents.automation.config import build_config
from tradingagents.research import (
    BacktestConfig,
    BroadMarketConfig,
    BroadMarketScreener,
    MarketDataWarehouse,
    MinerviniConfig,
    MinerviniScreener,
    PortfolioMinerviniBacktester,
    resolve_universe,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Review recent broad-universe opportunities missing from the fixed growth watchlist"
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=4,
        help="How many recent weeks to review",
    )
    parser.add_argument(
        "--benchmark",
        default="SPY",
        help="Benchmark symbol for market regime",
    )
    parser.add_argument(
        "--db",
        default=str(ROOT / "research_data" / "market_data.duckdb"),
        help="DuckDB path for market data cache",
    )
    parser.add_argument(
        "--results-dir",
        default=str(ROOT / "results" / "minervini"),
        help="Output directory",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force-refresh market data before analysis",
    )
    parser.add_argument(
        "--focus-symbol",
        default="LITE",
        help="Extra single-symbol diagnostic to export",
    )
    return parser.parse_args()


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_broad_slice(config: dict, benchmark: str) -> pd.DataFrame:
    screener = BroadMarketScreener(
        api_key=config["alpaca_api_key"],
        secret_key=config["alpaca_secret_key"],
        paper=config.get("paper_trading", True),
        config=BroadMarketConfig(
            min_price=float(config.get("broad_market_min_price", 10.0)),
            max_price=config.get("broad_market_max_price"),
            min_prev_volume=float(config.get("broad_market_min_prev_volume", 200_000)),
            min_prev_dollar_volume=float(
                config.get("broad_market_min_prev_dollar_volume", 25_000_000)
            ),
            min_avg_dollar_volume=float(
                config.get("broad_market_min_avg_dollar_volume", 20_000_000)
            ),
            max_seed_symbols=int(config.get("broad_market_max_seed_symbols", 600)),
            max_candidates=int(config.get("broad_market_max_candidates", 160)),
            snapshot_batch_size=int(config.get("broad_market_snapshot_batch_size", 200)),
            history_batch_size=int(config.get("broad_market_history_batch_size", 100)),
            history_period=str(config.get("broad_market_history_period", "1y")),
            exclude_funds=bool(config.get("broad_market_exclude_funds", True)),
            max_below_52w_high=float(config.get("broad_market_max_below_52w_high", 0.30)),
            min_above_52w_low=float(config.get("broad_market_min_above_52w_low", 0.25)),
        ),
    )
    return screener.build_candidates(benchmark=benchmark)


def build_portfolio_backtester(config: dict) -> PortfolioMinerviniBacktester:
    screener = MinerviniScreener(
        MinerviniConfig(
            min_rs_percentile=float(config.get("minervini_min_rs_percentile", 70.0)),
            min_above_52w_low=float(config.get("broad_market_min_above_52w_low", 0.25)),
            max_below_52w_high=float(config.get("broad_market_max_below_52w_high", 0.30)),
            require_fundamentals=False,
            require_market_uptrend=False,
            max_stage_number=int(config.get("minervini_max_stage_number", 3)),
            pivot_buffer_pct=float(config.get("minervini_pivot_buffer_pct", 0.0)),
            max_buy_zone_pct=float(config.get("minervini_max_buy_zone_pct", 0.07)),
        )
    )
    backtest = BacktestConfig(
        initial_cash=100_000.0,
        max_position_pct=float(config.get("max_position_pct", 0.12)),
        risk_per_trade=float(config.get("risk_per_trade", 0.012)),
        stop_loss_pct=float(config.get("default_stop_loss_pct", 0.08)),
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
        min_close_range_pct=float(config.get("minervini_min_close_range_pct", 0.55)),
        scale_exposure_in_weak_market=True,
        weak_market_position_scale=0.60,
        target_exposure_confirmed_uptrend=float(
            config.get("minervini_target_exposure_confirmed_uptrend", 0.72)
        ),
        target_exposure_uptrend_under_pressure=float(
            config.get("minervini_target_exposure_uptrend_under_pressure", 0.48)
        ),
        target_exposure_market_correction=float(
            config.get("minervini_target_exposure_market_correction", 0.0)
        ),
        allow_new_entries_in_correction=bool(
            config.get("minervini_allow_new_entries_in_correction", False)
        ),
        max_positions=int(config.get("max_open_positions", 6)),
    )
    return PortfolioMinerviniBacktester(screener=screener, config=backtest)


def main():
    args = parse_args()
    config = build_config({"trading_universe": "broad"})
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    end_dt = datetime.now()
    end_date = end_dt.strftime("%Y-%m-%d")
    trade_start_date = (end_dt - timedelta(weeks=args.weeks)).strftime("%Y-%m-%d")
    history_start_date = (end_dt - timedelta(days=max(450, args.weeks * 7 + 365))).strftime(
        "%Y-%m-%d"
    )
    stamp = end_date

    broad_df = build_broad_slice(config, benchmark=args.benchmark)
    broad_slice_path = results_dir / f"broad_miss_review_slice_{stamp}.csv"
    save_csv(broad_df, broad_slice_path)

    broad_symbols = broad_df["symbol"].dropna().astype(str).tolist() if not broad_df.empty else []
    growth_symbols = resolve_universe("growth")
    broad_only_symbols = sorted(set(broad_symbols) - set(growth_symbols))

    if not broad_symbols:
        print("Broad scanner returned no symbols.")
        return

    required_symbols = sorted(set(broad_symbols + growth_symbols + [args.benchmark]))
    warehouse = MarketDataWarehouse(args.db)
    try:
        available = set(warehouse.available_symbols())
        if args.refresh_data or not set(required_symbols).issubset(available):
            warehouse.fetch_and_store_daily_bars(required_symbols, history_start_date, end_date)

        benchmark_df = warehouse.get_daily_bars(args.benchmark, history_start_date, end_date)
        broad_data = {
            symbol: warehouse.get_daily_bars(symbol, history_start_date, end_date)
            for symbol in broad_symbols
        }
        broad_data = {k: v for k, v in broad_data.items() if v is not None and not v.empty}
        growth_data = {
            symbol: warehouse.get_daily_bars(symbol, history_start_date, end_date)
            for symbol in growth_symbols
        }
        growth_data = {k: v for k, v in growth_data.items() if v is not None and not v.empty}

        backtester = build_portfolio_backtester(config)
        broad_result = backtester.backtest_portfolio(
            broad_data,
            benchmark_df=benchmark_df,
            trade_start_date=trade_start_date,
        )
        growth_result = backtester.backtest_portfolio(
            growth_data,
            benchmark_df=benchmark_df,
            trade_start_date=trade_start_date,
        )

        broad_trades = broad_result.trades.copy()
        growth_trades = growth_result.trades.copy()
        broad_symbol_summary = broad_result.symbol_summary.copy()
        growth_symbol_summary = growth_result.symbol_summary.copy()

        broad_only_trades = (
            broad_trades[broad_trades["symbol"].isin(broad_only_symbols)].copy()
            if not broad_trades.empty
            else pd.DataFrame()
        )
        broad_only_symbol_summary = (
            broad_symbol_summary[broad_symbol_summary["symbol"].isin(broad_only_symbols)]
            .copy()
            .sort_values(["total_return", "total_pnl"], ascending=[False, False])
            if not broad_symbol_summary.empty
            else pd.DataFrame()
        )
        broad_only_candidates = broad_df[broad_df["symbol"].isin(broad_only_symbols)].copy()

        compare_rows = [
            {
                "window_weeks": args.weeks,
                "trade_start_date": trade_start_date,
                "end_date": end_date,
                "universe": "growth",
                "symbols_requested": len(growth_symbols),
                "symbols_available": len(growth_data),
                **growth_result.summary,
            },
            {
                "window_weeks": args.weeks,
                "trade_start_date": trade_start_date,
                "end_date": end_date,
                "universe": "broad_current_slice",
                "symbols_requested": len(broad_symbols),
                "symbols_available": len(broad_data),
                **broad_result.summary,
            },
        ]
        compare_df = pd.DataFrame(compare_rows)

        missed_summary = pd.DataFrame(
            [
                {
                    "window_weeks": args.weeks,
                    "trade_start_date": trade_start_date,
                    "end_date": end_date,
                    "broad_candidates": len(broad_symbols),
                    "growth_candidates": len(growth_symbols),
                    "broad_only_candidates": len(broad_only_symbols),
                    "broad_only_trades": int(len(broad_only_trades)),
                    "broad_only_symbols_traded": int(
                        broad_only_trades["symbol"].nunique()
                    )
                    if not broad_only_trades.empty
                    else 0,
                    "broad_only_positive_symbols": int(
                        (broad_only_symbol_summary["total_pnl"] > 0).sum()
                    )
                    if not broad_only_symbol_summary.empty
                    else 0,
                    "focus_symbol": args.focus_symbol.upper(),
                    "focus_symbol_in_broad_slice": args.focus_symbol.upper() in set(broad_symbols),
                }
            ]
        )

        focus_symbol = args.focus_symbol.upper()
        focus_metrics = pd.DataFrame()
        focus_trades = pd.DataFrame()
        if focus_symbol in broad_data:
            focus_result = backtester.backtest_symbol(
                focus_symbol,
                broad_data[focus_symbol],
                benchmark_df=benchmark_df,
                trade_start_date=trade_start_date,
            )
            focus_metrics = pd.DataFrame(
                [
                    {
                        key: value
                        for key, value in focus_result.items()
                        if key not in {"trades", "equity_curve"}
                    }
                ]
            )
            focus_trades = pd.DataFrame(focus_result.get("trades", []))

        save_csv(compare_df, results_dir / f"broad_miss_review_compare_{stamp}.csv")
        save_csv(broad_trades, results_dir / f"broad_miss_review_broad_trades_{stamp}.csv")
        save_csv(growth_trades, results_dir / f"broad_miss_review_growth_trades_{stamp}.csv")
        save_csv(
            broad_only_trades,
            results_dir / f"broad_miss_review_broad_only_trades_{stamp}.csv",
        )
        save_csv(
            broad_symbol_summary,
            results_dir / f"broad_miss_review_broad_symbol_summary_{stamp}.csv",
        )
        save_csv(
            broad_only_symbol_summary,
            results_dir / f"broad_miss_review_broad_only_symbol_summary_{stamp}.csv",
        )
        save_csv(
            broad_only_candidates,
            results_dir / f"broad_miss_review_broad_only_candidates_{stamp}.csv",
        )
        save_csv(missed_summary, results_dir / f"broad_miss_review_summary_{stamp}.csv")
        if not focus_metrics.empty:
            save_csv(
                focus_metrics,
                results_dir / f"broad_miss_review_{focus_symbol.lower()}_metrics_{stamp}.csv",
            )
        if not focus_trades.empty:
            save_csv(
                focus_trades,
                results_dir / f"broad_miss_review_{focus_symbol.lower()}_trades_{stamp}.csv",
            )

        print("Broad miss review complete.")
        print(missed_summary.to_string(index=False))
        print("\nTop broad-only symbols:")
        if broad_only_symbol_summary.empty:
            print("(none)")
        else:
            print(
                broad_only_symbol_summary.head(10).to_string(
                    index=False,
                    columns=[
                        "symbol",
                        "total_trades",
                        "win_rate",
                        "total_pnl",
                        "total_return",
                        "avg_trade_return",
                    ],
                )
            )
    finally:
        warehouse.close()


if __name__ == "__main__":
    main()
