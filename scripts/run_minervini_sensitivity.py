#!/usr/bin/env python3
"""Run baseline and more aggressive Minervini backtest variants side by side."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tradingagents.backtesting.screener import LARGE_CAP_UNIVERSE
from tradingagents.research import MarketDataWarehouse, MinerviniBacktester, MinerviniConfig, MinerviniScreener
from tradingagents.research.backtester import BacktestConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Compare strict vs aggressive Minervini backtests")
    parser.add_argument("--db", default=str(ROOT / "research_data" / "backtest_eval.duckdb"))
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "minervini"))
    parser.add_argument("--start", default="2025-03-23", help="Warmup data start date")
    parser.add_argument("--end", default="2026-03-23")
    parser.add_argument("--benchmark", default="SPY")
    return parser.parse_args()


def benchmark_return(benchmark_df: pd.DataFrame, trade_start_date: str) -> float:
    frame = benchmark_df.loc[benchmark_df.index >= trade_start_date]
    if frame.empty:
        return 0.0
    start_price = float(frame["close"].iloc[0])
    end_price = float(frame["close"].iloc[-1])
    if start_price <= 0:
        return 0.0
    return round((end_price / start_price) - 1.0, 4)


def compute_metrics(summary: pd.DataFrame, trades: pd.DataFrame, benchmark_df: pd.DataFrame, trade_start: str):
    active = summary[summary["total_trades"] > 0] if not summary.empty else summary
    trade_win_rate = float((trades["pnl"] > 0).mean()) if not trades.empty else 0.0
    avg_trade_return = float(trades["return_pct"].mean()) if not trades.empty else 0.0
    median_trade_return = float(trades["return_pct"].median()) if not trades.empty else 0.0
    avg_active_return = float(active["total_return"].mean()) if not active.empty else 0.0
    median_active_return = float(active["total_return"].median()) if not active.empty else 0.0
    positive_active_ratio = float((active["total_return"] > 0).mean()) if not active.empty else 0.0
    avg_max_drawdown = float(summary["max_drawdown"].mean()) if not summary.empty else 0.0

    return {
        "symbols_tested": int(len(summary)),
        "symbols_with_trades": int((summary["total_trades"] > 0).sum()) if not summary.empty else 0,
        "total_trades": int(len(trades)),
        "trade_win_rate": round(trade_win_rate, 4),
        "avg_trade_return": round(avg_trade_return, 4),
        "median_trade_return": round(median_trade_return, 4),
        "avg_active_symbol_return": round(avg_active_return, 4),
        "median_active_symbol_return": round(median_active_return, 4),
        "positive_active_symbol_ratio": round(positive_active_ratio, 4),
        "avg_max_drawdown": round(avg_max_drawdown, 4),
        "benchmark_return": benchmark_return(benchmark_df, trade_start),
    }


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()

    windows = {
        "three_month": "2025-12-23",
        "six_month": "2025-09-23",
    }
    variants = [
        {
            "name": "baseline_strict",
            "screener": MinerviniConfig(
                require_fundamentals=False,
                require_market_uptrend=False,
            ),
            "backtest": BacktestConfig(),
        },
        {
            "name": "aggressive_light",
            "screener": MinerviniConfig(
                require_fundamentals=False,
                require_market_uptrend=False,
                max_stage_number=3,
            ),
            "backtest": BacktestConfig(
                min_template_score=7,
                require_volume_surge=False,
                require_market_regime=False,
            ),
        },
        {
            "name": "aggressive_medium",
            "screener": MinerviniConfig(
                require_fundamentals=False,
                require_market_uptrend=False,
                max_stage_number=4,
                max_buy_zone_pct=0.08,
                pivot_buffer_pct=0.0,
            ),
            "backtest": BacktestConfig(
                min_template_score=6,
                require_volume_surge=False,
                require_market_regime=False,
                max_hold_days=75,
            ),
        },
        {
            "name": "public_champion",
            "screener": MinerviniConfig(
                require_fundamentals=False,
                require_market_uptrend=False,
                max_stage_number=3,
                max_buy_zone_pct=0.07,
                pivot_buffer_pct=0.0,
            ),
            "backtest": BacktestConfig(
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
            ),
        },
    ]

    warehouse = MarketDataWarehouse(args.db, read_only=True)
    try:
        symbols = [symbol for symbol in LARGE_CAP_UNIVERSE if not warehouse.get_daily_bars(symbol, args.start, args.end).empty]
        benchmark_df = warehouse.get_daily_bars(args.benchmark, args.start, args.end)
        data_by_symbol = {
            symbol: warehouse.get_daily_bars(symbol, args.start, args.end)
            for symbol in symbols
        }

        metric_rows = []
        for window_name, trade_start in windows.items():
            for variant in variants:
                screener = MinerviniScreener(variant["screener"])
                backtester = MinerviniBacktester(screener=screener, config=variant["backtest"])
                results = backtester.backtest_universe(
                    data_by_symbol,
                    benchmark_df=benchmark_df,
                    trade_start_date=trade_start,
                )
                summary = results["summary"]
                trades = results["trades"]
                metrics = compute_metrics(summary, trades, benchmark_df, trade_start)
                metric_rows.append(
                    {
                        "window": window_name,
                        "trade_start": trade_start,
                        "variant": variant["name"],
                        **metrics,
                    }
                )

                prefix = f"{variant['name']}_{window_name}_{stamp}"
                summary.to_csv(results_dir / f"{prefix}_summary.csv", index=False)
                if not trades.empty:
                    trades.to_csv(results_dir / f"{prefix}_trades.csv", index=False)

        metrics_df = pd.DataFrame(metric_rows)
        metrics_path = results_dir / f"aggressive_sensitivity_{stamp}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)

        print(metrics_df.to_string(index=False))
        print(f"\nSaved metrics to {metrics_path}")
    finally:
        warehouse.close()


if __name__ == "__main__":
    main()
