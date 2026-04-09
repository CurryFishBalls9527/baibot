#!/usr/bin/env python3
"""Run portfolio-level Minervini strategy experiments across named variants."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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


@dataclass
class ExperimentSpec:
    name: str
    universe: str
    screener_kwargs: Dict
    backtest_kwargs: Dict
    note: str = ""
    symbol_override: Optional[List[str]] = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run Minervini strategy experiment suite")
    parser.add_argument("--db", default=str(ROOT / "research_data" / "portfolio_eval.duckdb"))
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "minervini"))
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--end", default="2026-03-24")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--include-broad-current-slice", action="store_true")
    return parser.parse_args()


def save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _latest_broad_slice(results_dir: Path) -> list[str]:
    files = sorted(results_dir.glob("automation_coarse_candidates_*.csv"))
    if not files:
        return []
    frame = pd.read_csv(files[-1])
    if "symbol" not in frame.columns:
        return []
    return frame["symbol"].dropna().astype(str).tolist()


def build_experiments(results_dir: Path, include_broad_current_slice: bool) -> list[ExperimentSpec]:
    baseline_screener = {
        "require_fundamentals": False,
        "require_market_uptrend": False,
        "max_stage_number": 3,
        "max_buy_zone_pct": 0.07,
        "pivot_buffer_pct": 0.0,
    }
    baseline_backtest = {
        "max_position_pct": 0.12,
        "risk_per_trade": 0.012,
        "stop_loss_pct": 0.08,
        "trail_stop_pct": 0.12,
        "max_hold_days": 90,
        "min_template_score": 7,
        "require_volume_surge": False,
        "require_market_regime": False,
        "progressive_entries": True,
        "initial_entry_fraction": 0.50,
        "add_on_trigger_pct_1": 0.025,
        "add_on_trigger_pct_2": 0.05,
        "add_on_fraction_1": 0.30,
        "add_on_fraction_2": 0.20,
        "breakeven_trigger_pct": 0.05,
        "partial_profit_trigger_pct": 0.12,
        "partial_profit_fraction": 0.33,
        "use_ema21_exit": True,
        "use_close_range_filter": True,
        "min_close_range_pct": 0.55,
        "scale_exposure_in_weak_market": True,
        "weak_market_position_scale": 0.60,
        "target_exposure_confirmed_uptrend": 1.00,
        "target_exposure_uptrend_under_pressure": 0.60,
        "target_exposure_market_correction": 0.00,
        "allow_new_entries_in_correction": False,
        "max_positions": 6,
    }

    experiments = [
        ExperimentSpec(
            name="growth_baseline",
            universe="growth",
            screener_kwargs=baseline_screener,
            backtest_kwargs=baseline_backtest,
            note="current public_champion-style growth baseline",
        ),
        ExperimentSpec(
            name="growth_selective",
            universe="growth",
            screener_kwargs={
                **baseline_screener,
                "max_stage_number": 2,
                "max_buy_zone_pct": 0.05,
                "pivot_buffer_pct": 0.001,
            },
            backtest_kwargs={
                **baseline_backtest,
                "min_template_score": 8,
                "require_volume_surge": True,
                "trail_stop_pct": 0.10,
                "partial_profit_trigger_pct": 0.10,
                "partial_profit_fraction": 0.40,
                "min_close_range_pct": 0.60,
                "target_exposure_confirmed_uptrend": 0.90,
                "target_exposure_uptrend_under_pressure": 0.45,
                "max_positions": 5,
            },
            note="tighter quality filters and less capital concentration",
        ),
        ExperimentSpec(
            name="growth_flex",
            universe="growth",
            screener_kwargs={
                **baseline_screener,
                "max_stage_number": 4,
                "max_buy_zone_pct": 0.08,
            },
            backtest_kwargs={
                **baseline_backtest,
                "max_position_pct": 0.13,
                "risk_per_trade": 0.013,
                "min_template_score": 6,
                "min_close_range_pct": 0.50,
                "target_exposure_uptrend_under_pressure": 0.70,
                "target_exposure_market_correction": 0.15,
                "allow_new_entries_in_correction": True,
                "max_positions": 7,
            },
            note="more permissive regime and earlier entry posture",
        ),
        ExperimentSpec(
            name="growth_momentum_plus",
            universe="growth",
            screener_kwargs={
                **baseline_screener,
                "max_stage_number": 3,
                "max_buy_zone_pct": 0.06,
            },
            backtest_kwargs={
                **baseline_backtest,
                "max_position_pct": 0.14,
                "risk_per_trade": 0.014,
                "trail_stop_pct": 0.10,
                "partial_profit_trigger_pct": 0.15,
                "partial_profit_fraction": 0.25,
                "target_exposure_uptrend_under_pressure": 0.70,
                "target_exposure_market_correction": 0.10,
                "allow_new_entries_in_correction": True,
            },
            note="leans harder into momentum while keeping partial profit logic",
        ),
        ExperimentSpec(
            name="combined_baseline",
            universe="combined",
            screener_kwargs=baseline_screener,
            backtest_kwargs=baseline_backtest,
            note="same baseline on combined large-cap plus growth universe",
        ),
    ]

    if include_broad_current_slice:
        broad_symbols = _latest_broad_slice(results_dir)
        if broad_symbols:
            experiments.append(
                ExperimentSpec(
                    name="broad_current_slice",
                    universe="broad_current_slice",
                    screener_kwargs=baseline_screener,
                    backtest_kwargs=baseline_backtest,
                    note="current broad coarse-scan slice; survivorship-biased",
                    symbol_override=broad_symbols,
                )
            )
    return experiments


def build_windows(end_date: str) -> list[dict]:
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    return [
        {
            "window": "six_month",
            "start_date": (end_dt - timedelta(days=730)).strftime("%Y-%m-%d"),
            "trade_start_date": (end_dt - timedelta(days=183)).strftime("%Y-%m-%d"),
            "end_date": end_date,
        },
        {
            "window": "one_year",
            "start_date": (end_dt - timedelta(days=730)).strftime("%Y-%m-%d"),
            "trade_start_date": (end_dt - timedelta(days=365)).strftime("%Y-%m-%d"),
            "end_date": end_date,
        },
    ]


def _symbols_for_spec(spec: ExperimentSpec) -> list[str]:
    return spec.symbol_override or resolve_universe(spec.universe)


def _efficiency_score(summary: dict) -> tuple[float, float]:
    total_return = float(summary.get("total_return", 0.0))
    max_drawdown = float(summary.get("max_drawdown", 0.0))
    efficiency = total_return - (0.75 * max_drawdown)
    return_over_drawdown = total_return / max(max_drawdown, 0.01)
    return round(efficiency, 4), round(return_over_drawdown, 4)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.end

    experiments = build_experiments(results_dir, args.include_broad_current_slice)
    windows = build_windows(args.end)
    all_symbols = sorted(
        {
            symbol
            for spec in experiments
            for symbol in _symbols_for_spec(spec)
        }
    )
    global_start = min(window["start_date"] for window in windows)

    warehouse = MarketDataWarehouse(args.db)
    try:
        required = sorted(set(all_symbols + [args.benchmark]))
        available = set(warehouse.available_symbols())
        if args.refresh_data or not set(required).issubset(available):
            warehouse.fetch_and_store_daily_bars(required, global_start, args.end)

        data_cache = {
            symbol: warehouse.get_daily_bars(symbol, global_start, args.end)
            for symbol in all_symbols
        }
        benchmark_df = warehouse.get_daily_bars(args.benchmark, global_start, args.end)

        summary_rows: list[dict] = []
        for spec in experiments:
            symbols = _symbols_for_spec(spec)
            data_by_symbol = {
                symbol: data_cache[symbol]
                for symbol in symbols
                if symbol in data_cache and not data_cache[symbol].empty
            }
            if not data_by_symbol:
                continue

            screener = MinerviniScreener(MinerviniConfig(**spec.screener_kwargs))
            backtester = PortfolioMinerviniBacktester(
                screener=screener,
                config=BacktestConfig(**spec.backtest_kwargs),
            )

            for window in windows:
                result = backtester.backtest_portfolio(
                    data_by_symbol,
                    benchmark_df=benchmark_df,
                    trade_start_date=window["trade_start_date"],
                )
                efficiency, return_over_drawdown = _efficiency_score(result.summary)
                summary_row = {
                    **result.summary,
                    "experiment": spec.name,
                    "universe": spec.universe,
                    "note": spec.note,
                    "window": window["window"],
                    "symbols_requested": len(symbols),
                    "symbols_available": len(data_by_symbol),
                    "efficiency_score": efficiency,
                    "return_over_drawdown": return_over_drawdown,
                }
                summary_rows.append(summary_row)

                prefix = f"experiment_{spec.name}_{window['window']}_{stamp}"
                save(pd.DataFrame([summary_row]), results_dir / f"{prefix}_metrics.csv")
                save(result.trades, results_dir / f"{prefix}_trades.csv")
                save(result.equity_curve, results_dir / f"{prefix}_equity_curve.csv")
                save(result.daily_state, results_dir / f"{prefix}_daily_state.csv")
                save(result.symbol_summary, results_dir / f"{prefix}_symbol_summary.csv")

        summary_df = pd.DataFrame(summary_rows)
        if summary_df.empty:
            print("No experiment results were produced.")
            return

        ranking_df = summary_df.sort_values(
            ["window", "efficiency_score", "total_return", "trade_win_rate"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
        save(summary_df, results_dir / f"strategy_experiments_{stamp}_summary.csv")
        save(ranking_df, results_dir / f"strategy_experiments_{stamp}_ranking.csv")

        print(ranking_df.to_string(index=False))
        print(f"\nSaved experiment outputs to {results_dir}")
    finally:
        warehouse.close()


if __name__ == "__main__":
    main()
