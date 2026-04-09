#!/usr/bin/env python3
"""Grid search around the best growth momentum configuration."""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ProfitMode:
    name: str
    partial_profit_trigger_pct: float
    partial_profit_fraction: float
    max_hold_days: int
    trail_stop_pct: float


def parse_args():
    parser = argparse.ArgumentParser(description="Search risk-on growth variants")
    parser.add_argument("--db", default=str(ROOT / "research_data" / "portfolio_eval.duckdb"))
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "minervini"))
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--start", default="2023-03-25")
    parser.add_argument("--trade-start", default="2024-03-24")
    parser.add_argument("--end", default="2026-03-24")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--top-n", type=int, default=8)
    return parser.parse_args()


def save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _score(summary: dict) -> tuple[float, float]:
    total_return = float(summary.get("total_return", 0.0))
    benchmark_return = float(summary.get("benchmark_return", 0.0))
    max_drawdown = float(summary.get("max_drawdown", 0.0))
    alpha = total_return - benchmark_return
    efficiency = total_return - (0.75 * max_drawdown)
    return round(alpha, 4), round(efficiency, 4)


def _variant_name(
    *,
    max_position_pct: float,
    risk_per_trade: float,
    initial_entry_fraction: float,
    max_stage_number: int,
    min_template_score: int,
    profit_mode: ProfitMode,
) -> str:
    return (
        f"risk_on_p{int(max_position_pct * 100)}"
        f"_r{int(risk_per_trade * 1000)}"
        f"_i{int(initial_entry_fraction * 100)}"
        f"_s{max_stage_number}"
        f"_t{min_template_score}"
        f"_{profit_mode.name}"
    )


def _build_screener(stage: int) -> MinerviniScreener:
    return MinerviniScreener(
        MinerviniConfig(
            require_fundamentals=False,
            require_market_uptrend=False,
            max_stage_number=stage,
            max_buy_zone_pct=0.10,
            pivot_buffer_pct=0.0,
        )
    )


def _build_backtest_config(
    *,
    max_position_pct: float,
    risk_per_trade: float,
    initial_entry_fraction: float,
    min_template_score: int,
    profit_mode: ProfitMode,
) -> BacktestConfig:
    return BacktestConfig(
        max_position_pct=max_position_pct,
        risk_per_trade=risk_per_trade,
        stop_loss_pct=0.08,
        trail_stop_pct=profit_mode.trail_stop_pct,
        max_hold_days=profit_mode.max_hold_days,
        min_template_score=min_template_score,
        require_volume_surge=False,
        require_market_regime=False,
        progressive_entries=True,
        initial_entry_fraction=initial_entry_fraction,
        add_on_trigger_pct_1=0.015,
        add_on_trigger_pct_2=0.03,
        add_on_fraction_1=0.20,
        add_on_fraction_2=0.15,
        breakeven_trigger_pct=0.05,
        partial_profit_trigger_pct=profit_mode.partial_profit_trigger_pct,
        partial_profit_fraction=profit_mode.partial_profit_fraction,
        use_ema21_exit=True,
        use_close_range_filter=True,
        min_close_range_pct=0.45,
        scale_exposure_in_weak_market=True,
        weak_market_position_scale=0.60,
        target_exposure_confirmed_uptrend=1.00,
        target_exposure_uptrend_under_pressure=1.00,
        target_exposure_market_correction=0.00,
        allow_new_entries_in_correction=False,
        max_positions=12,
    )


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    symbols = resolve_universe("growth")
    warehouse = MarketDataWarehouse(args.db)
    try:
        required = sorted(set(symbols + [args.benchmark]))
        available = set(warehouse.available_symbols())
        if args.refresh_data or not set(required).issubset(available):
            warehouse.fetch_and_store_daily_bars(required, args.start, args.end)

        data_by_symbol = {
            symbol: warehouse.get_daily_bars(symbol, args.start, args.end)
            for symbol in symbols
        }
        data_by_symbol = {symbol: df for symbol, df in data_by_symbol.items() if not df.empty}
        benchmark_df = warehouse.get_daily_bars(args.benchmark, args.start, args.end)

        profit_modes = [
            ProfitMode(
                name="normal",
                partial_profit_trigger_pct=0.22,
                partial_profit_fraction=0.20,
                max_hold_days=90,
                trail_stop_pct=0.12,
            ),
            ProfitMode(
                name="runner",
                partial_profit_trigger_pct=0.30,
                partial_profit_fraction=0.10,
                max_hold_days=120,
                trail_stop_pct=0.14,
            ),
        ]
        search_space = list(
            itertools.product(
                [0.20, 0.22],
                [0.026, 0.030],
                [0.65, 0.75],
                [4, 5],
                [5, 6],
                profit_modes,
            )
        )

        rows = []
        for (
            max_position_pct,
            risk_per_trade,
            initial_entry_fraction,
            max_stage_number,
            min_template_score,
            profit_mode,
        ) in search_space:
            name = _variant_name(
                max_position_pct=max_position_pct,
                risk_per_trade=risk_per_trade,
                initial_entry_fraction=initial_entry_fraction,
                max_stage_number=max_stage_number,
                min_template_score=min_template_score,
                profit_mode=profit_mode,
            )
            backtester = PortfolioMinerviniBacktester(
                screener=_build_screener(max_stage_number),
                config=_build_backtest_config(
                    max_position_pct=max_position_pct,
                    risk_per_trade=risk_per_trade,
                    initial_entry_fraction=initial_entry_fraction,
                    min_template_score=min_template_score,
                    profit_mode=profit_mode,
                ),
            )
            result = backtester.backtest_portfolio(
                data_by_symbol,
                benchmark_df=benchmark_df,
                trade_start_date=args.trade_start,
            )
            alpha, efficiency = _score(result.summary)
            rows.append(
                {
                    **result.summary,
                    "variant": name,
                    "profit_mode": profit_mode.name,
                    "max_position_pct": max_position_pct,
                    "risk_per_trade": risk_per_trade,
                    "initial_entry_fraction": initial_entry_fraction,
                    "max_stage_number": max_stage_number,
                    "min_template_score": min_template_score,
                    "alpha_vs_spy": alpha,
                    "efficiency_score": efficiency,
                }
            )

        summary = pd.DataFrame(rows)
        ranking = summary.sort_values(
            ["alpha_vs_spy", "total_return", "efficiency_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        tag = args.end
        save(summary, results_dir / f"growth_risk_on_grid_{tag}_summary.csv")
        save(ranking, results_dir / f"growth_risk_on_grid_{tag}_ranking.csv")

        winners = ranking[ranking["alpha_vs_spy"] > 0].copy()
        if not winners.empty:
            save(winners, results_dir / f"growth_risk_on_grid_{tag}_winners.csv")

        top_variants = ranking.head(args.top_n)
        for row in top_variants.to_dict("records"):
            profit_mode = next(mode for mode in profit_modes if mode.name == row["profit_mode"])
            backtester = PortfolioMinerviniBacktester(
                screener=_build_screener(int(row["max_stage_number"])),
                config=_build_backtest_config(
                    max_position_pct=float(row["max_position_pct"]),
                    risk_per_trade=float(row["risk_per_trade"]),
                    initial_entry_fraction=float(row["initial_entry_fraction"]),
                    min_template_score=int(row["min_template_score"]),
                    profit_mode=profit_mode,
                ),
            )
            result = backtester.backtest_portfolio(
                data_by_symbol,
                benchmark_df=benchmark_df,
                trade_start_date=args.trade_start,
            )
            prefix = f"growth_risk_on_{row['variant']}_{tag}"
            save(pd.DataFrame([row]), results_dir / f"{prefix}_metrics.csv")
            save(result.trades, results_dir / f"{prefix}_trades.csv")
            save(result.daily_state, results_dir / f"{prefix}_daily_state.csv")
            save(result.symbol_summary, results_dir / f"{prefix}_symbol_summary.csv")

        if winners.empty:
            print("No risk-on grid variant beat SPY.")
        else:
            print("Risk-on grid winners:")
            print(
                winners[
                    [
                        "variant",
                        "total_return",
                        "benchmark_return",
                        "alpha_vs_spy",
                        "max_drawdown",
                    ]
                ].to_string(index=False)
            )
        print("\nTop ranking:")
        print(
            ranking[
                [
                    "variant",
                    "total_return",
                    "benchmark_return",
                    "alpha_vs_spy",
                    "max_drawdown",
                    "risk_per_trade",
                    "max_position_pct",
                    "initial_entry_fraction",
                    "max_stage_number",
                    "min_template_score",
                    "profit_mode",
                ]
            ]
            .head(args.top_n)
            .to_string(index=False)
        )
    finally:
        warehouse.close()


if __name__ == "__main__":
    main()
