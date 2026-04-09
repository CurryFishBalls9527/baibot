#!/usr/bin/env python3
"""Search a curated set of more aggressive growth-strategy variants."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

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
class VariantSpec:
    name: str
    screener_kwargs: Dict
    backtest_kwargs: Dict
    note: str


def parse_args():
    parser = argparse.ArgumentParser(description="Run aggressive growth strategy search")
    parser.add_argument("--db", default=str(ROOT / "research_data" / "portfolio_eval.duckdb"))
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "minervini"))
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--start", default="2023-03-25")
    parser.add_argument("--trade-start", default="2024-03-24")
    parser.add_argument("--end", default="2026-03-24")
    parser.add_argument("--refresh-data", action="store_true")
    return parser.parse_args()


def save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_variants() -> list[VariantSpec]:
    base_screener = {
        "require_fundamentals": False,
        "require_market_uptrend": False,
        "max_stage_number": 3,
        "max_buy_zone_pct": 0.07,
        "pivot_buffer_pct": 0.0,
    }
    base_backtest = {
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

    return [
        VariantSpec(
            name="baseline",
            screener_kwargs=base_screener,
            backtest_kwargs=base_backtest,
            note="current best deployable baseline",
        ),
        VariantSpec(
            name="high_exposure",
            screener_kwargs={**base_screener, "max_stage_number": 4, "max_buy_zone_pct": 0.09},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.18,
                "risk_per_trade": 0.020,
                "min_template_score": 6,
                "min_close_range_pct": 0.45,
                "target_exposure_uptrend_under_pressure": 0.85,
                "target_exposure_market_correction": 0.20,
                "allow_new_entries_in_correction": True,
                "max_positions": 10,
            },
            note="push capital deployment much harder",
        ),
        VariantSpec(
            name="high_exposure_no_ema",
            screener_kwargs={**base_screener, "max_stage_number": 4, "max_buy_zone_pct": 0.09},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.18,
                "risk_per_trade": 0.020,
                "min_template_score": 6,
                "trail_stop_pct": 0.15,
                "use_ema21_exit": False,
                "min_close_range_pct": 0.45,
                "target_exposure_uptrend_under_pressure": 0.85,
                "target_exposure_market_correction": 0.20,
                "allow_new_entries_in_correction": True,
                "max_positions": 10,
            },
            note="same as high_exposure but lets trends run longer",
        ),
        VariantSpec(
            name="volume_high_exposure",
            screener_kwargs={**base_screener, "max_stage_number": 4, "max_buy_zone_pct": 0.08},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.18,
                "risk_per_trade": 0.018,
                "min_template_score": 6,
                "require_volume_surge": True,
                "min_close_range_pct": 0.50,
                "target_exposure_uptrend_under_pressure": 0.80,
                "target_exposure_market_correction": 0.15,
                "allow_new_entries_in_correction": True,
                "max_positions": 8,
            },
            note="try to keep quality while increasing exposure",
        ),
        VariantSpec(
            name="trend_rider",
            screener_kwargs={**base_screener, "max_stage_number": 4, "max_buy_zone_pct": 0.09},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.16,
                "risk_per_trade": 0.018,
                "trail_stop_pct": 0.15,
                "max_hold_days": 120,
                "partial_profit_trigger_pct": 0.25,
                "partial_profit_fraction": 0.20,
                "target_exposure_uptrend_under_pressure": 0.90,
                "target_exposure_market_correction": 0.10,
                "allow_new_entries_in_correction": True,
                "max_positions": 8,
            },
            note="wider trailing exit and later profit taking",
        ),
        VariantSpec(
            name="minimal_partial",
            screener_kwargs={**base_screener, "max_stage_number": 4, "max_buy_zone_pct": 0.08},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.16,
                "risk_per_trade": 0.016,
                "trail_stop_pct": 0.14,
                "max_hold_days": 120,
                "partial_profit_trigger_pct": 0.40,
                "partial_profit_fraction": 0.10,
                "target_exposure_uptrend_under_pressure": 0.80,
                "target_exposure_market_correction": 0.10,
                "allow_new_entries_in_correction": True,
                "max_positions": 8,
            },
            note="effectively disables most early trimming",
        ),
        VariantSpec(
            name="aggressive_correction",
            screener_kwargs={**base_screener, "max_stage_number": 4, "max_buy_zone_pct": 0.10},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.18,
                "risk_per_trade": 0.020,
                "min_template_score": 6,
                "min_close_range_pct": 0.45,
                "target_exposure_uptrend_under_pressure": 1.00,
                "target_exposure_market_correction": 0.35,
                "allow_new_entries_in_correction": True,
                "max_positions": 10,
            },
            note="most aggressive attempt to avoid cash drag",
        ),
        VariantSpec(
            name="selective_high_conviction",
            screener_kwargs={**base_screener, "max_stage_number": 2, "max_buy_zone_pct": 0.05, "pivot_buffer_pct": 0.001},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.22,
                "risk_per_trade": 0.016,
                "min_template_score": 8,
                "require_volume_surge": True,
                "min_close_range_pct": 0.65,
                "partial_profit_trigger_pct": 0.15,
                "partial_profit_fraction": 0.25,
                "target_exposure_uptrend_under_pressure": 0.50,
                "max_positions": 4,
            },
            note="few trades, bigger bets, highest entry quality",
        ),
        VariantSpec(
            name="wide_net_quality",
            screener_kwargs={**base_screener, "max_stage_number": 4, "max_buy_zone_pct": 0.08},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.14,
                "risk_per_trade": 0.015,
                "min_template_score": 6,
                "require_volume_surge": True,
                "min_close_range_pct": 0.55,
                "target_exposure_uptrend_under_pressure": 0.75,
                "target_exposure_market_correction": 0.10,
                "allow_new_entries_in_correction": True,
                "max_positions": 8,
            },
            note="wider opportunity net but still demands real volume",
        ),
        VariantSpec(
            name="momentum_pyramid",
            screener_kwargs={**base_screener, "max_stage_number": 4, "max_buy_zone_pct": 0.08},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.16,
                "risk_per_trade": 0.018,
                "add_on_trigger_pct_1": 0.02,
                "add_on_trigger_pct_2": 0.04,
                "add_on_fraction_1": 0.35,
                "add_on_fraction_2": 0.25,
                "partial_profit_trigger_pct": 0.18,
                "partial_profit_fraction": 0.25,
                "target_exposure_uptrend_under_pressure": 0.80,
                "target_exposure_market_correction": 0.10,
                "allow_new_entries_in_correction": True,
                "max_positions": 8,
            },
            note="leans into winners faster through pyramiding",
        ),
        VariantSpec(
            name="momentum_no_correction",
            screener_kwargs={**base_screener, "max_stage_number": 4, "max_buy_zone_pct": 0.08},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.16,
                "risk_per_trade": 0.018,
                "add_on_trigger_pct_1": 0.02,
                "add_on_trigger_pct_2": 0.04,
                "add_on_fraction_1": 0.35,
                "add_on_fraction_2": 0.25,
                "partial_profit_trigger_pct": 0.18,
                "partial_profit_fraction": 0.25,
                "target_exposure_uptrend_under_pressure": 0.90,
                "target_exposure_market_correction": 0.00,
                "allow_new_entries_in_correction": False,
                "max_positions": 8,
            },
            note="same pyramid engine but avoids new correction entries",
        ),
        VariantSpec(
            name="momentum_full_pressure",
            screener_kwargs={**base_screener, "max_stage_number": 4, "max_buy_zone_pct": 0.08},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.17,
                "risk_per_trade": 0.020,
                "add_on_trigger_pct_1": 0.02,
                "add_on_trigger_pct_2": 0.04,
                "add_on_fraction_1": 0.35,
                "add_on_fraction_2": 0.25,
                "partial_profit_trigger_pct": 0.18,
                "partial_profit_fraction": 0.25,
                "target_exposure_uptrend_under_pressure": 1.00,
                "target_exposure_market_correction": 0.00,
                "allow_new_entries_in_correction": False,
                "max_positions": 10,
            },
            note="push to full deployment outside outright corrections",
        ),
        VariantSpec(
            name="momentum_fast_pyramid",
            screener_kwargs={**base_screener, "max_stage_number": 4, "max_buy_zone_pct": 0.08},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.18,
                "risk_per_trade": 0.022,
                "initial_entry_fraction": 0.65,
                "add_on_trigger_pct_1": 0.015,
                "add_on_trigger_pct_2": 0.03,
                "add_on_fraction_1": 0.20,
                "add_on_fraction_2": 0.15,
                "partial_profit_trigger_pct": 0.20,
                "partial_profit_fraction": 0.20,
                "target_exposure_uptrend_under_pressure": 0.90,
                "target_exposure_market_correction": 0.00,
                "allow_new_entries_in_correction": False,
                "max_positions": 10,
            },
            note="starts bigger and pyramids earlier in strong moves",
        ),
        VariantSpec(
            name="momentum_long_hold",
            screener_kwargs={**base_screener, "max_stage_number": 4, "max_buy_zone_pct": 0.08},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.16,
                "risk_per_trade": 0.018,
                "trail_stop_pct": 0.16,
                "max_hold_days": 150,
                "add_on_trigger_pct_1": 0.02,
                "add_on_trigger_pct_2": 0.04,
                "add_on_fraction_1": 0.35,
                "add_on_fraction_2": 0.25,
                "partial_profit_trigger_pct": 0.28,
                "partial_profit_fraction": 0.15,
                "target_exposure_uptrend_under_pressure": 0.90,
                "target_exposure_market_correction": 0.00,
                "allow_new_entries_in_correction": False,
                "max_positions": 8,
            },
            note="keeps the pyramid but gives leaders more room and time",
        ),
        VariantSpec(
            name="momentum_dense",
            screener_kwargs={**base_screener, "max_stage_number": 5, "max_buy_zone_pct": 0.09},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.14,
                "risk_per_trade": 0.020,
                "min_template_score": 6,
                "add_on_trigger_pct_1": 0.02,
                "add_on_trigger_pct_2": 0.04,
                "add_on_fraction_1": 0.35,
                "add_on_fraction_2": 0.25,
                "partial_profit_trigger_pct": 0.18,
                "partial_profit_fraction": 0.25,
                "target_exposure_uptrend_under_pressure": 1.00,
                "target_exposure_market_correction": 0.00,
                "allow_new_entries_in_correction": False,
                "max_positions": 12,
            },
            note="uses more slots to reduce cash drag without huge position size",
        ),
        VariantSpec(
            name="momentum_risk_on",
            screener_kwargs={**base_screener, "max_stage_number": 5, "max_buy_zone_pct": 0.10},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.20,
                "risk_per_trade": 0.026,
                "min_template_score": 5,
                "min_close_range_pct": 0.45,
                "initial_entry_fraction": 0.65,
                "add_on_trigger_pct_1": 0.015,
                "add_on_trigger_pct_2": 0.03,
                "add_on_fraction_1": 0.20,
                "add_on_fraction_2": 0.15,
                "partial_profit_trigger_pct": 0.22,
                "partial_profit_fraction": 0.20,
                "target_exposure_uptrend_under_pressure": 1.00,
                "target_exposure_market_correction": 0.05,
                "allow_new_entries_in_correction": False,
                "max_positions": 12,
            },
            note="strongest risk-on deployment without allowing correction entries",
        ),
        VariantSpec(
            name="momentum_clean_breakout",
            screener_kwargs={**base_screener, "max_stage_number": 4, "max_buy_zone_pct": 0.06},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.16,
                "risk_per_trade": 0.018,
                "min_template_score": 7,
                "min_close_range_pct": 0.65,
                "add_on_trigger_pct_1": 0.02,
                "add_on_trigger_pct_2": 0.04,
                "add_on_fraction_1": 0.35,
                "add_on_fraction_2": 0.25,
                "partial_profit_trigger_pct": 0.18,
                "partial_profit_fraction": 0.25,
                "target_exposure_uptrend_under_pressure": 0.90,
                "target_exposure_market_correction": 0.00,
                "allow_new_entries_in_correction": False,
                "max_positions": 8,
            },
            note="requires tighter close quality to avoid sloppy breakouts",
        ),
        VariantSpec(
            name="momentum_concentrated",
            screener_kwargs={**base_screener, "max_stage_number": 4, "max_buy_zone_pct": 0.08},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.22,
                "risk_per_trade": 0.024,
                "initial_entry_fraction": 0.60,
                "add_on_trigger_pct_1": 0.02,
                "add_on_trigger_pct_2": 0.04,
                "add_on_fraction_1": 0.25,
                "add_on_fraction_2": 0.15,
                "partial_profit_trigger_pct": 0.20,
                "partial_profit_fraction": 0.20,
                "target_exposure_uptrend_under_pressure": 0.90,
                "target_exposure_market_correction": 0.00,
                "allow_new_entries_in_correction": False,
                "max_positions": 6,
            },
            note="larger bets in fewer names to amplify the best setups",
        ),
    ]


def _score(summary: dict) -> tuple[float, float]:
    total_return = float(summary.get("total_return", 0.0))
    benchmark_return = float(summary.get("benchmark_return", 0.0))
    max_drawdown = float(summary.get("max_drawdown", 0.0))
    alpha = total_return - benchmark_return
    efficiency = total_return - (0.75 * max_drawdown)
    return round(alpha, 4), round(efficiency, 4)


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

        rows = []
        for spec in build_variants():
            screener = MinerviniScreener(MinerviniConfig(**spec.screener_kwargs))
            backtester = PortfolioMinerviniBacktester(
                screener=screener,
                config=BacktestConfig(**spec.backtest_kwargs),
            )
            result = backtester.backtest_portfolio(
                data_by_symbol,
                benchmark_df=benchmark_df,
                trade_start_date=args.trade_start,
            )
            alpha, efficiency = _score(result.summary)
            row = {
                **result.summary,
                "variant": spec.name,
                "note": spec.note,
                "alpha_vs_spy": alpha,
                "efficiency_score": efficiency,
            }
            rows.append(row)

            prefix = f"growth_search_{spec.name}_{args.end}"
            save(pd.DataFrame([row]), results_dir / f"{prefix}_metrics.csv")
            save(result.trades, results_dir / f"{prefix}_trades.csv")
            save(result.daily_state, results_dir / f"{prefix}_daily_state.csv")
            save(result.symbol_summary, results_dir / f"{prefix}_symbol_summary.csv")

        summary = pd.DataFrame(rows)
        ranking = summary.sort_values(
            ["alpha_vs_spy", "total_return", "efficiency_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        save(summary, results_dir / f"growth_strategy_search_{args.end}_summary.csv")
        save(ranking, results_dir / f"growth_strategy_search_{args.end}_ranking.csv")

        winners = ranking[ranking["alpha_vs_spy"] > 0]
        if winners.empty:
            print("No variant beat SPY in this search batch.")
        else:
            print("Variants beating SPY:")
            print(winners.to_string(index=False))
        print("\nFull ranking:")
        print(ranking.to_string(index=False))
    finally:
        warehouse.close()


if __name__ == "__main__":
    main()
