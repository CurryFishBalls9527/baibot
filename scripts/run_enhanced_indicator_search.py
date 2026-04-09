#!/usr/bin/env python3
"""Search indicator- and market-context-enhanced growth strategies."""

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
    build_market_context,
    resolve_universe,
)


@dataclass
class VariantSpec:
    name: str
    screener_kwargs: Dict
    backtest_kwargs: Dict
    note: str
    use_market_context: bool = True


def parse_args():
    parser = argparse.ArgumentParser(description="Run enhanced indicator strategy search")
    parser.add_argument("--db", default=str(ROOT / "research_data" / "long_horizon_eval.duckdb"))
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "minervini"))
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--trade-start", default="2021-01-04")
    parser.add_argument("--end", default="2026-03-24")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_variants() -> list[VariantSpec]:
    base_screener = {
        "require_fundamentals": False,
        "require_market_uptrend": False,
        "max_stage_number": 4,
        "max_buy_zone_pct": 0.08,
        "pivot_buffer_pct": 0.0,
    }
    base_backtest = {
        "max_position_pct": 0.20,
        "risk_per_trade": 0.024,
        "stop_loss_pct": 0.08,
        "trail_stop_pct": 0.12,
        "max_hold_days": 90,
        "min_template_score": 6,
        "require_volume_surge": False,
        "require_market_regime": False,
        "progressive_entries": True,
        "initial_entry_fraction": 0.65,
        "add_on_trigger_pct_1": 0.015,
        "add_on_trigger_pct_2": 0.03,
        "add_on_fraction_1": 0.20,
        "add_on_fraction_2": 0.15,
        "breakeven_trigger_pct": 0.05,
        "partial_profit_trigger_pct": 0.22,
        "partial_profit_fraction": 0.20,
        "use_ema21_exit": True,
        "use_close_range_filter": True,
        "min_close_range_pct": 0.50,
        "scale_exposure_in_weak_market": True,
        "weak_market_position_scale": 0.60,
        "target_exposure_confirmed_uptrend": 1.00,
        "target_exposure_uptrend_under_pressure": 0.60,
        "target_exposure_market_correction": 0.00,
        "allow_new_entries_in_correction": False,
        "max_positions": 10,
        "min_rsi_14": 52.0,
        "min_adx_14": 15.0,
        "max_atr_pct_14": 0.10,
        "min_roc_20": 0.02,
        "min_roc_60": 0.08,
        "min_roc_120": 0.12,
    }
    return [
        VariantSpec(
            name="control_risk_on",
            screener_kwargs={
                "require_fundamentals": False,
                "require_market_uptrend": False,
                "max_stage_number": 4,
                "max_buy_zone_pct": 0.10,
                "pivot_buffer_pct": 0.0,
            },
            backtest_kwargs={
                "max_position_pct": 0.22,
                "risk_per_trade": 0.030,
                "stop_loss_pct": 0.08,
                "trail_stop_pct": 0.12,
                "max_hold_days": 90,
                "min_template_score": 5,
                "require_volume_surge": False,
                "require_market_regime": False,
                "progressive_entries": True,
                "initial_entry_fraction": 0.75,
                "add_on_trigger_pct_1": 0.015,
                "add_on_trigger_pct_2": 0.03,
                "add_on_fraction_1": 0.20,
                "add_on_fraction_2": 0.15,
                "breakeven_trigger_pct": 0.05,
                "partial_profit_trigger_pct": 0.22,
                "partial_profit_fraction": 0.20,
                "use_ema21_exit": True,
                "use_close_range_filter": True,
                "min_close_range_pct": 0.45,
                "scale_exposure_in_weak_market": True,
                "weak_market_position_scale": 0.60,
                "target_exposure_confirmed_uptrend": 1.00,
                "target_exposure_uptrend_under_pressure": 1.00,
                "target_exposure_market_correction": 0.00,
                "allow_new_entries_in_correction": False,
                "max_positions": 12,
                "min_rsi_14": 0.0,
                "min_adx_14": 0.0,
                "max_atr_pct_14": 1.0,
                "min_roc_20": -1.0,
                "min_roc_60": -1.0,
                "min_roc_120": -1.0,
            },
            note="Current best 2-year risk-on control without extra indicator filters",
            use_market_context=False,
        ),
        VariantSpec(
            name="context_balanced",
            screener_kwargs=base_screener,
            backtest_kwargs=base_backtest,
            note="Balanced market-context + indicator version",
        ),
        VariantSpec(
            name="context_quality",
            screener_kwargs={**base_screener, "max_buy_zone_pct": 0.06},
            backtest_kwargs={
                **base_backtest,
                "min_template_score": 7,
                "require_volume_surge": True,
                "min_close_range_pct": 0.58,
                "target_exposure_uptrend_under_pressure": 0.45,
                "max_positions": 8,
                "min_rsi_14": 56.0,
                "min_adx_14": 18.0,
                "max_atr_pct_14": 0.09,
                "min_roc_20": 0.03,
                "min_roc_60": 0.10,
                "min_roc_120": 0.16,
            },
            note="Higher-quality breakout entries with tighter market deployment",
        ),
        VariantSpec(
            name="context_runner",
            screener_kwargs=base_screener,
            backtest_kwargs={
                **base_backtest,
                "trail_stop_pct": 0.14,
                "max_hold_days": 120,
                "partial_profit_trigger_pct": 0.30,
                "partial_profit_fraction": 0.10,
                "target_exposure_uptrend_under_pressure": 0.55,
                "min_rsi_14": 54.0,
                "min_adx_14": 16.0,
                "max_atr_pct_14": 0.10,
                "min_roc_20": 0.02,
                "min_roc_60": 0.08,
                "min_roc_120": 0.14,
            },
            note="Lets leaders run longer under stronger market conditions",
        ),
        VariantSpec(
            name="context_risk_on",
            screener_kwargs={**base_screener, "max_buy_zone_pct": 0.10},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.22,
                "risk_per_trade": 0.028,
                "initial_entry_fraction": 0.75,
                "min_template_score": 5,
                "min_close_range_pct": 0.45,
                "target_exposure_uptrend_under_pressure": 0.85,
                "max_positions": 12,
                "min_rsi_14": 50.0,
                "min_adx_14": 12.0,
                "max_atr_pct_14": 0.12,
                "min_roc_20": 0.01,
                "min_roc_60": 0.06,
                "min_roc_120": 0.10,
            },
            note="Aggressive deployment but still context-aware",
        ),
        VariantSpec(
            name="context_persistent_leaders",
            screener_kwargs={**base_screener, "max_buy_zone_pct": 0.08},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.24,
                "risk_per_trade": 0.028,
                "initial_entry_fraction": 0.70,
                "trail_stop_pct": 0.13,
                "max_hold_days": 120,
                "partial_profit_trigger_pct": 0.28,
                "partial_profit_fraction": 0.15,
                "target_exposure_uptrend_under_pressure": 0.70,
                "max_positions": 8,
                "min_rsi_14": 58.0,
                "min_adx_14": 20.0,
                "max_atr_pct_14": 0.09,
                "min_roc_20": 0.03,
                "min_roc_60": 0.12,
                "min_roc_120": 0.18,
            },
            note="Concentrates on the strongest persistent leaders",
        ),
        VariantSpec(
            name="context_low_vol_trend",
            screener_kwargs={**base_screener, "max_buy_zone_pct": 0.07},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.18,
                "risk_per_trade": 0.022,
                "target_exposure_uptrend_under_pressure": 0.50,
                "max_positions": 8,
                "min_rsi_14": 54.0,
                "min_adx_14": 16.0,
                "max_atr_pct_14": 0.07,
                "min_roc_20": 0.02,
                "min_roc_60": 0.08,
                "min_roc_120": 0.12,
            },
            note="Prefers smoother names with lower ATR and steady trend",
        ),
        VariantSpec(
            name="context_breakout_quality",
            screener_kwargs={**base_screener, "max_buy_zone_pct": 0.06},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.20,
                "risk_per_trade": 0.024,
                "require_volume_surge": True,
                "min_template_score": 7,
                "min_close_range_pct": 0.60,
                "target_exposure_uptrend_under_pressure": 0.40,
                "max_positions": 8,
                "min_rsi_14": 58.0,
                "min_adx_14": 20.0,
                "max_atr_pct_14": 0.09,
                "min_roc_20": 0.03,
                "min_roc_60": 0.12,
                "min_roc_120": 0.16,
            },
            note="Only high-quality volume-confirmed breakouts",
        ),
        VariantSpec(
            name="context_full_cycle",
            screener_kwargs={**base_screener, "max_buy_zone_pct": 0.09},
            backtest_kwargs={
                **base_backtest,
                "max_position_pct": 0.18,
                "risk_per_trade": 0.022,
                "initial_entry_fraction": 0.60,
                "trail_stop_pct": 0.11,
                "partial_profit_trigger_pct": 0.18,
                "partial_profit_fraction": 0.25,
                "target_exposure_uptrend_under_pressure": 0.65,
                "max_positions": 10,
                "min_rsi_14": 52.0,
                "min_adx_14": 14.0,
                "max_atr_pct_14": 0.11,
                "min_roc_20": 0.01,
                "min_roc_60": 0.07,
                "min_roc_120": 0.10,
            },
            note="Aimed at surviving different market phases without total shutdown",
        ),
    ]


def _score(summary: dict) -> tuple[float, float]:
    total_return = float(summary.get("total_return", 0.0))
    benchmark_return = float(summary.get("benchmark_return", 0.0))
    max_drawdown = float(summary.get("max_drawdown", 0.0))
    alpha = total_return - benchmark_return
    efficiency = total_return - (0.75 * max_drawdown)
    return round(alpha, 4), round(efficiency, 4)


def annual_breakdown(
    daily_state: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    trade_start_date: str,
    end_date: str,
) -> pd.DataFrame:
    if daily_state.empty or benchmark_df.empty:
        return pd.DataFrame()

    state = daily_state.copy()
    state["trade_date"] = pd.to_datetime(state["trade_date"])
    state = state[state["trade_date"] >= pd.Timestamp(trade_start_date)].copy()
    if state.empty:
        return pd.DataFrame()

    benchmark = benchmark_df.reset_index().rename(columns={"index": "trade_date"})[
        ["trade_date", "close"]
    ]
    benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"])
    benchmark = benchmark[benchmark["trade_date"] >= pd.Timestamp(trade_start_date)].copy()

    merged = state.merge(benchmark, on="trade_date", how="inner")
    if merged.empty:
        return pd.DataFrame()

    overall_start = pd.Timestamp(trade_start_date)
    overall_end = pd.Timestamp(end_date)
    rows = []
    for year, group in merged.groupby(merged["trade_date"].dt.year):
        group = group.sort_values("trade_date")
        strategy_return = (float(group["equity"].iloc[-1]) / float(group["equity"].iloc[0])) - 1.0
        spy_return = (float(group["close"].iloc[-1]) / float(group["close"].iloc[0])) - 1.0
        is_partial = year in {overall_start.year, overall_end.year}
        rows.append(
            {
                "year": int(year),
                "start_date": group["trade_date"].iloc[0].date().isoformat(),
                "end_date": group["trade_date"].iloc[-1].date().isoformat(),
                "is_partial_year": bool(is_partial),
                "strategy_return": round(strategy_return, 4),
                "spy_return": round(spy_return, 4),
                "alpha_vs_spy": round(strategy_return - spy_return, 4),
                "mean_exposure": round(float(group["actual_exposure"].mean()), 4),
                "avg_market_score": round(float(group["market_score"].dropna().mean()), 2)
                if "market_score" in group.columns and not group["market_score"].dropna().empty
                else None,
            }
        )
    return pd.DataFrame(rows)


def summarize_years(annual_df: pd.DataFrame) -> dict:
    if annual_df.empty:
        return {
            "full_years_tested": 0,
            "positive_full_year_alpha_count": 0,
            "positive_full_year_alpha_ratio": 0.0,
            "min_full_year_alpha": 0.0,
            "avg_full_year_alpha": 0.0,
        }
    full_years = annual_df[~annual_df["is_partial_year"]].copy()
    if full_years.empty:
        return {
            "full_years_tested": 0,
            "positive_full_year_alpha_count": 0,
            "positive_full_year_alpha_ratio": 0.0,
            "min_full_year_alpha": 0.0,
            "avg_full_year_alpha": 0.0,
        }
    positive_count = int((full_years["alpha_vs_spy"] > 0).sum())
    return {
        "full_years_tested": int(len(full_years)),
        "positive_full_year_alpha_count": positive_count,
        "positive_full_year_alpha_ratio": round(positive_count / len(full_years), 4),
        "min_full_year_alpha": round(float(full_years["alpha_vs_spy"].min()), 4),
        "avg_full_year_alpha": round(float(full_years["alpha_vs_spy"].mean()), 4),
    }


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    universe = resolve_universe("growth")
    context_symbols = [args.benchmark, "QQQ", "IWM", "SMH", "^VIX"]
    warehouse = MarketDataWarehouse(args.db)
    try:
        required = sorted(set(universe + context_symbols))
        available = set(warehouse.available_symbols())
        if args.refresh_data or not set(required).issubset(available):
            warehouse.fetch_and_store_daily_bars(required, args.start, args.end)

        data_by_symbol = {
            symbol: warehouse.get_daily_bars(symbol, args.start, args.end)
            for symbol in universe
        }
        data_by_symbol = {symbol: df for symbol, df in data_by_symbol.items() if not df.empty}
        benchmark_df = warehouse.get_daily_bars(args.benchmark, args.start, args.end)
        context_frames = {
            symbol: warehouse.get_daily_bars(symbol, args.start, args.end)
            for symbol in context_symbols
        }
        market_context_df = build_market_context(context_frames)

        rows = []
        annual_outputs: dict[str, pd.DataFrame] = {}
        cached_results: dict[str, tuple] = {}
        for spec in build_variants():
            screener = MinerviniScreener(MinerviniConfig(**spec.screener_kwargs))
            backtester = PortfolioMinerviniBacktester(
                screener=screener,
                config=BacktestConfig(**spec.backtest_kwargs),
            )
            result = backtester.backtest_portfolio(
                data_by_symbol,
                benchmark_df=benchmark_df,
                market_context_df=market_context_df if spec.use_market_context else None,
                trade_start_date=args.trade_start,
            )
            alpha, efficiency = _score(result.summary)
            annual_df = annual_breakdown(
                result.daily_state,
                benchmark_df,
                trade_start_date=args.trade_start,
                end_date=result.summary.get("end_date", args.end),
            )
            year_stats = summarize_years(annual_df)
            row = {
                **result.summary,
                **year_stats,
                "variant": spec.name,
                "note": spec.note,
                "uses_market_context": spec.use_market_context,
                "alpha_vs_spy": alpha,
                "efficiency_score": efficiency,
            }
            rows.append(row)
            annual_outputs[spec.name] = annual_df
            cached_results[spec.name] = (row, result)

        summary = pd.DataFrame(rows)
        ranking = summary.sort_values(
            [
                "positive_full_year_alpha_count",
                "min_full_year_alpha",
                "alpha_vs_spy",
                "total_return",
                "efficiency_score",
            ],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)

        tag = args.end
        save(summary, results_dir / f"enhanced_indicator_search_{tag}_summary.csv")
        save(ranking, results_dir / f"enhanced_indicator_search_{tag}_ranking.csv")

        top_variants = ranking.head(args.top_n)["variant"].tolist()
        for variant in top_variants:
            row, result = cached_results[variant]
            prefix = f"enhanced_indicator_{variant}_{tag}"
            save(pd.DataFrame([row]), results_dir / f"{prefix}_metrics.csv")
            save(result.trades, results_dir / f"{prefix}_trades.csv")
            save(result.daily_state, results_dir / f"{prefix}_daily_state.csv")
            save(result.symbol_summary, results_dir / f"{prefix}_symbol_summary.csv")
            save(annual_outputs[variant], results_dir / f"{prefix}_annual_returns.csv")

        winners = ranking[
            (ranking["alpha_vs_spy"] > 0)
            & (ranking["positive_full_year_alpha_count"] >= 3)
        ].copy()
        if not winners.empty:
            save(winners, results_dir / f"enhanced_indicator_search_{tag}_winners.csv")

        print("Top ranking:")
        print(
            ranking[
                [
                    "variant",
                    "total_return",
                    "benchmark_return",
                    "alpha_vs_spy",
                    "max_drawdown",
                    "positive_full_year_alpha_count",
                    "full_years_tested",
                    "min_full_year_alpha",
                    "avg_full_year_alpha",
                    "uses_market_context",
                ]
            ].to_string(index=False)
        )
        if winners.empty:
            print("\nNo enhanced variant cleared both total alpha and multi-year consistency filters.")
        else:
            print("\nConsistency winners:")
            print(
                winners[
                    [
                        "variant",
                        "total_return",
                        "benchmark_return",
                        "alpha_vs_spy",
                        "positive_full_year_alpha_count",
                        "full_years_tested",
                        "min_full_year_alpha",
                    ]
                ].to_string(index=False)
            )
    finally:
        warehouse.close()


if __name__ == "__main__":
    main()
