#!/usr/bin/env python3
"""Search portable-alpha overlays on top of a base stock strategy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tradingagents.research import MarketDataWarehouse, build_market_context


def parse_args():
    parser = argparse.ArgumentParser(description="Search ETF overlays over idle strategy cash")
    parser.add_argument(
        "--base-daily-state",
        default=str(
            ROOT
            / "results"
            / "minervini"
            / "enhanced_indicator_control_risk_on_2026-03-24_daily_state.csv"
        ),
    )
    parser.add_argument("--db", default=str(ROOT / "research_data" / "long_horizon_eval.duckdb"))
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "minervini"))
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--trade-start", default="2021-01-04")
    parser.add_argument("--end", default="2026-03-24")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--refresh-data", action="store_true")
    return parser.parse_args()


def save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _annual_breakdown(frame: pd.DataFrame, trade_start: str, end_date: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame[frame["trade_date"] >= pd.Timestamp(trade_start)].copy()
    overall_start = pd.Timestamp(trade_start)
    overall_end = pd.Timestamp(end_date)

    rows = []
    for year, group in frame.groupby(frame["trade_date"].dt.year):
        group = group.sort_values("trade_date")
        strategy_return = (float(group["equity"].iloc[-1]) / float(group["equity"].iloc[0])) - 1.0
        spy_return = (float(group["spy_close"].iloc[-1]) / float(group["spy_close"].iloc[0])) - 1.0
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
                "mean_overlay_weight": round(float(group["overlay_weight"].mean()), 4),
                "mean_stock_exposure": round(float(group["actual_exposure"].mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def _year_stats(annual_df: pd.DataFrame) -> dict:
    full_years = annual_df[~annual_df["is_partial_year"]].copy()
    if full_years.empty:
        return {
            "full_years_tested": 0,
            "positive_full_year_alpha_count": 0,
            "positive_full_year_alpha_ratio": 0.0,
            "min_full_year_alpha": 0.0,
            "avg_full_year_alpha": 0.0,
        }
    positive = int((full_years["alpha_vs_spy"] > 0).sum())
    return {
        "full_years_tested": int(len(full_years)),
        "positive_full_year_alpha_count": positive,
        "positive_full_year_alpha_ratio": round(positive / len(full_years), 4),
        "min_full_year_alpha": round(float(full_years["alpha_vs_spy"].min()), 4),
        "avg_full_year_alpha": round(float(full_years["alpha_vs_spy"].mean()), 4),
    }


def _gate_weight(row: pd.Series, gate: str, idle_weight: float, fraction: float) -> float:
    if idle_weight <= 0:
        return 0.0
    score = row.get("market_score")
    score = float(score) if pd.notna(score) else None
    regime = row.get("market_regime")
    if gate == "always":
        return idle_weight * fraction
    if gate == "confirmed_only":
        return idle_weight * fraction if regime == "confirmed_uptrend" else 0.0
    if gate == "not_correction":
        return idle_weight * fraction if regime != "market_correction" else 0.0
    if gate == "score_gte_5":
        return idle_weight * fraction if score is not None and score >= 5 else 0.0
    if gate == "score_gte_6":
        return idle_weight * fraction if score is not None and score >= 6 else 0.0
    return 0.0


def _simulate_overlay(
    state: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    overlay_df: pd.DataFrame,
    gate: str,
    fraction: float,
) -> tuple[pd.DataFrame, dict]:
    frame = state.copy().sort_values("trade_date")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    bench = benchmark_df.reset_index().rename(columns={"index": "trade_date", "close": "spy_close"})
    overlay = overlay_df.reset_index().rename(columns={"index": "trade_date", "close": "overlay_close"})
    bench["trade_date"] = pd.to_datetime(bench["trade_date"])
    overlay["trade_date"] = pd.to_datetime(overlay["trade_date"])
    overlay["overlay_return"] = overlay["overlay_close"].pct_change().fillna(0.0)
    bench["spy_return"] = bench["spy_close"].pct_change().fillna(0.0)

    merged = (
        frame.merge(bench[["trade_date", "spy_close", "spy_return"]], on="trade_date", how="inner")
        .merge(overlay[["trade_date", "overlay_close", "overlay_return"]], on="trade_date", how="inner")
        .copy()
    )
    merged["base_return"] = merged["equity"].pct_change().fillna(0.0)
    merged["idle_weight"] = (1.0 - merged["actual_exposure"]).clip(lower=0.0)
    merged["overlay_weight"] = merged.apply(
        lambda row: _gate_weight(row, gate, float(row["idle_weight"]), fraction),
        axis=1,
    )
    merged["total_return"] = merged["base_return"] + (
        merged["overlay_weight"].shift(1).fillna(0.0) * merged["overlay_return"]
    )

    equity = []
    current_equity = float(merged["equity"].iloc[0])
    peak = current_equity
    max_drawdown = 0.0
    for _, row in merged.iterrows():
        if equity:
            current_equity *= 1.0 + float(row["total_return"])
        peak = max(peak, current_equity)
        max_drawdown = max(max_drawdown, (peak - current_equity) / peak if peak else 0.0)
        equity.append(current_equity)

    merged["equity"] = [round(float(value), 2) for value in equity]
    total_return = (float(merged["equity"].iloc[-1]) / float(merged["equity"].iloc[0])) - 1.0
    spy_return = (float(merged["spy_close"].iloc[-1]) / float(merged["spy_close"].iloc[0])) - 1.0
    summary = {
        "total_return": round(total_return, 4),
        "benchmark_return": round(spy_return, 4),
        "alpha_vs_spy": round(total_return - spy_return, 4),
        "max_drawdown": round(float(max_drawdown), 4),
        "mean_overlay_weight": round(float(merged["overlay_weight"].mean()), 4),
        "mean_stock_exposure": round(float(merged["actual_exposure"].mean()), 4),
    }
    return merged, summary


def main():
    args = parse_args()
    base_daily_state_path = Path(args.base_daily_state)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    state = pd.read_csv(base_daily_state_path)
    state["trade_date"] = pd.to_datetime(state["trade_date"])

    warehouse = MarketDataWarehouse(args.db)
    try:
        overlay_symbols = [args.benchmark, "QQQ", "SMH", "SSO", "QLD", "TQQQ"]
        context_symbols = [args.benchmark, "QQQ", "IWM", "SMH", "^VIX"]
        required = sorted(set(overlay_symbols + context_symbols))
        available = set(warehouse.available_symbols())
        if args.refresh_data or not set(required).issubset(available):
            warehouse.fetch_and_store_daily_bars(required, args.start, args.end)

        benchmark_df = warehouse.get_daily_bars(args.benchmark, args.start, args.end)
        overlay_frames = {
            symbol: warehouse.get_daily_bars(symbol, args.start, args.end)
            for symbol in required
        }
        market_context_df = build_market_context({symbol: overlay_frames[symbol] for symbol in context_symbols})
        if not market_context_df.empty:
            state = state.merge(
                market_context_df.reset_index().rename(columns={"index": "trade_date"}),
                on="trade_date",
                how="left",
                suffixes=("", "_ctx"),
            )
            if "market_regime_ctx" in state.columns:
                state["market_regime"] = state["market_regime_ctx"]
                state = state.drop(columns=["market_regime_ctx"])
            if "market_confirmed_uptrend_ctx" in state.columns:
                state["market_confirmed_uptrend"] = state["market_confirmed_uptrend_ctx"]
                state = state.drop(columns=["market_confirmed_uptrend_ctx"])

        rows = []
        annual_outputs = {}
        detailed_outputs = {}
        for overlay_symbol in overlay_symbols:
            overlay_df = overlay_frames.get(overlay_symbol)
            if overlay_df is None or overlay_df.empty:
                continue
            for gate in ["always", "confirmed_only", "not_correction", "score_gte_5", "score_gte_6"]:
                for fraction in [0.5, 0.75, 1.0]:
                    simulated, summary = _simulate_overlay(
                        state,
                        benchmark_df=benchmark_df,
                        overlay_df=overlay_df,
                        gate=gate,
                        fraction=fraction,
                    )
                    annual_df = _annual_breakdown(
                        simulated,
                        trade_start=args.trade_start,
                        end_date=args.end,
                    )
                    row = {
                        **summary,
                        **_year_stats(annual_df),
                        "overlay_symbol": overlay_symbol,
                        "gate": gate,
                        "fraction": fraction,
                        "variant": f"{overlay_symbol}_{gate}_{int(fraction * 100)}",
                    }
                    rows.append(row)
                    annual_outputs[row["variant"]] = annual_df
                    detailed_outputs[row["variant"]] = simulated

        summary_df = pd.DataFrame(rows)
        ranking = summary_df.sort_values(
            [
                "positive_full_year_alpha_count",
                "min_full_year_alpha",
                "alpha_vs_spy",
                "total_return",
            ],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

        tag = args.end
        save(summary_df, results_dir / f"overlay_portable_alpha_{tag}_summary.csv")
        save(ranking, results_dir / f"overlay_portable_alpha_{tag}_ranking.csv")
        winners = ranking[
            (ranking["alpha_vs_spy"] > 0)
            & (ranking["positive_full_year_alpha_count"] >= 3)
        ].copy()
        if not winners.empty:
            save(winners, results_dir / f"overlay_portable_alpha_{tag}_winners.csv")

        for variant in ranking.head(5)["variant"].tolist():
            save(
                detailed_outputs[variant],
                results_dir / f"overlay_portable_alpha_{variant}_{tag}_daily_state.csv",
            )
            save(
                annual_outputs[variant],
                results_dir / f"overlay_portable_alpha_{variant}_{tag}_annual_returns.csv",
            )

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
                ]
            ].head(15).to_string(index=False)
        )
        if winners.empty:
            print("\nNo overlay variant cleared the multi-year consistency filter.")
        else:
            print("\nOverlay winners:")
            print(winners.to_string(index=False))
    finally:
        warehouse.close()


if __name__ == "__main__":
    main()
