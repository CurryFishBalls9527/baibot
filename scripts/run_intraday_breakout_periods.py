#!/usr/bin/env python3
"""Run the mechanical intraday breakout prototype across standard periods.

Research-only harness for establishing an intraday baseline before optimization.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradingagents.research.intraday_backtester import (  # noqa: E402
    IntradayBacktestConfig,
    IntradayBreakoutBacktester,
)
from tradingagents.research.intraday_universe import (  # noqa: E402
    IntradayUniverseFilterConfig,
    filter_symbols_by_tradability,
)


PERIODS = {
    "2023_2025": {
        "begin": "2023-01-01",
        "end": "2025-12-30",
        "db": "research_data/intraday_30m.duckdb",
        "db_broad": "research_data/intraday_30m_broad.duckdb",
        "db_15m": "research_data/intraday_15m.duckdb",
    },
    "2020": {
        "begin": "2020-01-01",
        "end": "2020-12-31",
        "db": "research_data/intraday_30m_2020.duckdb",
        "db_broad": "research_data/intraday_30m_broad_2020.duckdb",
        "db_15m": "research_data/intraday_15m_2020.duckdb",
    },
    "2018": {
        "begin": "2018-01-01",
        "end": "2018-12-31",
        "db": "research_data/intraday_30m_2018.duckdb",
        "db_broad": "research_data/intraday_30m_broad_2018.duckdb",
        "db_15m": "research_data/intraday_15m_2018.duckdb",
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Run intraday breakout prototype across periods")
    p.add_argument("--universe", default="research_data/spike_universe.json")
    p.add_argument("--broad", action="store_true", help="Use intraday_30m_broad*.duckdb per period")
    p.add_argument("--db-override", default=None, help="Override DB path for all selected periods (e.g. intraday_15m.duckdb)")
    p.add_argument("--symbols", nargs="*")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--interval", type=int, choices=[5, 15, 30], default=30)
    p.add_argument("--cash", type=float, default=100_000)
    p.add_argument("--max-positions", type=int, default=6)
    p.add_argument("--max-position-pct", type=float, default=0.10)
    p.add_argument("--stop-loss-pct", type=float, default=0.03)
    p.add_argument("--trail-stop-pct", type=float, default=0.04)
    p.add_argument("--use-atr-stops", action="store_true",
                   help="Use ATR-based stop and trail instead of fixed % stops.")
    p.add_argument("--atr-period-bars", type=int, default=14)
    p.add_argument("--atr-stop-multiplier", type=float, default=1.0)
    p.add_argument("--atr-trail-multiplier", type=float, default=2.0)
    p.add_argument("--opening-range-bars", type=int, default=1)
    p.add_argument("--min-opening-range-pct", type=float, default=0.0)
    p.add_argument("--min-breakout-distance-pct", type=float, default=0.0)
    p.add_argument("--require-above-prior-high", action="store_true")
    p.add_argument("--latest-entry-bar", type=int, default=None)
    p.add_argument("--use-midday-entry-window", action="store_true")
    p.add_argument("--midday-entry-earliest-bar", type=int, default=1)
    p.add_argument("--midday-entry-latest-bar", type=int, default=9)
    p.add_argument("--max-trades-per-symbol-day", type=int, default=1)
    p.add_argument("--min-volume-ratio", type=float, default=1.5)
    p.add_argument("--continuation-min-volume-ratio", type=float, default=None)
    p.add_argument("--continuation-max-vwap-distance-pct", type=float, default=None)
    p.add_argument("--continuation-latest-entry-bar", type=int, default=None)
    p.add_argument("--expansion-min-volume-ratio", type=float, default=None)
    p.add_argument("--expansion-min-breakout-distance-pct", type=float, default=None)
    p.add_argument("--expansion-min-vwap-distance-pct", type=float, default=None)
    p.add_argument("--expansion-latest-entry-bar", type=int, default=None)
    p.add_argument("--expansion-failed-follow-through-bars", type=int, default=None)
    p.add_argument("--expansion-failed-follow-through-min-return-pct", type=float, default=None)
    p.add_argument("--use-expansion-confirmation-entry", action="store_true")
    p.add_argument("--expansion-confirmation-max-pullback-pct", type=float, default=None)
    p.add_argument("--expansion-confirmation-reclaim-buffer-pct", type=float, default=0.0)
    p.add_argument("--expansion-confirmation-max-bars-after-signal", type=int, default=None)
    p.add_argument("--disable-continuation-setup", action="store_true")
    p.add_argument("--disable-overextended-setup", action="store_true")
    p.add_argument("--disable-expansion-setup", action="store_true")
    p.add_argument("--allow-pullback-vwap", action="store_true")
    p.add_argument("--pullback-vwap-touch-tolerance-pct", type=float, default=0.002)
    p.add_argument("--pullback-vwap-touch-lookback-bars", type=int, default=4)
    p.add_argument("--pullback-vwap-reclaim-min-pct", type=float, default=0.001)
    p.add_argument("--pullback-vwap-min-session-trend-pct", type=float, default=0.005)
    p.add_argument("--pullback-vwap-min-volume-ratio", type=float, default=1.2)
    p.add_argument("--pullback-vwap-earliest-entry-bar", type=int, default=3)
    p.add_argument("--pullback-vwap-latest-entry-bar", type=int, default=10)
    p.add_argument("--pullback-vwap-min-distance-from-or-high-pct", type=float, default=None)
    p.add_argument("--pullback-vwap-max-position-pct", type=float, default=None)
    p.add_argument("--allow-gap-reclaim-long", action="store_true")
    p.add_argument("--gap-reclaim-min-gap-down-pct", type=float, default=0.015)
    p.add_argument("--gap-reclaim-max-gap-down-pct", type=float, default=0.06)
    p.add_argument("--gap-reclaim-min-reclaim-fraction", type=float, default=0.5)
    p.add_argument("--gap-reclaim-min-volume-ratio", type=float, default=1.3)
    p.add_argument("--gap-reclaim-earliest-entry-bar", type=int, default=1)
    p.add_argument("--gap-reclaim-latest-entry-bar", type=int, default=4)
    p.add_argument("--gap-reclaim-disable-above-session-open", action="store_true")
    p.add_argument("--gap-reclaim-max-position-pct", type=float, default=None)
    p.add_argument("--gap-reclaim-trail-stop-pct", type=float, default=None)
    p.add_argument("--gap-reclaim-trail-activation-return-pct", type=float, default=None)
    p.add_argument("--allow-nr4-breakout", action="store_true")
    p.add_argument("--nr4-lookback-days", type=int, default=4)
    p.add_argument("--nr4-earliest-entry-bar", type=int, default=1)
    p.add_argument("--nr4-latest-entry-bar", type=int, default=12)
    p.add_argument("--nr4-min-volume-ratio", type=float, default=1.3)
    p.add_argument("--nr4-min-breakout-distance-pct", type=float, default=0.0)
    p.add_argument("--nr4-max-position-pct", type=float, default=None)
    p.add_argument("--allow-orb-breakout", action="store_true")
    p.add_argument("--orb-range-bars", type=int, default=2)
    p.add_argument("--orb-min-volume-ratio", type=float, default=1.5)
    p.add_argument("--orb-min-breakout-distance-pct", type=float, default=0.001)
    p.add_argument("--orb-earliest-entry-bar", type=int, default=2)
    p.add_argument("--orb-latest-entry-bar", type=int, default=10)
    p.add_argument("--orb-disable-above-vwap", action="store_true")
    p.add_argument("--execution-half-spread-bps", type=float, default=0.0)
    p.add_argument("--execution-stop-slippage-bps", type=float, default=0.0)
    p.add_argument("--execution-impact-coeff-bps", type=float, default=0.0)
    p.add_argument("--allow-relative-volume-filter", action="store_true")
    p.add_argument("--relative-volume-lookback-days", type=int, default=20)
    p.add_argument("--relative-volume-top-k", type=int, default=20)
    p.add_argument("--benchmark-symbol", default="SPY")
    p.add_argument("--min-relative-strength-pct", type=float, default=None)
    p.add_argument("--min-entry-strength-breakout-pct", type=float, default=None)
    p.add_argument("--min-entry-strength-vwap-distance-pct", type=float, default=None)
    p.add_argument("--allow-below-vwap", action="store_true")
    p.add_argument("--no-flatten", action="store_true")
    p.add_argument("--daily-trend-filter", action="store_true")
    p.add_argument("--daily-trend-sma", type=int, default=20)
    p.add_argument("--daily-db", default="research_data/market_data.duckdb")
    p.add_argument("--apply-tradability-filter", action="store_true")
    p.add_argument("--tradability-lookback-days", type=int, default=60)
    p.add_argument("--min-median-close", type=float, default=20.0)
    p.add_argument("--min-median-dollar-volume", type=float, default=50000000.0)
    p.add_argument("--min-trading-days", type=int, default=40)
    p.add_argument("--periods", nargs="*", default=["2023_2025", "2020", "2018"])
    p.add_argument("--out", default="results/intraday_breakout/period_matrix.json")
    return p.parse_args()


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        ).strip()
    except Exception:
        return None


def load_symbols(args) -> list[str]:
    if args.symbols:
        symbols = args.symbols
    else:
        payload = json.loads(Path(args.universe).read_text())
        symbols = payload["symbols"] if isinstance(payload, dict) else payload
    if args.limit:
        symbols = symbols[: args.limit]
    return symbols


def build_config(args) -> IntradayBacktestConfig:
    return IntradayBacktestConfig(
        initial_cash=args.cash,
        max_positions=args.max_positions,
        max_position_pct=args.max_position_pct,
        stop_loss_pct=args.stop_loss_pct,
        trail_stop_pct=args.trail_stop_pct,
        use_atr_stops=args.use_atr_stops,
        atr_period_bars=args.atr_period_bars,
        atr_stop_multiplier=args.atr_stop_multiplier,
        atr_trail_multiplier=args.atr_trail_multiplier,
        opening_range_bars=args.opening_range_bars,
        min_opening_range_pct=args.min_opening_range_pct,
        min_breakout_distance_pct=args.min_breakout_distance_pct,
        require_above_prior_high=args.require_above_prior_high,
        latest_entry_bar_in_session=args.latest_entry_bar,
        use_midday_entry_window=args.use_midday_entry_window,
        midday_entry_earliest_bar=args.midday_entry_earliest_bar,
        midday_entry_latest_bar=args.midday_entry_latest_bar,
        max_trades_per_symbol_per_day=args.max_trades_per_symbol_day,
        min_volume_ratio=args.min_volume_ratio,
        continuation_min_volume_ratio=args.continuation_min_volume_ratio,
        continuation_max_distance_from_vwap_pct=args.continuation_max_vwap_distance_pct,
        continuation_latest_entry_bar_in_session=args.continuation_latest_entry_bar,
        expansion_min_volume_ratio=args.expansion_min_volume_ratio,
        expansion_min_breakout_distance_pct=args.expansion_min_breakout_distance_pct,
        expansion_min_distance_from_vwap_pct=args.expansion_min_vwap_distance_pct,
        expansion_latest_entry_bar_in_session=args.expansion_latest_entry_bar,
        expansion_failed_follow_through_bars=args.expansion_failed_follow_through_bars,
        expansion_failed_follow_through_min_return_pct=args.expansion_failed_follow_through_min_return_pct,
        use_expansion_confirmation_entry=args.use_expansion_confirmation_entry,
        expansion_confirmation_max_pullback_pct=args.expansion_confirmation_max_pullback_pct,
        expansion_confirmation_reclaim_buffer_pct=args.expansion_confirmation_reclaim_buffer_pct,
        expansion_confirmation_max_bars_after_signal=args.expansion_confirmation_max_bars_after_signal,
        allow_continuation_setup=not args.disable_continuation_setup,
        allow_overextended_setup=not args.disable_overextended_setup,
        allow_expansion_setup=not args.disable_expansion_setup,
        allow_pullback_vwap=args.allow_pullback_vwap,
        pullback_vwap_touch_tolerance_pct=args.pullback_vwap_touch_tolerance_pct,
        pullback_vwap_touch_lookback_bars=args.pullback_vwap_touch_lookback_bars,
        pullback_vwap_reclaim_min_pct=args.pullback_vwap_reclaim_min_pct,
        pullback_vwap_min_session_trend_pct=args.pullback_vwap_min_session_trend_pct,
        pullback_vwap_min_volume_ratio=args.pullback_vwap_min_volume_ratio,
        pullback_vwap_earliest_entry_bar=args.pullback_vwap_earliest_entry_bar,
        pullback_vwap_latest_entry_bar=args.pullback_vwap_latest_entry_bar,
        pullback_vwap_min_distance_from_or_high_pct=args.pullback_vwap_min_distance_from_or_high_pct,
        pullback_vwap_max_position_pct=args.pullback_vwap_max_position_pct,
        allow_gap_reclaim_long=args.allow_gap_reclaim_long,
        gap_reclaim_min_gap_down_pct=args.gap_reclaim_min_gap_down_pct,
        gap_reclaim_max_gap_down_pct=args.gap_reclaim_max_gap_down_pct,
        gap_reclaim_min_reclaim_fraction=args.gap_reclaim_min_reclaim_fraction,
        gap_reclaim_min_volume_ratio=args.gap_reclaim_min_volume_ratio,
        gap_reclaim_earliest_entry_bar=args.gap_reclaim_earliest_entry_bar,
        gap_reclaim_latest_entry_bar=args.gap_reclaim_latest_entry_bar,
        gap_reclaim_require_above_session_open=not args.gap_reclaim_disable_above_session_open,
        gap_reclaim_max_position_pct=args.gap_reclaim_max_position_pct,
        gap_reclaim_trail_stop_pct=args.gap_reclaim_trail_stop_pct,
        gap_reclaim_trail_activation_return_pct=args.gap_reclaim_trail_activation_return_pct,
        allow_nr4_breakout=args.allow_nr4_breakout,
        nr4_lookback_days=args.nr4_lookback_days,
        nr4_earliest_entry_bar=args.nr4_earliest_entry_bar,
        nr4_latest_entry_bar=args.nr4_latest_entry_bar,
        nr4_min_volume_ratio=args.nr4_min_volume_ratio,
        nr4_min_breakout_distance_pct=args.nr4_min_breakout_distance_pct,
        nr4_max_position_pct=args.nr4_max_position_pct,
        allow_orb_breakout=args.allow_orb_breakout,
        orb_range_bars=args.orb_range_bars,
        orb_min_volume_ratio=args.orb_min_volume_ratio,
        orb_min_breakout_distance_pct=args.orb_min_breakout_distance_pct,
        orb_earliest_entry_bar=args.orb_earliest_entry_bar,
        orb_latest_entry_bar=args.orb_latest_entry_bar,
        orb_require_above_vwap=not args.orb_disable_above_vwap,
        execution_half_spread_bps=args.execution_half_spread_bps,
        execution_stop_slippage_bps=args.execution_stop_slippage_bps,
        execution_impact_coeff_bps=args.execution_impact_coeff_bps,
        allow_relative_volume_filter=args.allow_relative_volume_filter,
        relative_volume_lookback_days=args.relative_volume_lookback_days,
        relative_volume_top_k=args.relative_volume_top_k,
        benchmark_symbol=args.benchmark_symbol,
        min_relative_strength_pct=args.min_relative_strength_pct,
        min_entry_strength_breakout_pct=args.min_entry_strength_breakout_pct,
        min_entry_strength_vwap_distance_pct=args.min_entry_strength_vwap_distance_pct,
        require_above_vwap=not args.allow_below_vwap,
        flatten_at_close=not args.no_flatten,
        interval_minutes=args.interval,
        daily_trend_filter=args.daily_trend_filter,
        daily_trend_sma=args.daily_trend_sma,
        daily_db_path=args.daily_db,
    )


def main():
    args = parse_args()
    symbols = load_symbols(args)
    cfg = build_config(args)
    bt = IntradayBreakoutBacktester(cfg)

    rows = []
    details = {}
    commit_sha = git_sha()
    for period_name in args.periods:
        period = PERIODS[period_name]
        run_symbols = list(symbols)
        tradability_diagnostics = []
        if args.apply_tradability_filter:
            run_symbols, diagnostics = filter_symbols_by_tradability(
                run_symbols,
                as_of_date=period["end"],
                config=IntradayUniverseFilterConfig(
                    daily_db_path=args.daily_db,
                    lookback_days=args.tradability_lookback_days,
                    min_median_close=args.min_median_close,
                    min_median_dollar_volume=args.min_median_dollar_volume,
                    min_trading_days=args.min_trading_days,
                ),
            )
            tradability_diagnostics = diagnostics.to_dict("records") if not diagnostics.empty else []
        if args.db_override:
            db_path = args.db_override
        elif args.interval == 15 and "db_15m" in period:
            db_path = period["db_15m"]
        elif args.broad:
            db_path = period["db_broad"]
        else:
            db_path = period["db"]
        result = bt.backtest_portfolio(run_symbols, period["begin"], period["end"], db_path)
        rows.append({"period": period_name, **result.summary})
        details[period_name] = {
            "metadata": {
                **result.metadata,
                "period": period_name,
                "git_commit": commit_sha,
                "input_symbols": symbols,
                "run_symbols": run_symbols,
                "apply_tradability_filter": bool(args.apply_tradability_filter),
            },
            "summary": result.summary,
            "tradability_diagnostics": tradability_diagnostics,
            "symbol_summary": result.symbol_summary.to_dict("records") if not result.symbol_summary.empty else [],
            "setup_summary": result.setup_summary.to_dict("records") if not result.setup_summary.empty else [],
            "candidate_log": result.candidate_log.to_dict("records") if not result.candidate_log.empty else [],
            "trades": result.trades.to_dict("records") if not result.trades.empty else [],
            "equity_curve": result.equity_curve.to_dict("records") if not result.equity_curve.empty else [],
            "daily_state": result.daily_state.to_dict("records") if not result.daily_state.empty else [],
        }

    print()
    print("=" * 96)
    print("  INTRADAY MECHANICAL BREAKOUT — PERIOD MATRIX")
    print("=" * 96)
    for row in rows:
        print(
            f"  {row['period']:<10}  return {row['total_return_pct']:>7.2f}%"
            f"  dd {row['max_drawdown_pct']:>6.2f}%"
            f"  trades {row['total_trades']:>5}"
            f"  win {row['win_rate']*100:>5.1f}%"
            f"  avg bars {row['avg_bars_held']:>4.1f}"
        )
    print("=" * 96)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "strategy_name": "intraday_mechanical_breakout",
                    "strategy_version": "prototype_v2",
                    "periods": args.periods,
                    "symbols": symbols,
                    "git_commit": commit_sha,
                    "config": cfg.__dict__,
                },
                "rows": rows,
                "details": details,
            },
            indent=2,
        )
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
